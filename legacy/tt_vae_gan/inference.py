"""Legacy VAE-GAN proof-of-concept - inference script (superseded).

Part of the early trumpet-to-violin timbre-transfer experiment we built before
switching to the diffusion model. Kept for reference and learning only; the
final pipeline does not use it. See legacy/README for the full context.
"""

# Forked from ebadawy/voice_conversion/src/inference.py
# Inference script for VAE-GAN timbre transfer with sliding-window overlap

import argparse
import os
import glob
import numpy as np
import sys
from tqdm import tqdm

from torch.autograd import Variable
import torch
import soundfile as sf
import skimage.metrics
from statistics import mean

# Use relative imports when run as module, absolute when run as script
try:
    from .models import Encoder, Generator, ResidualBlock
    from .utils import (preprocess_wav, melspectrogram, to_numpy,
                        plot_mel_transfer_infer, reconstruct_waveform)
    from .params_config import get_params
except ImportError:
    from models import Encoder, Generator, ResidualBlock
    from utils import (preprocess_wav, melspectrogram, to_numpy,
                       plot_mel_transfer_infer, reconstruct_waveform)
    from params_config import get_params

# Dynamic sample_rate from active params (pipeline=22050Hz, original=16000Hz)
sample_rate = get_params().sample_rate


def load_inference_model(model_name, epoch, trg_id, src_id=None,
                         img_height=128, img_width=128, channels=1,
                         n_downsample=2, dim=32, device=None):
    """
    Load a trained VAE-GAN model for inference.
    
    Args:
        model_name: Name of the saved model
        epoch: Training epoch to load
        trg_id: Target generator ID (e.g., '1' for G1=trumpet, '2' for G2=violin)
        src_id: Source generator ID for SSIM evaluation (optional)
        img_height: Mel bin count (128 for original, 80 for pipeline)
        img_width: Frame count per window (128)
        channels: Number of channels (1 for mono mel)
        n_downsample: Number of downsampling layers in encoder
        dim: Base filter count
        device: torch device
    
    Returns:
        dict with 'encoder', 'G_trg', and optionally 'G_src'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cuda = device.type == 'cuda'
    shared_dim = dim * 2 ** n_downsample

    # Validate checkpoint files exist
    encoder_path = "saved_models/%s/encoder_%02d.pth" % (model_name, epoch)
    trg_path = "saved_models/%s/G%s_%02d.pth" % (model_name, trg_id, epoch)
    assert os.path.exists(encoder_path), \
        f'Encoder checkpoint not found: {encoder_path}'
    assert os.path.exists(trg_path), \
        f'Target generator checkpoint not found: {trg_path}'

    # Initialize models
    encoder = Encoder(dim=dim, in_channels=channels, n_downsample=n_downsample)
    G_trg = Generator(dim=dim, out_channels=channels,
                      n_upsample=n_downsample,
                      shared_block=ResidualBlock(features=shared_dim))

    G_src = None
    if src_id:
        src_path = "saved_models/%s/G%s_%02d.pth" % (model_name, src_id, epoch)
        assert os.path.exists(src_path), \
            f'Source generator checkpoint not found: {src_path}'
        G_src = Generator(dim=dim, out_channels=channels,
                          n_upsample=n_downsample,
                          shared_block=ResidualBlock(features=shared_dim))

    if cuda:
        encoder = encoder.cuda()
        G_trg = G_trg.cuda()
        if G_src:
            G_src = G_src.cuda()

    # Load weights
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    G_trg.load_state_dict(torch.load(trg_path, map_location=device))
    if G_src:
        G_src.load_state_dict(torch.load(src_path, map_location=device))

    # Set to eval mode
    encoder.eval()
    G_trg.eval()
    if G_src:
        G_src.eval()

    return {
        'encoder': encoder,
        'G_trg': G_trg,
        'G_src': G_src,
        'device': device,
    }


def infer_window(model_dict, S, img_height, img_width, channels=1):
    """
    Inference on a single spectrogram window.
    
    Args:
        model_dict: dict from load_inference_model()
        S: numpy array of shape (img_height, img_width), [0,1] normalized
        img_height: expected mel bin count
        img_width: expected frame count
        channels: number of channels
    
    Returns:
        dict with 'fake' and optionally 'recon', 'cyclic'
    """
    device = model_dict['device']
    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.Tensor

    encoder = model_dict['encoder']
    G_trg = model_dict['G_trg']
    G_src = model_dict['G_src']

    S_tensor = torch.from_numpy(S).view(1, channels, img_height, img_width)
    X = Variable(S_tensor.type(Tensor))

    ret = {}
    with torch.no_grad():
        mu, Z = encoder(X)
        fake_X = G_trg(Z)
        ret['fake'] = to_numpy(fake_X)

        if G_src:
            recon_X = G_src(Z)
            ret['recon'] = to_numpy(recon_X)

            mu_, Z_ = encoder(fake_X)
            cyclic_X = G_src(Z_)
            ret['cyclic'] = to_numpy(cyclic_X)

    return ret


def audio_infer(wav_path, model_dict, img_height=128, img_width=128,
                n_overlap=4, channels=1, output_root=None, plot=True):
    """
    Full audio inference with sliding window and overlap averaging.
    
    Args:
        wav_path: Path to input WAV file
        model_dict: dict from load_inference_model()
        img_height: mel bin count
        img_width: frame count per window
        n_overlap: overlap factor (4 = 75% overlap)
        channels: number of channels
        output_root: directory to save outputs
        plot: whether to save spectrogram plot
    
    Returns:
        dict with 'spect_src', 'spect_trg', 'wav_trg'
    """
    # Load audio and preprocess
    sample = preprocess_wav(wav_path)
    spect_src = melspectrogram(sample)

    # Pad for consistent overlap
    spect_src = np.pad(spect_src, ((0, 0), (img_width, img_width)), 'constant')
    spect_trg = np.zeros(spect_src.shape)
    spect_recon = np.zeros(spect_src.shape)
    spect_cyclic = np.zeros(spect_src.shape)

    length = spect_src.shape[1]
    hop = img_width // n_overlap

    G_src = model_dict['G_src']

    for i in tqdm(range(0, length, hop), desc='Inference'):
        x = i + img_width

        # Get cropped spectro of right dims
        if x <= length:
            S = spect_src[:, i:x]
        else:
            S = spect_src[:, i:]
            S = np.pad(S, ((0, 0), (x - length, 0)), 'constant')

        ret = infer_window(model_dict, S, img_height, img_width, channels)
        T = ret['fake']
        R = ret.get('recon')
        C = ret.get('cyclic')

        # Add parts with overlap averaging
        for j in range(0, img_width, hop):
            y = j + hop
            if i + y > length:
                break

            t = T[:, j:y]
            spect_trg[:, i + j:i + y] += t / n_overlap

            if G_src:
                spect_recon[:, i + j:i + y] += R[:, j:y] / n_overlap
                spect_cyclic[:, i + j:i + y] += C[:, j:y] / n_overlap

    # Remove initial padding
    spect_src = spect_src[:, img_width:-img_width]
    spect_trg = spect_trg[:, img_width:-img_width]
    if G_src:
        spect_recon = spect_recon[:, img_width:-img_width]
        spect_cyclic = spect_cyclic[:, img_width:-img_width]

    result = {
        'spect_src': spect_src,
        'spect_trg': spect_trg,
    }

    # Save outputs if output_root specified
    if output_root:
        f = os.path.basename(wav_path)
        wavname = os.path.splitext(f)[0]
        fname = 'transferred_%s' % wavname

        if plot:
            os.makedirs(os.path.join(output_root, 'plots'), exist_ok=True)
            plot_mel_transfer_infer(
                os.path.join(output_root, 'plots', '%s.png' % fname),
                spect_src, spect_trg)

        # Reconstruct with Griffin-Lim
        print('Reconstructing with Griffin-Lim...')
        wav_trg = reconstruct_waveform(spect_trg)
        result['wav_trg'] = wav_trg

        os.makedirs(os.path.join(output_root, 'gen'), exist_ok=True)
        os.makedirs(os.path.join(output_root, 'ref'), exist_ok=True)
        sf.write(os.path.join(output_root, 'gen', '%s_gen.wav' % fname),
                 wav_trg, sample_rate)
        sf.write(os.path.join(output_root, 'ref', '%s_ref.wav' % fname),
                 sample, sample_rate)

    return result


# ─── SSIM evaluation ────────────────────────────────────────────────────────────

def compute_ssim(spect_src, spect_recon):
    """Compute Structural Similarity Index between spectrograms."""
    return skimage.metrics.structural_similarity(
        spect_src, spect_recon, data_range=1)


# ─── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=99,
                        help="saved version based on epoch to test from")
    parser.add_argument("--model_name", type=str, required=True,
                        help="name of the model")
    parser.add_argument("--trg_id", type=str, required=True,
                        help="id of the generator for target domain")
    parser.add_argument("--src_id", type=str, default=None,
                        help="id of the generator for source domain "
                             "(specify for recon/cyclic SSIM evaluation)")
    parser.add_argument("--wav", type=str, default=None,
                        help="path to wav file for input to transfer")
    parser.add_argument("--wavdir", type=str, default=None,
                        help="path to directory of wav files")
    parser.add_argument("--plot", type=int, default=1,
                        help="plot spectrograms (disable with -1)")
    parser.add_argument("--n_overlap", type=int, default=4,
                        help="number of overlaps per slice")
    parser.add_argument("--img_height", type=int, default=128,
                        help="size of image height (mel bins)")
    parser.add_argument("--img_width", type=int, default=128,
                        help="size of image width (frames)")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--n_downsample", type=int, default=2,
                        help="number downsampling layers in encoder")
    parser.add_argument("--dim", type=int, default=32,
                        help="number of filters in first encoder layer")

    opt = parser.parse_args()
    print(opt)

    assert opt.wav or opt.wavdir, \
        'Please specify an input wav file or directory'
    assert not (opt.wav and opt.wavdir), \
        'Cannot specify both --wav and --wavdir, choose one'

    # Load model
    model_dict = load_inference_model(
        opt.model_name, opt.epoch, opt.trg_id, opt.src_id,
        opt.img_height, opt.img_width, opt.channels,
        opt.n_downsample, opt.dim)

    # Prepare output directory
    root = 'out_infer/%s_%d_G%s' % (opt.model_name, opt.epoch, opt.trg_id)
    if opt.src_id:
        root += '_S%s' % opt.src_id

    ssim_recon = []
    ssim_cyclic = []

    # Run inference
    wav_files = []
    if opt.wav:
        wav_files = [opt.wav]
    elif opt.wavdir:
        wav_files = glob.glob(os.path.join(opt.wavdir, '*.wav'))

    for i, wav_path in enumerate(wav_files):
        print('[File %d/%d] %s' % (i + 1, len(wav_files), wav_path))
        audio_infer(
            wav_path, model_dict,
            img_height=opt.img_height,
            img_width=opt.img_width,
            n_overlap=opt.n_overlap,
            channels=opt.channels,
            output_root=root,
            plot=(opt.plot != -1))

    # Display average SSIM
    if opt.src_id and ssim_recon:
        print('Average SSIM for recon: %0.2f' % mean(ssim_recon))
        print('Average SSIM for cyclic: %0.2f' % mean(ssim_cyclic))
