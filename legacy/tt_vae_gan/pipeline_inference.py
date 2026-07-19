"""
Pipeline Inference Bridge
==========================
Connects our DSP preprocessing pipeline (dsp_preprocessor.py) to the
VAE-GAN timbre transfer model (models/tt_vae_gan/).

Key responsibilities:
1. Convert our pipeline mels [-1,1] → model mels [0,1]
2. Run sliding-window inference through the VAE-GAN
3. Convert model output [0,1] → pipeline mels [-1,1]
4. Optionally reconstruct audio via BigVGAN vocoder

Normalization mapping:
    Pipeline (dsp_preprocessor):  [-1, 1]  (min-max per-segment)
    VAE-GAN model:                [0, 1]   (global normalize: clip((S - min_level_db) / -min_level_db, 0, 1))

Usage:
    python -m models.tt_vae_gan.pipeline_inference \\
        --input_mel path/to/mel.npy \\
        --model_name initial --epoch 490 --trg_id 2
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.autograd import Variable

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tt_vae_gan.models import Encoder, Generator, ResidualBlock
from models.tt_vae_gan.utils import to_numpy, plot_mel_transfer_infer
from models.tt_vae_gan.params_config import get_params as _get_params


# ─── Normalization converters ───────────────────────────────────────────────────

def pipeline_to_model(mel_pipeline: np.ndarray) -> np.ndarray:
    """
    Convert pipeline [-1, 1] normalized mel to model [0, 1] normalization.
    
    Pipeline: normalized = 2 * (S - min) / (max - min) - 1
    Model:    normalized = clip((S - min_level_db) / -min_level_db, 0, 1)
    
    Simple linear mapping: model_mel = (pipeline_mel + 1) / 2
    """
    return (mel_pipeline + 1.0) / 2.0


def model_to_pipeline(mel_model: np.ndarray) -> np.ndarray:
    """
    Convert model [0, 1] normalized mel back to pipeline [-1, 1] normalization.
    
    Simple linear mapping: pipeline_mel = 2 * model_mel - 1
    """
    return 2.0 * mel_model - 1.0


def model_to_db(mel_01: np.ndarray, min_level_db: float = -100) -> np.ndarray:
    """
    Convert model [0, 1] mel to dB-scale magnitude.
    
    Reverses: mel_01 = clip((S_db - min_level_db) / -min_level_db, 0, 1)
    → S_db = mel_01 * -min_level_db + min_level_db = mel_01 * 100 - 100
    """
    return np.clip(mel_01, 0, 1) * (-min_level_db) + min_level_db


def db_to_linear(S_db: np.ndarray) -> np.ndarray:
    """Convert dB-scale mel to linear amplitude."""
    return np.power(10.0, S_db * 0.05)


# ─── Vocoder synthesis ──────────────────────────────────────────────────────────

def reconstruct_with_bigvgan(mel_01: np.ndarray,
                             vocoder_name: str = 'bigvgan_22k') -> np.ndarray:
    """
    Synthesize audio from model-normalized mel [0,1] using BigVGAN.
    
    Chain: mel [0,1] → dB → linear amplitude → BigVGAN native log-mel → audio
    
    Args:
        mel_01: Model output in [0, 1], shape (n_mels, n_frames)
        vocoder_name: 'bigvgan_22k' or 'bigvgan_24k'
    
    Returns:
        audio: numpy array in [-1, 1]
    """
    try:
        from postprocessing.vocoder_factory import create_vocoder
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT))
        from postprocessing.vocoder_factory import create_vocoder
    
    vocoder = create_vocoder(vocoder_name)
    
    # Convert model mel to linear-scale amplitude
    S_db = model_to_db(mel_01)
    S_linear = db_to_linear(S_db)
    
    # BigVGAN expects log-magnitude mel: log(clamp(S_linear, min=1e-5))
    # Compute the log mel as BigVGAN's native format
    log_mel = np.log(np.maximum(S_linear, 1e-5))
    
    # Convert to tensor: (1, n_mels, frames)
    mel_tensor = torch.from_numpy(log_mel).float().unsqueeze(0)
    
    audio = vocoder.mel_to_audio(mel_tensor)
    return audio


def reconstruct_with_griffinlim(mel_01: np.ndarray,
                                n_iter: int = 32) -> np.ndarray:
    """
    Synthesize audio from model-normalized mel [0,1] using Griffin-Lim.
    
    Simple but lower quality than neural vocoder.
    
    Args:
        mel_01: Model output in [0, 1], shape (n_mels, n_frames)
        n_iter: Griffin-Lim iterations
    
    Returns:
        audio: numpy array
    """
    from models.tt_vae_gan.utils import reconstruct_waveform
    return reconstruct_waveform(mel_01, n_iter=n_iter)


# ─── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_name: str, epoch: int, trg_id: str,
               src_id: str = None,
               img_height: int = 128, img_width: int = 128,
               channels: int = 1, n_downsample: int = 2, dim: int = 32,
               models_dir: str = None):
    """
    Load a trained VAE-GAN model.
    
    Args:
        model_name: Name of the saved model (e.g., 'initial')
        epoch: Training epoch to load (e.g., 490 for URMP pretrained)
        trg_id: Target generator ID ('1'=trumpet, '2'=violin for URMP)
        src_id: Source generator ID for cyclic reconstruction (optional)
        img_height: Mel bins (128 for original URMP, 80 for our pipeline)
        img_width: Frame count per inference window
        channels: Image channels (1 for mono mel)
        n_downsample: Encoder downsampling layers
        dim: Base filter count
        models_dir: Override for saved_models directory location
    
    Returns:
        dict with model components and metadata
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    
    shared_dim = dim * 2 ** n_downsample
    
    # Build paths
    encoder_path = os.path.join(models_dir, model_name,
                                f"encoder_{epoch:02d}.pth")
    trg_path = os.path.join(models_dir, model_name,
                            f"G{trg_id}_{epoch:02d}.pth")
    
    for p, name in [(encoder_path, 'Encoder'), (trg_path, 'Target Generator')]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{name} checkpoint not found: {p}\n"
                f"Download pretrained weights from: "
                f"https://drive.google.com/drive/folders/1Nq3tKE-kcoMOw5AYEa0qWddxwbUYL8aA")
    
    # Initialize
    encoder = Encoder(dim=dim, in_channels=channels, n_downsample=n_downsample)
    G_trg = Generator(dim=dim, out_channels=channels,
                      n_upsample=n_downsample,
                      shared_block=ResidualBlock(features=shared_dim))
    
    G_src = None
    if src_id:
        src_path = os.path.join(models_dir, model_name,
                                f"G{src_id}_{epoch:02d}.pth")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source generator not found: {src_path}")
        G_src = Generator(dim=dim, out_channels=channels,
                          n_upsample=n_downsample,
                          shared_block=ResidualBlock(features=shared_dim))
    
    # Move to device
    encoder = encoder.to(device)
    G_trg = G_trg.to(device)
    if G_src:
        G_src = G_src.to(device)
    
    # Load weights
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    G_trg.load_state_dict(torch.load(trg_path, map_location=device))
    if G_src:
        G_src.load_state_dict(torch.load(src_path, map_location=device))
    
    # Eval mode
    encoder.eval()
    G_trg.eval()
    if G_src:
        G_src.eval()
    
    return {
        'encoder': encoder,
        'G_trg': G_trg,
        'G_src': G_src,
        'device': device,
        'img_height': img_height,
        'img_width': img_width,
        'channels': channels,
    }


# ─── Sliding-window inference ───────────────────────────────────────────────────

def infer_mel(mel_01: np.ndarray, model_dict: dict,
              n_overlap: int = 4,
              return_timing: bool = False):
    """
    Run sliding-window inference on a full mel spectrogram.
    
    The input mel must be in [0, 1] range (model normalization).
    Output is also in [0, 1] range.
    
    Args:
        mel_01: Mel spectrogram in [0, 1], shape (n_mels, n_frames)
        model_dict: dict from load_model()
        n_overlap: Overlap factor (4 = 75% overlap, hop = width/4)
        return_timing: If True, return (mel, timing_info) where timing_info
            is a dict with per-patch latency measurements.
    
    Returns:
        If return_timing=False: Transferred mel spectrogram in [0, 1]
        If return_timing=True:  (transferred_mel, timing_info) where
            timing_info = {
                'patch_latencies': list of float (seconds per patch),
                'total_time': float (total inference time in seconds),
                'n_patches': int,
                'patch_width_frames': int,
                'hop_frames': int,
                'input_frames': int,
            }
    """
    import time as _time
    
    device = model_dict['device']
    encoder = model_dict['encoder']
    G_trg = model_dict['G_trg']
    img_height = model_dict['img_height']
    img_width = model_dict['img_width']
    channels = model_dict['channels']
    
    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
    
    # Validate dimensions
    assert mel_01.shape[0] == img_height, \
        f"Mel height mismatch: got {mel_01.shape[0]}, expected {img_height}. " \
        f"Use img_height=128 for pretrained URMP or img_height=80 for pipeline."
    
    # Pad for consistent overlap
    padded = np.pad(mel_01, ((0, 0), (img_width, img_width)), mode='constant')
    output = np.zeros_like(padded)
    
    length = padded.shape[1]
    hop = img_width // n_overlap
    
    patch_latencies = []
    t_total_start = _time.perf_counter()
    
    with torch.no_grad():
        for i in tqdm(range(0, length, hop), desc='VAE-GAN inference',
                      leave=False):
            t_patch_start = _time.perf_counter()
            x = i + img_width
            
            # Extract window
            if x <= length:
                S = padded[:, i:x]
            else:
                S = padded[:, i:]
                S = np.pad(S, ((0, 0), (0, x - length)), mode='constant')
            
            # Forward pass
            S_tensor = torch.from_numpy(S).float()
            S_tensor = S_tensor.view(1, channels, img_height, img_width)
            # Rescale [0,1] → [-1,1] to match training data range
            S_tensor = S_tensor * 2.0 - 1.0
            X = S_tensor.to(device)
            
            mu, Z = encoder(X)
            fake_X = G_trg(Z)
            T = to_numpy(fake_X)
            # Rescale Tanh output [-1,1] → [0,1]
            T = (T + 1.0) / 2.0
            
            # Overlap-add
            for j in range(0, img_width, hop):
                y = j + hop
                if i + y > length:
                    break
                output[:, i + j:i + y] += T[:, j:y] / n_overlap
            
            patch_latencies.append(_time.perf_counter() - t_patch_start)
    
    t_total = _time.perf_counter() - t_total_start
    
    # Remove padding
    output = output[:, img_width:-img_width]
    result = np.clip(output, 0, 1)
    
    if return_timing:
        timing_info = {
            'patch_latencies': patch_latencies,
            'total_time': t_total,
            'n_patches': len(patch_latencies),
            'patch_width_frames': img_width,
            'hop_frames': hop,
            'input_frames': mel_01.shape[1],
        }
        return result, timing_info
    return result


# ─── Full pipeline inference ────────────────────────────────────────────────────

def transfer_mel_pipeline(mel_pipeline: np.ndarray, model_dict: dict,
                          n_overlap: int = 4) -> np.ndarray:
    """
    End-to-end timbre transfer for a pipeline mel spectrogram.
    
    Handles normalization conversion automatically:
        pipeline [-1,1] → model [0,1] → inference → model [0,1] → pipeline [-1,1]
    
    Args:
        mel_pipeline: Mel from our DSP pipeline, shape (n_mels, n_frames), [-1,1]
        model_dict: dict from load_model()
        n_overlap: Overlap factor
    
    Returns:
        Transferred mel in [-1, 1], same shape as input
    """
    # Convert to model normalization
    mel_01 = pipeline_to_model(mel_pipeline)
    
    # Run inference
    transferred_01 = infer_mel(mel_01, model_dict, n_overlap)
    
    # Convert back to pipeline normalization
    return model_to_pipeline(transferred_01)


def transfer_segments(segment_dir: str, model_dict: dict,
                      output_dir: str, n_overlap: int = 4):
    """
    Transfer all mel segments in a directory.
    
    Expects .npy files with mel spectrograms from our DSP pipeline.
    
    Args:
        segment_dir: Directory containing *_mel.npy files
        model_dict: dict from load_model()
        output_dir: Directory to save transferred mels
        n_overlap: Overlap factor
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mel_files = sorted(Path(segment_dir).glob('*_mel.npy'))
    if not mel_files:
        print(f"No *_mel.npy files found in {segment_dir}")
        return
    
    print(f"Transferring {len(mel_files)} segments...")
    for mel_path in tqdm(mel_files, desc='Segments'):
        mel_pipeline = np.load(mel_path)
        
        transferred = transfer_mel_pipeline(mel_pipeline, model_dict, n_overlap)
        
        out_name = mel_path.stem.replace('_mel', '_transferred_mel') + '.npy'
        np.save(os.path.join(output_dir, out_name), transferred)
    
    print(f"Saved {len(mel_files)} transferred segments to {output_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pipeline inference bridge for VAE-GAN timbre transfer")
    
    # Input
    parser.add_argument("--input_mel", type=str, default=None,
                        help="Path to a single .npy mel spectrogram")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of *_mel.npy segment files")
    
    # Model
    parser.add_argument("--model_name", type=str, default="pipeline_urmp",
                        help="Name of the saved model")
    parser.add_argument("--epoch", type=int, default=500,
                        help="Training epoch to load (500 for pipeline URMP)")
    parser.add_argument("--trg_id", type=str, default="2",
                        help="Target generator ID (1=trumpet, 2=violin for URMP)")
    parser.add_argument("--src_id", type=str, default=None,
                        help="Source generator ID for cyclic eval")
    parser.add_argument("--models_dir", type=str, default=None,
                        help="Override saved_models directory location")
    
    # Architecture (must match training)
    parser.add_argument("--img_height", type=int, default=80,
                        help="Mel bins (80=our pipeline, 128=original URMP)")
    parser.add_argument("--img_width", type=int, default=128,
                        help="Frames per inference window")
    parser.add_argument("--n_overlap", type=int, default=4,
                        help="Overlap factor for sliding window")
    parser.add_argument("--dim", type=int, default=32,
                        help="Base filter count")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="transferred_output",
                        help="Directory to save transferred mels")
    parser.add_argument("--plot", action="store_true",
                        help="Save before/after mel plots")
    parser.add_argument("--vocoder", type=str, default=None,
                        choices=["griffinlim", "bigvgan_22k", "bigvgan_24k"],
                        help="Synthesize audio with a vocoder (default: mel only)")
    
    args = parser.parse_args()
    
    assert args.input_mel or args.input_dir, \
        "Specify --input_mel or --input_dir"
    assert not (args.input_mel and args.input_dir), \
        "Specify only one of --input_mel or --input_dir"
    
    # Load model
    print(f"Loading model: {args.model_name} (epoch {args.epoch}, G{args.trg_id})")
    model_dict = load_model(
        args.model_name, args.epoch, args.trg_id, args.src_id,
        img_height=args.img_height, img_width=args.img_width,
        dim=args.dim, models_dir=args.models_dir)
    print(f"Model loaded on {model_dict['device']}")
    
    if args.input_dir:
        transfer_segments(args.input_dir, model_dict,
                          args.output_dir, args.n_overlap)
    else:
        mel_pipeline = np.load(args.input_mel)
        print(f"Input mel shape: {mel_pipeline.shape}, "
              f"range: [{mel_pipeline.min():.3f}, {mel_pipeline.max():.3f}]")
        
        transferred = transfer_mel_pipeline(mel_pipeline, model_dict,
                                            args.n_overlap)
        
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, 'transferred_mel.npy')
        np.save(out_path, transferred)
        print(f"Saved transferred mel to {out_path}")
        print(f"Output range: [{transferred.min():.3f}, {transferred.max():.3f}]")
        
        if args.plot:
            plot_path = os.path.join(args.output_dir, 'transfer_comparison.png')
            mel_01_in = pipeline_to_model(mel_pipeline)
            mel_01_out = pipeline_to_model(transferred)
            plot_mel_transfer_infer(plot_path, mel_01_in, mel_01_out)
            print(f"Saved comparison plot to {plot_path}")
        
        # Vocoder synthesis
        if args.vocoder:
            import soundfile as sf
            mel_01_out = pipeline_to_model(transferred)
            if args.vocoder == 'griffinlim':
                audio = reconstruct_with_griffinlim(mel_01_out)
            else:
                audio = reconstruct_with_bigvgan(mel_01_out, args.vocoder)
            
            wav_path = os.path.join(args.output_dir,
                                    f'transferred_{args.vocoder}.wav')
            # Use sample rate from active params
            params = _get_params()
            sf.write(wav_path, audio, params.sample_rate)
            print(f"Synthesized audio: {wav_path} "
                  f"({len(audio)/params.sample_rate:.1f}s)")
