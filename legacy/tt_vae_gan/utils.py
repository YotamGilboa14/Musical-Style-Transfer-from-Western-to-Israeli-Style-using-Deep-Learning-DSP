"""Legacy VAE-GAN proof-of-concept - helper utilities (superseded).

Part of the early trumpet-to-violin timbre-transfer experiment we built before
switching to the diffusion model. Kept for reference and learning only; the
final pipeline does not use it. See legacy/README for the full context.
"""

# SOURCE:
# - https://github.com/CorentinJ/Real-Time-Voice-Cloning
# - https://github.com/r9y9/wavenet_vocoder
# Forked from ebadawy/voice_conversion/src/utils.py

from scipy.ndimage import binary_dilation
import os
import math
import numpy as np
from pathlib import Path
from typing import Optional, Union
import librosa
import struct
from scipy.signal import lfilter
import soundfile as sf
import matplotlib.pyplot as plt

# ─── Dynamic parameter loading ──────────────────────────────────────────────────
# Uses params_config switcher so we can swap between original (128-mel, 16kHz)
# and pipeline (80-mel, 22050Hz) params via env var or explicit call.
try:
    from .params_config import get_params as _get_params
    _p = _get_params()
except ImportError:
    # Fallback for running as standalone script
    import params as _p

# Import all param values into this module's namespace
sample_rate = _p.sample_rate
n_fft = _p.n_fft
num_mels = _p.num_mels
num_samples = _p.num_samples
hop_length = _p.hop_length
win_length = _p.win_length
fmin = _p.fmin
min_level_db = _p.min_level_db
ref_level_db = _p.ref_level_db
bits = _p.bits
mu_law = _p.mu_law
peak_norm = _p.peak_norm
preemphasis = _p.preemphasis
audio_norm_target_dBFS = _p.audio_norm_target_dBFS
# fmax: only in pipeline params, default None for original
fmax = getattr(_p, 'fmax', None)
# VAD params
vad_window_length = _p.vad_window_length
vad_moving_average_width = _p.vad_moving_average_width
vad_max_silence_length = _p.vad_max_silence_length

try:
    import webrtcvad
except ImportError:
    print("Warning: Unable to import 'webrtcvad'. "
          "This package enables noise removal and is recommended.")
    webrtcvad = None

int16_max = (2 ** 15) - 1


# ─── Waveform preprocessing ────────────────────────────────────────────────────

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder 
    to a waveform either on disk or in memory. The waveform will be resampled 
    to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file, or the waveform as
        a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the
        waveform before preprocessing. If passing a filepath, the sampling rate
        will be automatically detected and this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sample_rate:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sample_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad:
        wav = trim_long_silences(wav)

    return wav


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sample_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav),
                           *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(
            pcm_wave[window_start * 2:window_end * 2],
            sample_rate=sample_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2),
                                       array,
                                       np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool_)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.mean(wav ** 2)
    if rms == 0:
        return wav
    dBFS_change = target_dBFS - 10 * np.log10(rms)
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


# ─── File listing (cross-platform) ──────────────────────────────────────────────

def ls(path):
    """List files matching a path pattern (cross-platform)."""
    import glob as _glob
    if '*' in path or '?' in path:
        return [os.path.basename(f) for f in _glob.glob(path)]
    # Plain directory listing
    if os.path.isdir(path):
        return os.listdir(path)
    # Legacy: pipe-grep style "dir | grep .wav"
    if '|' in path:
        parts = path.split('|')
        dir_path = parts[0].strip()
        grep_pattern = parts[1].replace('grep', '').strip()
        return [f for f in os.listdir(dir_path) if grep_pattern in f]
    return []


# ─── Numeric utilities ──────────────────────────────────────────────────────────

def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


# ─── Audio I/O ──────────────────────────────────────────────────────────────────

def load_wav(path):
    return librosa.load(path, sr=sample_rate)[0]


def save_wav(x, path):
    sf.write(path, x.astype(np.float32), sample_rate)


def split_signal(x):
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2**15


def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


# ─── Spectrogram functions ──────────────────────────────────────────────────────

def linear_to_mel(spectrogram):
    kwargs = dict(S=spectrogram, sr=sample_rate, n_fft=n_fft,
                  n_mels=num_mels, fmin=fmin)
    if fmax is not None:
        kwargs['fmax'] = fmax
    return librosa.feature.melspectrogram(**kwargs)


def normalize(S):
    """Normalize spectrogram to [0, 1] range."""
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def denormalize(S):
    """Denormalize from [0, 1] back to dB scale."""
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - ref_level_db
    return normalize(S)


def melspectrogram(y):
    """Compute a normalized mel spectrogram in [0, 1] range."""
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)


def stft(y):
    return librosa.stft(
        y=y,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def pre_emphasis(x):
    return lfilter([1, -preemphasis], [1], x)


def de_emphasis(x):
    return lfilter([1], [1, -preemphasis], x)


# ─── Mu-law ─────────────────────────────────────────────────────────────────────

def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


# ─── Griffin-Lim reconstruction ─────────────────────────────────────────────────

def reconstruct_waveform(mel, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    denormalized = denormalize(mel)
    amp_mel = db_to_amp(denormalized)
    kwargs = dict(M=amp_mel, power=1, sr=sample_rate,
                   n_fft=n_fft, fmin=fmin)
    if fmax is not None:
        kwargs['fmax'] = fmax
    S = librosa.feature.inverse.mel_to_stft(**kwargs)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=hop_length, win_length=win_length)
    return wav


# ─── Tensor utilities ───────────────────────────────────────────────────────────

def to_numpy(batch):
    batch = batch.detach().cpu().numpy()
    batch = np.squeeze(batch)
    return batch


# ─── Plotting utilities ─────────────────────────────────────────────────────────

def plot_mel_transfer_train(save_path, curr_epoch, mel_in, mel_cyclic,
                            mel_out, mel_target):
    """Visualises melspectrogram style transfer in training, with target."""
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

    ax[0, 0].imshow(mel_in, interpolation="None")
    ax[0, 0].invert_yaxis()
    ax[0, 0].set(title='Input')
    ax[0, 0].set_ylabel('Mels')
    ax[0, 0].axes.xaxis.set_ticks([])

    ax[1, 0].imshow(mel_cyclic, interpolation="None")
    ax[1, 0].invert_yaxis()
    ax[1, 0].set(title='Cyclic Reconstruction')
    ax[1, 0].set_xlabel('Frames')
    ax[1, 0].set_ylabel('Mels')

    ax[0, 1].imshow(mel_out, interpolation="None")
    ax[0, 1].invert_yaxis()
    ax[0, 1].set(title='Output')
    ax[0, 1].axes.yaxis.set_ticks([])
    ax[0, 1].axes.xaxis.set_ticks([])

    ax[1, 1].imshow(mel_target, interpolation="None")
    ax[1, 1].invert_yaxis()
    ax[1, 1].set(title='Target')
    ax[1, 1].set_xlabel('Frames')
    ax[1, 1].axes.yaxis.set_ticks([])

    fig.suptitle('Epoch ' + str(curr_epoch))
    plt.savefig(save_path)
    plt.close()


def plot_batch_train(modelname, direction, curr_epoch, SRC, cyclic_SRC,
                     fake_TRGT, real_TRGT):
    SRC, cyclic_SRC = to_numpy(SRC), to_numpy(cyclic_SRC)
    fake_TRGT, real_TRGT = to_numpy(fake_TRGT), to_numpy(real_TRGT)
    i = 1
    for src, cyclic_src, fake_target, real_target in zip(
            SRC, cyclic_SRC, fake_TRGT, real_TRGT):
        fname = "out_train/%s/%s/%s_%02d_%s.png" % (
            modelname, direction, direction, curr_epoch, i)
        plot_mel_transfer_train(fname, curr_epoch, src, cyclic_src,
                                fake_target, real_target)
        i += 1


def plot_mel_transfer_eval(save_path, mel_in, mel_out):
    """Visualises melspectrogram style transfer in testing."""
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(5, 3))

    ax[0].imshow(mel_in, interpolation="None")
    ax[0].invert_yaxis()
    ax[0].set(title='Input')
    ax[0].set_ylabel('Mels')
    ax[0].set_xlabel('Frames')

    ax[1].imshow(mel_out, interpolation="None")
    ax[1].invert_yaxis()
    ax[1].set(title='Output')
    ax[1].set_xlabel('Frames')
    ax[1].axes.yaxis.set_ticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_batch_eval(modelname, direction, batchno, SRC, fake_TRGT):
    SRC, fake_TRGT = to_numpy(SRC), to_numpy(fake_TRGT)
    i = 1
    for src, fake_target in zip(SRC, fake_TRGT):
        fname = "out_eval/%s/%s/%s_%04d_%s.png" % (
            modelname, direction, direction, batchno, i)
        plot_mel_transfer_eval(fname, src, fake_target)
        i += 1


def wav_batch_eval(modelname, direction, batchno, SRC, fake_TRGT):
    SRC, fake_TRGT = to_numpy(SRC), to_numpy(fake_TRGT)
    i = 1
    for src, fake_target in zip(SRC, fake_TRGT):
        name = "out_eval/%s/%s/%s_%04d_%s" % (
            modelname, direction, direction, batchno, i)

        ref = reconstruct_waveform(src)
        ref_fname = name + '_ref.wav'
        sf.write(ref_fname, ref, sample_rate)

        out = reconstruct_waveform(fake_target)
        out_fname = name + '_out.wav'
        sf.write(out_fname, out, sample_rate)
        i += 1


def plot_mel_transfer_infer(save_path, mel_in, mel_out):
    """Visualises melspectrogram style transfer in inference (full-length)."""
    fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True)

    ax[0].imshow(mel_in, interpolation="None", aspect='auto')
    ax[0].set(title='Input')
    ax[0].set_ylabel('Mels')
    ax[0].axes.xaxis.set_ticks([])

    ax[1].imshow(mel_out, interpolation="None", aspect='auto')
    ax[1].set(title='Output')
    ax[1].set_ylabel('Mels')
    ax[1].set_xlabel('Frames')

    ax[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
