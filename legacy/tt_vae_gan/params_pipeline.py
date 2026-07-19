"""
Pipeline-adapted parameters for the VAE-GAN timbre transfer model.
These match our DSP preprocessing pipeline (dsp_preprocessor.py).

Usage:
    When training a NEW model on our pipeline data, import from here
    instead of params.py. The pretrained URMP weights use params.py.
"""

# ─── Audio ──────────────────────────────────────────────────────────────────────
sample_rate = 22050

# Number of spectrogram frames in a partial utterance
partials_n_frames = 160

# Number of spectrogram frames at inference
inference_n_frames = 80

# ─── Mel-filterbank (matches DSPConfig defaults) ───────────────────────────────
n_fft = 1024
num_mels = 80
num_samples = 128           # input spect shape: num_mels × num_samples

hop_length = 256            # ~11.6 ms at 22050 Hz
win_length = 1024           # ~46.4 ms at 22050 Hz

fmin = 0
fmax = 8000
min_level_db = -100
ref_level_db = 20

bits = 9
mu_law = True
peak_norm = False

# ─── Voice Activation Detection ────────────────────────────────────────────────
vad_window_length = 30
vad_moving_average_width = 8
vad_max_silence_length = 16

# ─── Audio volume normalisation ─────────────────────────────────────────────────
audio_norm_target_dBFS = -30

# ─── Pre-emphasis ───────────────────────────────────────────────────────────────
preemphasis = 0.97

# ─── Normalization note ─────────────────────────────────────────────────────────
# Our pipeline uses [-1, 1] normalization (dsp_preprocessor.py)
# The original voice_conversion code uses [0, 1] normalization
# The pipeline_inference.py bridge handles the conversion:
#   pipeline [-1,1] → model [0,1] → pipeline [-1,1]
