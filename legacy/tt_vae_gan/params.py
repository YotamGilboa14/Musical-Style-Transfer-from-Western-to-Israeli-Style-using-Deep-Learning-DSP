"""
Hyperparameters for the VAE-GAN timbre transfer model.
Forked from ebadawy/voice_conversion/src/params.py

NOTE: These are the ORIGINAL parameters used to train the pretrained URMP weights.
      Keep these values for sanity-check inference with pretrained weights.
      For our pipeline (22050 Hz, 80 mels), see params_pipeline.py.
"""

# ─── Audio ──────────────────────────────────────────────────────────────────────
sample_rate = 16000

# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms

# Number of spectrogram frames at inference
inference_n_frames = 80     #  800 ms

# ─── Mel-filterbank ─────────────────────────────────────────────────────────────
n_fft = 2048
num_mels = 128
num_samples = 128           # input spect shape: num_mels × num_samples

hop_length = int(0.0125 * sample_rate)   # 200 samples  (12.5 ms)
win_length = int(0.05 * sample_rate)     # 800 samples  (50 ms)

fmin = 40
min_level_db = -100
ref_level_db = 20

bits = 9
mu_law = True
peak_norm = False

# ─── Voice Activation Detection ────────────────────────────────────────────────
vad_window_length = 30          # ms — must be 10, 20, or 30
vad_moving_average_width = 8
vad_max_silence_length = 16

# ─── Audio volume normalisation ─────────────────────────────────────────────────
audio_norm_target_dBFS = -30

# ─── Pre-emphasis ───────────────────────────────────────────────────────────────
preemphasis = 0.97
