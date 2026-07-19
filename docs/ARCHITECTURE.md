# Architecture

This is the model side of the project: how we turn "predict the noise" into
"generate a styled mel-spectrogram." The code lives in `model/`.

## Overview

We generate audio in the **mel-spectrogram** domain, not on the raw waveform.
Raw audio has about 22,000 numbers per second, far too much to generate
directly. A mel-spectrogram is a compact, image-like picture of sound: it warps
the frequency axis to match human hearing, keeps only 80 frequency bands, and
log-compresses the magnitudes. We use 22,050 Hz, 80 mel bins, hop 256 and an
8 kHz cap because those match what the BigVGAN vocoder expects on the way back
to audio.

The generator is a **denoising diffusion model** (DDPM), the same family of
models behind AI image generators. It never draws a mel in one shot; it learns
to remove a little noise at a time, and generation runs that process backwards
from pure noise.

```
noisy mel  x_t  ─┐
piano-roll score ─┼─►  U-Net denoiser  ─►  predicted noise  ─►  (DDIM step) ─►  x_{t-1}
timestep t       │            ▲
style ID         ┘         FiLM (t, style)
```

## The U-Net denoiser (`model/unet.py`, `model/blocks.py`)

The denoiser is a **1-D U-Net** that runs along the time axis and treats the 80
mel bins as feature channels. A U-Net has an encoder that step by step shrinks
the time axis while widening the channels, a bottleneck in the middle, and a
decoder that grows the time axis back. **Skip connections** copy each encoder
level across to the matching decoder level, so the network keeps fine detail
from the input while still reasoning about the whole segment at coarser levels.

Concrete configuration (`configs/default.yaml`):

- base width 160 channels, multipliers `[1, 2, 3, 4]` → levels of 160 / 320 /
  480 / 640 channels;
- 2 residual blocks per encoder level, 3 per decoder level;
- self-attention at the three deeper levels (8 heads);
- GroupNorm (32 groups), SiLU activations, dropout 0.1;
- about **32.3 million** parameters (32,287,120).

Two building blocks are worth explaining:

- **Residual block** — a small stack of convolutions whose output is added back
  to its own input. That shortcut lets each block learn a small correction
  instead of reproducing its input from scratch, which is what makes deep
  networks trainable.
- **Self-attention** — convolutions only see a small local window, so they miss
  long-range structure. Attention lets every time position look at every other
  and weigh how relevant it is, which helps keep phrasing and rhythm consistent
  across a segment. We only switch it on at the deeper, shorter levels where it
  is affordable.

Two things turn this plain denoiser into a **style-transfer** model: the pitch
score enters as extra input channels next to the noisy mel (so the network
always sees which notes should sound), and the timestep + style are injected
into every block through FiLM.

## Conditioning: content + style

- **Content** — a pitch transcription of the source, encoded as a 256-channel
  piano-roll (2 × 128 MIDI pitches, onset + sustain) with the same time length
  as the mel. It enters the U-Net as extra input channels. It keeps pitch only
  and does not separate instruments, which matters when reading the F1 metric
  (see `docs/RESULTS.md`).
- **Style / version** — a small learned embedding (a trainable lookup table)
  with one extra "null" slot used for classifier-free guidance.
- The timestep is turned into a vector by a sinusoidal embedding + MLP. Time and
  style are concatenated into a 256-d conditioning vector `C`.

### FiLM (`model/film.py`)

Every residual block is modulated by `C` through a **FiLM** (Feature-wise Linear
Modulation) layer:

```
gamma = 1 + scale(C)
beta  = shift(C)
h'    = gamma * h + beta
```

The `1 +` matters: at the start of training the two linear layers output values
near zero, so `gamma ≈ 1` and `beta ≈ 0`, which makes `h' = h`. The block starts
as an identity and only gradually learns how the timestep and style should
reshape the features, which makes early training more stable.

## Classifier-free guidance (CFG)

CFG lets us decide at generation time how strongly each condition is applied,
using a single network trained both with and without its conditions. During
training we randomly drop the score (set it to zeros) and/or the style (replace
it with the null token). At sampling we run the denoiser three times per step —
unconditional, score-only, and style-only — and combine the two directions
separately:

```
eps_hat = eps_uncond + w_s (eps_score - eps_uncond) + w_v (eps_version - eps_uncond)
```

with `w_s = w_v = 1.25`. Splitting the two lets us control content-faithfulness
and style-strength independently.

## Diffusion process (`model/diffusion.py`)

- **Noise schedule** — cosine (Nichol & Dhariwal 2021), `T = 1000`, which spreads
  the noise more evenly than a linear schedule.
- **Forward step (`q_sample`)** — a closed-form jump straight to a noisy version
  of a clean mel, used during training.
- **Loss** — L1 on the predicted noise (we found it gives sharper mels than L2).
- **Sampling (`ddim_sample`)** — DDIM with `eta = 0` (deterministic) over 100 or
  200 strided steps, which is what makes inference fast.
- **Long audio** — songs are longer than the 5-second (430-frame) segments, so
  we split into overlapping windows and blend the boundaries inside the DDIM
  loop, which removes clicks at the segment joins.

## Vocoder

The generated mel is converted back to a waveform by **BigVGAN v2**, a GAN-based
neural vocoder trained on 22 kHz audio with 80 mel bins — which is exactly why
our DSP front-end targets those parameters. A HiFi-GAN implementation is kept in
`postprocessing/` as a reference vocoder we compared against, but BigVGAN is the
one used for all results.
