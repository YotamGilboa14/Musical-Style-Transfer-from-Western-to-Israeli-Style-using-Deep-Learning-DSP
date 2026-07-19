# `model/` — Active multi-version diffusion U-Net

This is the **current active model code** for the score-conditioned diffusion U-Net. Not to be confused with [../models/](../models/) (plural), which holds the older VAE-GAN POC code.

The architecture is designed for multiple style versions. The final trained plan (`Israeli_3style`, `n_versions=3`) is:

| Version ID | Meaning |
|---|---|
| `0` | Slakh rock sanity/baseline style (trained) |
| `1` | Israeli artists (trained) |
| `2` | Israeli military bands (trained) |
| `3` (`n_versions`) | Null / unconditional CFG token, not a real style |

## Files

| File | What |
|---|---|
| `unet.py` | 1D U-Net (time + freq jointly via flattened mel + score). FiLM-conditioned, 4 levels (base 160 -> 160/320/480/640), ~32.3 M params. |
| `diffusion.py` | `GaussianDiffusion` — forward q-sampler, reverse DDIM sampler, classifier-free guidance. |
| `embeddings.py` | Sinusoidal time embedding + version embedding (FiLM-mapped). |
| `film.py` | FiLM modulation block — applies version+time conditioning to feature maps. |
| `blocks.py` | Residual blocks, attention, up/down samplers used by `unet.py`. |

## Inputs / outputs

- **Mel input:** `[B, 80, 430]` — 5 s @ sr=22050, hop=256, normalized to `[-1, 1]`.
- **Score input (conditioning):** `[B, 256, 430]` — 2-channel × 128-pitch piano roll, flattened.
- **Version conditioning:** integer in `[0, n_versions-1]`. For Slakh-only sanity runs, `n_versions = 1` and real style `0` is Slakh. For Slakh + first Israeli training, `n_versions` must become `2` so Israeli can use style `1`.
- **Output:** predicted noise of same shape as mel input.

See [../configs/default.yaml](../configs/default.yaml) for hyperparameters and [../ENGINEERING_DECISIONS.md](../ENGINEERING_DECISIONS.md) §13, §19, §24 for design rationale.
