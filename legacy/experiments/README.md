# Experiments & Milestones

This folder contains completed experiments and milestone demos that contributed to the active diffusion pipeline. These experiments are historical context: they explain how the team chose preprocessing, vocoding, and modeling directions, but they are not the current Israeli training path.

Only this top-level experiment README is present right now. If we want every experiment folder to have its own README, that should be done as a separate documentation task.

## Folder Contents

### `online_pipeline/`
**Phase 1–2 milestone** — The original online processing pipeline (`music_pipeline.py`) that orchestrated YouTube download → DSP preprocessing → Google Drive upload. This was our first end-to-end pipeline and is now superseded by `process_song_offline.py`, which adds Demucs source separation, mel normalization, and vocoder round-trip testing. The individual components (youtube_downloader, gdrive_uploader) remain active in `preprocessing/` for future integration.

### `ddsp_experiment/`
**Phase 4 milestone** — Google DDSP (Differentiable Digital Signal Processing) timbre transfer experiment on Colab. Used pretrained DDSP models (violin, trumpet, flute, etc.) to perform timbre transfer. This experiment validated the concept of mel-based timbre transfer and informed the architecture of the VAE-GAN model (Phase 4B).

### `vocoder_tests/`
**Phase 3B milestone** — A/B/C comparison of three vocoders: HiFi-GAN UNIVERSAL_V1, BigVGAN v2 22kHz, and BigVGAN v2 24kHz. This experiment helped us choose BigVGAN as the primary vocoder for the pipeline based on audio quality metrics and listening tests.
