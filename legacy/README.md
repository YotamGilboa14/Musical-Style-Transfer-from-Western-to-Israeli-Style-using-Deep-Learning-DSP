# Legacy — early experiments we kept for reference

This is a student project, and a lot of what we learned came from trying
approaches that did not make it into the final system. We keep those here rather
than deleting them, so the history of the project is honest and so the code is
available to learn from. **None of this is used by the final diffusion pipeline**
— for that, see the top-level `model/`, `preprocessing/` and `postprocessing/`.

## What's here

### `tt_vae_gan/`
Our first serious attempt: a CycleGAN-style **VAE-GAN** for timbre transfer,
forked and adapted from `ebadawy/voice_conversion` (which itself builds on UNIT
and other open-source projects — see the headers in each file). We used it to
prove to ourselves that mel-spectrogram-domain transfer was feasible on a single
instrument (trumpet → violin) before we moved to the diffusion model. It works
on roughly monophonic input, which is exactly the limitation that pushed us to
diffusion for polyphonic, multi-instrument music.

- `legacy/notebooks/tt_vae_gan_transfer.ipynb` — the Colab notebook that ran it.

### `experiments/`
Smaller explorations from early on:

- `ddsp_experiment/` — a DDSP timbre-transfer experiment. DDSP also assumes
  roughly monophonic input, another data point that led us to diffusion.
- `vocoder_tests/` — comparing vocoders before we settled on BigVGAN.
- `online_pipeline/` — an early end-to-end pipeline sketch.

### `audio_tp_midi_poc.py`
A proof-of-concept audio-to-MIDI step, later replaced by Basic-Pitch.

### `run_poc_transfer.py`
The runner script for the early proof-of-concept transfer.

## Why keep it

Two reasons. First, it documents *why* the final design looks the way it does:
each of these approaches taught us something (mostly that single-instrument
methods do not carry over to full polyphonic mixes), and the diffusion model in
the main tree is the answer to those lessons. Second, as students seeing these
tools for the first time, we found it useful to be able to go back and read the
older, simpler code — so we left it readable rather than removing it.
