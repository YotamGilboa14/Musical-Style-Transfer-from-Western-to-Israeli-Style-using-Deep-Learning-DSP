# Musical Style Transfer — Western → Israeli, with a Diffusion Model

Final-year Electrical Engineering project. We take a Western pop or rock song
and re-render it in an Israeli musical style, while keeping the original melody
and chords recognizable. In plain terms: we change *how* a song sounds — its
instruments and texture — without changing *what* is played.

The system is a single **conditional diffusion model** that generates
mel-spectrograms (a compact, image-like picture of sound), conditioned on two
things:

- a **content** signal — a pitch transcription of the source song (a piano-roll
  of the notes to keep), and
- a **style** signal — a small learned style ID that says which style to paint
  on top.

A neural vocoder (**BigVGAN**) then turns the generated spectrogram back into
audio you can play. One model holds three styles at once and we pick the style
at generation time:

- **v0** — Western rock (Slakh), used as a clean reference,
- **v1** — a blend of Israeli artists,
- **v2** — Israeli military-band songs.

Because no ready-made Israeli-style dataset exists, the first and biggest part
of the project was building our own — downloading songs, transcribing them, and
turning them into training tensors. Working with only a few hours of
hand-collected music per style is the core challenge of this project.

## Live demo

An interactive showcase (block diagrams, DSP examples, training gallery and
audio you can play) is published with this repo — see the **`showcase/`** folder
and the GitHub Pages link *(added once published)*.

## Documentation

Start with the README, then the topic docs:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the diffusion U-Net, FiLM
  conditioning and classifier-free guidance.
- [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) — how the dataset is built
  (download → transcribe → DSP → tensors) and augmented.
- [docs/TRAINING.md](docs/TRAINING.md) — how to train and reproduce, and the
  local ↔ Drive ↔ Colab split.
- [docs/RESULTS.md](docs/RESULTS.md) — FAD / F1 / latency, with the honest
  caveat about the F1 number.

## Repository layout

| Path | What it is |
|------|------------|
| `model/` | the diffusion U-Net: blocks, FiLM, embeddings, diffusion process |
| `data/` | the PyTorch `Dataset` that feeds training |
| `preprocessing/` | dataset building: download, transcription, DSP, augmentation, splitting |
| `postprocessing/` | vocoder wrappers and evaluation (FAD, note-level F1, latency) |
| `configs/` | model config and the per-run / per-version YAML specs |
| `colab/` | guided Colab notebooks for ingest, training and postprocessing |
| `tests/` | end-to-end and augmentation sanity tests |
| `docs/` | the topic documentation above |
| `examples/` | a small set of curated input/output audio examples |
| `showcase/` | the interactive HTML showcase (hosted via GitHub Pages) |
| `legacy/` | early experiments we kept for reference — see `legacy/README.md` |
| `train.py`, `inference.py` | training loop and inference entry points |
| `process_song_offline.py` | one-song preprocessing driver (download → tensors) |

## Getting started

The project runs across three environments (a dependency conflict forced the
split — see `docs/TRAINING.md`):

- **`ml_env`** (local, Python + PyTorch + CUDA) — model code, DSP driver,
  figure/deliverable builders.
- **`basic_pitch_env`** (local, Python 3.10 + TensorFlow) — Basic-Pitch
  transcription only.
- **Google Colab** (A100 GPU) — heavy training, DDIM inference and BigVGAN
  vocoding.

Install the local requirements and run the smoke test:

```bash
pip install -r requirements_ml.txt
python smoke_test_local.py
```

Training and inference run on Colab; open the notebooks in `colab/` and follow
the per-cell instructions (each code cell has a short "what this does / inputs /
outputs / action required / runtime" header).

## Results at a glance

Four targets, three met (see [docs/RESULTS.md](docs/RESULTS.md) for the full
story):

| Target | Result | Met? |
|--------|--------|------|
| Audio realism — FAD ≤ 9 | ~2.2–2.5 | yes |
| Per-style realism — group-FAD ≤ 7 | ~2.2–2.5 | yes |
| Speed — faster than real-time | RTF 0.13 (100 steps) / 0.20 (200 steps) | yes |
| Content preservation — note-F1 ≥ 30% | ~3–4.5% | no (metric-limited) |

The low F1 is mostly a property of the metric: it compares two automatic
transcriptions, both noisy, and our score tracks pitch only. Lining the notes up
visually (piano-roll overlays) shows the melody clearly survives the transfer.

## Authors

Yotam Gilboa and Gal Geva. Supervised by Dr. Lior Arbel, Tel Aviv University.

## Acknowledgements

We build on open-source and published work, in particular Ben-Maman et al.
(2024) on multi-aspect diffusion music synthesis, the BigVGAN vocoder, the
Basic-Pitch transcriber, and the Slakh2100 dataset. See `docs/` and the
references in the project book for details.

## License

Released under the MIT License — see [LICENSE](LICENSE).
