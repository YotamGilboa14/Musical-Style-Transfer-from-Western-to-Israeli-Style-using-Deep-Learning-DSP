# Training and reproducing

Training and inference code is in `train.py`, `inference.py` and the notebooks
in `colab/`. This doc explains how the pieces fit together and how to reproduce
a run.

## Where each part runs, and why

The project is split across three environments on purpose, because of a
dependency conflict — Basic-Pitch needs a TensorFlow stack pinned to Python 3.10,
which clashes with our PyTorch/CUDA environment and with Colab's default runtime.

- **`ml_env`** (local, Python + PyTorch + CUDA) — model code, the DSP driver,
  and the figure/deliverable builders.
- **`basic_pitch_env`** (local, Python 3.10 + TensorFlow) — Basic-Pitch
  transcription only, called as a subprocess.
- **Google Colab** (A100 GPU) — heavy training, DDIM inference and BigVGAN
  vocoding.

We keep only light DSP and transcription local (little GPU, would waste Colab
time), push storage to Google Drive (our PCs have limited disk for a growing
dataset plus every checkpoint), and run GPU-heavy work on Colab. Results come
back to the local machine for evaluation and figure building.

## What the model is trained to do

During training (the DDPM view) we take a clean mel, pick a random noise level
`t` between 1 and 1000, add exactly that much noise, and ask the network to
predict the noise it sees — while also showing it the piano-roll and the style.
Because `t` is random every time, over many updates the network learns to undo
noise at every level. It never generates a full mel during training; it only
learns to estimate noise, which is a stable regression target.

Generation runs the same process backwards (the DDIM view): start from pure
noise and repeatedly ask "what noise do you see?", subtract a portion, and step
to a slightly cleaner mel, until a full mel appears. DDIM lets us take a few
large deterministic steps (100 or 200) instead of all 1000.

## Slakh first (architecture sanity check)

Before committing to the small Israeli data, we trained the exact same
architecture on Slakh, a large Western multi-track dataset with clean
hand-crafted MIDI. If the design could not learn clean mels there, it would have
no chance on the scarcer material. The Slakh run converged and produced
recognizable mels, so we knew the U-Net, conditioning and diffusion schedule all
worked before spending effort on the harder low-resource styles. Slakh then
stayed in the final model as the v0 reference style.

## Final run settings (`configs/default.yaml`)

- `T = 1000`, cosine schedule (`s = 0.008`), L1 loss;
- batch size 32, AdamW, learning rate `1e-4 → 1e-5` (linear decay from 70% of
  training), weight decay `1e-4`, gradient clip 1.0, EMA 0.999, bf16 on A100;
- 250,000 steps, checkpoint every 2,000 steps (all kept);
- CFG dropout: score 0.10, version 0.10, both 0.05;
- three styles: `v0` Slakh Western rock (reference), `v1` Israeli artists,
  `v2` Israeli military-band; `n_versions = 3`, null token index 3.

Full hyperparameters are in `configs/default.yaml`.

## Reproducing a run

1. Build the dataset (see `docs/DATA_PIPELINE.md`) and upload the tensors to
   Drive.
2. Open `colab/data_ingest_israeli.ipynb` and run it once per Israeli version to
   produce the tensors and the train/val/test splits.
3. Open `colab/train_israeli.ipynb`, mount Drive, build the combined manifest,
   and run training. Every code cell has a short header telling you what it does,
   its inputs/outputs, the action required, and the expected runtime.
4. Use `colab/postprocessing.ipynb` for batch inference across candidate
   checkpoints, in-Colab FAD/latency, and best-step selection; the note-level F1
   is computed locally in `basic_pitch_env`.

A quick local sanity check (no GPU training) is available with:

```bash
python smoke_test_local.py
```
