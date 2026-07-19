# `data/` — PyTorch dataset code

| File | What |
|---|---|
| `dataset.py` | `MelPianoRollDataset` — loads mel + piano-roll `.pt` segments from a manifest CSV, applies augmentation, returns `dict(mel, piano_roll, version_id)`. |

The dataset is fed by manifests produced by `preprocessing/batch_ingest.py` and split by `preprocessing/split_dataset.py`. Augmentation is wired in here; the augmentation logic itself lives in [../preprocessing/augmentation.py](../preprocessing/augmentation.py).

## Version labels

The active plan is multi-version:

- `version_id=0` is Slakh rock.
- `version_id=1` is the first Israeli style.
- Future styles append as `2`, `3`, and so on.
- The null/unconditional CFG token is `n_versions`; it should not appear as a real manifest style.

Before training, every manifest should be checked against `configs/default.yaml` so no row points to the null token by mistake.
