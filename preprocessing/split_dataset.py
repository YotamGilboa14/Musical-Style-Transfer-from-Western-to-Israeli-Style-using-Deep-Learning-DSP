"""
split_dataset.py — song-grouped train/val/test split
=====================================================

Reads dataset_manifest.csv and splits by song_id (never by segment),
so no song leaks across splits.

Output: train.csv, val.csv, test.csv  in --out_dir

Usage
-----
python preprocessing/split_dataset.py \
    --manifest  /content/drive/MyDrive/MusicProject/data/dataset_manifest.csv \
    --out_dir   /content/drive/MyDrive/MusicProject/data \
    --train 0.8 --val 0.1 --test 0.1 \
    --seed 42
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def split_dataset(
    manifest_path: Path,
    out_dir: Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    group_by: list[str] | None = None,
) -> dict:
    """Split manifest CSV by song group.

    By default groups by ``song_id`` (one group per processed song variant).
    Pass ``group_by=['artist', 'album', 'song_name']`` to keep a base song and
    all of its augmented variants together in the same split (no augmentation
    leakage across train/val/test).  Returns dict with counts.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    df = pd.read_csv(manifest_path)

    if group_by:
        missing = [c for c in group_by if c not in df.columns]
        if missing:
            raise ValueError(f"Manifest is missing group_by column(s): {missing}")
        group_key = "_group_key"
        df[group_key] = df[group_by].astype(str).agg("\u241f".join, axis=1)
    else:
        if "song_id" not in df.columns:
            raise ValueError("Manifest is missing 'song_id' column")
        group_key = "song_id"

    rng = np.random.default_rng(seed)
    song_ids = np.array(sorted(df[group_key].unique()))
    rng.shuffle(song_ids)

    n = len(song_ids)
    n_train = max(1, int(round(n * train_frac)))
    n_val   = max(1, int(round(n * val_frac)))
    # test gets the remainder (handles rounding)
    n_test  = n - n_train - n_val
    if n_test < 0:
        # edge case: very few songs → give at least 1 to test, shrink val
        n_val  = max(1, n_val + n_test)
        n_test = n - n_train - n_val

    train_ids = set(song_ids[:n_train])
    val_ids   = set(song_ids[n_train: n_train + n_val])
    test_ids  = set(song_ids[n_train + n_val:])

    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split_name, id_set in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_df = df[df[group_key].isin(id_set)].copy()
        split_df = split_df.drop(columns=["_group_key"], errors="ignore")
        out_path = out_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        results[split_name] = {
            "songs": len(id_set),
            "segments": len(split_df),
            "duration_h": split_df["duration_s"].sum() / 3600 if "duration_s" in split_df.columns else None,
            "path": str(out_path),
        }
        print(f"{split_name:5s}  songs={len(id_set):3d}  segments={len(split_df):5d}"
              + (f"  {split_df['duration_s'].sum()/3600:.2f} h" if "duration_s" in split_df.columns else ""))

    # Leakage assertion
    assert train_ids.isdisjoint(val_ids),  "LEAK: train ∩ val"
    assert train_ids.isdisjoint(test_ids), "LEAK: train ∩ test"
    assert val_ids.isdisjoint(test_ids),   "LEAK: val ∩ test"
    print("OK: No song-level leakage across splits")

    return results


def main() -> None:
    """Parse CLI arguments and write train/val/test split CSV files."""
    parser = argparse.ArgumentParser(description="Song-grouped dataset split.")
    parser.add_argument("--manifest", required=True,
                        help="Path to consolidated dataset_manifest.csv")
    parser.add_argument("--out_dir",  required=True,
                        help="Directory to write train.csv / val.csv / test.csv")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val",   type=float, default=0.1)
    parser.add_argument("--test",  type=float, default=0.1)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--group-by", nargs="+", default=None,
                        help="Columns forming the split group key "
                             "(e.g. artist album song_name keeps all augmented "
                             "variants of a song together). Default: song_id.")
    args = parser.parse_args()

    split_dataset(
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
        train_frac=args.train,
        val_frac=args.val,
        test_frac=args.test,
        seed=args.seed,
        group_by=args.group_by,
    )


if __name__ == "__main__":
    main()
