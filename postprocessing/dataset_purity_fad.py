"""Dataset-purity check via leave-one-song-out FAD.

Question (advisor): are all the songs we collected for a style really one
coherent "version" - or did some outliers sneak in? We answer it with the same
distance the evaluation uses: VGGish embeddings + Frechet distance.

Method
------
For every ORIGINAL (non-augmented) song of a dataset (Israeli_Artists v1,
Israeli_Military v2):
  1. embed up to N seconds from the middle of the song (0.96 s VGGish windows,
     128-D each);
  2. compute the leave-one-song-out FAD: the Frechet distance between this
     song's embedding cloud and the pooled cloud of ALL OTHER songs in the
     same dataset. Small = the song sits inside its dataset's distribution.

Reference scale (how small is small?)
-------------------------------------
A single song (about 60 windows) measured against a pool of ~75-100 songs
always shows a larger FAD than a pool-vs-pool comparison, purely for sample-
size reasons - so the between-dataset pool distance is reported as context
but NOT used as the per-song threshold. Instead:
  * a song is flagged as an outlier by the standard Tukey fence on its own
    dataset's LOSO distribution: FAD > Q3 + 1.5 * IQR;
  * the absolute VGGish-FAD practice bands (used since Kilgour et al. 2019 and
    in our own POC tooling) anchor the scale: < 5 near-indistinguishable,
    < 15 same-family. A dataset whose songs ALL sit below 5 is coherent.

Outputs (default --out-dir <version root>/_dataset_purity):
  purity_<dataset>.png            per-song LOSO FAD bar chart + threshold line
  dataset_purity_summary.json     all numbers, machine-readable

Run locally (ml_env, Drive mounted at G:):
  .\\ml_env\\Scripts\\python.exe -m postprocessing.dataset_purity_fad ^
      --manifest-dir "G:/My Drive/MusicProject/versions/Israeli_3style" ^
      --source-pool  "G:/My Drive/MusicProject/SourcePool"
  (use --max-songs 3 for a quick dry run)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import torch

from postprocessing.fad_eval import (
    get_embedder,
    extract_embeddings_from_audio,
    compute_statistics,
    frechet_distance,
)

DATASETS = {"1": "Israeli_Artists", "2": "Israeli_Military"}
_QUALITY_BAND_GOOD = 15.0   # same bands as fad_visualize.py (POC tooling)


def _collect_songs(manifest_dir: Path) -> Dict[str, List[dict]]:
    """Unique ORIGINAL songs per dataset from the training manifests.

    Returns {version_id: [ {artist, album, song_name}, ... ]}.
    """
    songs: Dict[str, Dict[str, dict]] = {v: {} for v in DATASETS}
    for name in ("combined_train.csv", "combined_val.csv"):
        p = manifest_dir / name
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                vid = row.get("version_id")
                if vid not in DATASETS:
                    continue
                if (row.get("aug_tag") or "orig") != "orig":
                    continue
                key = f"{row['artist']}__{row['album']}__{row['song_name']}"
                songs[vid].setdefault(key, {
                    "artist": row["artist"],
                    "album": row["album"],
                    "song_name": row["song_name"],
                })
    return {v: list(d.values()) for v, d in songs.items()}


def _song_wav(source_pool: Path, s: dict) -> Optional[Path]:
    """Locate the original WAV for one song in the source pool."""
    d = source_pool / s["artist"] / s["album"] / s["song_name"]
    exact = d / f"{s['song_name']}.wav"
    if exact.exists():
        return exact
    if d.exists():
        wavs = sorted(d.glob("*.wav"))
        if wavs:
            return wavs[0]
    return None


def _embed_song(wav: Path, cache_dir: Path, *, embedder, device,
                seconds: float, sr: int = 22050) -> Optional[np.ndarray]:
    """Embed the middle ``seconds`` of one song; cache to .npy for re-runs."""
    cache = cache_dir / (wav.parent.name + "__" + wav.stem + ".npy")
    if cache.exists():
        return np.load(cache)
    try:
        total = librosa.get_duration(path=str(wav))
        offset = max(0.0, (total - seconds) / 2.0)
        audio, _ = librosa.load(str(wav), sr=sr, mono=True,
                                offset=offset, duration=seconds)
        emb = extract_embeddings_from_audio(audio, sr=sr, embedder=embedder,
                                            device=device)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb
    except Exception as e:  # noqa: BLE001
        print(f"  !! failed on {wav.name}: {e}")
        return None


def _loso_fads(embs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Leave-one-song-out FAD for every song in one dataset."""
    keys = list(embs)
    out: Dict[str, float] = {}
    for k in keys:
        rest = np.concatenate([embs[j] for j in keys if j != k], axis=0)
        mu_r, sig_r = compute_statistics(rest)
        mu_s, sig_s = compute_statistics(embs[k])
        out[k] = frechet_distance(mu_r, sig_r, mu_s, sig_s)
    return out


def _plot_dataset(name: str, fads: Dict[str, float], threshold: float,
                  out_png: Path) -> None:
    """Sorted per-song LOSO FAD bars with the Tukey outlier fence."""
    items = sorted(fads.items(), key=lambda kv: kv[1])
    labels = [k.split("__")[-1][:28] for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#C62828" if v > threshold else "#2E7D32" for v in vals]

    fig_h = max(5.0, 0.22 * len(items))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    y = np.arange(len(items))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.axvline(threshold, color="#333", linestyle="--", linewidth=1.6,
               label=f"outlier fence (Q3 + 1.5 IQR) = {threshold:.2f}")
    ax.axvspan(0, 5.0, color="#2E7D32", alpha=0.06,
               label="FAD < 5: near-indistinguishable band")
    ax.set_xlabel("leave-one-song-out FAD (lower = fits its dataset better)")
    ax.set_title(f"Dataset purity - {name}: every song vs. the rest of its "
                 f"own dataset\n(n={len(items)} songs; red = statistical "
                 f"outlier within the dataset)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Leave-one-song-out FAD purity check")
    ap.add_argument("--manifest-dir", type=Path, required=True,
                    help="folder holding combined_train.csv / combined_val.csv")
    ap.add_argument("--source-pool", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <manifest-dir>/_dataset_purity")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("deliverables/_purity_cache"))
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="audio seconds embedded per song (from the middle)")
    ap.add_argument("--max-songs", type=int, default=0,
                    help="dry-run cap per dataset (0 = all)")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or (args.manifest_dir / "_dataset_purity")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = get_embedder(device, use_pretrained=True)
    if not getattr(embedder, "_fad_is_pretrained", False):
        print("WARNING: pretrained VGGish unavailable - falling back. "
              "Numbers will not be comparable to the finalist metrics.")

    songs_by_vid = _collect_songs(args.manifest_dir)
    embs_by_ds: Dict[str, Dict[str, np.ndarray]] = {}
    for vid, ds_name in DATASETS.items():
        songs = songs_by_vid.get(vid, [])
        if args.max_songs:
            songs = songs[: args.max_songs]
        print(f"[{ds_name}] embedding {len(songs)} songs "
              f"({args.seconds:.0f}s each) ...")
        embs: Dict[str, np.ndarray] = {}
        for i, s in enumerate(songs, 1):
            wav = _song_wav(args.source_pool, s)
            if wav is None:
                print(f"  ({i}/{len(songs)}) MISSING wav: "
                      f"{s['artist']}/{s['album']}/{s['song_name']}")
                continue
            key = f"{s['artist']}__{s['song_name']}"
            emb = _embed_song(wav, args.cache_dir, embedder=embedder,
                              device=device, seconds=args.seconds)
            if emb is not None and len(emb):
                embs[key] = emb
            if i % 10 == 0:
                print(f"  ({i}/{len(songs)}) done")
        embs_by_ds[ds_name] = embs

    # Between-dataset reference distance (pooled Artists vs pooled Military).
    between = None
    names = list(embs_by_ds)
    if len(names) == 2 and all(embs_by_ds[n] for n in names):
        a = np.concatenate(list(embs_by_ds[names[0]].values()), axis=0)
        b = np.concatenate(list(embs_by_ds[names[1]].values()), axis=0)
        between = frechet_distance(*compute_statistics(a), *compute_statistics(b))
        print(f"between-dataset FAD ({names[0]} vs {names[1]}): {between:.2f}")

    summary: Dict[str, dict] = {
        "method": "leave-one-song-out FAD, VGGish 128-D, "
                  f"{args.seconds:.0f}s from song middle",
        "embedder": getattr(embedder, "_fad_embedder_label", "?"),
        "between_dataset_fad": between,
        "between_dataset_note": "pool-vs-pool distance; context only - not "
                                "comparable to single-song LOSO values",
        "threshold_rule": "Tukey fence per dataset: Q3 + 1.5*IQR; absolute "
                          "anchor: <5 near-indistinguishable, <15 same-family",
        "datasets": {},
    }
    for ds_name, embs in embs_by_ds.items():
        if len(embs) < 3:
            print(f"[{ds_name}] too few songs embedded, skipping")
            continue
        fads = _loso_fads(embs)
        vals = np.array(list(fads.values()))
        q1, q3 = np.percentile(vals, [25, 75])
        thr = float(q3 + 1.5 * (q3 - q1))
        outliers = {k: v for k, v in fads.items() if v > thr}
        _plot_dataset(ds_name, fads, thr, out_dir / f"purity_{ds_name}.png")
        summary["datasets"][ds_name] = {
            "n_songs": len(fads),
            "loso_fad_median": float(np.median(vals)),
            "loso_fad_mean": float(vals.mean()),
            "loso_fad_max": float(vals.max()),
            "n_below_5": int((vals < 5.0).sum()),
            "tukey_fence": thr,
            "n_outliers": len(outliers),
            "outliers": {k: round(v, 3) for k, v in
                         sorted(outliers.items(), key=lambda kv: -kv[1])},
            "per_song": {k: round(v, 3) for k, v in
                         sorted(fads.items(), key=lambda kv: -kv[1])},
        }
        print(f"[{ds_name}] median {np.median(vals):.2f}, max {vals.max():.2f}, "
              f"fence {thr:.2f}, outliers: {len(outliers)}, "
              f"songs under FAD 5: {(vals < 5.0).sum()}/{len(vals)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "dataset_purity_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
