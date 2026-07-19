"""Embedding-cluster visualizations: real vs. generated, 2-D and 3-D.

Advisor request: besides the FAD bell curves, show the *samples themselves* as
colored dots - the real songs of each dataset and the generated audio of each
style - so the scattering and the similarities are directly visible (same
spirit as the latency visualization).

Every dot is one 0.96 s VGGish embedding (128-D), the exact representation FAD
is computed in. We draw:
  clusters_2d_pca.png    PCA to the top-2 variance directions
  clusters_2d_tsne.png   t-SNE (preserves local neighborhoods; distances
                         between far clusters are NOT meaningful)
  clusters_3d_pca.png    PCA to 3 components, static 3-D view
  embedding_clusters_summary.json

Groups:
  real Israeli_Artists / real Israeli_Military  (source-pool songs, cached by
      dataset_purity_fad.py - run that first, or this script fills the cache)
  generated -> Artists / Military / Slakh_v0    (demo renders, ddim100)
Real Slakh audio is not stored on Drive (tensors only), so the Western
reference appears as generated-only.

Run locally (ml_env, Drive at G:):
  .\\ml_env\\Scripts\\python.exe -m postprocessing.embedding_cluster_viz ^
      --manifest-dir "G:/My Drive/MusicProject/versions/Israeli_3style" ^
      --source-pool  "G:/My Drive/MusicProject/SourcePool" ^
      --renders-root "G:/My Drive/MusicProject/versions/Israeli_3style/demo_external/_renders"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from postprocessing.fad_eval import get_embedder
from postprocessing.dataset_purity_fad import (
    _collect_songs, _song_wav, _embed_song, DATASETS,
)

# group -> (color, marker, alpha, size)
_STYLE = {
    "real Israeli Artists":      ("#1565C0", "o", 0.25, 10),
    "real Israeli Military":     ("#2E7D32", "o", 0.25, 10),
    "generated -> Artists":      ("#EF6C00", "D", 0.55, 16),
    "generated -> Military":     ("#C62828", "D", 0.55, 16),
    "generated -> Slakh rock":   ("#6A1B9A", "D", 0.55, 16),
}

_GEN_DIRS = {
    "generated -> Artists":    "demo_Israeli_Artists_ddim100",
    "generated -> Military":   "demo_Israeli_Military_ddim100",
    "generated -> Slakh rock": "demo_Slakh_v0_ddim100",
}


def _subsample(arr: np.ndarray, n: int, seed: int = 7) -> np.ndarray:
    if len(arr) <= n:
        return arr
    rng = np.random.default_rng(seed)
    return arr[rng.choice(len(arr), size=n, replace=False)]


def _gather_groups(args, embedder, device) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[np.ndarray]] = {}

    # Real datasets (shares the purity cache).
    songs_by_vid = _collect_songs(args.manifest_dir)
    for vid, ds in DATASETS.items():
        gname = f"real {ds.replace('_', ' ')}"
        for s in songs_by_vid.get(vid, []):
            wav = _song_wav(args.source_pool, s)
            if wav is None:
                continue
            emb = _embed_song(wav, args.cache_dir, embedder=embedder,
                              device=device, seconds=args.seconds)
            if emb is not None and len(emb):
                groups.setdefault(gname, []).append(emb)
        print(f"[{gname}] {len(groups.get(gname, []))} songs embedded")

    # Generated renders (whole-song transfers, ddim100).
    gen_cache = args.cache_dir / "_generated"
    for gname, sub in _GEN_DIRS.items():
        audio_dir = args.renders_root / sub / "audio"
        wavs = sorted(audio_dir.glob("*role_transferred.wav"))
        for wav in wavs:
            emb = _embed_song(wav, gen_cache, embedder=embedder,
                              device=device, seconds=args.seconds)
            if emb is not None and len(emb):
                groups.setdefault(gname, []).append(emb)
        print(f"[{gname}] {len(wavs)} renders embedded")

    return {g: np.concatenate(v, axis=0) for g, v in groups.items() if v}


def _plot_2d(coords: Dict[str, np.ndarray], title: str, xlab: str, ylab: str,
             out_png: Path, note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for g, pts in coords.items():
        c, m, a, s = _STYLE.get(g, ("#555", "o", 0.4, 12))
        ax.scatter(pts[:, 0], pts[:, 1], c=c, marker=m, alpha=a, s=s,
                   label=f"{g} ({len(pts)})", edgecolors="none")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(title)
    leg = ax.legend(loc="best", fontsize=9, framealpha=0.92)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.grid(True, alpha=0.2)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8.5,
                color="#555", va="bottom")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_3d(coords: Dict[str, np.ndarray], explained, out_png: Path) -> None:
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    for g, pts in coords.items():
        c, m, a, s = _STYLE.get(g, ("#555", "o", 0.4, 12))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, marker=m,
                   alpha=min(a + 0.1, 0.7), s=s, label=f"{g} ({len(pts)})",
                   edgecolors="none", depthshade=False)
    ax.set_xlabel(f"PC 1 ({explained[0]*100:.0f}%)")
    ax.set_ylabel(f"PC 2 ({explained[1]*100:.0f}%)")
    ax.set_zlabel(f"PC 3 ({explained[2]*100:.0f}%)")
    ax.set_title("Real vs. generated audio in VGGish embedding space - 3-D PCA\n"
                 "(every dot = 0.96 s of audio)")
    leg = ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    # Look roughly along PC2 so PC1 (the axis that carries the real-vs-generated
    # separation, ~38% of variance) runs left-to-right and faces the viewer.
    ax.view_init(elev=12, azim=-72)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Embedding cluster visualizations")
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--source-pool", type=Path, required=True)
    ap.add_argument("--renders-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <manifest-dir>/_embedding_clusters")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("deliverables/_purity_cache"))
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--max-points-per-group", type=int, default=1200)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or (args.manifest_dir / "_embedding_clusters")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = get_embedder(device, use_pretrained=True)

    groups = _gather_groups(args, embedder, device)
    sub = {g: _subsample(a, args.max_points_per_group) for g, a in groups.items()}
    names = list(sub)
    stacked = np.concatenate([sub[g] for g in names], axis=0)
    splits = np.cumsum([len(sub[g]) for g in names])[:-1]

    # --- 2-D PCA ---
    from sklearn.decomposition import PCA
    p2 = PCA(n_components=2).fit(stacked)
    parts = np.split(p2.transform(stacked), splits)
    _plot_2d(dict(zip(names, parts)),
             "Real vs. generated audio in VGGish embedding space - 2-D PCA\n"
             "(every dot = 0.96 s of audio)",
             f"PC 1 ({p2.explained_variance_ratio_[0]*100:.0f}% of variance)",
             f"PC 2 ({p2.explained_variance_ratio_[1]*100:.0f}% of variance)",
             out_dir / "clusters_2d_pca.png",
             note="PCA keeps the 2 directions with the most variance out of 128.")

    # --- 2-D t-SNE ---
    from sklearn.manifold import TSNE
    ts = TSNE(n_components=2, perplexity=35, init="pca", random_state=7,
              max_iter=1000)
    t_parts = np.split(ts.fit_transform(stacked), splits)
    _plot_2d(dict(zip(names, t_parts)),
             "Real vs. generated audio in VGGish embedding space - t-SNE\n"
             "(neighborhood-preserving map; every dot = 0.96 s of audio)",
             "t-SNE axis 1", "t-SNE axis 2",
             out_dir / "clusters_2d_tsne.png",
             note="t-SNE keeps neighbors close; distances between far clusters "
                  "are not meaningful.")

    # --- 3-D PCA ---
    p3 = PCA(n_components=3).fit(stacked)
    parts3 = np.split(p3.transform(stacked), splits)
    _plot_3d(dict(zip(names, parts3)), p3.explained_variance_ratio_,
             out_dir / "clusters_3d_pca.png")

    summary = {
        "points_per_group": {g: int(len(a)) for g, a in groups.items()},
        "plotted_per_group": {g: int(len(a)) for g, a in sub.items()},
        "pca2_explained": [float(x) for x in p2.explained_variance_ratio_],
        "pca3_explained": [float(x) for x in p3.explained_variance_ratio_],
        "note": "real Slakh audio not stored on Drive - Western reference "
                "appears as generated-only",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "embedding_clusters_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
