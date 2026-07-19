"""
Build FAD + F1 visualizations for the 12 finalist checkpoints.
=============================================================
These are the held-out-song metrics (the demo/external songs have FAD/F1
disabled — they have no target-style reference — so quantitative metrics come
from the held-out inference runs, NOT the demo renders).

For each finalist (config, step) this produces:
  * Per-step FAD   — Fréchet distance between the real style embedding cloud
                     (fad_real/<style>/*.wav) and that step's generated wavs,
                     computed in the 128-D VGGish space. This is the same
                     "vector distance model" used for the trumpet->violin POC
                     (postprocessing/fad_eval.py).
  * Per-step PCA scatter figure (POC-style) — real vs generated embeddings
                     projected to 2D with Gaussian 2sigma/3sigma ellipses,
                     annotated with FAD + cosine similarity.
  * Per-step mean F1 — averaged over the step's songs, read from the run's
                     metrics.json (Basic-Pitch note-level F1).

Outputs (default --out-dir <version-root>/_finalist_metrics):
  fad_pca/<config>__step_<N>.png   per-step PCA scatter figures (12)
  fad_by_step.png                  grouped bar chart: FAD by step
  f1_by_step.png                   grouped bar chart: mean-F1 by step
  finalist_metrics_summary.json    machine-readable roll-up
  index.html                       browsable gallery of all of the above

Usage:
  python -m postprocessing.build_finalist_metrics \
    --version-root "G:/My Drive/MusicProject/versions/Israeli_3style"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import torch
from sklearn.decomposition import PCA

from postprocessing.fad_eval import (
    get_embedder,
    extract_embeddings_from_audio,
    compute_statistics,
    frechet_distance,
)
from postprocessing.fad_visualize import _draw_confidence_ellipse
from postprocessing.results_visualizations import _HTML_HEAD, _h, _rel

_DPI = 200

# ── The 12 finalists (4 configs x 3 steps) ──────────────────────────────────
# Steps are listed best-first per config. Military configs have 2 held-out
# songs; Artists configs have 3.
FINALISTS: List[Dict] = [
    {
        "config": "Artists_ddim100",
        "run": "Israeli_Artists_step_search_20260607",
        "style": "Israeli_Artists",
        "ddim": 100,
        "steps": [224000, 212000, 238000],
    },
    {
        "config": "Military_ddim100",
        "run": "Israeli_Military_step_search_20260607",
        "style": "Israeli_Military",
        "ddim": 100,
        "steps": [242000, 238000, 248000],
    },
    {
        "config": "Artists_ddim200",
        "run": "Israeli_Artists_step_search_20260706_ddim200",
        "style": "Israeli_Artists",
        "ddim": 200,
        "steps": [242000, 248000, 232000],
    },
    {
        "config": "Military_ddim200",
        "run": "Israeli_Military_step_search_20260706_ddim200",
        "style": "Israeli_Military",
        "ddim": 200,
        "steps": [238000, 224000, 232000],
    },
]

# Consistent colours per config for the grouped bar charts.
_CONFIG_COLORS = {
    "Artists_ddim100": "#1565C0",
    "Military_ddim100": "#2E7D32",
    "Artists_ddim200": "#6A1B9A",
    "Military_ddim200": "#C62828",
}


# ── Embedding helpers ───────────────────────────────────────────────────────
def _embed_files(files: List[Path], *, embedder, device, sr: int) -> Dict[str, np.ndarray]:
    """Return {stem: (n_windows, 128)} VGGish embeddings for each wav."""
    per_file: Dict[str, np.ndarray] = {}
    for f in files:
        audio, _ = librosa.load(str(f), sr=sr, mono=True)
        per_file[f.stem] = extract_embeddings_from_audio(
            audio, sr=sr, embedder=embedder, device=device
        )
    return per_file


def _step_wavs(audio_dir: Path, step: int) -> List[Path]:
    """Return the transferred-audio WAVs for one training step, sorted by name."""
    return sorted(audio_dir.glob(f"*__step_{step}__*role_transferred.wav"))


def _mean_f1_for_step(metrics_json: Path, step: int) -> Dict:
    """Average the per-stem F1 for one step from a run's metrics.json."""
    if not metrics_json.exists():
        return {"mean_f1": None, "per_song_f1": {}}
    with metrics_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    tag = f"__step_{step}__"
    per_song = {}
    for stem, vals in data.items():
        if tag in stem and isinstance(vals, dict) and vals.get("f1") is not None:
            song = stem.split("__step_")[0]
            per_song[song] = float(vals["f1"])
    mean_f1 = float(np.mean(list(per_song.values()))) if per_song else None
    return {"mean_f1": mean_f1, "per_song_f1": per_song}


def _load_f1_pairs(run_dir: Path) -> List[Dict]:
    """Load the per-pair F1 records (precision/recall/matched counts) for a run."""
    pair_json = run_dir / "metrics" / "f1_per_pair.json"
    if not pair_json.exists():
        return []
    with pair_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("per_pair", [])


def _agg_pr_for_step(pairs: List[Dict], step: int) -> Dict:
    """Micro-average precision/recall/F1 for one step (pool note counts across
    songs), plus per-song F1. Micro-averaging weights by note count, which is
    more principled than averaging per-song ratios."""
    rows = [p for p in pairs if p.get("step") == step]
    if not rows:
        return {"precision": None, "recall": None, "f1": None, "per_song_f1": {}}
    tot_matched = sum(int(p.get("matched", 0)) for p in rows)
    tot_pred = sum(int(p.get("n_predicted", 0)) for p in rows)
    tot_ref = sum(int(p.get("n_reference", 0)) for p in rows)
    precision = tot_matched / tot_pred if tot_pred else 0.0
    recall = tot_matched / tot_ref if tot_ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    per_song_f1 = {p["song"]: float(p.get("f1", 0.0)) for p in rows}
    return {"precision": precision, "recall": recall, "f1": f1,
            "per_song_f1": per_song_f1}


# ── PCA scatter (POC-style) ─────────────────────────────────────────────────
def _plot_pca_scatter(real_per_file, gen_per_file, fad, cos_sim,
                      out_png: Path, title: str) -> None:
    """Scatter real vs generated embeddings in 2-D PCA space.

    We squeeze the 128-D VGGish embeddings down to their two biggest directions
    (PCA) so we can actually see them, colour real clouds blue and generated
    clouds red, and draw 2-sigma / 3-sigma ellipses around each. If the two
    clouds overlap, the generated audio lives in the same region as the real
    audio, which is what a low FAD is telling us numerically.
    """
    all_real = np.concatenate(list(real_per_file.values()), axis=0)
    all_gen = np.concatenate(list(gen_per_file.values()), axis=0)

    combined = np.vstack([all_real, all_gen])
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    explained = pca.explained_variance_ratio_
    real_2d = combined_2d[: len(all_real)]
    gen_2d = combined_2d[len(all_real):]

    fig, ax = plt.subplots(figsize=(12, 9))
    real_colors = ["#2196F3", "#1565C0", "#0D47A1", "#42A5F5"]
    gen_colors = ["#F44336", "#E91E63", "#FF5722", "#FF7043"]

    idx = 0
    for i, (name, emb) in enumerate(real_per_file.items()):
        n = emb.shape[0]
        pts = real_2d[idx:idx + n]
        idx += n
        ax.scatter(pts[:, 0], pts[:, 1], c=real_colors[i % len(real_colors)],
                   alpha=0.25, s=12, label=f"Real: {name} ({n})")

    idx = 0
    for i, (name, emb) in enumerate(gen_per_file.items()):
        n = emb.shape[0]
        pts = gen_2d[idx:idx + n]
        idx += n
        short = name.split("__step_")[0]
        ax.scatter(pts[:, 0], pts[:, 1], c=gen_colors[i % len(gen_colors)],
                   alpha=0.35, s=18, marker="D", label=f"Gen: {short} ({n})")

    mu_r, cov_r = np.mean(real_2d, axis=0), np.cov(real_2d, rowvar=False)
    mu_g, cov_g = np.mean(gen_2d, axis=0), np.cov(gen_2d, rowvar=False)
    _draw_confidence_ellipse(ax, mu_r, cov_r, n_std=2.0, edgecolor="#1565C0",
                             linewidth=2.5, facecolor="#2196F3", alpha=0.08,
                             linestyle="-", label="Real 2σ")
    _draw_confidence_ellipse(ax, mu_r, cov_r, n_std=3.0, edgecolor="#1565C0",
                             linewidth=1.5, facecolor="none", alpha=0.3,
                             linestyle="--")
    _draw_confidence_ellipse(ax, mu_g, cov_g, n_std=2.0, edgecolor="#C62828",
                             linewidth=2.5, facecolor="#F44336", alpha=0.08,
                             linestyle="-", label="Generated 2σ")
    _draw_confidence_ellipse(ax, mu_g, cov_g, n_std=3.0, edgecolor="#C62828",
                             linewidth=1.5, facecolor="none", alpha=0.3,
                             linestyle="--")

    ax.set_xlabel(f"PC 1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC 2 ({explained[1] * 100:.1f}% var)")
    ax.set_title(f"{title}\nFAD = {fad:.4f}   |   cosine sim = {cos_sim:.4f}")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)


def _gaussian_pdf(x, mu, std):
    """Plain 1-D Gaussian density, used to draw the Fisher-axis bell curves."""
    std = max(float(std), 1e-8)
    return np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def _plot_fisher_bells(real_per_file, gen_per_file, fad, cos_sim,
                       out_png: Path, title: str) -> None:
    """1D Gaussian bell curves along the Fisher axis (line joining the means).

    Projects the 128-D VGGish embeddings onto the unit vector connecting the
    real and generated means, then plots the two 1D Gaussian densities so the
    overlap (or separation) is directly readable.
    """
    all_real = np.concatenate(list(real_per_file.values()), axis=0)
    all_gen = np.concatenate(list(gen_per_file.values()), axis=0)

    mu_real = np.mean(all_real, axis=0)
    mu_gen = np.mean(all_gen, axis=0)
    direction = mu_gen - mu_real
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-10:
        direction = np.ones_like(direction)
        dir_norm = np.linalg.norm(direction)
    direction = direction / dir_norm

    proj_real = all_real @ direction
    proj_gen = all_gen @ direction
    mu_r, std_r = float(np.mean(proj_real)), float(np.std(proj_real))
    mu_g, std_g = float(np.mean(proj_gen)), float(np.std(proj_gen))

    lo = min(mu_r - 4 * std_r, mu_g - 4 * std_g)
    hi = max(mu_r + 4 * std_r, mu_g + 4 * std_g)
    x = np.linspace(lo, hi, 500)
    pdf_real = _gaussian_pdf(x, mu_r, std_r)
    pdf_gen = _gaussian_pdf(x, mu_g, std_g)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(x, pdf_real, alpha=0.25, color="#2196F3")
    ax.plot(x, pdf_real, color="#1565C0", linewidth=2.5,
            label=f"Real ({len(proj_real)} win, \u03bc={mu_r:.2f}, \u03c3={std_r:.2f})")
    ax.fill_between(x, pdf_gen, alpha=0.25, color="#F44336")
    ax.plot(x, pdf_gen, color="#C62828", linewidth=2.5,
            label=f"Generated ({len(proj_gen)} win, \u03bc={mu_g:.2f}, \u03c3={std_g:.2f})")

    ax.axvline(mu_r, color="#1565C0", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(mu_g, color="#C62828", linewidth=1.5, linestyle="--", alpha=0.7)

    y_arrow = max(pdf_real.max(), pdf_gen.max()) * 0.85
    ax.annotate("", xy=(mu_g, y_arrow), xytext=(mu_r, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=2))
    ax.text((mu_r + mu_g) / 2, y_arrow * 1.05,
            f"separation = {abs(mu_g - mu_r):.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333")

    ax.set_xlabel("Projection onto Fisher axis (128-D \u2192 1-D)")
    ax.set_ylabel("probability density")
    ax.set_title(f"{title}\nFAD = {fad:.4f}   |   cosine sim = {cos_sim:.4f}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)

def _plot_style_overlay(style: str, real_emb: np.ndarray,
                        gens: List[Dict], out_png: Path) -> None:
    """Overlay every finalist of one style against its ground-truth bell.

    All bells share ONE axis (per style): the unit vector from the real mean to
    the pooled-generated mean. That makes the finalists directly comparable —
    the closest bell to the black ground-truth bell is the lowest-FAD version.

    gens: list of {"label", "emb", "fad"} for the 6 finalists of this style.
    """
    mu_real = np.mean(real_emb, axis=0)
    pooled_gen = np.concatenate([g["emb"] for g in gens], axis=0)
    direction = np.mean(pooled_gen, axis=0) - mu_real
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-10:
        direction = np.ones_like(direction)
        dir_norm = np.linalg.norm(direction)
    direction = direction / dir_norm

    proj_real = real_emb @ direction
    mu_r, std_r = float(np.mean(proj_real)), float(np.std(proj_real))

    gens_sorted = sorted(gens, key=lambda g: g["fad"])  # best (lowest FAD) first
    projs = [g["emb"] @ direction for g in gens_sorted]
    all_mu = [mu_r] + [float(np.mean(p)) for p in projs]
    all_sd = [std_r] + [float(np.std(p)) for p in projs]
    lo = min(m - 4 * s for m, s in zip(all_mu, all_sd))
    hi = max(m + 4 * s for m, s in zip(all_mu, all_sd))
    x = np.linspace(lo, hi, 600)

    fig, ax = plt.subplots(figsize=(14, 7))
    # Ground-truth bell (black, filled) — the reference every finalist chases.
    ax.fill_between(x, _gaussian_pdf(x, mu_r, std_r), alpha=0.18, color="#000000")
    ax.plot(x, _gaussian_pdf(x, mu_r, std_r), color="#000000", linewidth=3.5,
            label=f"GROUND TRUTH {style} (\u03bc={mu_r:.2f})")
    ax.axvline(mu_r, color="#000000", linewidth=1.5, linestyle=":", alpha=0.6)

    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(gens_sorted)))
    for i, (g, p) in enumerate(zip(gens_sorted, projs)):
        mu_g, std_g = float(np.mean(p)), float(np.std(p))
        closest = i == 0
        star = "\u2605 " if closest else ""
        ax.plot(x, _gaussian_pdf(x, mu_g, std_g), color=cmap[i],
                linewidth=3.0 if closest else 1.8,
                linestyle="-" if closest else "--",
                label=f"{star}{g['label']} (FAD {g['fad']:.3f})")

    ax.set_xlabel("Projection onto per-style Fisher axis (128-D \u2192 1-D)")
    ax.set_ylabel("probability density")
    ax.set_title(f"{style} \u2014 all 6 finalists vs ground truth (shared axis)\n"
                 f"\u2605 closest bell = lowest FAD = best match")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)

# ── Bar charts ───────────────────────────────────────────────────────────────────────────
def _bar_by_step(records: List[Dict], value_key: str, ylabel: str,
                 title: str, out_png: Path) -> None:
    """Draw a bar chart with one bar per finalist (config + step).

    Bars are coloured by config so the two samplers/styles are easy to tell
    apart. Missing values become NaN so the bar is simply left blank.
    """
    labels = [f"{r['config']}\nstep {r['step']}" for r in records]
    values = [r[value_key] if r[value_key] is not None else np.nan for r in records]
    colors = [_CONFIG_COLORS.get(r["config"], "#555555") for r in records]

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(records))
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    for rect, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)


def _plot_f1_fad_scatter(records: List[Dict], out_png: Path) -> None:
    """Trade-off scatter: x=FAD (lower better), y=F1 (higher better), one point
    per finalist. The Pareto frontier (non-dominated: lower FAD AND higher F1)
    is highlighted — those are the defensible 'best' picks."""
    pts = [r for r in records if r.get("mean_f1") is not None]
    fads = np.array([r["fad"] for r in pts])
    f1s = np.array([r["mean_f1"] for r in pts])

    # Pareto frontier: a point is non-dominated if no other has both lower FAD
    # and higher F1.
    pareto = []
    for i, r in enumerate(pts):
        dominated = any((fads[j] <= fads[i] and f1s[j] >= f1s[i] and
                         (fads[j] < fads[i] or f1s[j] > f1s[i]))
                        for j in range(len(pts)))
        if not dominated:
            pareto.append(i)
    pareto.sort(key=lambda i: fads[i])

    fig, ax = plt.subplots(figsize=(12, 8))
    for r, fa, f1 in zip(pts, fads, f1s):
        ax.scatter(fa, f1, s=90, color=_CONFIG_COLORS.get(r["config"], "#555"),
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"{r['config'].replace('_ddim', ' d')} s{r['step']}",
                    (fa, f1), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    if pareto:
        ax.plot(fads[pareto], f1s[pareto], "--", color="#333", linewidth=1.5,
                zorder=2, label="Pareto frontier")
        ax.scatter(fads[pareto], f1s[pareto], s=220, facecolor="none",
                   edgecolor="#333", linewidth=2.0, zorder=4)

    # De-duplicate config legend.
    seen = {}
    for r in pts:
        seen.setdefault(r["config"], _CONFIG_COLORS.get(r["config"], "#555"))
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c,
                          label=cfg, markersize=9) for cfg, c in seen.items()]
    handles.append(plt.Line2D([0], [0], linestyle="--", color="#333",
                              label="Pareto frontier"))
    ax.legend(handles=handles, loc="best", fontsize=9)

    ax.set_xlabel("FAD  (lower = more realistic)")
    ax.set_ylabel("mean note-F1  (higher = better score fidelity)")
    ax.set_title("F1 vs FAD trade-off — 12 finalist checkpoints\n"
                 "top-left is best (low FAD, high F1); circled = Pareto-optimal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)


def _plot_pr_f1_bars(pr_records: List[Dict], out_png: Path) -> None:
    """Grouped precision / recall / F1 bars per finalist — explains *why* F1 is
    low (precision = false positives, recall = misses)."""
    rows = [r for r in pr_records if r.get("f1") is not None]
    labels = [f"{r['config'].replace('_ddim', ' d')}\ns{r['step']}" for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]

    x = np.arange(len(rows))
    w = 0.27
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x - w, prec, w, label="precision", color="#1E88E5")
    ax.bar(x, rec, w, label="recall", color="#43A047")
    ax.bar(x + w, f1, w, label="F1", color="#E53935")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("score (0–1)")
    ax.set_title("Precision / Recall / F1 per finalist (micro-averaged over songs)\n"
                 "note-level, pitch + onset ±50 ms match")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)


def _plot_f1_heatmap(style: str, config_steps: List[Dict], out_png: Path) -> None:
    """Per-song × step F1 heatmap for one style (rows=songs, cols=finalist steps
    across both ddim configs). Shows which songs are easy/hard and the best step
    per song."""
    # Collect union of songs and the ordered column list.
    songs = sorted({s for cs in config_steps for s in cs["per_song_f1"].keys()})
    cols = [f"{cs['config'].replace('_ddim', ' d')}\ns{cs['step']}"
            for cs in config_steps]
    mat = np.full((len(songs), len(config_steps)), np.nan)
    for j, cs in enumerate(config_steps):
        for i, song in enumerate(songs):
            if song in cs["per_song_f1"]:
                mat[i, j] = cs["per_song_f1"][song]

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.2), max(3, len(songs))))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(np.arange(len(songs)))
    ax.set_yticklabels(songs, fontsize=9)
    for i in range(len(songs)):
        for j in range(len(config_steps)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="note-F1")
    ax.set_title(f"{style} — per-song F1 by finalist step")
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)


# ── Main build ──────────────────────────────────────────────────────────────
def build_finalist_metrics(version_root: Path, out_dir: Path, *,
                           sr: int = 22050, use_pretrained: bool = True) -> Dict:
    """Compute and plot every finalist metric for one version.

    For each finalist (a config x training-step pair) this embeds the generated
    and the real held-out audio with VGGish, computes FAD and cosine similarity,
    and renders the PCA scatter, the Fisher bell curves, the per-step FAD/F1 bar
    charts and a self-contained index.html. Returns the summary dict that is
    also written to finalist_metrics_summary.json.
    """
    version_root = Path(version_root)
    inference_runs = version_root / "inference_runs"
    fad_real_root = version_root / "fad_real"
    out_dir = Path(out_dir)
    pca_dir = out_dir / "fad_pca"
    bells_dir = out_dir / "fad_bells"
    overlay_dir = out_dir / "fad_bells_overlay"
    pca_dir.mkdir(parents=True, exist_ok=True)
    bells_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = get_embedder(device, use_pretrained=use_pretrained)
    pretrained = bool(getattr(embedder, "_fad_is_pretrained", False))
    embedder_label = getattr(embedder, "_fad_embedder_label", "unknown")
    print(f"[finalist-metrics] embedder: {embedder_label}  device={device}")

    # Real embeddings per style (computed once, reused across steps/configs).
    real_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _real_for(style: str) -> Dict[str, np.ndarray]:
        # Embed the real held-out wavs for a style once and cache them, since the
        # same real set is reused across every step and config of that style.
        if style not in real_cache:
            files = sorted((fad_real_root / style).glob("*.wav"))
            if not files:
                raise FileNotFoundError(f"No real wavs in {fad_real_root / style}")
            print(f"[finalist-metrics] embedding {len(files)} real '{style}' wavs")
            real_cache[style] = _embed_files(files, embedder=embedder,
                                             device=device, sr=sr)
        return real_cache[style]

    summary: Dict = {
        "embedder": embedder_label,
        "pretrained": pretrained,
        "sr": sr,
        "configs": {},
    }
    records: List[Dict] = []
    # Per-style accumulator for the shared-axis overlay: style -> real_emb + gens.
    style_overlay: Dict[str, Dict] = {}
    # Precision/recall/F1 rows for the grouped bars, and per-style rows for the
    # per-song heatmaps.
    pr_records: List[Dict] = []
    style_f1: Dict[str, List[Dict]] = {}

    for fin in FINALISTS:
        cfg = fin["config"]
        run_dir = inference_runs / fin["run"]
        audio_dir = run_dir / "audio"
        metrics_json = run_dir / "metrics.json"
        f1_pairs = _load_f1_pairs(run_dir)

        # Config-level pooled FAD (constant across stems in metrics.json).
        pooled_fad = None
        if metrics_json.exists():
            with metrics_json.open("r", encoding="utf-8") as fh:
                mdata = json.load(fh)
            for vals in mdata.values():
                if isinstance(vals, dict) and vals.get("fad") is not None:
                    pooled_fad = float(vals["fad"])
                    break

        real_per_file = _real_for(fin["style"])
        all_real = np.concatenate(list(real_per_file.values()), axis=0)
        mu_r, sigma_r = compute_statistics(all_real)
        if fin["style"] not in style_overlay:
            style_overlay[fin["style"]] = {"real_emb": all_real, "gens": []}

        cfg_entry = {
            "run": fin["run"],
            "style": fin["style"],
            "ddim": fin["ddim"],
            "pooled_fad": pooled_fad,
            "steps": {},
        }

        for step in fin["steps"]:
            gen_files = _step_wavs(audio_dir, step)
            if not gen_files:
                print(f"[finalist-metrics] WARN no gen wavs for {cfg} step {step}")
                continue
            gen_per_file = _embed_files(gen_files, embedder=embedder,
                                        device=device, sr=sr)
            all_gen = np.concatenate(list(gen_per_file.values()), axis=0)
            mu_g, sigma_g = compute_statistics(all_gen)
            fad = frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
            cos_sim = float((mu_r @ mu_g) /
                            (np.linalg.norm(mu_r) * np.linalg.norm(mu_g) + 1e-8))

            f1_info = _mean_f1_for_step(metrics_json, step)

            pca_png = pca_dir / f"{cfg}__step_{step}.png"
            _plot_pca_scatter(real_per_file, gen_per_file, fad, cos_sim,
                              pca_png, title=f"{cfg} — step {step}")
            bells_png = bells_dir / f"{cfg}__step_{step}.png"
            _plot_fisher_bells(real_per_file, gen_per_file, fad, cos_sim,
                               bells_png, title=f"{cfg} — step {step}")

            print(f"[finalist-metrics] {cfg} step {step}: "
                  f"FAD={fad:.4f} cos={cos_sim:.4f} "
                  f"F1={f1_info['mean_f1']}")

            cfg_entry["steps"][str(step)] = {
                "fad": fad,
                "cosine_sim": cos_sim,
                "mean_f1": f1_info["mean_f1"],
                "per_song_f1": f1_info["per_song_f1"],
                "n_gen_files": len(gen_files),
                "pca_png": _rel(pca_png, out_dir),
                "bells_png": _rel(bells_png, out_dir),
            }
            records.append({
                "config": cfg,
                "step": step,
                "fad": fad,
                "mean_f1": f1_info["mean_f1"],
                "pca_png": _rel(pca_png, out_dir),
                "bells_png": _rel(bells_png, out_dir),
            })
            style_overlay[fin["style"]]["gens"].append({
                "label": f"{cfg} s{step}",
                "emb": all_gen,
                "fad": fad,
            })
            pr = _agg_pr_for_step(f1_pairs, step)
            pr_records.append({
                "config": cfg,
                "step": step,
                "precision": pr["precision"],
                "recall": pr["recall"],
                "f1": pr["f1"],
            })
            style_f1.setdefault(fin["style"], []).append({
                "config": cfg,
                "step": step,
                "per_song_f1": pr["per_song_f1"],
            })

        summary["configs"][cfg] = cfg_entry

    # Grouped bar charts across all finalists.
    fad_png = out_dir / "fad_by_step.png"
    f1_png = out_dir / "f1_by_step.png"
    _bar_by_step(records, "fad", "FAD (lower is better)",
                 "Per-step FAD — 12 finalist checkpoints (held-out songs)", fad_png)
    _bar_by_step(records, "mean_f1", "mean note-F1 (higher is better)",
                 "Per-step mean F1 — 12 finalist checkpoints (held-out songs)", f1_png)

    # Per-style overlay: all finalists of a style vs its ground-truth bell.
    overlay_pngs: List[str] = []
    for style, data in style_overlay.items():
        ov_png = overlay_dir / f"overlay__{style}.png"
        _plot_style_overlay(style, data["real_emb"], data["gens"], ov_png)
        overlay_pngs.append(_rel(ov_png, out_dir))
        summary.setdefault("overlays", {})[style] = _rel(ov_png, out_dir)

    # F1-focused presentation charts.
    scatter_png = out_dir / "f1_vs_fad_scatter.png"
    pr_png = out_dir / "precision_recall_f1.png"
    _plot_f1_fad_scatter(records, scatter_png)
    _plot_pr_f1_bars(pr_records, pr_png)
    heatmap_pngs: List[str] = []
    for style, cs in style_f1.items():
        hm_png = out_dir / f"f1_heatmap__{style}.png"
        _plot_f1_heatmap(style, cs, hm_png)
        heatmap_pngs.append(_rel(hm_png, out_dir))
        summary.setdefault("f1_heatmaps", {})[style] = _rel(hm_png, out_dir)
    summary["f1_vs_fad_scatter"] = _rel(scatter_png, out_dir)
    summary["precision_recall_f1"] = _rel(pr_png, out_dir)

    summary_path = out_dir / "finalist_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _write_index_html(out_dir, records, summary, fad_png, f1_png, overlay_pngs,
                      scatter_png, pr_png, heatmap_pngs)
    print(f"[finalist-metrics] wrote {out_dir / 'index.html'}")
    return summary


def _write_index_html(out_dir: Path, records, summary, fad_png, f1_png,
                      overlay_pngs, scatter_png=None, pr_png=None,
                      heatmap_pngs=None) -> None:
    """Stitch every finalist plot and the summary table into one index.html."""
    parts = [_HTML_HEAD.format(title="Finalist Metrics — FAD &amp; F1 (held-out)")]
    note = ("VGGish 128-D embedding FAD (same vector-distance model as the "
            "trumpet→violin POC). Embedder: "
            f"<code>{_h(summary['embedder'])}</code>. "
            "FAD/F1 are on held-out target-style songs — the 5 external demo "
            "songs have no reference, so they are qualitative-only.")
    parts.append(f'<p class="note">{note}</p>')

    parts.append("<h2>Per-step FAD</h2>")
    parts.append(f'<img src="{_rel(fad_png, out_dir)}" alt="FAD by step">')
    parts.append("<h2>Per-step mean F1</h2>")
    parts.append(f'<img src="{_rel(f1_png, out_dir)}" alt="F1 by step">')

    if scatter_png is not None:
        parts.append("<h2>F1 vs FAD trade-off</h2>")
        parts.append('<p class="note">Each point is one finalist. Top-left is '
                     'best (low FAD, high F1); circled points are Pareto-optimal '
                     '(nothing beats them on both axes).</p>')
        parts.append(f'<img src="{_rel(scatter_png, out_dir)}" alt="F1 vs FAD">')
    if pr_png is not None:
        parts.append("<h2>Precision / Recall / F1</h2>")
        parts.append('<p class="note">Decomposes F1: precision = share of '
                     'transcribed notes that are correct; recall = share of '
                     'reference notes recovered (micro-averaged over songs).</p>')
        parts.append(f'<img src="{_rel(pr_png, out_dir)}" alt="precision recall F1">')
    if heatmap_pngs:
        parts.append("<h2>Per-song F1 heatmap</h2>")
        for hm in heatmap_pngs:
            parts.append(f'<img src="{_h(hm)}" alt="F1 heatmap">')

    # Auto-include piano-roll match overlays if the companion script has run.
    pianoroll_dir = out_dir / "f1_pianoroll"
    pianoroll_pngs = sorted(pianoroll_dir.glob("*.png")) if pianoroll_dir.exists() else []
    if pianoroll_pngs:
        parts.append("<h2>Piano-roll match overlay (best finalist per style)</h2>")
        parts.append('<p class="note">Note-level view of one finalist per style: '
                     '<b style="color:#2E7D32">green</b> = matched, '
                     '<b style="color:#C62828">red</b> = missed reference (drives '
                     'low recall), <b style="color:#EF6C00">orange</b> = false '
                     'positive. Cropped to the busiest window for readability.</p>')
        for pr_png2 in pianoroll_pngs:
            parts.append(f'<img src="{_rel(pr_png2, out_dir)}" alt="piano-roll overlay">')

    parts.append("<h2>Ground-truth overlay (all finalists per style, shared axis)</h2>")
    parts.append('<p class="note">Black bell = real target-style distribution. '
                 'Each coloured bell is one finalist projected onto the same '
                 'per-style axis; the \u2605 solid bell is the closest match '
                 '(lowest FAD).</p>')
    for ov in overlay_pngs:
        parts.append(f'<img src="{_h(ov)}" alt="style overlay">')

    parts.append("<h2>Summary table</h2>")
    parts.append("<table><tr><th>Config</th><th>Step</th><th>FAD</th>"
                 "<th>cosine</th><th>mean F1</th><th>pooled FAD</th></tr>")
    for cfg, entry in summary["configs"].items():
        for step, s in entry["steps"].items():
            f1 = "—" if s["mean_f1"] is None else f"{s['mean_f1']:.4f}"
            pooled = "—" if entry["pooled_fad"] is None else f"{entry['pooled_fad']:.4f}"
            parts.append(
                f"<tr><td>{_h(cfg)}</td><td>{_h(step)}</td>"
                f"<td>{s['fad']:.4f}</td><td>{s['cosine_sim']:.4f}</td>"
                f"<td>{f1}</td><td>{pooled}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>PCA scatter (real vs generated embeddings)</h2>")
    for r in records:
        cap = f"{r['config']} — step {r['step']} — FAD {r['fad']:.4f}"
        if r["mean_f1"] is not None:
            cap += f" — F1 {r['mean_f1']:.4f}"
        parts.append(f"<h3>{_h(cap)}</h3>")
        parts.append(f'<img src="{_h(r["pca_png"])}" alt="{_h(cap)}">')

    parts.append("<h2>Gaussian bell curves (Fisher 1D projection)</h2>")
    for r in records:
        cap = f"{r['config']} — step {r['step']} — FAD {r['fad']:.4f}"
        parts.append(f"<h3>{_h(cap)}</h3>")
        parts.append(f'<img src="{_h(r["bells_png"])}" alt="{_h(cap)}">')

    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    """CLI entry point: parse arguments and run build_finalist_metrics."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version-root", required=True,
                    help="Path to versions/Israeli_3style")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default <version-root>/_finalist_metrics)")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Force the deterministic fallback embedder")
    args = ap.parse_args()

    version_root = Path(args.version_root)
    out_dir = Path(args.out_dir) if args.out_dir else version_root / "_finalist_metrics"
    build_finalist_metrics(version_root, out_dir, sr=args.sr,
                           use_pretrained=not args.no_pretrained)


if __name__ == "__main__":
    main()
