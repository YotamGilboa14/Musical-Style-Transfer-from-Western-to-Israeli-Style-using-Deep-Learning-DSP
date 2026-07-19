"""Input-side dataset visualizations for the data-quality gate.

Every function:
  * accepts the same manifest ``df`` + ``manifest_root`` as
    :mod:`preprocessing.data_quality`;
  * uses matplotlib only (no seaborn, no holoviews);
  * saves a standalone PNG at ``save_path`` when given;
  * returns the matplotlib ``Figure`` so the caller can ``display(fig)``
    in a notebook;
  * targets a presentation-friendly ~1600x900 image at ``dpi=200`` with
    ``bbox_inches='tight'``.

These plots are reused by every style notebook (Slakh, Israeli, future).

Also provides :func:`plot_preprocessing_demo` — a single 6-panel PNG that
visualises every stage of the DSP preprocessing block (raw WAV → resample →
mel filter bank → log-mel → normalized mel + 5 s segments) for slides.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from preprocessing.data_quality import _dct_axis0, _resolve  # internal helpers


_DEFAULT_DPI = 200
_FIGSIZE_WIDE = (16, 9)


def _save_and_return(fig: plt.Figure, save_path: Optional[str | Path]) -> plt.Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=_DEFAULT_DPI)
    return fig


def plot_mel_grid(
    df: pd.DataFrame,
    manifest_root: str | Path,
    n: int = 8,
    seed: int = 0,
    save_path: Optional[str | Path] = None,
    title: str = "Sample mel spectrograms",
) -> plt.Figure:
    """Plot ``n`` random mel segments as a 2-row grid."""
    manifest_root = Path(manifest_root)
    rng = np.random.default_rng(seed)
    n = min(n, len(df))
    idxs = rng.choice(len(df), size=n, replace=False)
    cols = max(1, math.ceil(n / 2))
    fig, axes = plt.subplots(2, cols, figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
    axes = np.atleast_2d(axes).ravel()
    for ax, i in zip(axes, idxs):
        row = df.iloc[int(i)]
        p = _resolve(manifest_root, row["segment_path"])
        mel = torch.load(p, weights_only=True, map_location="cpu").numpy()
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma", vmin=-1, vmax=1)
        label = row.get("song_id", row["segment_path"])
        ax.set_title(str(label)[-40:], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(idxs):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_piano_roll_grid(
    df: pd.DataFrame,
    manifest_root: str | Path,
    n: int = 8,
    seed: int = 0,
    save_path: Optional[str | Path] = None,
    title: str = "Sample piano rolls (onset+sustain composite)",
) -> plt.Figure:
    """Plot ``n`` random piano rolls. Channels are summed for visualization."""
    manifest_root = Path(manifest_root)
    rng = np.random.default_rng(seed)
    n = min(n, len(df))
    idxs = rng.choice(len(df), size=n, replace=False)
    cols = max(1, math.ceil(n / 2))
    fig, axes = plt.subplots(2, cols, figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
    axes = np.atleast_2d(axes).ravel()
    for ax, i in zip(axes, idxs):
        row = df.iloc[int(i)]
        p = _resolve(manifest_root, row["score_path"])
        pr = torch.load(p, weights_only=True, map_location="cpu").numpy()
        composite = pr.sum(axis=0)  # [128, 430]
        ax.imshow(composite, aspect="auto", origin="lower", cmap="Greens", vmin=0, vmax=2)
        label = row.get("song_id", row["score_path"])
        ax.set_title(str(label)[-40:], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(idxs):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_segment_length_histogram(
    df: pd.DataFrame,
    manifest_root: str | Path,
    sample_n: int = 256,
    seed: int = 0,
    save_path: Optional[str | Path] = None,
    title: str = "Segment time-frame distribution",
) -> plt.Figure:
    """Histogram of mel time-frame counts (should be a single spike at 430)."""
    manifest_root = Path(manifest_root)
    rng = np.random.default_rng(seed)
    sample_n = min(sample_n, len(df))
    idxs = rng.choice(len(df), size=sample_n, replace=False)
    frames = []
    for i in idxs:
        p = _resolve(manifest_root, df.iloc[int(i)]["segment_path"])
        try:
            mel = torch.load(p, weights_only=True, map_location="cpu")
            frames.append(int(mel.shape[-1]))
        except Exception:  # noqa: BLE001
            continue
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
    if frames:
        ax.hist(frames, bins=20, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("frames per segment")
    ax.set_ylabel("count")
    ax.set_title(f"{title}  (n={len(frames)} sampled)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_mfcc_similarity_heatmap(
    df: pd.DataFrame,
    manifest_root: str | Path,
    sample_per_song: int = 3,
    n_mfcc: int = 20,
    seed: int = 2,
    save_path: Optional[str | Path] = None,
    title: str = "Per-song MFCC cosine similarity",
) -> plt.Figure:
    """Per-song mean-MFCC cosine similarity heatmap."""
    manifest_root = Path(manifest_root)
    if "song_id" not in df.columns:
        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
        ax.axis("off")
        ax.text(0.5, 0.5, "no song_id column — skipping", ha="center", va="center", fontsize=14)
        return _save_and_return(fig, save_path)
    rng = np.random.default_rng(seed)
    song_vecs: dict[str, np.ndarray] = {}
    for song_id, group in df.groupby("song_id"):
        take = min(sample_per_song, len(group))
        idxs = rng.choice(len(group), size=take, replace=False)
        seg_means = []
        for j in idxs:
            row = group.iloc[int(j)]
            p = _resolve(manifest_root, row["segment_path"])
            try:
                mel = torch.load(p, weights_only=True, map_location="cpu").numpy()
            except Exception:  # noqa: BLE001
                continue
            mfcc = _dct_axis0(mel)[:n_mfcc, :]
            seg_means.append(mfcc.mean(axis=1))
        if seg_means:
            song_vecs[str(song_id)] = np.mean(np.stack(seg_means), axis=0)
    songs = list(song_vecs.keys())
    M = np.stack([song_vecs[s] for s in songs]) if songs else np.zeros((0, n_mfcc))
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
    if len(songs) >= 2:
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
        Mn = M / norms
        sim = Mn @ Mn.T
        im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.set_xticks(range(len(songs)))
        ax.set_yticks(range(len(songs)))
        ax.set_xticklabels([s[-20:] for s in songs], rotation=90, fontsize=7)
        ax.set_yticklabels([s[-20:] for s in songs], fontsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "fewer than 2 songs", ha="center", va="center", fontsize=14)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_dataset_stats_panel(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
    title: str = "Dataset statistics",
) -> plt.Figure:
    """One-glance panel: segment count, total hours, version distribution,
    per-song segment counts."""
    fig, axes = plt.subplots(2, 2, figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)

    # --- top-left: scalar stats text panel ---
    ax = axes[0, 0]
    ax.axis("off")
    n_rows = len(df)
    n_songs = df["song_id"].nunique() if "song_id" in df.columns else float("nan")
    if "duration_s" in df.columns and df["duration_s"].notna().any():
        total_h = float(pd.to_numeric(df["duration_s"], errors="coerce").fillna(5.0).sum()) / 3600.0
    else:
        total_h = float(n_rows * 5.0 / 3600.0)
    versions = sorted(df["version_id"].unique().tolist()) if "version_id" in df.columns else []
    lines = [
        f"segments:  {n_rows:,}",
        f"songs:     {n_songs}",
        f"total:     {total_h:.2f} h",
        f"versions:  {versions}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=13)
    ax.set_title("scalars", fontsize=11, loc="left")

    # --- top-right: version distribution ---
    ax = axes[0, 1]
    if "version_id" in df.columns:
        counts = df["version_id"].value_counts().sort_index()
        ax.bar([str(v) for v in counts.index], counts.values, color="#2ca02c")
        ax.set_xlabel("version_id")
        ax.set_ylabel("segments")
    ax.set_title("version distribution", fontsize=11, loc="left")

    # --- bottom-left: per-song segment count histogram ---
    ax = axes[1, 0]
    if "song_id" in df.columns:
        per_song = df.groupby("song_id").size()
        ax.hist(per_song.values, bins=30, color="#ff7f0e", edgecolor="white")
        ax.set_xlabel("segments per song")
        ax.set_ylabel("count of songs")
    ax.set_title("segments per song", fontsize=11, loc="left")

    # --- bottom-right: split distribution if available ---
    ax = axes[1, 1]
    if "__split" in df.columns:
        counts = df["__split"].value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#9467bd")
        ax.set_xlabel("split")
        ax.set_ylabel("segments")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no split column", ha="center", va="center", fontsize=12)
    ax.set_title("train/val/test split", fontsize=11, loc="left")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Raw-WAV preprocessing demo (single PNG visualising every DSP stage)
# ──────────────────────────────────────────────────────────────────────────────

def plot_preprocessing_demo(
    wav_path: str | Path,
    midi_path: Optional[str | Path] = None,
    save_path: Optional[str | Path] = None,
    cfg=None,
    max_seconds: float = 30.0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Render a 6-panel figure showing the DSP preprocessing block end-to-end.

    Panels (figsize 16x9, dpi 200):
      1. Raw waveform at native SR (downsampling input)
      2. Linear STFT spectrogram of the raw signal (full bandwidth — pre-LPF)
      3. Resampled mono waveform at ``cfg.sample_rate`` (downsampled input)
      4. Mel filter bank overlay (visualises ``fmax`` as an LPF)
      5. Log-mel in dB, shape ``(n_mels, T)``
      6. Normalized mel in [-1, 1] with 5 s segment-boundary overlay

    If ``midi_path`` is provided, a companion piano-roll panel (onset+sustain
    composite) is rendered as a subplot below the main grid.

    Parameters
    ----------
    wav_path : str | Path
        Path to the raw WAV (any SR, mono or stereo).
    midi_path : str | Path, optional
        Optional MIDI file aligned with the WAV. Adds a 7th piano-roll panel.
    save_path : str | Path, optional
        If given, the figure is saved as PNG (dpi=200, bbox_inches='tight').
    cfg : DSPConfig, optional
        DSP config object; defaults to ``DSPConfig()`` (22050 Hz, 80 mels,
        fmax=8000, hop=256, segment=5 s).
    max_seconds : float
        Hard cap on the analysed duration to keep the figure readable
        regardless of song length (default 30 s).
    title : str, optional
        Optional figure suptitle. Defaults to the WAV filename.
    """
    import librosa  # local import: dataset_visualizations is also imported in light contexts

    from preprocessing.dsp_preprocessor import (
        DSPConfig,
        extract_mel_spectrogram,
        load_and_resample_audio,
        load_midi_to_piano_roll,
        normalize_mel,
    )

    if cfg is None:
        cfg = DSPConfig()

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    # ── (1) Raw audio at native SR ────────────────────────────────────────
    y_raw, sr_raw = librosa.load(str(wav_path), sr=None, mono=True,
                                 duration=max_seconds)
    t_raw = np.arange(len(y_raw)) / float(sr_raw)

    # ── (2) Linear STFT of raw signal (full bandwidth, log magnitude) ─────
    # n_fft chosen relative to native SR so the time grid resembles the mel.
    n_fft_raw = 2048
    hop_raw = n_fft_raw // 4
    S_raw = np.abs(librosa.stft(y_raw, n_fft=n_fft_raw, hop_length=hop_raw))
    S_raw_db = librosa.amplitude_to_db(S_raw + 1e-9, ref=np.max)

    # ── (3) Resample to model SR ───────────────────────────────────────────
    y_target = load_and_resample_audio(wav_path, cfg.sample_rate)
    max_samples = int(max_seconds * cfg.sample_rate)
    y_target = y_target[:max_samples]
    t_target = np.arange(len(y_target)) / float(cfg.sample_rate)

    # ── (4) Mel filter bank (visualises fmax LPF) ─────────────────────────
    mel_basis = librosa.filters.mel(
        sr=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
        fmin=cfg.fmin, fmax=cfg.fmax,
    )

    # ── (5) Log-mel in dB (project pipeline) ──────────────────────────────
    mel_db = extract_mel_spectrogram(y_target, cfg)

    # ── (6) Normalised mel ∈ [-1, 1] ──────────────────────────────────────
    mel_norm, m_min, m_max = normalize_mel(mel_db)

    has_midi = midi_path is not None and Path(midi_path).exists()
    piano_roll = None
    if has_midi:
        try:
            duration = len(y_target) / float(cfg.sample_rate)
            piano_roll = load_midi_to_piano_roll(Path(midi_path), cfg, duration)
        except Exception:  # noqa: BLE001
            has_midi = False
            piano_roll = None

    # ── Figure layout ─────────────────────────────────────────────────────
    n_rows = 4 if has_midi else 3
    fig = plt.figure(figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
    gs = fig.add_gridspec(n_rows, 2, hspace=0.55, wspace=0.18)

    # Panel 1: raw waveform
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_raw, y_raw, linewidth=0.5, color="#1f77b4")
    ax.set_xlim(0, t_raw[-1] if len(t_raw) else max_seconds)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"1. Raw waveform — native SR = {sr_raw} Hz, mono", fontsize=10, loc="left")

    # Panel 2: raw linear spectrogram (full bandwidth, pre-LPF)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(S_raw_db, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, t_raw[-1] if len(t_raw) else max_seconds, 0, sr_raw / 2])
    ax.axhline(cfg.fmax, color="cyan", linestyle="--", linewidth=1.0,
               label=f"fmax = {int(cfg.fmax)} Hz (future LPF)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    ax.set_title(f"2. Raw linear STFT (pre-LPF) — full {sr_raw // 2} Hz bandwidth",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="dB")

    # Panel 3: resampled waveform at target SR
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_target, y_target, linewidth=0.5, color="#2ca02c")
    ax.set_xlim(0, t_target[-1] if len(t_target) else max_seconds)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"3. Resampled waveform — {cfg.sample_rate} Hz, mono "
                 f"({len(y_target):,} samples)", fontsize=10, loc="left")

    # Panel 4: mel filter bank (LPF preview)
    ax = fig.add_subplot(gs[1, 1])
    freqs = np.linspace(0, cfg.sample_rate / 2, mel_basis.shape[1])
    for i in range(0, mel_basis.shape[0], 4):  # every 4th filter to avoid clutter
        ax.plot(freqs, mel_basis[i], linewidth=0.6, alpha=0.7)
    ax.axvline(cfg.fmax, color="red", linestyle="--", linewidth=1.0,
               label=f"fmax = {int(cfg.fmax)} Hz")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0, cfg.sample_rate / 2)
    ax.set_xlabel("Hz")
    ax.set_ylabel("filter weight")
    ax.set_title(f"4. Mel filter bank — {cfg.n_mels} mels, fmin={int(cfg.fmin)}, "
                 f"fmax={int(cfg.fmax)} (acts as LPF)", fontsize=10, loc="left")

    # Panel 5: log-mel in dB
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("frames (hop=256)")
    ax.set_ylabel("mel bin")
    ax.set_title(f"5. Log-mel (dB) — shape {tuple(mel_db.shape)}, "
                 f"range [{mel_db.min():.1f}, {mel_db.max():.1f}] dB",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="dB")

    # Panel 6: normalised mel + 5 s segment boundaries
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(mel_norm, aspect="auto", origin="lower", cmap="magma",
                   vmin=-1, vmax=1)
    seg_frames = cfg.segment_frames
    T = mel_norm.shape[-1]
    for k in range(seg_frames, T, seg_frames):
        ax.axvline(k, color="cyan", linestyle="--", linewidth=0.8)
    ax.set_xlabel("frames (hop=256)")
    ax.set_ylabel("mel bin")
    ax.set_title(f"6. Normalised mel ∈ [-1, 1] — segment length = "
                 f"{seg_frames} frames ({cfg.segment_duration:.0f} s); "
                 f"orig dB range [{m_min:.1f}, {m_max:.1f}]",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="[-1, 1]")

    # Panel 7 (optional): piano-roll companion
    if has_midi and piano_roll is not None:
        ax = fig.add_subplot(gs[3, :])
        composite = np.asarray(piano_roll).sum(axis=0)  # [128, T]
        im = ax.imshow(composite, aspect="auto", origin="lower", cmap="Greens",
                       vmin=0, vmax=2)
        ax.set_xlabel("frames (hop=256, aligned with mel)")
        ax.set_ylabel("MIDI pitch")
        ax.set_title(f"7. Piano-roll companion (onset+sustain) — shape "
                     f"{tuple(piano_roll.shape)}", fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="onset+sustain")

    fig.suptitle(title or f"Preprocessing demo — {wav_path.name}",
                 fontsize=13, fontweight="bold")
    return _save_and_return(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI: produce a preprocessing demo PNG for any WAV (+ optional MIDI)
# ──────────────────────────────────────────────────────────────────────────────

def _main() -> int:
    """CLI entry point for one-off preprocessing demo PNGs.

    Examples::

        python -m preprocessing.dataset_visualizations --demo path/to/song.wav \\
            --midi path/to/song.mid --out song_demo.png
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="Render a 6-panel preprocessing-demo PNG for one WAV "
                    "(downsample, LPF, mel, normalize, segment)."
    )
    ap.add_argument("--demo", required=True, help="Path to the raw WAV")
    ap.add_argument("--midi", default=None, help="Optional aligned MIDI")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="Trim analysed duration (default: 30 s)")
    args = ap.parse_args()

    fig = plot_preprocessing_demo(
        wav_path=args.demo,
        midi_path=args.midi,
        save_path=args.out,
        max_seconds=args.max_seconds,
    )
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
