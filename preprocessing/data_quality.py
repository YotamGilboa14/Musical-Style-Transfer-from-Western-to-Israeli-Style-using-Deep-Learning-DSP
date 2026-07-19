"""Pre-training data-quality gate for style-transfer training datasets.

This module provides reusable checks that verify a preprocessed dataset
(manifest CSV + segment tensors on disk) is fit for training. It is intended
to be the SAME gate for every style version — Slakh rock v0, Israeli v1,
and any future v2+. Notebook integration is a thin layer that calls
:func:`run_full_gate` and renders the resulting :class:`GateReport`.

Manifest schema (matches data.dataset.MelPianoRollDataset):
    segment_path   relative path to mel tensor   [80, 430]
    score_path     relative path to piano roll   [2, 128, 430]
    version_id     int
    (optional)     song_id, artist, album, song_name, segment_idx, duration_s

Each check returns a :class:`GateResult`. :func:`run_full_gate` aggregates
all results into a :class:`GateReport` whose ``assert_pass()`` raises if any
required check returned ``FAIL`` (``WARN`` does not block).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Severity constants
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

# Default expected shapes — keep in sync with configs/default.yaml.
EXPECTED_MEL_SHAPE: Tuple[int, int] = (80, 430)
EXPECTED_PR_SHAPE: Tuple[int, int, int] = (2, 128, 430)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    """Single quality-gate check outcome."""

    name: str
    status: str  # PASS | FAIL | WARN
    value: object = None
    threshold: object = None
    message: str = ""

    def as_row(self) -> dict:
        return {
            "check": self.name,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class GateReport:
    """Aggregated gate report with one row per check."""

    results: List[GateResult] = field(default_factory=list)

    # ------------- introspection -------------
    @property
    def overall(self) -> str:
        if any(r.status == FAIL for r in self.results):
            return FAIL
        if any(r.status == WARN for r in self.results):
            return WARN
        return PASS

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.as_row() for r in self.results])

    # ------------- console / asserts -------------
    def print_summary(self) -> None:
        df = self.to_dataframe()
        print("=" * 78)
        print(f"  DATA QUALITY GATE — overall: {self.overall}")
        print("=" * 78)
        if df.empty:
            print("  (no checks ran)")
            return
        for _, row in df.iterrows():
            marker = {"PASS": "OK", "FAIL": "!!", "WARN": "**"}.get(row["status"], "  ")
            v = row["value"]
            t = row["threshold"]
            tail = f"  value={v}" if v is not None else ""
            tail += f"  threshold={t}" if t is not None else ""
            print(f"  [{marker}] {row['check']:<32} {row['status']:<5}{tail}")
            if row["message"]:
                print(f"        {row['message']}")
        print("=" * 78)

    def assert_pass(self) -> None:
        """Raise RuntimeError if any check is FAIL. WARNs are allowed."""
        fails = [r for r in self.results if r.status == FAIL]
        if fails:
            names = ", ".join(r.name for r in fails)
            raise RuntimeError(
                f"Data-quality gate FAILED on {len(fails)} check(s): {names}. "
                f"Refusing to start training. Run report.print_summary() for details."
            )

    # ------------- persistence -------------
    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"overall": self.overall, "results": [r.as_row() for r in self.results]},
                fh,
                indent=2,
                default=str,
            )

    def write_html(self, path: str | Path, title: str = "Data Quality Gate") -> None:
        """Write a small standalone HTML table. No JS, no external CSS."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        colors = {"PASS": "#1a7f37", "FAIL": "#cf222e", "WARN": "#9a6700"}
        rows_html = []
        for r in self.results:
            color = colors.get(r.status, "#333")
            rows_html.append(
                f"<tr>"
                f"<td>{_h(r.name)}</td>"
                f"<td style='color:{color};font-weight:600'>{r.status}</td>"
                f"<td>{_h(r.value)}</td>"
                f"<td>{_h(r.threshold)}</td>"
                f"<td>{_h(r.message)}</td>"
                f"</tr>"
            )
        overall_color = colors.get(self.overall, "#333")
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_h(title)}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; padding:24px; color:#1f2328 }}
table {{ border-collapse:collapse; width:100%; max-width:1100px }}
th, td {{ padding:8px 12px; border-bottom:1px solid #d0d7de; vertical-align:top; text-align:left }}
th {{ background:#f6f8fa }}
.overall {{ font-size:20px; font-weight:700; color:{overall_color}; margin:12px 0 24px 0 }}
</style></head><body>
<h1>{_h(title)}</h1>
<div class="overall">Overall: {self.overall}</div>
<table>
<thead><tr><th>Check</th><th>Status</th><th>Value</th><th>Threshold</th><th>Message</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
</body></html>
"""
        path.write_text(html, encoding="utf-8")

    def write_png_table(self, path: str | Path, title: str = "Data Quality Gate") -> None:
        """Render the report as a matplotlib table PNG (for slide decks)."""
        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        if df.empty:
            df = pd.DataFrame([{"check": "(none)", "status": "PASS", "value": "", "threshold": "", "message": ""}])
        fig, ax = plt.subplots(figsize=(13, 0.5 + 0.45 * len(df)), dpi=200)
        ax.axis("off")
        ax.set_title(f"{title}  —  overall: {self.overall}", loc="left", fontsize=13, fontweight="bold")
        cell_text = df[["check", "status", "value", "threshold", "message"]].astype(str).values.tolist()
        table = ax.table(
            cellText=cell_text,
            colLabels=["check", "status", "value", "threshold", "message"],
            loc="upper left",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)
        # Color status cells
        color_map = {"PASS": "#dafbe1", "FAIL": "#ffebe9", "WARN": "#fff8c5"}
        for i, status in enumerate(df["status"].tolist(), start=1):
            cell = table[(i, 1)]
            cell.set_facecolor(color_map.get(status, "#ffffff"))
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close(fig)


def _h(x) -> str:
    """Minimal HTML escape."""
    if x is None:
        return ""
    s = str(x)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: Tuple[str, ...] = ("segment_path", "score_path", "version_id")


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


def _is_frame_jitter(shape: tuple, expected: tuple, frame_tol: int) -> bool:
    """True if ``shape`` matches ``expected`` except for a small time-axis delta.

    All dimensions except the last must match exactly, and the last (time/frame)
    dimension must differ by 1..``frame_tol`` frames. This identifies benign
    augmentation framing jitter that the dataset loader clamps at load time.
    """
    if len(shape) != len(expected):
        return False
    if shape[:-1] != expected[:-1]:
        return False
    delta = abs(shape[-1] - expected[-1])
    return 0 < delta <= frame_tol


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_manifest_sanity(
    df: pd.DataFrame,
    expected_version_id: Optional[int] = None,
) -> GateResult:
    """Required columns, no NaN in required cols, unique segment paths."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return GateResult(
            "manifest_sanity", FAIL,
            value=f"missing={missing}", threshold=list(REQUIRED_COLUMNS),
            message="Manifest is missing required columns.",
        )
    nan_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    if any(nan_counts.values()):
        return GateResult(
            "manifest_sanity", FAIL,
            value=nan_counts, threshold="all=0",
            message="Required columns contain NaN values.",
        )
    n_dupes = int(df["segment_path"].duplicated().sum())
    if n_dupes > 0:
        return GateResult(
            "manifest_sanity", FAIL,
            value=n_dupes, threshold=0,
            message=f"{n_dupes} duplicate segment_path entries.",
        )
    if expected_version_id is not None:
        bad = df[df["version_id"].astype(int) != int(expected_version_id)]
        if len(bad) > 0:
            return GateResult(
                "manifest_sanity", FAIL,
                value=int(len(bad)), threshold=0,
                message=f"{len(bad)} rows have version_id != {expected_version_id}.",
            )
    return GateResult(
        "manifest_sanity", PASS,
        value=f"n_rows={len(df)}",
        message="Required columns present, no NaN, paths unique.",
    )


def check_duration(
    df: pd.DataFrame,
    min_hours: float = 3.0,
    fallback_segment_seconds: float = 5.0,
) -> GateResult:
    """Total dataset duration in hours must be >= min_hours.

    Uses ``duration_s`` column if present, otherwise assumes
    ``fallback_segment_seconds`` per row.
    """
    if "duration_s" in df.columns and df["duration_s"].notna().any():
        total_s = float(pd.to_numeric(df["duration_s"], errors="coerce").fillna(fallback_segment_seconds).sum())
        source = "duration_s column"
    else:
        total_s = float(len(df) * fallback_segment_seconds)
        source = f"fallback {fallback_segment_seconds}s/segment"
    total_h = total_s / 3600.0
    status = PASS if total_h >= min_hours else FAIL
    return GateResult(
        "duration",
        status,
        value=f"{total_h:.2f} h ({len(df)} segments)",
        threshold=f">= {min_hours} h",
        message=f"Source: {source}.",
    )


def check_segment_length_distribution(
    df: pd.DataFrame,
    manifest_root: Path,
    expected_frames: int = EXPECTED_MEL_SHAPE[1],
    sample_n: int = 64,
    rng_seed: int = 0,
    frame_tol: int = 2,
) -> GateResult:
    """All inspected mels share the expected time-frame count.

    Samples ``sample_n`` rows to keep the check fast on big datasets. A time
    axis off by <= ``frame_tol`` frames is treated as benign augmentation
    framing jitter (the dataset loader clamps it) and reported as WARN, not
    FAIL; larger deltas or wrong ndim FAIL.
    """
    if len(df) == 0:
        return GateResult("segment_length", FAIL, value=0, threshold=expected_frames,
                          message="Empty manifest.")
    rng = np.random.default_rng(rng_seed)
    idxs = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    bad = []
    jitter = []
    for i in idxs:
        row = df.iloc[int(i)]
        p = _resolve(manifest_root, row["segment_path"])
        try:
            t = torch.load(p, weights_only=True, map_location="cpu")
            if t.ndim != 2:
                bad.append((str(p), tuple(t.shape)))
            elif t.shape[1] != expected_frames:
                if abs(t.shape[1] - expected_frames) <= frame_tol:
                    jitter.append((str(p), tuple(t.shape)))
                else:
                    bad.append((str(p), tuple(t.shape)))
        except Exception as e:  # noqa: BLE001
            bad.append((str(p), f"load_error: {e!r}"))
    if bad:
        return GateResult(
            "segment_length", FAIL,
            value=f"{len(bad)}/{len(idxs)} bad",
            threshold=f"frames=={expected_frames}",
            message=f"First mismatch: {bad[0]}",
        )
    if jitter:
        return GateResult(
            "segment_length", WARN,
            value=f"{len(jitter)}/{len(idxs)} within +/-{frame_tol} frames",
            threshold=f"frames=={expected_frames}",
            message=f"Benign framing jitter (clamped at load). First: {jitter[0]}",
        )
    return GateResult(
        "segment_length", PASS,
        value=f"{len(idxs)} sampled, all frames={expected_frames}",
        threshold=f"frames=={expected_frames}",
    )


def check_tensor_health(
    df: pd.DataFrame,
    manifest_root: Path,
    expected_mel_shape: Tuple[int, int] = EXPECTED_MEL_SHAPE,
    expected_pr_shape: Tuple[int, int, int] = EXPECTED_PR_SHAPE,
    sample_n: int = 64,
    rng_seed: int = 1,
    frame_tol: int = 2,
) -> Tuple[GateResult, pd.DataFrame]:
    """Open ``sample_n`` mel/score pairs and verify shape, dtype, contiguous.

    Returns (result, broken_df). ``broken_df`` lists rows that failed
    inspection (useful for the visualizer or a follow-up ``fix_non_contiguous``).
    A time axis off by <= ``frame_tol`` frames (all other dims matching) is
    benign augmentation framing jitter that the loader clamps, so it is a WARN
    rather than a FAIL.
    """
    if len(df) == 0:
        return GateResult("tensor_health", FAIL, value=0, threshold="n>0",
                          message="Empty manifest."), pd.DataFrame()
    rng = np.random.default_rng(rng_seed)
    idxs = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    broken = []
    n_noncontig = 0
    for i in idxs:
        row = df.iloc[int(i)]
        for col, expected_shape in (("segment_path", expected_mel_shape),
                                    ("score_path", expected_pr_shape)):
            p = _resolve(manifest_root, row[col])
            try:
                t = torch.load(p, weights_only=True, map_location="cpu")
                problems = []
                serious = False
                if tuple(t.shape) != tuple(expected_shape):
                    if _is_frame_jitter(tuple(t.shape), tuple(expected_shape), frame_tol):
                        problems.append(
                            f"frame_jitter shape={tuple(t.shape)} ~ {expected_shape} "
                            "(clamped at load)")
                    else:
                        problems.append(f"shape={tuple(t.shape)} != {expected_shape}")
                        serious = True
                if t.dtype != torch.float32:
                    problems.append(f"dtype={t.dtype} != float32")
                    serious = True
                if not t.is_contiguous():
                    n_noncontig += 1
                    problems.append("non-contiguous")
                if problems:
                    broken.append({
                        "row": int(i),
                        "column": col,
                        "path": str(p),
                        "problems": "; ".join(problems),
                        "serious": serious,
                    })
            except Exception as e:  # noqa: BLE001
                broken.append({
                    "row": int(i),
                    "column": col,
                    "path": str(p),
                    "problems": f"load_error: {e!r}",
                    "serious": True,
                })
    broken_df = pd.DataFrame(broken)
    # Serious = wrong dtype, large/structural shape mismatch, or load error.
    # Frame jitter (<= frame_tol) and non-contiguous alone are WARN (fixable).
    serious = broken_df[broken_df["serious"]] if not broken_df.empty else broken_df
    if not serious.empty:
        return GateResult(
            "tensor_health", FAIL,
            value=f"{len(serious)}/{len(idxs)*2} broken (mel+pr)",
            threshold="0",
            message=f"First: {serious.iloc[0].to_dict()}",
        ), broken_df
    n_jitter = int(broken_df["problems"].str.contains("frame_jitter").sum()) if not broken_df.empty else 0
    if n_jitter > 0 or n_noncontig > 0:
        parts = []
        if n_jitter:
            parts.append(f"{n_jitter} frame-jitter (clamped at load)")
        if n_noncontig:
            parts.append(f"{n_noncontig} non-contiguous")
        return GateResult(
            "tensor_health", WARN,
            value="; ".join(parts),
            threshold="0",
            message="Benign: loader clamps frames; call fix_non_contiguous(...) to repair storage.",
        ), broken_df
    return GateResult(
        "tensor_health", PASS,
        value=f"{len(idxs)} sampled, shapes/dtype/contiguous OK",
    ), broken_df


def fix_non_contiguous(
    df: pd.DataFrame,
    manifest_root: Path,
    dry_run: bool = True,
) -> int:
    """Re-save any non-contiguous mel/score tensors as ``.contiguous()``.

    Returns the number of files that were (or would be) fixed.
    """
    fixed = 0
    for i in range(len(df)):
        row = df.iloc[i]
        for col in ("segment_path", "score_path"):
            p = _resolve(manifest_root, row[col])
            try:
                t = torch.load(p, weights_only=True, map_location="cpu")
            except Exception:  # noqa: BLE001
                continue
            if not t.is_contiguous():
                fixed += 1
                if not dry_run:
                    torch.save(t.contiguous(), p)
    return fixed


def check_mfcc_consistency(
    df: pd.DataFrame,
    manifest_root: Path,
    sample_per_song: int = 3,
    z_threshold: float = 3.0,
    n_mfcc: int = 20,
    rng_seed: int = 2,
) -> Tuple[GateResult, pd.DataFrame]:
    """Approximate MFCC from the saved mel and flag stylistic outlier songs.

    We compute MFCC = DCT(log-mel) per segment, take the mean MFCC vector per
    song, compute pairwise cosine similarities, and flag any song whose mean
    similarity z-score is below ``-z_threshold``. This is a coarse but cheap
    style-consistency proxy that does not need raw audio.
    """
    if "song_id" not in df.columns:
        return GateResult(
            "mfcc_consistency", WARN,
            value="no song_id column", threshold="present",
            message="Skipping — manifest has no song_id column.",
        ), pd.DataFrame()
    rng = np.random.default_rng(rng_seed)
    song_mean_mfcc: dict[str, np.ndarray] = {}
    n_failed_loads = 0
    for song_id, group in df.groupby("song_id"):
        take = min(sample_per_song, len(group))
        idxs = rng.choice(len(group), size=take, replace=False)
        segs: List[np.ndarray] = []
        for j in idxs:
            row = group.iloc[int(j)]
            p = _resolve(manifest_root, row["segment_path"])
            try:
                mel = torch.load(p, weights_only=True, map_location="cpu").numpy()
            except Exception:  # noqa: BLE001
                n_failed_loads += 1
                continue
            # mel in [-1, 1] is already in a log-scale-ish domain (preprocessing
            # log-mel normalized). Treat as log-mel directly for the DCT.
            mfcc = _dct_axis0(mel)[:n_mfcc, :]  # [n_mfcc, frames]
            segs.append(mfcc.mean(axis=1))  # mean across time -> [n_mfcc]
        if segs:
            song_mean_mfcc[str(song_id)] = np.mean(np.stack(segs), axis=0)
    songs = list(song_mean_mfcc.keys())
    if len(songs) < 3:
        return GateResult(
            "mfcc_consistency", WARN,
            value=f"only {len(songs)} songs", threshold=">=3",
            message="Need at least 3 songs to flag outliers.",
        ), pd.DataFrame()
    M = np.stack([song_mean_mfcc[s] for s in songs])  # [S, n_mfcc]
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    Mn = M / norms
    sim = Mn @ Mn.T  # cosine similarity matrix
    # mean off-diagonal similarity per song
    np.fill_diagonal(sim, np.nan)
    mean_sim = np.nanmean(sim, axis=1)
    z = (mean_sim - mean_sim.mean()) / (mean_sim.std() + 1e-9)
    outliers = pd.DataFrame({
        "song_id": songs,
        "mean_cosine_sim": mean_sim,
        "z_score": z,
    }).sort_values("z_score")
    flagged = outliers[outliers["z_score"] < -z_threshold]
    if len(flagged) > 0:
        return GateResult(
            "mfcc_consistency", WARN,
            value=f"{len(flagged)} outlier song(s)",
            threshold=f"z >= -{z_threshold}",
            message=f"Outliers: {', '.join(flagged['song_id'].tolist())}",
        ), outliers
    return GateResult(
        "mfcc_consistency", PASS,
        value=f"{len(songs)} songs, no outliers (z >= -{z_threshold})",
    ), outliers


def _dct_axis0(x: np.ndarray) -> np.ndarray:
    """Type-II DCT along axis 0 (numpy-only, dependency-light)."""
    n = x.shape[0]
    k = np.arange(n)[:, None]
    n_idx = np.arange(n)[None, :]
    basis = np.cos(np.pi * (n_idx + 0.5) * k / n)
    return basis @ x


def check_rms_distribution(
    df: pd.DataFrame,
    manifest_root: Path,
    sample_n: int = 64,
    silence_db: float = -45.0,
    rng_seed: int = 3,
) -> GateResult:
    """WARN-level check that flags segments whose mel energy looks silent.

    Computes mean-squared mel value per segment, converts to dB. Flags
    segments whose dB falls below ``silence_db``.
    """
    if len(df) == 0:
        return GateResult("rms_distribution", WARN, value=0, message="Empty manifest.")
    rng = np.random.default_rng(rng_seed)
    idxs = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    dbs = []
    for i in idxs:
        row = df.iloc[int(i)]
        p = _resolve(manifest_root, row["segment_path"])
        try:
            mel = torch.load(p, weights_only=True, map_location="cpu").numpy()
        except Exception:  # noqa: BLE001
            continue
        # mel is normalized log-mel in [-1,1]; map to "energy-like" value
        energy = float(np.mean(mel ** 2))
        dbs.append(10.0 * math.log10(energy + 1e-12))
    if not dbs:
        return GateResult("rms_distribution", WARN, value="no readable mels",
                          message="Could not load any mel for RMS check.")
    dbs_arr = np.array(dbs)
    n_silent = int((dbs_arr < silence_db).sum())
    msg = f"min={dbs_arr.min():.1f} dB, mean={dbs_arr.mean():.1f} dB, max={dbs_arr.max():.1f} dB"
    if n_silent > 0:
        return GateResult(
            "rms_distribution", WARN,
            value=f"{n_silent}/{len(dbs)} below {silence_db} dB",
            threshold=f">= {silence_db} dB",
            message=msg,
        )
    return GateResult(
        "rms_distribution", PASS,
        value=f"{len(dbs)} sampled, all >= {silence_db} dB",
        message=msg,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_full_gate(
    manifest_df: pd.DataFrame,
    manifest_root: str | Path,
    *,
    expected_version_id: Optional[int] = None,
    min_hours: float = 3.0,
    expected_mel_shape: Tuple[int, int] = EXPECTED_MEL_SHAPE,
    expected_pr_shape: Tuple[int, int, int] = EXPECTED_PR_SHAPE,
    sample_n_health: int = 64,
    sample_per_song_mfcc: int = 3,
    silence_db: float = -45.0,
    frame_tol: int = 2,
) -> GateReport:
    """Run every check and return a single :class:`GateReport`.

    Args:
        manifest_df: pandas DataFrame already loaded from the (combined)
            manifest CSV — typically train+val+test concatenated.
        manifest_root: directory used to resolve relative paths in the
            manifest (usually the manifest CSV's parent).
        expected_version_id: if set, enforce all rows match.
        min_hours: minimum total dataset duration (FAIL otherwise).
    """
    manifest_root = Path(manifest_root)
    results: List[GateResult] = []

    sanity = check_manifest_sanity(manifest_df, expected_version_id=expected_version_id)
    results.append(sanity)
    # If the manifest itself is broken there's no point inspecting tensors.
    if sanity.status == FAIL:
        return GateReport(results=results)

    results.append(check_duration(manifest_df, min_hours=min_hours))
    results.append(check_segment_length_distribution(
        manifest_df, manifest_root,
        expected_frames=expected_mel_shape[1],
        sample_n=sample_n_health,
        frame_tol=frame_tol,
    ))
    tensor_res, _ = check_tensor_health(
        manifest_df, manifest_root,
        expected_mel_shape=expected_mel_shape,
        expected_pr_shape=expected_pr_shape,
        sample_n=sample_n_health,
        frame_tol=frame_tol,
    )
    results.append(tensor_res)
    mfcc_res, _ = check_mfcc_consistency(
        manifest_df, manifest_root,
        sample_per_song=sample_per_song_mfcc,
    )
    results.append(mfcc_res)
    results.append(check_rms_distribution(
        manifest_df, manifest_root,
        silence_db=silence_db,
    ))
    return GateReport(results=results)


def load_combined_manifest(splits_dir: str | Path) -> Tuple[pd.DataFrame, Path]:
    """Concatenate train/val/test split CSVs into one DataFrame.

    Returns (combined_df, manifest_root) where ``manifest_root`` is
    ``splits_dir`` (paths in split CSVs are usually relative to it).
    """
    splits_dir = Path(splits_dir)
    parts = []
    for name in ("train.csv", "val.csv", "test.csv"):
        p = splits_dir / name
        if p.exists():
            df = pd.read_csv(p)
            df["__split"] = name.replace(".csv", "")
            parts.append(df)
    if not parts:
        raise FileNotFoundError(
            f"No split CSVs found under {splits_dir}. Expected train.csv/val.csv/test.csv."
        )
    return pd.concat(parts, ignore_index=True), splits_dir
