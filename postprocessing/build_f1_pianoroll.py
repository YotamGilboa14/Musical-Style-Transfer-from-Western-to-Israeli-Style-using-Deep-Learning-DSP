"""
F1 Piano-Roll Match Overlay
===========================
Renders a note-level piano-roll that visualises *why* a finalist gets the F1
score it does. For a chosen (run, step, song) it:

    1. transcribes the generated WAV with Basic-Pitch (via
       ``f1_eval.compute_f1_detailed``),
    2. greedily matches predicted notes to the reference MIDI
       (pitch + onset ±tolerance), and
    3. draws a piano-roll cropped to the busiest window, colouring:

        - matched notes            → green   (model reproduced the pitch/time)
        - missed reference notes   → red     (in the score, not in the audio)
        - false-positive predicts  → orange  (in the audio, not in the score)

Because the reference is a full-mix Basic-Pitch transcription (~1500 notes),
the whole song is unreadable, so we crop to the ~DEFAULT_WINDOW_S seconds that
contain the most matched notes.

This script MUST run in the Basic-Pitch environment (Python 3.10), e.g.::

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    .\\basic_pitch_env\\Scripts\\python.exe -m postprocessing.build_f1_pianoroll \\
        --version-root "G:\\My Drive\\MusicProject\\versions\\Israeli_3style"

Author: Yotam & Gal — StyleTransfer Music Project
Date: July 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from postprocessing.f1_eval import compute_f1_detailed

# ── Configuration ────────────────────────────────────────────────────────────
# The best finalist (lowest per-step FAD) for each style. One overlay per
# held-out song of that style — a small, presentation-sized set.
BEST_FINALISTS = [
    {"config": "Artists_ddim100", "run": "Israeli_Artists_step_search_20260607",
     "style": "Israeli_Artists", "step": 224000},
    {"config": "Military_ddim100", "run": "Israeli_Military_step_search_20260607",
     "style": "Israeli_Military", "step": 238000},
]

DEFAULT_WINDOW_S = 14.0   # crop width for readability
_DPI = 200
_MATCH_C = "#2E7D32"      # green
_MISS_C = "#C62828"       # red
_FP_C = "#EF6C00"         # orange


def _load_pairs_for_step(run_dir: Path, step: int) -> List[Dict]:
    """Return the per-song F1 records for one training step.

    Reads the run's metrics/f1_per_pair.json and keeps only the rows whose
    step matches. Returns an empty list if the file does not exist yet.
    """
    pair_json = run_dir / "metrics" / "f1_per_pair.json"
    if not pair_json.exists():
        return []
    with pair_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [p for p in data.get("per_pair", []) if p.get("step") == step]


def _best_window(matched_ref_onsets: List[float], span: float,
                 window: float) -> Tuple[float, float]:
    """Pick the [t0, t0+window] slice containing the most matched onsets."""
    if not matched_ref_onsets:
        return 0.0, min(window, span)
    onsets = sorted(matched_ref_onsets)
    best_t0, best_count = onsets[0], 0
    for start in onsets:
        count = sum(1 for o in onsets if start <= o < start + window)
        if count > best_count:
            best_count, best_t0 = count, start
    # Centre the window a little before the first matched onset in the run.
    t0 = max(0.0, best_t0 - 0.5)
    return t0, t0 + window


def _draw_overlay(detail: Dict, out_png: Path, *, title: str,
                  window_s: float = DEFAULT_WINDOW_S) -> Dict:
    """Draw one piano-roll overlay for a single (run, step, song).

    Matched notes are green, reference notes the model missed are red, and notes
    the model added that are not in the score (false positives) are orange. The
    plot is cropped to the busiest window so it stays readable. Returns the per
    colour counts and the window used.
    """
    predicted = detail["predicted_notes"]
    reference = detail["reference_notes"]
    pairs = detail["matched_pairs"]
    matched_pred = {p for p, _ in pairs}
    matched_ref = {r for _, r in pairs}

    span = max([n[2] for n in reference] + [n[2] for n in predicted] + [1.0])
    matched_ref_onsets = [reference[r][1] for _, r in pairs]
    t0, t1 = _best_window(matched_ref_onsets, span, window_s)

    def _in_win(onset: float, offset: float) -> bool:
        # Keep a note if any part of it falls inside the cropped [t0, t1] window.
        return offset >= t0 and onset <= t1

    fig, ax = plt.subplots(figsize=(14, 7))
    pitches_in_view: List[int] = []

    # Missed reference notes (red) — in the score but not recovered.
    n_missed = 0
    for idx, (pitch, onset, offset) in enumerate(reference):
        if idx in matched_ref or not _in_win(onset, offset):
            continue
        n_missed += 1
        pitches_in_view.append(pitch)
        ax.add_patch(mpatches.Rectangle(
            (onset, pitch - 0.4), max(offset - onset, 0.03), 0.8,
            facecolor=_MISS_C, edgecolor="none", alpha=0.55))

    # False-positive predicted notes (orange) — in the audio but not the score.
    n_fp = 0
    for idx, (pitch, onset, offset) in enumerate(predicted):
        if idx in matched_pred or not _in_win(onset, offset):
            continue
        n_fp += 1
        pitches_in_view.append(pitch)
        ax.add_patch(mpatches.Rectangle(
            (onset, pitch - 0.4), max(offset - onset, 0.03), 0.8,
            facecolor=_FP_C, edgecolor="none", alpha=0.55))

    # Matched notes (green) — drawn last so they sit on top.
    n_match = 0
    for pred_idx, ref_idx in pairs:
        pitch, onset, offset = reference[ref_idx]
        if not _in_win(onset, offset):
            continue
        n_match += 1
        pitches_in_view.append(pitch)
        ax.add_patch(mpatches.Rectangle(
            (onset, pitch - 0.4), max(offset - onset, 0.03), 0.8,
            facecolor=_MATCH_C, edgecolor="black", linewidth=0.4))

    if pitches_in_view:
        lo, hi = min(pitches_in_view) - 2, max(pitches_in_view) + 2
    else:
        lo, hi = 48, 84
    ax.set_xlim(t0, t1)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MIDI pitch")

    legend = [
        mpatches.Patch(color=_MATCH_C, label=f"matched ({n_match})"),
        mpatches.Patch(color=_MISS_C, label=f"missed reference ({n_missed})"),
        mpatches.Patch(color=_FP_C, label=f"false positive ({n_fp})"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    ax.set_title(
        f"{title}\nprecision={detail['precision']:.3f}  "
        f"recall={detail['recall']:.3f}  F1={detail['f1']:.3f}  "
        f"(window {t0:.1f}–{t1:.1f}s of {span:.0f}s)")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return {"n_match": n_match, "n_missed": n_missed, "n_fp": n_fp,
            "window": [round(t0, 2), round(t1, 2)]}


def build_pianoroll_overlays(version_root: Path, out_dir: Path,
                             basic_pitch_python: str,
                             onset_tolerance_s: float = 0.05,
                             window_s: float = DEFAULT_WINDOW_S) -> Dict:
    """Build one overlay PNG per held-out song for each best finalist.

    For every finalist in BEST_FINALISTS we transcribe its generated WAV with
    Basic-Pitch, match it against the reference MIDI, and draw the coloured
    overlay. A summary JSON is written next to the images and returned.
    """
    version_root = Path(version_root)
    inference_runs = version_root / "inference_runs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict = {"onset_tolerance_s": onset_tolerance_s,
                     "window_s": window_s, "overlays": []}

    for fin in BEST_FINALISTS:
        run_dir = inference_runs / fin["run"]
        audio_dir = run_dir / "audio"
        pairs = _load_pairs_for_step(run_dir, fin["step"])
        if not pairs:
            print(f"[f1-pianoroll] WARN no pairs for {fin['config']} "
                  f"step {fin['step']}")
            continue

        for pair in pairs:
            stem = pair["stem"]
            song = pair["song"]
            ref_midi = Path(pair["reference_midi"])
            gen_wav = audio_dir / f"{stem}.wav"
            if not gen_wav.exists():
                print(f"[f1-pianoroll] WARN missing wav {gen_wav}")
                continue
            if not ref_midi.exists():
                print(f"[f1-pianoroll] WARN missing ref midi {ref_midi}")
                continue

            print(f"[f1-pianoroll] transcribing {fin['config']} "
                  f"step {fin['step']} — {song}")
            detail = compute_f1_detailed(
                generated_wav=gen_wav, reference_midi=ref_midi,
                basic_pitch_python=basic_pitch_python,
                onset_tolerance_s=onset_tolerance_s)

            out_png = out_dir / f"{fin['config']}__step_{fin['step']}__{song}.png"
            title = f"{fin['config']} step {fin['step']} — {song}"
            stats = _draw_overlay(detail, out_png, title=title,
                                  window_s=window_s)
            print(f"[f1-pianoroll]   {out_png.name}: "
                  f"match={stats['n_match']} miss={stats['n_missed']} "
                  f"fp={stats['n_fp']}")
            summary["overlays"].append({
                "config": fin["config"], "step": fin["step"], "song": song,
                "png": out_png.name, "precision": detail["precision"],
                "recall": detail["recall"], "f1": detail["f1"],
                **stats,
            })

    summary_path = out_dir / "f1_pianoroll_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[f1-pianoroll] wrote {summary_path}")
    return summary


def main() -> None:
    """CLI entry point: parse arguments and build the overlays for one version."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <version-root>/_finalist_metrics/f1_pianoroll")
    ap.add_argument("--basic-pitch-python", type=str, default=None,
                    help="Python executable with basic_pitch (default: current)")
    ap.add_argument("--onset-tolerance-s", type=float, default=0.05)
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    args = ap.parse_args()

    import sys
    bp_python = args.basic_pitch_python or sys.executable
    out_dir = (args.out_dir if args.out_dir is not None
               else args.version_root / "_finalist_metrics" / "f1_pianoroll")
    build_pianoroll_overlays(
        args.version_root, out_dir, basic_pitch_python=bp_python,
        onset_tolerance_s=args.onset_tolerance_s, window_s=args.window_s)


if __name__ == "__main__":
    main()
