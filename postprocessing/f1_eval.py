"""
Note-Level Transcription F1 Evaluation
========================================
Measures how well the model's generated audio reproduces the input score,
by transcribing the generated WAV with Basic-Pitch and comparing the
resulting notes against the reference MIDI.

Pipeline:
    generated WAV  →  Basic-Pitch subprocess  →  predicted MIDI
    reference MIDI →  parse notes             →  reference notes
    greedy match by (pitch, onset ± tolerance) → precision / recall / F1

Metric details:
    - A predicted note matches a reference note if:
        1. pitch is identical (semitone integer, 0-127)
        2. |onset_pred - onset_ref| ≤ onset_tolerance_s (default 50 ms)
    - Each reference note is matched at most once (greedy, sorted by onset).
    - Precision  = matched / n_predicted
    - Recall     = matched / n_reference
    - F1         = 2 * P * R / (P + R)   (0.0 if both are 0)

Target values:
    - Slakh sanity run:    optional after the 150k hearing-test decision
    - Israeli training:    F1 ≥ 0.25

Usage:
    from postprocessing.f1_eval import compute_f1, evaluate_f1_from_manifest

    result = compute_f1(
        generated_wav=Path("samples/step_30000_seg0.wav"),
        reference_midi=Path("data/song/song.mid"),
    )
    print(result["f1"])

    # CLI:
    python postprocessing/f1_eval.py \\
        --generated_wav samples/step_30000_seg0.wav \\
        --reference_midi data/song/song.mid \\
        --out_json f1_result.json

Author: Yotam & Gal — StyleTransfer Music Project
Date: May 2026
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pretty_midi


# ---------------------------------------------------------------------------
# Note parsing helpers
# ---------------------------------------------------------------------------

def _midi_to_note_list(midi_path: Path) -> List[Tuple[int, float, float]]:
    """
    Parse a MIDI file into a list of (pitch, onset_s, offset_s) tuples.

    All instruments are merged.  Drum tracks are excluded.

    Args:
        midi_path: path to a .mid / .midi file

    Returns:
        Sorted list of (pitch: int, onset_s: float, offset_s: float),
        ordered by onset_s ascending.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: List[Tuple[int, float, float]] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append((note.pitch, note.start, note.end))
    notes.sort(key=lambda n: n[1])
    return notes


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def compute_f1(
    generated_wav: Path,
    reference_midi: Path,
    basic_pitch_python: str = sys.executable,
    onset_tolerance_s: float = 0.05,
    tmp_dir: Optional[Path] = None,
) -> dict:
    """
    Compute note-level precision, recall, and F1 for one generated WAV.

    Args:
        generated_wav:      path to the generated .wav file
        reference_midi:     path to the ground-truth .mid file
        basic_pitch_python: Python executable that has basic_pitch installed
                            (default: current interpreter).  On Colab use the
                            path to the Python 3.10 basic_pitch_env executable.
        onset_tolerance_s:  onset matching window in seconds (default 50 ms)
        tmp_dir:            directory for Basic-Pitch output MIDI; a temporary
                            directory is created and cleaned up if not provided

    Returns:
        dict with keys:
            precision, recall, f1          (float, 0-1)
            n_predicted, n_reference, matched  (int)
            generated_wav, reference_midi  (str paths)
            basic_pitch_python             (str)
    """
    generated_wav = Path(generated_wav)
    reference_midi = Path(reference_midi)

    if not generated_wav.exists():
        raise FileNotFoundError(f"Generated WAV not found: {generated_wav}")
    if not reference_midi.exists():
        raise FileNotFoundError(f"Reference MIDI not found: {reference_midi}")

    # ------------------------------------------------------------------
    # Step 1: transcribe generated WAV with Basic-Pitch
    # ------------------------------------------------------------------
    use_tmp = tmp_dir is None
    if use_tmp:
        _tmp_obj = tempfile.TemporaryDirectory()
        bp_out_dir = Path(_tmp_obj.name)
    else:
        bp_out_dir = Path(tmp_dir)
        bp_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Locate the basic-pitch CLI. This is intentionally a subprocess: the
        # main PyTorch environment and Basic-Pitch/TensorFlow environment do not
        # have to be import-compatible for the metric to run.
        #   1. Next to the given Python executable (local venvs, Windows)
        #   2. PATH lookup via shutil.which  (Colab: /usr/local/bin/basic-pitch)
        #   3. python -m basic_pitch          (fallback — works anywhere)
        scripts_dir = Path(basic_pitch_python).parent
        cli_name = "basic-pitch.exe" if sys.platform == "win32" else "basic-pitch"
        basic_pitch_cli: Optional[Path] = scripts_dir / cli_name
        if not basic_pitch_cli.exists():
            basic_pitch_cli = scripts_dir / "basic-pitch"  # no .exe variant
        if not basic_pitch_cli.exists():
            found = shutil.which("basic-pitch")
            basic_pitch_cli = Path(found) if found else None

        if basic_pitch_cli is not None:
            cmd = [str(basic_pitch_cli), str(bp_out_dir), str(generated_wav)]
        else:
            # Module fallback: python -m basic_pitch <out_dir> <wav>
            cmd = [basic_pitch_python, "-m", "basic_pitch", str(bp_out_dir), str(generated_wav)]
        bp_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=bp_env,
        )

        # Basic-Pitch writes: <out_dir>/<stem>_basic_pitch.mid
        stem = generated_wav.stem
        predicted_midi_path = bp_out_dir / f"{stem}_basic_pitch.mid"

        # Fallback: find any .mid in the output dir
        if not predicted_midi_path.exists():
            candidates = list(bp_out_dir.glob("*.mid")) + list(bp_out_dir.glob("*.midi"))
            if candidates:
                predicted_midi_path = candidates[0]
            else:
                # Only now treat a non-zero exit as a real failure
                raise RuntimeError(
                    f"Basic-Pitch failed (exit {result.returncode}):\n"
                    f"  stderr: {result.stderr[:800]}"
                )

        # ------------------------------------------------------------------
        # Step 2: parse both MIDIs
        # ------------------------------------------------------------------
        predicted_notes = _midi_to_note_list(predicted_midi_path)
        reference_notes = _midi_to_note_list(reference_midi)

    finally:
        if use_tmp:
            _tmp_obj.cleanup()

    # ------------------------------------------------------------------
    # Step 3: greedy matching. We compare MIDI notes rather than audio samples:
    # this asks whether the generated sound still contains the intended pitches
    # at roughly the intended times.
    # ------------------------------------------------------------------
    matched = _greedy_match(predicted_notes, reference_notes, onset_tolerance_s)

    n_pred = len(predicted_notes)
    n_ref = len(reference_notes)

    precision = matched / n_pred if n_pred > 0 else 0.0
    recall = matched / n_ref if n_ref > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "n_predicted": n_pred,
        "n_reference": n_ref,
        "matched": matched,
        "generated_wav": str(generated_wav),
        "reference_midi": str(reference_midi),
        "basic_pitch_python": basic_pitch_python,
        "onset_tolerance_s": onset_tolerance_s,
    }


def _greedy_match(
    predicted: List[Tuple[int, float, float]],
    reference: List[Tuple[int, float, float]],
    tolerance: float,
) -> int:
    """
    Greedy note matching: each reference note matched at most once.

    Both lists are sorted by onset.  For each predicted note (in onset order),
    find the earliest unmatched reference note with the same pitch whose onset
    is within ±tolerance seconds.

    Returns:
        Number of matched pairs (int).
    """
    # Index reference notes by pitch for fast lookup. Matching only considers
    # notes of the same MIDI pitch, then chooses the nearest onset inside the
    # tolerance window.
    from collections import defaultdict
    ref_by_pitch: dict = defaultdict(list)
    for idx, (pitch, onset, offset) in enumerate(reference):
        ref_by_pitch[pitch].append((onset, idx))

    used: set = set()
    matched = 0

    for pred_pitch, pred_onset, _ in predicted:
        candidates = ref_by_pitch.get(pred_pitch, [])
        best_idx = None
        best_dist = float("inf")
        for ref_onset, ref_idx in candidates:
            if ref_idx in used:
                continue
            dist = abs(pred_onset - ref_onset)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = ref_idx
        if best_idx is not None:
            used.add(best_idx)
            matched += 1

    return matched


def _greedy_match_labeled(
    predicted: List[Tuple[int, float, float]],
    reference: List[Tuple[int, float, float]],
    tolerance: float,
) -> List[Tuple[int, int]]:
    """
    Same greedy note matching as ``_greedy_match``, but returns the matched
    (predicted_index, reference_index) pairs instead of only a count.

    Used for the piano-roll match overlay, which needs to colour each note as
    matched / missed (reference) / false-positive (predicted).
    """
    from collections import defaultdict
    ref_by_pitch: dict = defaultdict(list)
    for idx, (pitch, onset, offset) in enumerate(reference):
        ref_by_pitch[pitch].append((onset, idx))

    used: set = set()
    pairs: List[Tuple[int, int]] = []

    for pred_idx, (pred_pitch, pred_onset, _) in enumerate(predicted):
        candidates = ref_by_pitch.get(pred_pitch, [])
        best_idx = None
        best_dist = float("inf")
        for ref_onset, ref_idx in candidates:
            if ref_idx in used:
                continue
            dist = abs(pred_onset - ref_onset)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = ref_idx
        if best_idx is not None:
            used.add(best_idx)
            pairs.append((pred_idx, best_idx))

    return pairs


def compute_f1_detailed(
    generated_wav: Path,
    reference_midi: Path,
    basic_pitch_python: str = sys.executable,
    onset_tolerance_s: float = 0.05,
    tmp_dir: Optional[Path] = None,
    save_predicted_midi: Optional[Path] = None,
) -> dict:
    """
    Like ``compute_f1`` but also returns the parsed note lists and the matched
    (predicted_index, reference_index) pairs, for the piano-roll overlay.

    Returns the same metric keys as ``compute_f1`` plus:
        predicted_notes : List[(pitch, onset_s, offset_s)]
        reference_notes : List[(pitch, onset_s, offset_s)]
        matched_pairs   : List[(pred_idx, ref_idx)]
    If ``save_predicted_midi`` is given, the Basic-Pitch output MIDI is copied
    there before the temporary directory is cleaned up.
    """
    generated_wav = Path(generated_wav)
    reference_midi = Path(reference_midi)

    if not generated_wav.exists():
        raise FileNotFoundError(f"Generated WAV not found: {generated_wav}")
    if not reference_midi.exists():
        raise FileNotFoundError(f"Reference MIDI not found: {reference_midi}")

    use_tmp = tmp_dir is None
    if use_tmp:
        _tmp_obj = tempfile.TemporaryDirectory()
        bp_out_dir = Path(_tmp_obj.name)
    else:
        bp_out_dir = Path(tmp_dir)
        bp_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        scripts_dir = Path(basic_pitch_python).parent
        cli_name = "basic-pitch.exe" if sys.platform == "win32" else "basic-pitch"
        basic_pitch_cli: Optional[Path] = scripts_dir / cli_name
        if not basic_pitch_cli.exists():
            basic_pitch_cli = scripts_dir / "basic-pitch"
        if not basic_pitch_cli.exists():
            found = shutil.which("basic-pitch")
            basic_pitch_cli = Path(found) if found else None

        if basic_pitch_cli is not None:
            cmd = [str(basic_pitch_cli), str(bp_out_dir), str(generated_wav)]
        else:
            cmd = [basic_pitch_python, "-m", "basic_pitch",
                   str(bp_out_dir), str(generated_wav)]
        bp_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", env=bp_env,
        )

        stem = generated_wav.stem
        predicted_midi_path = bp_out_dir / f"{stem}_basic_pitch.mid"
        if not predicted_midi_path.exists():
            candidates = (list(bp_out_dir.glob("*.mid")) +
                          list(bp_out_dir.glob("*.midi")))
            if candidates:
                predicted_midi_path = candidates[0]
            else:
                raise RuntimeError(
                    f"Basic-Pitch failed (exit {result.returncode}):\n"
                    f"  stderr: {result.stderr[:800]}"
                )

        predicted_notes = _midi_to_note_list(predicted_midi_path)
        reference_notes = _midi_to_note_list(reference_midi)

        if save_predicted_midi is not None:
            save_predicted_midi = Path(save_predicted_midi)
            save_predicted_midi.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(predicted_midi_path, save_predicted_midi)

    finally:
        if use_tmp:
            _tmp_obj.cleanup()

    pairs = _greedy_match_labeled(predicted_notes, reference_notes,
                                  onset_tolerance_s)
    matched = len(pairs)
    n_pred = len(predicted_notes)
    n_ref = len(reference_notes)
    precision = matched / n_pred if n_pred > 0 else 0.0
    recall = matched / n_ref if n_ref > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "n_predicted": n_pred,
        "n_reference": n_ref,
        "matched": matched,
        "generated_wav": str(generated_wav),
        "reference_midi": str(reference_midi),
        "basic_pitch_python": basic_pitch_python,
        "onset_tolerance_s": onset_tolerance_s,
        "predicted_notes": predicted_notes,
        "reference_notes": reference_notes,
        "matched_pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Local-transcription variant (skip Basic-Pitch subprocess)
# ---------------------------------------------------------------------------

def compute_f1_from_midi(
    predicted_midi: Path,
    reference_midi: Path,
    onset_tolerance_s: float = 0.05,
) -> dict:
    """
    Compute note-level F1 from an already-transcribed predicted MIDI.

    Use this when Basic-Pitch was run locally (e.g. Python ≤3.10 environment)
    and the resulting MIDI has been uploaded to the evaluation directory.

    Args:
        predicted_midi:     path to the Basic-Pitch output .mid file
        reference_midi:     path to the ground-truth .mid file
        onset_tolerance_s:  onset matching window in seconds (default 50 ms)

    Returns:
        Same dict as ``compute_f1`` (without ``generated_wav`` /
        ``basic_pitch_python`` keys).
    """
    predicted_midi = Path(predicted_midi)
    reference_midi = Path(reference_midi)

    if not predicted_midi.exists():
        raise FileNotFoundError(f"Predicted MIDI not found: {predicted_midi}")
    if not reference_midi.exists():
        raise FileNotFoundError(f"Reference MIDI not found: {reference_midi}")

    predicted_notes = _midi_to_note_list(predicted_midi)
    reference_notes = _midi_to_note_list(reference_midi)

    matched = _greedy_match(predicted_notes, reference_notes, onset_tolerance_s)

    n_pred = len(predicted_notes)
    n_ref = len(reference_notes)
    precision = matched / n_pred if n_pred > 0 else 0.0
    recall = matched / n_ref if n_ref > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "n_predicted": n_pred,
        "n_reference": n_ref,
        "matched": matched,
        "predicted_midi": str(predicted_midi),
        "reference_midi": str(reference_midi),
        "onset_tolerance_s": onset_tolerance_s,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_f1_from_manifest(
    manifest_csv: Path,
    generated_wav_dir: Path,
    basic_pitch_python: str = sys.executable,
    onset_tolerance_s: float = 0.05,
) -> pd.DataFrame:
    """
    Evaluate F1 for every segment in a manifest CSV.

    Expects the manifest to have at least columns:
        segment_path   — path to the mel .pt file (stem used to find WAV)
        score_path     — path to the piano-roll .pt file
                         (used to derive the MIDI path: same stem, .mid extension)

    The function looks for generated WAVs by matching the segment_path stem
    against WAV files in generated_wav_dir.

    Args:
        manifest_csv:       path to test.csv (or any manifest CSV)
        generated_wav_dir:  directory containing generated WAVs named by
                            segment stem, e.g. segment_0000.wav
        basic_pitch_python: Python executable with basic_pitch installed
        onset_tolerance_s:  onset matching window

    Returns:
        DataFrame with one row per segment + summary appended as last row
        (song_id == "__summary__").
    """
    manifest_csv = Path(manifest_csv)
    generated_wav_dir = Path(generated_wav_dir)

    df = pd.read_csv(manifest_csv)
    required = {"segment_path", "score_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    rows = []
    for _, row in df.iterrows():
        seg_stem = Path(row["segment_path"]).stem          # e.g. segment_0000
        generated_wav = generated_wav_dir / f"{seg_stem}.wav"

        # Derive MIDI path from score_path (piano_rolls/segment_0000.pt → .mid)
        score_dir = Path(row["score_path"]).parent.parent  # song root
        midi_candidates = list(score_dir.glob("*.mid")) + list(score_dir.glob("*.midi"))

        if not generated_wav.exists():
            rows.append({**row.to_dict(), "f1": None, "precision": None,
                          "recall": None, "error": "generated_wav_missing"})
            continue
        if not midi_candidates:
            rows.append({**row.to_dict(), "f1": None, "precision": None,
                          "recall": None, "error": "reference_midi_missing"})
            continue

        reference_midi = midi_candidates[0]
        try:
            res = compute_f1(generated_wav, reference_midi,
                             basic_pitch_python=basic_pitch_python,
                             onset_tolerance_s=onset_tolerance_s)
            rows.append({**row.to_dict(),
                          "f1": res["f1"],
                          "precision": res["precision"],
                          "recall": res["recall"],
                          "n_predicted": res["n_predicted"],
                          "n_reference": res["n_reference"],
                          "matched": res["matched"],
                          "error": None})
        except Exception as exc:
            rows.append({**row.to_dict(), "f1": None, "precision": None,
                          "recall": None, "error": str(exc)[:200]})

    result_df = pd.DataFrame(rows)

    # Summary row
    valid = result_df["f1"].dropna()
    summary = {
        "song_id": "__summary__",
        "f1": valid.mean() if len(valid) else float("nan"),
        "precision": result_df["precision"].dropna().mean() if len(valid) else float("nan"),
        "recall": result_df["recall"].dropna().mean() if len(valid) else float("nan"),
        "n_predicted": result_df["n_predicted"].dropna().sum() if len(valid) else 0,
        "n_reference": result_df["n_reference"].dropna().sum() if len(valid) else 0,
        "matched": result_df["matched"].dropna().sum() if len(valid) else 0,
        "error": f"{(result_df['error'].notna()).sum()} errors / {len(result_df)} total",
    }
    result_df = pd.concat([result_df, pd.DataFrame([summary])], ignore_index=True)

    return result_df


# ---------------------------------------------------------------------------
# Run-directory batch evaluation (Israeli pipeline, local-only)
# ---------------------------------------------------------------------------

def _parse_stable_stem(stem: str) -> Optional[dict]:
    """Parse ``{song}__step_{N}__style_{target}__role_{role}`` back into fields.

    Returns None when the stem doesn't match the expected schema (e.g. legacy
    files). The double-underscore separator is what makes this safe even when
    song names contain a single underscore.
    """
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    song = parts[0]
    try:
        step = int(parts[1].removeprefix("step_"))
    except ValueError:
        return None
    style = parts[2].removeprefix("style_")
    role = parts[3].removeprefix("role_")
    return {"song": song, "step": step, "target_style": style, "role": role}


def evaluate_run_dir(
    run_dir: Path,
    song_filter: Optional[str] = None,
    basic_pitch_python: str = sys.executable,
    onset_tolerance_s: float = 0.05,
) -> dict:
    """Score every (or one) generated WAV under an ``inference_runs/<run_id>/``.

    Reads ``run_spec.copy.yaml`` for the ``song → reference_midi`` mapping and
    ``audio/*.wav`` for the generated files. Writes
    ``metrics/f1_per_pair.json`` (list of per-stem results) and updates the
    ``f1`` column of the project-wide ``inference_runs/_index.csv`` in place.

    F1 requires Basic-Pitch (Python 3.10 + TF) which the project ships in
    ``basic_pitch_env`` — this branch is therefore **local-only**.

    Args:
        run_dir:            ``versions/<v>/inference_runs/<run_id>/`` directory
        song_filter:        score only this song name (default: all)
        basic_pitch_python: Python executable with basic_pitch installed
        onset_tolerance_s:  onset matching window (default 50 ms)

    Returns:
        dict with keys ``run_id``, ``scored``, ``skipped``, ``mean_f1``,
        ``per_pair`` (list of dicts), ``index_updated`` (int rows changed).
    """
    import yaml as _yaml

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    spec_path = run_dir / "run_spec.copy.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"run_spec.copy.yaml missing under {run_dir} — was this dir produced by "
            f"run_inference_batch.py?"
        )
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = _yaml.safe_load(fh)

    # Specs are authored on Colab, so song `midi` paths are Colab-absolute
    # (/content/drive/MyDrive/MusicProject/...). Remap that prefix to the local
    # Drive root so the local F1 pass resolves them (paths.py is the single
    # source of truth for both roots).
    try:
        from paths import DRIVE_ROOT_COLAB, DRIVE_ROOT_LOCAL
        _colab_prefix = DRIVE_ROOT_COLAB.as_posix()
    except Exception:  # noqa: BLE001 — paths.py unavailable; skip remap
        DRIVE_ROOT_LOCAL = None
        _colab_prefix = None

    def _remap_colab_to_local(raw: Path) -> Optional[Path]:
        if DRIVE_ROOT_LOCAL is None or _colab_prefix is None:
            return None
        raw_posix = raw.as_posix()
        if raw_posix.startswith(_colab_prefix):
            rel = raw_posix[len(_colab_prefix):].lstrip("/")
            return DRIVE_ROOT_LOCAL / rel
        return None

    spec_songs = spec.get("songs", []) or []
    # Songs in run_spec.copy.yaml carry `midi` paths relative to the original
    # spec_dir on the run author's machine. Try as-is first, then resolve
    # relative to run_dir as a fallback for portable runs.
    song_midi: dict[str, Path] = {}
    for s in spec_songs:
        name = s["name"]
        midi_raw = Path(s["midi"])
        candidates = [midi_raw] if midi_raw.is_absolute() else [
            Path(midi_raw),
            (run_dir / midi_raw).resolve(),
            (run_dir.parent / midi_raw).resolve(),
        ]
        remapped = _remap_colab_to_local(midi_raw)
        if remapped is not None:
            candidates.insert(0, remapped)
        chosen = next((p for p in candidates if p.exists()), None)
        if chosen is None:
            print(f"  ⚠ reference MIDI not found for {name}: tried {[str(c) for c in candidates]}")
            continue
        song_midi[name] = chosen

    audio_dir = run_dir / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio/ missing under {run_dir}")

    per_pair: list[dict] = []
    skipped: list[dict] = []
    for wav in sorted(audio_dir.glob("*.wav")):
        parsed = _parse_stable_stem(wav.stem)
        if parsed is None:
            skipped.append({"stem": wav.stem, "reason": "unparseable_stem"})
            continue
        if song_filter is not None and parsed["song"] != song_filter:
            continue
        ref = song_midi.get(parsed["song"])
        if ref is None:
            skipped.append({"stem": wav.stem, "reason": "reference_midi_missing"})
            continue
        try:
            res = compute_f1(
                generated_wav=wav,
                reference_midi=ref,
                basic_pitch_python=basic_pitch_python,
                onset_tolerance_s=onset_tolerance_s,
            )
            per_pair.append({
                "stem": wav.stem,
                **parsed,
                "f1": res["f1"],
                "precision": res["precision"],
                "recall": res["recall"],
                "n_predicted": res["n_predicted"],
                "n_reference": res["n_reference"],
                "matched": res["matched"],
                "reference_midi": str(ref),
            })
            print(f"  ✓ {wav.stem}: F1={res['f1']:.4f}")
        except Exception as exc:  # noqa: BLE001
            skipped.append({"stem": wav.stem, "reason": str(exc)[:200]})
            print(f"  ✗ {wav.stem}: {exc}")

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_json = metrics_dir / "f1_per_pair.json"
    payload = {
        "run_id": spec.get("run_id", run_dir.name),
        "song_filter": song_filter,
        "onset_tolerance_s": onset_tolerance_s,
        "per_pair": per_pair,
        "skipped": skipped,
        "mean_f1": (sum(p["f1"] for p in per_pair) / len(per_pair)) if per_pair else None,
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Wrote {out_json}  ({len(per_pair)} scored, {len(skipped)} skipped)")

    # ── Merge f1 into per-stem metrics.json (consumed by select_best_step) ─
    metrics_json = run_dir / "metrics.json"
    if metrics_json.exists():
        with open(metrics_json, "r", encoding="utf-8") as fh:
            per_stem = json.load(fh)
        merged = 0
        for p in per_pair:
            entry = per_stem.get(p["stem"])
            if entry is not None:
                entry["f1"] = p["f1"]
                merged += 1
        with open(metrics_json, "w", encoding="utf-8") as fh:
            json.dump(per_stem, fh, indent=2)
        print(f"  Merged f1 into {merged} stems of {metrics_json}")
    else:
        print(f"  (no metrics.json at {metrics_json} — skipped f1 merge)")

    # ── Update _index.csv (parent of run_dir) in place ───────────────────
    index_path = run_dir.parent / "_index.csv"
    updated = 0
    if index_path.exists():
        df = pd.read_csv(index_path)
        run_id = payload["run_id"]
        # Build (song, step, style, role) → f1 lookup
        f1_by_key = {(p["song"], int(p["step"]), p["target_style"], p["role"]): p["f1"]
                     for p in per_pair}
        for i, row in df.iterrows():
            if str(row.get("run_id")) != str(run_id):
                continue
            key = (str(row["song"]), int(row["step"]),
                   str(row["target_style"]), str(row["role"]))
            if key in f1_by_key:
                df.at[i, "f1"] = f1_by_key[key]
                updated += 1
        df.to_csv(index_path, index=False)
        print(f"  Updated {updated} rows in {index_path}")
    else:
        print(f"  (no _index.csv at {index_path} — skipped index update)")

    return {
        "run_id": payload["run_id"],
        "scored": len(per_pair),
        "skipped": len(skipped),
        "mean_f1": payload["mean_f1"],
        "per_pair": per_pair,
        "index_updated": updated,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_single_pair(args) -> int:
    res = compute_f1(
        generated_wav=Path(args.generated_wav),
        reference_midi=Path(args.reference_midi),
        basic_pitch_python=args.basic_pitch_python,
        onset_tolerance_s=args.onset_tolerance_s,
    )

    print("\n=== F1 Evaluation Result ===")
    print(f"  Precision:   {res['precision']:.4f}")
    print(f"  Recall:      {res['recall']:.4f}")
    print(f"  F1:          {res['f1']:.4f}")
    print(f"  Predicted:   {res['n_predicted']} notes")
    print(f"  Reference:   {res['n_reference']} notes")
    print(f"  Matched:     {res['matched']} notes")
    print(f"  Tolerance:   {res['onset_tolerance_s']*1000:.0f} ms")
    print()

    target_slakh = 0.30
    target_israeli = 0.25
    print(f"  Target (Slakh gate):    F1 ≥ {target_slakh:.2f}  →  "
          f"{'PASS' if res['f1'] >= target_slakh else 'FAIL'}")
    print(f"  Target (Israeli gate):  F1 ≥ {target_israeli:.2f}  →  "
          f"{'PASS' if res['f1'] >= target_israeli else 'FAIL'}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"\n  Results saved to {out_path}")
    return 0


def _cli_run_dir(args) -> int:
    summary = evaluate_run_dir(
        run_dir=Path(args.run_dir),
        song_filter=args.song,
        basic_pitch_python=args.basic_pitch_python,
        onset_tolerance_s=args.onset_tolerance_s,
    )
    print("\n=== Run-dir F1 summary ===")
    print(f"  run_id        : {summary['run_id']}")
    print(f"  scored        : {summary['scored']}")
    print(f"  skipped       : {summary['skipped']}")
    print(f"  mean F1       : "
          f"{summary['mean_f1']:.4f}" if summary['mean_f1'] is not None else "  mean F1       : n/a")
    print(f"  index updated : {summary['index_updated']} rows")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute note-level transcription F1. Two modes:\n"
            "  (1) single pair — pass --generated_wav and --reference_midi\n"
            "  (2) run-dir batch — pass --run-dir to score every output of an "
            "inference run (local-only; needs Basic-Pitch).\n"
            "Optional --song filters mode (2) to a single song."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode (1) args (optional unless --run-dir is absent)
    parser.add_argument("--generated_wav", type=str, default=None,
                        help="[mode 1] Path to generated .wav file")
    parser.add_argument("--reference_midi", type=str, default=None,
                        help="[mode 1] Path to reference .mid file")
    parser.add_argument("--out_json", type=str, default=None,
                        help="[mode 1] Path to write JSON result (optional)")
    # Mode (2) args
    parser.add_argument("--run-dir", dest="run_dir", type=str, default=None,
                        help="[mode 2] Path to versions/<v>/inference_runs/<run_id>/. "
                             "Local-only: needs Basic-Pitch (basic_pitch_env on Windows).")
    parser.add_argument("--song", type=str, default=None,
                        help="[mode 2] Score only this song name (optional filter)")
    # Shared
    parser.add_argument("--basic_pitch_python", type=str, default=sys.executable,
                        help="Python executable with basic_pitch installed "
                             "(default: current interpreter)")
    parser.add_argument("--onset_tolerance_s", type=float, default=0.05,
                        help="Onset matching tolerance in seconds (default: 0.05)")
    args = parser.parse_args()

    if args.run_dir is not None:
        sys.exit(_cli_run_dir(args))
    if args.generated_wav is None or args.reference_midi is None:
        parser.error("either --run-dir, or both --generated_wav and --reference_midi, are required")
    sys.exit(_cli_single_pair(args))
