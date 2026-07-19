"""
run_inference_batch.py — Cross-product inference runner with per-run traceability.

Reads ``run_spec.yaml`` and iterates ``(song × step × style × role)``, calling
:func:`inference.synthesize` for each combination. Every produced file is named
deterministically so a later script (or a human) can identify it without
opening the YAML:

    {song}__step_{N}__style_{target}__role_{role}.wav
    {song}__step_{N}__style_{target}__role_{role}.mel.pt
    {song}__step_{N}__style_{target}__role_{role}.basic_pitch.mid   (optional, local)

All outputs land under::

    versions/<v>/inference_runs/<run_id>/
        audio/   *.wav
        mels/    *.mel.pt
        midi/    *.basic_pitch.mid     (only if Basic-Pitch is available)
        run_spec.copy.yaml
        timing_infos.json
        metrics.json
        _summary.csv                   (one row per produced file)

In addition, a project-wide append-only index is updated at::

    versions/<v>/inference_runs/_index.csv
    columns: run_id, song, target_style, role, step, fad, f1, latency_p50,
             listening_rank, run_date, checkpoint_path

The F1 step is **best-effort** because it requires Basic-Pitch (Python 3.10 +
TF), which only runs locally. On Colab, F1 columns are left empty; the
operator runs the F1 pass on Windows afterwards.

run_spec.yaml schema
--------------------
    run_id: Israeli_Shalom_Arik_step_search_2026_03_01
    version_id: 1
    target_style: Israeli            # human label; the version_id picks the channel
    style_version_id: 1              # version index passed to UNet conditioning
    output_root: G:/My Drive/MusicProject/versions/Israeli_Shalom_Arik/inference_runs
    checkpoint_template: runs/Israeli_Shalom_Arik/step_{step}.pt
    steps: [10000, 20000, 30000]
    songs:
      - name: AuSep_1_tpt_33_Elise
        midi: benchmark_output/AuSep_1_tpt_33_Elise/reference.mid
        duration: 30.0
    roles: [transferred]             # tag (e.g. transferred, reference, null)
    sampling:
      ddim_steps: 100
      cfg_score: 1.25
      cfg_version: 1.25
      mel_min: -80.0
      mel_max: 0.0
    metrics:
      f1:
        enable: true
        reference_midi_key: midi      # use song.midi as the reference
        onset_tolerance_s: 0.05
      fad:
        enable: true
        real_dir: benchmark_output/fad_real/
      latency:
        enable: true
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import load_checkpoint, synthesize  # noqa: E402


_INDEX_FIELDS = [
    "run_id", "song", "target_style", "role", "step",
    "fad", "f1", "latency_p50", "listening_rank",
    "run_date", "checkpoint_path",
]


def _stable_name(song: str, step: int, style: str, role: str) -> str:
    """Build the deterministic filename stem used for every artifact.

    Format: ``{song}__step_{N}__style_{target}__role_{role}``. The double
    underscore separator makes it easy to parse back later::

        song, step, style, role = name.split("__")
    """
    return f"{song}__step_{step}__style_{style}__role_{role}"


def _resolve(base: Path, p: str | Path) -> Path:
    """Resolve ``p`` relative to ``base`` if it isn't absolute."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def run_batch(spec_path: Path) -> dict:
    """Execute one batch inference run from a YAML spec."""
    spec_path = spec_path.resolve()
    spec_dir = spec_path.parent
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    run_id = spec["run_id"]
    target_style = spec.get("target_style", "unknown")
    style_version_id = int(spec.get("style_version_id", spec.get("version_id", 0)))
    output_root = _resolve(spec_dir, spec["output_root"])
    run_dir = output_root / run_id
    audio_dir = run_dir / "audio"
    mels_dir = run_dir / "mels"
    midi_dir = run_dir / "midi"
    for d in (audio_dir, mels_dir, midi_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Snapshot the spec so re-runs can be diffed against this exact config.
    with open(run_dir / "run_spec.copy.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)

    checkpoint_template = spec["checkpoint_template"]
    steps = list(spec["steps"])
    songs = spec["songs"]
    roles = spec.get("roles", ["transferred"])
    sampling = spec.get("sampling", {})
    metrics_cfg = spec.get("metrics", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summary_rows: list[dict] = []
    timing_infos: dict[str, Any] = {}
    metrics_out: dict[str, Any] = {}

    # Iterate steps in the outer loop: load each checkpoint exactly once.
    for step in steps:
        ckpt = _resolve(spec_dir, checkpoint_template.format(step=step))
        if not ckpt.exists():
            print(f"  ✗ checkpoint missing: {ckpt}")
            continue
        print(f"\n=== step {step}  ({ckpt.name}) ===")
        model, diffusion, model_cfg = load_checkpoint(str(ckpt), device)

        for song in songs:
            song_name = song["name"]
            midi_path = _resolve(spec_dir, song["midi"])
            duration = float(song.get("duration", 30.0))

            for role in roles:
                stem = _stable_name(song_name, step, target_style, role)
                wav_out = audio_dir / f"{stem}.wav"
                mel_out = mels_dir / f"{stem}.mel.pt"

                if wav_out.exists() and mel_out.exists():
                    print(f"  • {stem}  (skip — exists)")
                else:
                    print(f"  • {stem}")
                    # Time the full MIDI→WAV synthesis so the render pass
                    # doubles as the latency/RTF benchmark (Mission 3c). CUDA is
                    # async, so synchronize before/after for a true wall-clock.
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    _t0 = time.perf_counter()
                    audio = synthesize(
                        model, diffusion, model_cfg,
                        midi_path=str(midi_path),
                        version_id=style_version_id,
                        duration_s=duration,
                        cfg_score=float(sampling.get("cfg_score", 1.25)),
                        cfg_version=float(sampling.get("cfg_version", 1.25)),
                        n_ddim_steps=int(sampling.get("ddim_steps", 100)),
                        mel_min=float(sampling.get("mel_min", -80.0)),
                        mel_max=float(sampling.get("mel_max", 0.0)),
                        device=device,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    _infer_s = time.perf_counter() - _t0
                    _audio_s = len(audio) / 22050.0
                    timing_infos[stem] = {
                        "song": song_name,
                        "step": int(step),
                        "ddim_steps": int(sampling.get("ddim_steps", 100)),
                        "infer_s": round(_infer_s, 4),
                        "audio_s": round(_audio_s, 4),
                        "rtf": round(_infer_s / max(_audio_s, 1e-9), 5),
                    }
                    sf.write(str(wav_out), audio, samplerate=22050)
                    # Recompute mel from generated WAV for downstream eval
                    from preprocessing.dsp_preprocessor import (
                        DSPConfig, extract_mel_spectrogram,
                    )
                    cfg_dsp = DSPConfig()
                    mel = extract_mel_spectrogram(audio.astype(np.float32), cfg_dsp)
                    torch.save(torch.from_numpy(mel).float(), mel_out)

                summary_rows.append({
                    "stem": stem, "song": song_name, "step": step,
                    "target_style": target_style, "role": role,
                    "wav": str(wav_out), "mel": str(mel_out),
                    "checkpoint": str(ckpt),
                })

        # Free the GPU between checkpoints
        del model, diffusion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Metrics (best-effort) ────────────────────────────────────────────
    metrics_out = _compute_metrics(
        summary_rows, songs, run_dir, audio_dir, spec_dir, metrics_cfg,
        timing_infos,
    )

    # ── Write run summary CSV + metrics.json ─────────────────────────────
    with open(run_dir / "_summary.csv", "w", newline="", encoding="utf-8") as fh:
        if summary_rows:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, indent=2, default=str)
    with open(run_dir / "timing_infos.json", "w", encoding="utf-8") as fh:
        json.dump(timing_infos, fh, indent=2)

    # ── Append to project-wide _index.csv ─────────────────────────────────
    index_path = output_root / "_index.csv"
    write_header = not index_path.exists()
    now = _dt.datetime.utcnow().isoformat() + "Z"
    with open(index_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_INDEX_FIELDS)
        if write_header:
            w.writeheader()
        for row in summary_rows:
            key = row["stem"]
            per = metrics_out.get(key, {})
            w.writerow({
                "run_id": run_id,
                "song": row["song"],
                "target_style": row["target_style"],
                "role": row["role"],
                "step": row["step"],
                "fad": per.get("fad"),
                "f1": per.get("f1"),
                "latency_p50": per.get("latency_p50"),
                "listening_rank": "",
                "run_date": now,
                "checkpoint_path": row["checkpoint"],
            })

    print(f"\nWrote: {run_dir}")
    print(f"Index appended: {index_path}")
    return {"run_dir": str(run_dir), "n_outputs": len(summary_rows)}


def _compute_metrics(
    summary_rows: list[dict],
    songs: list[dict],
    run_dir: Path,
    audio_dir: Path,
    spec_dir: Path,
    metrics_cfg: dict,
    timing_infos: dict | None = None,
) -> dict:
    """Per-stem metrics (FAD, F1, latency). Each block is best-effort.

    F1 needs Basic-Pitch (local-only); the function silently skips it on
    Colab. FAD uses VGGish, which runs fine on Colab. Latency is read from
    ``timing_infos`` captured live during synthesis.
    """
    out: dict[str, dict] = {row["stem"]: {} for row in summary_rows}

    # --- FAD (group-level, but we replicate the value per stem for indexing)
    fad_cfg = metrics_cfg.get("fad", {})
    if fad_cfg.get("enable"):
        try:
            from postprocessing.fad_eval import compute_fad
            real_dir = _resolve(spec_dir, fad_cfg["real_dir"])
            fad_value = compute_fad(str(real_dir), str(audio_dir),
                                    use_pretrained=True)
            for stem in out:
                out[stem]["fad"] = float(fad_value)
            print(f"  FAD = {fad_value:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ FAD skipped: {exc}")

    # --- F1 (per song, requires Basic-Pitch)
    f1_cfg = metrics_cfg.get("f1", {})
    if f1_cfg.get("enable"):
        try:
            from postprocessing.f1_eval import compute_f1
            song_midi = {s["name"]: _resolve(spec_dir, s["midi"]) for s in songs}
            tol = float(f1_cfg.get("onset_tolerance_s", 0.05))
            for row in summary_rows:
                stem = row["stem"]
                wav = Path(row["wav"])
                ref = song_midi.get(row["song"])
                if not ref or not ref.exists():
                    continue
                try:
                    res = compute_f1(str(wav), str(ref), onset_tolerance_s=tol)
                    out[stem]["f1"] = float(res.get("f1", 0.0))
                except Exception as exc:  # noqa: BLE001
                    print(f"  ⚠ F1 skipped for {stem}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ F1 module unavailable (likely no Basic-Pitch on this env): {exc}")

    # --- Latency (from live timing captured during synthesis)
    lat_cfg = metrics_cfg.get("latency", {})
    if lat_cfg.get("enable") and timing_infos:
        rtfs = []
        for row in summary_rows:
            stem = row["stem"]
            t = timing_infos.get(stem)
            if t and "rtf" in t:
                out[stem]["latency_p50"] = t["rtf"]
                rtfs.append(t["rtf"])
        if rtfs:
            print(f"  latency: {len(rtfs)} timed renders, mean RTF = {sum(rtfs)/len(rtfs):.4f}")

    return out


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--run-spec", required=True, type=Path)
    args = ap.parse_args()
    summary = run_batch(args.run_spec)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
