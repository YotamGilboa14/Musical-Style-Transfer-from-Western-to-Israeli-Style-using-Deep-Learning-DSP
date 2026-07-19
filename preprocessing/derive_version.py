"""
derive_version.py — Build a training *version* from the immutable ``source_pool/``.

The data architecture splits responsibilities between two roots:

    source_pool/<artist>/<album>/<song>/
        <song>.wav            (raw WAV at native SR — written once)
        <song>.mid            (Basic-Pitch transcription — written once, locally)
        metadata.json         (url, native_sr, duration, ingest_date, ...)
        augmented/            (deterministic WAV+MIDI pairs)
            <song>_ps+2.wav   (librosa pitch shift)
            <song>_ps+2.mid   (pretty_midi transpose)
            ...
            aug_spec.json

    versions/<v>/processed_data/<artist>/<album>/<song>/[<aug_tag>/]
        mels/segment_NNNN.pt
        piano_rolls/segment_NNNN.pt
        manifest_song.csv
        dsp_config.json
        preprocessing_demo.png         (best-effort)
    versions/<v>/manifest.csv          (concatenation of all manifest_song.csv)
    versions/<v>/version_spec.copy.yaml

A *version* is a cheap derivative of the pool: pick a subset of songs, pick a
DSPConfig, and re-run DSP on each (song, augmentation) pair. The pool is never
duplicated and Basic-Pitch is never re-run.

This script is Colab-friendly (no Basic-Pitch dependency) and intended to run
on top of Google Drive Desktop mounts on Windows or ``/content/drive/MyDrive``
on Colab.

CLI
---
    python -m preprocessing.derive_version \\
        --source-pool G:/My Drive/MusicProject/source_pool \\
        --version-spec configs/version_israeli_v1.yaml \\
        --out-version-dir G:/My Drive/MusicProject/versions/israeli_v1

version_spec.yaml schema
------------------------
    version_id: 1
    style_name: Israeli
    songs:
      - artist: Arik_Einstein
        album: Singles
        song_name: Atur_Mitzchek
        include_augmented: true   # default true
    dsp:                            # optional DSPConfig overrides
      sample_rate: 22050
      n_mels: 80
      fmax: 8000
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Make sibling packages resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.dsp_preprocessor import (  # noqa: E402
    DSPConfig,
    extract_mel_spectrogram,
    load_and_resample_audio,
    load_midi_to_piano_roll,
    normalize_mel,
    segment_data,
)

_MANIFEST_FIELDS = [
    "segment_path", "score_path", "version_id", "song_id",
    "artist", "album", "song_name", "aug_tag", "segment_idx", "duration_s",
]


def _build_cfg(overrides: Optional[dict]) -> DSPConfig:
    """Build a DSPConfig honouring optional overrides from version_spec.dsp."""
    cfg = DSPConfig()
    if overrides:
        valid = {f.name for f in dataclasses.fields(DSPConfig)}
        for k, v in overrides.items():
            if k in valid:
                setattr(cfg, k, v)
            else:
                print(f"  ⚠ ignored unknown DSPConfig key: {k}")
    return cfg


def _validate_version_spec(spec: dict) -> None:
    """Fail fast with a clear message if a version spec is malformed.

    Catches the most common YAML mistakes BEFORE we spin up any DSP work:
      * missing required top-level keys
      * songs missing artist/album/song_name (or empty strings)
      * unknown keys inside `dsp:` (typos like `sr` vs `sample_rate`)
      * non-list `songs` / `held_out_songs`
    """
    if not isinstance(spec, dict):
        raise ValueError(f"version_spec must be a mapping, got {type(spec).__name__}")

    for key in ("version_id", "style_name", "songs"):
        if key not in spec:
            raise ValueError(f"version_spec missing required key: {key!r}")

    try:
        int(spec["version_id"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"version_id must be an int, got {spec['version_id']!r}") from exc

    if not isinstance(spec["style_name"], str) or not spec["style_name"].strip():
        raise ValueError("style_name must be a non-empty string")

    songs = spec.get("songs") or []
    if not isinstance(songs, list):
        raise ValueError(f"`songs` must be a list, got {type(songs).__name__}")

    for i, s in enumerate(songs):
        if not isinstance(s, dict):
            raise ValueError(f"songs[{i}] must be a mapping, got {type(s).__name__}")
        for k in ("artist", "album", "song_name"):
            v = s.get(k)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"songs[{i}].{k} must be a non-empty string (got {v!r})")

    held = spec.get("held_out_songs", [])
    if held is not None and not isinstance(held, list):
        raise ValueError(f"`held_out_songs` must be a list or null, got {type(held).__name__}")

    overrides = spec.get("dsp")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError(f"`dsp` must be a mapping, got {type(overrides).__name__}")
        valid_fields = {f.name for f in dataclasses.fields(DSPConfig)}
        unknown = sorted(set(overrides) - valid_fields)
        if unknown:
            raise ValueError(
                f"`dsp` contains unknown DSPConfig fields: {unknown}. "
                f"Valid fields: {sorted(valid_fields)}"
            )


def _process_pair(
    wav_path: Path,
    midi_path: Path,
    out_dir: Path,
    cfg: DSPConfig,
    *,
    artist: str,
    album: str,
    song_name: str,
    aug_tag: str,
    version_id: int,
    version_root: Path,
) -> tuple[int, list[dict]]:
    """Run DSP on one (WAV, MIDI) pair and write segments + per-song manifest.

    Returns (n_segments, manifest_rows). Skips entirely if the per-pair manifest
    already exists AND every tensor it references is present on disk (idempotent
    re-run support). If the manifest exists but any tensor is missing (e.g. an
    interrupted write to a flaky filesystem), the pair is fully reprocessed.
    """
    mels_dir = out_dir / "mels"
    pr_dir = out_dir / "piano_rolls"
    manifest_path = out_dir / "manifest_song.csv"
    if manifest_path.exists():
        # Read existing rows so the top-level manifest stays consistent
        with open(manifest_path, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # Verify every referenced tensor actually exists before trusting the
        # manifest — guards against interrupted writes (Drive/FUSE flakiness).
        all_present = bool(rows) and all(
            (version_root / r["segment_path"]).exists()
            and (version_root / r["score_path"]).exists()
            for r in rows
        )
        if all_present:
            return len(rows), rows
        print(f"    ⚠ {song_name}/{aug_tag}: manifest present but tensors "
              f"missing — reprocessing")
        manifest_path.unlink()

    mels_dir.mkdir(parents=True, exist_ok=True)
    pr_dir.mkdir(parents=True, exist_ok=True)

    y = load_and_resample_audio(wav_path, cfg.sample_rate)
    duration = len(y) / float(cfg.sample_rate)
    mel = extract_mel_spectrogram(y, cfg)
    mel_norm, m_min, m_max = normalize_mel(mel)
    pr = load_midi_to_piano_roll(midi_path, cfg, duration)
    segments = segment_data(mel_norm, pr, cfg)
    if not segments:
        raise RuntimeError(f"no segments produced from {wav_path}")

    song_id = f"{artist}__{album}__{song_name}__{aug_tag}"
    rows: list[dict] = []
    for idx, (mel_seg, pr_seg) in enumerate(segments):
        mel_file = mels_dir / f"segment_{idx:04d}.pt"
        pr_file = pr_dir / f"segment_{idx:04d}.pt"
        torch.save(torch.from_numpy(np.ascontiguousarray(mel_seg)).float().clone(), mel_file)
        torch.save(torch.from_numpy(np.ascontiguousarray(pr_seg)).float().clone(), pr_file)
        rows.append({
            "segment_path": str(mel_file.relative_to(version_root)).replace("\\", "/"),
            "score_path":   str(pr_file.relative_to(version_root)).replace("\\", "/"),
            "version_id":   version_id,
            "song_id":      song_id,
            "artist":       artist,
            "album":        album,
            "song_name":    song_name,
            "aug_tag":      aug_tag,
            "segment_idx":  idx,
            "duration_s":   cfg.segment_duration,
        })

    with open(out_dir / "dsp_config.json", "w", encoding="utf-8") as fh:
        json.dump({
            **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(DSPConfig)},
            "mel_min": float(m_min), "mel_max": float(m_max),
        }, fh, indent=2)

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    return len(rows), rows


def derive_version(
    source_pool: Path,
    version_spec: dict,
    out_version_dir: Path,
) -> dict:
    """Materialise one training version from ``source_pool/`` per ``version_spec``.

    See module docstring for the spec schema. Returns a summary dict.
    """
    _validate_version_spec(version_spec)
    version_id = int(version_spec.get("version_id", 0))
    style_name = str(version_spec.get("style_name", "unknown"))
    cfg = _build_cfg(version_spec.get("dsp"))
    songs = version_spec.get("songs", []) or []

    out_version_dir.mkdir(parents=True, exist_ok=True)
    processed_root = out_version_dir / "processed_data"
    processed_root.mkdir(parents=True, exist_ok=True)

    # Persist a copy of the spec alongside the materialised data — TA-grading
    # reproducibility: every version directory knows how it was produced.
    with open(out_version_dir / "version_spec.copy.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(version_spec, fh, sort_keys=False)

    all_rows: list[dict] = []
    summary = {
        "version_id": version_id, "style_name": style_name,
        "songs_ok": [], "songs_failed": [], "total_segments": 0,
    }

    n_songs = len(songs)
    print(f"Deriving version {version_id} ({style_name}): {n_songs} songs to process",
          flush=True)
    for song_i, entry in enumerate(songs, start=1):
        artist = entry["artist"]
        album = entry["album"]
        song_name = entry["song_name"]
        include_aug = bool(entry.get("include_augmented", True))

        song_src = source_pool / artist / album / song_name
        wav = song_src / f"{song_name}.wav"
        mid = song_src / f"{song_name}.mid"
        if not wav.exists() or not mid.exists():
            print(f"  [{song_i}/{n_songs}] ✗ missing WAV/MIDI in pool: {song_src}",
                  flush=True)
            summary["songs_failed"].append(str(song_src))
            continue

        print(f"  [{song_i}/{n_songs}] ▶ {artist}/{album}/{song_name}", flush=True)
        song_out = processed_root / artist / album / song_name
        try:
            # Original (no augmentation)
            n, rows = _process_pair(
                wav, mid, song_out / "orig", cfg,
                artist=artist, album=album, song_name=song_name,
                aug_tag="orig", version_id=version_id,
                version_root=out_version_dir,
            )
            print(f"    orig: {n} segments", flush=True)
            all_rows.extend(rows)

            # Best-effort preprocessing demo for slides — never gate the run on it
            demo_path = song_out / "preprocessing_demo.png"
            if not demo_path.exists():
                try:
                    from preprocessing.dataset_visualizations import plot_preprocessing_demo
                    import matplotlib.pyplot as _plt
                    fig = plot_preprocessing_demo(
                        wav_path=wav, midi_path=mid,
                        save_path=demo_path, cfg=cfg,
                    )
                    _plt.close(fig)
                except Exception as exc:  # noqa: BLE001
                    print(f"    ⚠ preprocessing_demo skipped: {exc}")

            # Augmentations
            if include_aug:
                aug_dir = song_src / "augmented"
                if aug_dir.exists():
                    # Pair WAV + MIDI by stem
                    for aug_wav in sorted(aug_dir.glob("*.wav")):
                        aug_mid = aug_wav.with_suffix(".mid")
                        if not aug_mid.exists():
                            print(f"    ⚠ missing aug MIDI for {aug_wav.name}")
                            continue
                        aug_tag = aug_wav.stem.replace(f"{song_name}_", "")
                        n, rows = _process_pair(
                            aug_wav, aug_mid, song_out / aug_tag, cfg,
                            artist=artist, album=album, song_name=song_name,
                            aug_tag=aug_tag, version_id=version_id,
                            version_root=out_version_dir,
                        )
                        print(f"    {aug_tag}: {n} segments", flush=True)
                        all_rows.extend(rows)

            summary["songs_ok"].append(f"{artist}/{album}/{song_name}")
            print(f"    ✓ done ({song_i}/{n_songs}) — running total: "
                  f"{len(all_rows)} segments", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    ✗ failed: {exc}", flush=True)
            summary["songs_failed"].append(f"{artist}/{album}/{song_name}: {exc}")

    # Concatenated manifest at the version root
    top_manifest = out_version_dir / "manifest.csv"
    with open(top_manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    summary["total_segments"] = len(all_rows)
    summary["manifest"] = str(top_manifest)
    print(f"\nVersion {version_id} ({style_name}): {summary['total_segments']} segments "
          f"across {len(summary['songs_ok'])} songs")
    return summary


def _main() -> int:
    ap = argparse.ArgumentParser(
        description="Materialise a training version from source_pool/."
    )
    ap.add_argument("--source-pool", required=True, type=Path)
    ap.add_argument("--version-spec", required=True, type=Path,
                    help="YAML file describing version_id, songs, dsp overrides")
    ap.add_argument("--out-version-dir", required=True, type=Path,
                    help="versions/<v>/  output root")
    args = ap.parse_args()

    if not args.source_pool.exists():
        print(f"source pool not found: {args.source_pool}")
        return 1
    with open(args.version_spec, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    summary = derive_version(args.source_pool, spec, args.out_version_dir)
    summary_path = args.out_version_dir / "derive_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {summary_path}")
    return 0 if not summary["songs_failed"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
