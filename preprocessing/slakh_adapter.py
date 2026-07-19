"""
slakh_adapter.py — Convert Slakh2100 tracks into training tensors
==================================================================

Slakh2100 folder layout (per track):
  Track00001/
    mix.flac              ← full-mix audio
    metadata.yaml         ← BPM, time signatures, etc.
    MIDI/
      S01.mid             ← one MIDI file per instrument stem
      S02.mid
      ...

This adapter:
  1. Loads mix.flac  →  mel-spectrogram  [80, T]
  2. Merges all MIDI files  →  piano roll  [2, 128, T]  (onset + sustain)
  3. Segments both into 5-second chunks  [80, 430] / [2, 128, 430]
  4. Saves  segment_NNNN.pt  files under the same directory layout as
     process_song_offline.py uses for Israeli data.
  5. Writes  manifest_song.csv  so batch_ingest.py can aggregate everything
     into a consolidated  dataset_manifest.csv.

The output tensor format is IDENTICAL to what process_song_offline.py
produces, so dataset.py and the training loop see no difference between
Slakh and Israeli data.

Usage (CLI)
-----------
# Process a single Slakh track:
python preprocessing/slakh_adapter.py \
    --track_dir  /path/to/slakh2100/train/Track00001 \
    --out_root   /content/drive/MyDrive/MusicProject/slakh_processed

# Process a whole split (Slakh comes pre-split into train/validation/test):
python preprocessing/slakh_adapter.py \
    --slakh_split_dir  /path/to/slakh2100/train \
    --out_root         /content/drive/MyDrive/MusicProject/slakh_processed \
    --manifest_out     /content/drive/MyDrive/MusicProject/data/slakh_manifest.csv \
    --max_tracks       60        # limit for a sanity-training run

Importable API
--------------
from preprocessing.slakh_adapter import process_slakh_track, adapt_slakh_split
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import numpy as np
import librosa
import pretty_midi

# Allow running from repo root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the same DSPConfig so all hyperparameters match exactly
from preprocessing.dsp_preprocessor import DSPConfig

# ---------------------------------------------------------------------------
# Manifest schema — must match process_song_offline._MANIFEST_FIELDS
# ---------------------------------------------------------------------------
_MANIFEST_FIELDS = [
    "segment_path", "score_path", "version_id", "song_id",
    "artist", "album", "song_name", "segment_idx", "duration_s",
    "transcription_confidence",
]

# Slakh is its own "version" — version_id=0 means "no style conditioning"
_SLAKH_VERSION_ID = 0


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio(audio_path: Path, cfg: DSPConfig) -> np.ndarray:
    """Load audio file and resample to cfg.sample_rate.  Returns mono float32."""
    y, _ = librosa.load(str(audio_path), sr=cfg.sample_rate, mono=True)
    return y.astype(np.float32)


def _audio_to_mel(y: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Convert waveform to log-mel spectrogram, normalized to [-1, 1]."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # Normalize to [-1, 1] — same as process_song_offline does
    min_v, max_v = log_mel.min(), log_mel.max()
    if max_v - min_v > 1e-6:
        log_mel = 2.0 * (log_mel - min_v) / (max_v - min_v) - 1.0
    return log_mel  # [n_mels, T]


# ---------------------------------------------------------------------------
# MIDI helpers
# ---------------------------------------------------------------------------

def _merge_midi_files(midi_dir: Path) -> pretty_midi.PrettyMIDI | None:
    """Merge all .mid files in midi_dir into a single PrettyMIDI object."""
    mid_files = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.midi"))
    if not mid_files:
        return None

    merged = pretty_midi.PrettyMIDI()
    for mf in mid_files:
        try:
            pm = pretty_midi.PrettyMIDI(str(mf))
            for inst in pm.instruments:
                if not inst.is_drum:
                    merged.instruments.append(inst)
        except Exception as e:
            print(f"    Warning: could not load {mf.name}: {e}")
    return merged


def _midi_to_piano_roll(pm: pretty_midi.PrettyMIDI, cfg: DSPConfig,
                         n_frames: int) -> np.ndarray:
    """
    Convert merged PrettyMIDI to a 2-channel piano roll [2, 128, n_frames].
      Channel 0: onset  (1.0 where a note starts, 0 otherwise)
      Channel 1: sustain (1.0 where a note is active)
    """
    fs = cfg.midi_fs  # frames per second

    # pretty_midi's get_piano_roll returns a [128, T] array (sustain)
    sustain = pm.get_piano_roll(fs=fs)  # [128, T_midi]

    # Pad or trim to match audio length
    if sustain.shape[1] < n_frames:
        pad = np.zeros((128, n_frames - sustain.shape[1]), dtype=np.float32)
        sustain = np.concatenate([sustain, pad], axis=1)
    sustain = sustain[:, :n_frames]
    sustain = (sustain > 0).astype(np.float32)

    # Onset: 1 only at the frame where each note begins
    onset = np.zeros_like(sustain)
    for inst in pm.instruments:
        for note in inst.notes:
            frame = int(note.start * fs)
            if 0 <= frame < n_frames and 0 <= note.pitch < 128:
                onset[note.pitch, frame] = 1.0

    return np.stack([onset, sustain], axis=0)  # [2, 128, n_frames]


# ---------------------------------------------------------------------------
# Core per-track processor
# ---------------------------------------------------------------------------

def process_slakh_track(
    track_dir: Path,
    out_root: Path,
    skip_if_exists: bool = True,
) -> dict:
    """Process one Slakh track folder → mel + piano roll tensors.

    Args:
        track_dir:      Path to Track00001/ (or similar) inside slakh2100/
        out_root:       Root for output; mirrors track name as subdirectory.
        skip_if_exists: Skip if manifest_song.csv already present.

    Returns dict with keys: status, song_dir, manifest_path, n_segments, error
    """
    cfg = DSPConfig()
    track_name = track_dir.name  # e.g. "Track00001"
    song_id    = f"slakh__{track_name}"

    song_dir       = out_root / "slakh" / track_name
    processed_dir  = song_dir / "processed_data"
    mels_dir       = processed_dir / "mels"
    prs_dir        = processed_dir / "piano_rolls"
    manifest_path  = processed_dir / "manifest_song.csv"

    if skip_if_exists and manifest_path.exists():
        print(f"  Skipped: {track_name} — already processed")
        return {
            "status": "skipped",
            "song_dir": str(song_dir),
            "manifest_path": str(manifest_path),
            "n_segments": 0,
            "error": "",
        }

    # --- Locate audio ---
    audio_path = None
    for candidate in ["mix.flac", "mix.wav", "mix.mp3"]:
        p = track_dir / candidate
        if p.exists():
            audio_path = p
            break
    if audio_path is None:
        return {"status": "failed", "song_dir": str(song_dir),
                "manifest_path": "", "n_segments": 0,
                "error": f"No mix audio found in {track_dir}"}

    # --- Locate MIDI ---
    midi_dir = track_dir / "MIDI"
    if not midi_dir.exists():
        return {"status": "failed", "song_dir": str(song_dir),
                "manifest_path": "", "n_segments": 0,
                "error": f"No MIDI/ directory in {track_dir}"}

    print(f"  Processing {track_name} …")

    try:
        # Load audio
        y = _load_audio(audio_path, cfg)
        log_mel = _audio_to_mel(y, cfg)          # [80, T]
        T = log_mel.shape[1]

        # Merge & convert MIDI
        pm = _merge_midi_files(midi_dir)
        if pm is None:
            return {"status": "failed", "song_dir": str(song_dir),
                    "manifest_path": "", "n_segments": 0,
                    "error": "No valid MIDI files"}
        piano_roll = _midi_to_piano_roll(pm, cfg, T)  # [2, 128, T]

        # Segment
        seg_frames = cfg.segment_frames  # 430
        n_full     = T // seg_frames
        if n_full == 0:
            return {"status": "failed", "song_dir": str(song_dir),
                    "manifest_path": "", "n_segments": 0,
                    "error": f"Track too short ({T} frames < {seg_frames})"}

        mels_dir.mkdir(parents=True, exist_ok=True)
        prs_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for idx in range(n_full):
            s, e = idx * seg_frames, (idx + 1) * seg_frames
            # .copy() forces a contiguous numpy array before wrapping with torch.
            # Without it, torch.from_numpy gets a non-contiguous VIEW of the full
            # track array, so torch.save stores the entire track's storage (~10-20 MB)
            # instead of just the 430-frame segment (~134 KB / ~430 KB).
            mel_seg = torch.from_numpy(log_mel[:, s:e].copy()).float()       # [80, 430]
            pr_seg  = torch.from_numpy(piano_roll[:, :, s:e].copy()).float()  # [2, 128, 430]

            seg_name = f"segment_{idx:04d}.pt"
            mel_path = mels_dir / seg_name
            pr_path  = prs_dir  / seg_name
            torch.save(mel_seg, mel_path)
            torch.save(pr_seg,  pr_path)

            # Store relative paths (relative to out_root, same as Israeli pipeline)
            records.append({
                "segment_path": str(mel_path.relative_to(out_root)),
                "score_path":   str(pr_path.relative_to(out_root)),
                "version_id":   _SLAKH_VERSION_ID,
                "song_id":      song_id,
                "artist":       "slakh",
                "album":        "slakh2100",
                "song_name":    track_name,
                "segment_idx":  idx,
                "duration_s":   cfg.segment_duration,
                "transcription_confidence": 1.0,  # ground-truth MIDI => always 1.0
            })

        # Write per-track manifest
        with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(records)

        print(f"  ✓ {n_full} segments  →  {processed_dir}")
        return {
            "status": "ok",
            "song_dir": str(song_dir),
            "manifest_path": str(manifest_path),
            "n_segments": n_full,
            "error": "",
        }

    except Exception as exc:
        return {"status": "failed", "song_dir": str(song_dir),
                "manifest_path": "", "n_segments": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Batch adapter for a whole split dir
# ---------------------------------------------------------------------------

def adapt_slakh_split(
    slakh_split_dir: Path,
    out_root: Path,
    manifest_out: Path | None = None,
    max_tracks: int | None = None,
    track_ids: list[str] | None = None,
    skip_if_exists: bool = True,
) -> dict:
    """Process every track in a Slakh split directory.

    Args:
        slakh_split_dir: e.g.  .../slakh2100/train/
        out_root:        output root (Drive path or local)
        manifest_out:    where to write consolidated manifest CSV
        max_tracks:      cap number of tracks (useful for sanity runs)
        track_ids:       explicit list of track IDs to process (overrides max_tracks)
        skip_if_exists:  skip already-processed tracks

    Returns summary dict.
    """
    track_dirs = sorted(
        p for p in slakh_split_dir.iterdir()
        if p.is_dir() and p.name.startswith("Track")
    )
    if track_ids is not None:
        allowed = set(track_ids)
        track_dirs = [p for p in track_dirs if p.name in allowed]
    elif max_tracks:
        track_dirs = track_dirs[:max_tracks]

    total = len(track_dirs)
    print(f"Found {total} tracks in {slakh_split_dir}")

    stats = {"ok": 0, "skipped": 0, "failed": 0}

    if manifest_out:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        # Clear or prepare manifest file
        _manifest_written_header = False

    for i, td in enumerate(track_dirs):
        print(f"\n[{i+1}/{total}] {td.name}")
        result = process_slakh_track(td, out_root, skip_if_exists)
        stats[result["status"]] += 1

        if manifest_out and result["status"] in ("ok", "skipped") \
                and result.get("manifest_path"):
            song_manifest = Path(result["manifest_path"])
            if song_manifest.exists():
                write_header = not manifest_out.exists() or manifest_out.stat().st_size == 0
                with open(song_manifest, newline="", encoding="utf-8") as src, \
                     open(manifest_out, "a", newline="", encoding="utf-8") as dst:
                    reader = csv.DictReader(src)
                    writer = csv.DictWriter(dst, fieldnames=_MANIFEST_FIELDS)
                    if write_header:
                        writer.writeheader()
                    for row in reader:
                        writer.writerow({k: row.get(k, "") for k in _MANIFEST_FIELDS})

    print(f"\n{'='*50}")
    print(f"Slakh adapt complete — ok={stats['ok']}  "
          f"skipped={stats['skipped']}  failed={stats['failed']}")
    if manifest_out:
        print(f"Manifest: {manifest_out}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse Slakh adapter CLI arguments and process one track or a split."""
    parser = argparse.ArgumentParser(
        description="Convert Slakh2100 tracks to training tensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--track_dir", help="Process a single Slakh track directory")
    grp.add_argument("--slakh_split_dir", help="Process all tracks in a split dir")

    parser.add_argument("--out_root",    required=True,
                        help="Root output directory for processed tensors")
    parser.add_argument("--manifest_out", default=None,
                        help="(batch mode) Path to write consolidated manifest CSV")
    parser.add_argument("--max_tracks",  type=int, default=None,
                        help="(batch mode) Cap number of tracks processed")
    parser.add_argument("--track_ids_file", default=None,
                        help="(batch mode) Text file with one track ID per line to process")
    parser.add_argument("--no_skip",     action="store_true",
                        help="Re-process already-processed tracks")
    args = parser.parse_args()

    out_root = Path(args.out_root)

    if args.track_dir:
        result = process_slakh_track(
            Path(args.track_dir), out_root,
            skip_if_exists=not args.no_skip,
        )
        print(f"\nStatus   : {result['status']}")
        print(f"Song dir : {result['song_dir']}")
        if result["status"] == "ok":
            print(f"Segments : {result['n_segments']}")
        elif result["status"] == "failed":
            print(f"Error    : {result['error']}")
            sys.exit(1)
    else:
        track_ids = None
        if args.track_ids_file:
            with open(args.track_ids_file) as f:
                track_ids = [l.strip() for l in f if l.strip()]
        adapt_slakh_split(
            slakh_split_dir=Path(args.slakh_split_dir),
            out_root=out_root,
            manifest_out=Path(args.manifest_out) if args.manifest_out else None,
            max_tracks=args.max_tracks,
            track_ids=track_ids,
            skip_if_exists=not args.no_skip,
        )


if __name__ == "__main__":
    main()
