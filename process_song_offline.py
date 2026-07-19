"""
Song Processing Library + CLI
==============================

Processes a single song through the complete preprocessing pipeline:
  1. Resolve metadata  (auto-detect from yt-dlp if fields are blank)
  2. Download audio    (YouTube → WAV, if --url given)
  3. Source separation (optional Demucs, off by default for Israeli pipeline)
  4. MIDI transcription via Basic-Pitch on the full-mix WAV
  5. DSP: mel extraction → normalize → segment
  6. Save segment tensors  (segment_NNNN.pt) + manifest_song.csv

Two execution paths are supported. The current Israeli workflow uses Path B:
preprocess one URL locally, upload every artifact to Drive, then move on to
the next URL.

    PATH A — Fully on Colab (historical/optional):
    process_song(row, out_root=Path("/content/drive/MyDrive/MusicProject/MusicProjectData"))
    → writes directly to Google Drive; nothing lands on local disk.

    PATH B — Local preprocessing → upload → clean (current Israeli path):
    process_song(row, out_root=Path("/tmp/local_out"))
    → then call  preprocessing/drive_sync.py  to push to Drive and optionally
      delete the local song folder.

Public API (importable from batch_ingest, notebooks, etc.):
    from process_song_offline import process_song

CLI (single-song convenience wrapper):
    python process_song_offline.py --url "https://youtu.be/..." \\
        --out_root /path/to/MusicProjectData

    python process_song_offline.py --artist "Arik_Einstein" \\
        --song "Atur_Mitzchek" --source_wav downloads/song.wav \\
        --out_root /path/to/MusicProjectData

    # Re-enable Demucs:
    python process_song_offline.py --url "..." --out_root ... --separate_stems

Authors: Yotam & Gal — StyleTransfer Music Project
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# Force UTF-8 on stdout/stderr so the progress glyphs (✓ ✗ → …) used below do
# not raise UnicodeEncodeError on Windows' default cp1252 console. Without this,
# a single print() crash gets caught by the batch loop and mislabels an
# otherwise-successful song as "failed".
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Make sure sibling packages resolve whether run as script or imported as module
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.dsp_preprocessor import (
    DSPConfig,
    load_and_resample_audio,
    extract_mel_spectrogram,
    normalize_mel,
    load_midi_to_piano_roll,
    segment_data,
    create_visualization,
)
from preprocessing.source_separator import separate_stems as _run_demucs, check_stems_exist


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _transcribe_to_midi(
    input_wav: Path,
    output_midi: Path,
    project_root: Path,
    label: str = "full_mix",
) -> bool:
    """Run Basic-Pitch on *input_wav* and move result to *output_midi*.

    Returns True on success, False on failure (caller decides whether to abort).
    """
    if output_midi.exists():
        print(f"  [{label}] MIDI already exists: {output_midi.name}")
        return True

    basic_pitch_python = project_root / "basic_pitch_env" / "Scripts" / "python.exe"
    if not basic_pitch_python.exists():
        basic_pitch_python = Path(sys.executable)

    basic_pitch_script = project_root / "preprocessing" / "audio_tp_midi_poc.py"
    midi_output_dir = project_root / "midi_output"
    expected_midi = midi_output_dir / f"{input_wav.stem}_basic_pitch.mid"
    if expected_midi.exists():
        expected_midi.unlink()

    cmd = [str(basic_pitch_python), str(basic_pitch_script), str(input_wav)]
    print(f"  [{label}] Running Basic-Pitch on {input_wav.name} …")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))

    # Basic-Pitch writes into midi_output/ — find and relocate the file
    if expected_midi.exists():
        shutil.move(str(expected_midi), str(output_midi))
        print(f"  [{label}] ✓ MIDI saved: {output_midi.name}")
        return True

    if midi_output_dir.exists():
        candidates = sorted(midi_output_dir.glob("*.mid"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            shutil.move(str(candidates[0]), str(output_midi))
            print(f"  [{label}] ✓ MIDI moved: {output_midi.name}")
            return True

    print(f"  [{label}] ✗ Transcription failed")
    print(f"    stdout: {result.stdout[:300] or 'empty'}")
    print(f"    stderr: {result.stderr[:300] or 'empty'}")
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MANIFEST_FIELDS = [
    "segment_path", "score_path", "version_id", "song_id",
    "artist", "album", "song_name", "segment_idx", "duration_s",
]


def process_song(
    row: dict,
    out_root: Path,
    skip_if_exists: bool = True,
    separate_stems: bool = False,
    source_pool_mode: bool = False,
) -> dict:
    """Process one song through the full preprocessing pipeline.

    Parameters
    ----------
    row : dict
        One row of batch_songs.csv as a plain dict.  Expected keys (new 5-col
        schema): ``artist``, ``album``, ``song``, ``url``, ``notes``.
        Legacy key ``song_name`` is still accepted for back-compat.
        ``version_id`` is injected by the caller (batch_ingest --version_id
        or process_song_offline --version_id); the CSV no longer carries it.
        ``artist`` / ``album`` / ``song`` may be empty — they are
        auto-filled from yt-dlp when *url* is provided.
        An optional ``source_wav`` key may point to a pre-downloaded WAV.
    out_root : Path
        Root of MusicProjectData/ (or ``source_pool/`` if ``source_pool_mode``).
        Works with any path: a local temp dir, a mounted Drive path in Colab.
    skip_if_exists : bool
        Return ``"skipped"`` immediately when manifest_song.csv already
        exists for this song (idempotent re-runs). In ``source_pool_mode``
        the marker is ``metadata.json`` instead.
    separate_stems : bool
        Run Demucs before MIDI transcription.  Default False (Israeli
        pipeline passes full-mix WAV directly to Basic-Pitch).
    source_pool_mode : bool
        When True, treat ``out_root`` as the immutable ``source_pool/`` root:
        skip DSP (steps 4-8) and instead write ``<song>.wav``, ``<song>.mid``,
        ``metadata.json``, and a populated ``augmented/`` folder (WAV+MIDI
        pairs produced via :func:`preprocessing.augmentation.augment_song`).
        Versions are derived later by ``preprocessing/derive_version.py``.

    Returns
    -------
    dict  with keys:
        ``status``        – ``"ok"`` | ``"skipped"`` | ``"failed"``
        ``song_dir``      – absolute path to the song's output folder
        ``manifest_path`` – path to manifest_song.csv (empty string in
        source-pool mode or on fail)
        ``n_segments``    – number of 5-second segments produced (0 in
        source-pool mode)
        ``error``         – error message string or None
    """
    project_root = Path(__file__).parent

    try:
        url        = (row.get("url", "")        or "").strip()
        version_id = int(row.get("version_id", 0))
        artist     = (row.get("artist", "")     or "").strip()
        album      = (row.get("album",  "")     or "").strip()
        song_name  = (row.get("song") or row.get("song_name", "") or "").strip()
        source_wav = (row.get("source_wav", "") or "").strip()

        # ── 0a. Resolve missing metadata from YouTube ──────────────────────
        # The CSV can stay lightweight during curation: if artist/song fields
        # are blank, yt-dlp metadata fills a stable folder name before any files
        # are written. Explicit CSV values always win when provided.
        yt_meta = None
        if url and (not artist or not song_name):
            print("=" * 60)
            print("STEP 0a: Fetching YouTube metadata")
            from preprocessing.youtube_downloader import extract_youtube_metadata
            yt_meta = extract_youtube_metadata(url)
            print(f"  Artist : {yt_meta['artist']}")
            print(f"  Song   : {yt_meta['song']}")
            print(f"  Album  : {yt_meta['album']}")

        artist    = artist    or (yt_meta["artist"] if yt_meta else "Unknown_Artist")
        album     = album     or (yt_meta["album"]  if yt_meta else "Singles")
        song_name = song_name or (yt_meta["song"]   if yt_meta else "Unknown_Song")

        # ── 0b. Skip-if-exists check ───────────────────────────────────────
        # The completion marker depends on mode:
        #   • full DSP mode  → processed_data/manifest_song.csv
        #   • source-pool    → metadata.json (DSP is deferred to derive_version.py)
        song_dir      = out_root / artist / album / song_name
        processed_dir = song_dir / "processed_data"
        manifest_path = processed_dir / "manifest_song.csv"
        metadata_path = song_dir / "metadata.json"

        if skip_if_exists:
            if source_pool_mode and metadata_path.exists():
                print(f"  Skipped: {artist}/{song_name} — metadata.json already exists (source-pool)")
                return {
                    "status": "skipped", "song_dir": str(song_dir),
                    "manifest_path": "", "n_segments": 0, "error": None,
                }
            if (not source_pool_mode) and manifest_path.exists():
                print(f"  Skipped: {artist}/{song_name} — manifest_song.csv already exists")
                return {
                    "status": "skipped", "song_dir": str(song_dir),
                    "manifest_path": str(manifest_path),
                    "n_segments": 0, "error": None,
                }

        # ── 0c. Download from YouTube ──────────────────────────────────────
        # We keep the original WAV alongside processed tensors so later checks
        # can listen, re-transcribe, or rebuild mels without downloading again.
        if url and not source_wav:
            print("=" * 60)
            print("STEP 0c: Downloading from YouTube")
            from preprocessing.youtube_downloader import download_youtube_audio
            dl_dir = project_root / "youtube_downloads"
            downloaded_path, _ = download_youtube_audio(
                url=url, output_dir=str(dl_dir),
                audio_format="wav", audio_quality="best",
                cookies_file=os.environ.get('YTDLP_COOKIES_FILE'),
            )
            source_wav = str(downloaded_path)
            print(f"  ✓ Downloaded: {downloaded_path}")

        # ── 1. Directory structure ─────────────────────────────────────────
        print("=" * 60)
        print("STEP 1: Setting up folder structure")
        mels_dir        = processed_dir / "mels"
        piano_rolls_dir = processed_dir / "piano_rolls"
        for d in [mels_dir, piano_rolls_dir]:
            d.mkdir(parents=True, exist_ok=True)

        wav_path = song_dir / f"{song_name}.wav"
        if source_wav:
            src = Path(source_wav)
            if not src.is_absolute():
                src = project_root / src
            if src.exists() and not wav_path.exists():
                shutil.copy2(str(src), str(wav_path))
                print(f"  Copied WAV → {wav_path}")

        if not wav_path.exists():
            raise FileNotFoundError(
                f"WAV not found at {wav_path}. "
                "Supply --source_wav or a YouTube --url."
            )
        print(f"  Song dir : {song_dir}")
        print(f"  WAV      : {wav_path.name}")

        # ── 2. Optional source separation (Demucs) ─────────────────────────
        # The active Israeli path leaves Demucs off: Basic-Pitch sees the full
        # mix, which matches how the real user-facing input will arrive.
        print("=" * 60)
        dsp_input_wav = wav_path
        if separate_stems:
            print("STEP 2: Source Separation (Demucs)")
            stems_dir = song_dir / "stems"
            stems_dir.mkdir(parents=True, exist_ok=True)
            if not check_stems_exist(stems_dir):
                _run_demucs(
                    audio_path=wav_path, output_dir=stems_dir,
                    model_name="htdemucs_ft", device="cpu",
                )
                print("  ✓ Separation complete")
            else:
                print("  Stems already exist — skipping")
            other_stem = stems_dir / "other.wav"
            if other_stem.exists():
                dsp_input_wav = other_stem
                print(f"  DSP input: {other_stem.name}")
        else:
            print("STEP 2: Source separation OFF (full-mix → Basic-Pitch)")

        # ── 3. MIDI transcription ──────────────────────────────────────────
        # Basic-Pitch lives in the Python 3.10 env, so this helper launches a
        # subprocess instead of importing TensorFlow into the Python 3.12 stack.
        print("=" * 60)
        print("STEP 3: MIDI Transcription (Basic-Pitch on full-mix WAV)")
        midi_path = song_dir / f"{song_name}.mid"
        if not _transcribe_to_midi(wav_path, midi_path, project_root):
            raise RuntimeError("Basic-Pitch MIDI transcription failed")

        # ── Source-pool short-circuit ──────────────────────────────────────
        # In source-pool mode the song folder is the *immutable input* for any
        # number of downstream versions. We deliberately do NOT run DSP here —
        # that work belongs in derive_version.py so each version can pick its
        # own DSPConfig + subset of source_pool songs without duplicating
        # WAV/MIDI bytes. We still write the deterministic augmented WAV+MIDI
        # pairs once, locally, while Basic-Pitch is on the same machine.
        if source_pool_mode:
            print("=" * 60)
            print("STEP 4 (source-pool): writing metadata.json + augmented pairs")
            import datetime as _dt
            import soundfile as _sf
            from preprocessing.augmentation import augment_song, DEFAULT_AUGMENTATIONS

            info = _sf.info(str(wav_path))
            metadata = {
                "artist": artist,
                "album": album,
                "song_name": song_name,
                "url": url or None,
                "native_sr": int(info.samplerate),
                "channels": int(info.channels),
                "duration_s": float(info.frames) / float(info.samplerate),
                "ingest_date": _dt.datetime.utcnow().isoformat() + "Z",
                "transcription_model": "basic_pitch",
                "wav": wav_path.name,
                "midi": midi_path.name,
            }
            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
            print(f"  ✓ metadata.json  ({metadata['duration_s']:.1f}s @ {metadata['native_sr']} Hz)")

            # Augmentations: deterministic MIDI + librosa WAV. Disable via
            # row['augment'] == 0 / 'false' / '' (lets users curate per song).
            aug_flag = str(row.get("augment", "")).strip().lower()
            if aug_flag in ("0", "false", "no", "off"):
                print("  Augmentation: SKIPPED (row.augment is falsy)")
                aug_result = {"produced": [], "skipped": [], "errors": []}
            else:
                print(f"  Augmentation: {len(DEFAULT_AUGMENTATIONS)} variants → augmented/")
                aug_result = augment_song(
                    song_dir,
                    wav_name=wav_path.name,
                    midi_name=midi_path.name,
                    skip_if_exists=skip_if_exists,
                )
                print(f"    produced: {aug_result['produced']}")
                if aug_result["skipped"]:
                    print(f"    skipped:  {aug_result['skipped']}")
                if aug_result["errors"]:
                    print(f"    errors:   {aug_result['errors']}")

            print("=" * 60)
            print(f"DONE (source-pool)  {artist} / {album} / {song_name}")
            return {
                "status": "ok", "song_dir": str(song_dir),
                "manifest_path": "", "n_segments": 0, "error": None,
                "metadata_path": str(metadata_path),
                "augmented": aug_result,
            }

        # ── 4. DSP ─────────────────────────────────────────────────────────
        # The model never sees waveform audio directly. This block converts WAV
        # to normalized mel targets and converts MIDI to piano-roll conditioning
        # on the same time grid, so each segment lines up frame-by-frame.
        print("=" * 60)
        print("STEP 4: DSP — mel extraction + piano roll + segmentation")
        cfg = DSPConfig()

        y        = load_and_resample_audio(dsp_input_wav, cfg.sample_rate)
        duration = len(y) / cfg.sample_rate
        print(f"  Duration : {duration:.2f} s")

        mel_spec           = extract_mel_spectrogram(y, cfg)
        mel_norm, m_min, m_max = normalize_mel(mel_spec)
        print(f"  Mel      : {mel_spec.shape}  [{m_min:.2f}, {m_max:.2f}] → [-1, 1]")

        piano_roll = load_midi_to_piano_roll(midi_path, cfg, duration)
        print(f"  Piano roll: {piano_roll.shape}")

        segments = segment_data(mel_norm, piano_roll, cfg)
        if not segments:
            raise RuntimeError("No segments produced — audio may be too short")
        print(f"  Segments : {len(segments)} × {cfg.segment_duration} s")

        # ── 5. Save tensors ────────────────────────────────────────────────
        song_id = f"{artist}__{album}__{song_name}"
        records = []
        for idx, (mel_seg, piano_seg) in enumerate(segments):
            mel_file   = mels_dir        / f"segment_{idx:04d}.pt"
            piano_file = piano_rolls_dir / f"segment_{idx:04d}.pt"
            # np.ascontiguousarray + .clone() prevents torch.save from persisting
            # the entire underlying storage when mel_seg / piano_seg are views
            # into the full-song arrays (would inflate each .pt by ~50×).
            mel_t   = torch.from_numpy(np.ascontiguousarray(mel_seg)).float().clone()
            piano_t = torch.from_numpy(np.ascontiguousarray(piano_seg)).float().clone()
            torch.save(mel_t,   mel_file)
            torch.save(piano_t, piano_file)
            records.append({
                "segment_path": str(mel_file.relative_to(out_root)),
                "score_path":   str(piano_file.relative_to(out_root)),
                "version_id":   version_id,
                "song_id":      song_id,
                "artist":       artist,
                "album":        album,
                "song_name":    song_name,
                "segment_idx":  idx,
                "duration_s":   cfg.segment_duration,
            })

        # ── 6. DSP config JSON ─────────────────────────────────────────────
        # Store mel min/max per song. Inference needs these values to reverse
        # normalization before the vocoder turns generated mels back into WAV.
        with open(processed_dir / "dsp_config.json", "w") as fh:
            json.dump({
                "sample_rate": cfg.sample_rate, "hop_length": cfg.hop_length,
                "n_fft": cfg.n_fft, "n_mels": cfg.n_mels,
                "fmin": cfg.fmin, "fmax": cfg.fmax,
                "segment_duration": cfg.segment_duration,
                "mel_min": float(m_min), "mel_max": float(m_max),
            }, fh, indent=2)

        # ── 7. Per-song manifest ───────────────────────────────────────────
        # The manifest is the contract with training: every row points to one
        # mel tensor, the matching score tensor, and the version/style ID.
        with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(records)
        print(f"  ✓ manifest_song.csv  ({len(records)} rows)")

        # ── 8. Visualization (best-effort, non-critical) ───────────────────
        # A failed PNG should not invalidate a good dataset item; it is only a
        # human inspection aid for checking mel/score alignment.
        try:
            create_visualization(
                segments[0][0], segments[0][1], cfg,
                song_name, processed_dir / "visualization.png",
                mel_min=m_min, mel_max=m_max,
            )
        except Exception:
            pass

        print("=" * 60)
        print(f"DONE  {artist} / {album} / {song_name}  ({len(segments)} segments)")
        return {
            "status": "ok", "song_dir": str(song_dir),
            "manifest_path": str(manifest_path),
            "n_segments": len(segments), "error": None,
        }

    except Exception as exc:
        return {
            "status": "failed", "song_dir": str(out_root),
            "manifest_path": "", "n_segments": 0, "error": str(exc),
        }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse single-song CLI arguments and call process_song()."""
    parser = argparse.ArgumentParser(
        description="Process a single song through the preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect metadata from a YouTube URL:
  python process_song_offline.py --url "https://youtu.be/..." \
      --out_root /data/MusicProjectData

  # Re-enable Demucs source separation:
  python process_song_offline.py --url "https://youtu.be/..." \
      --out_root /data/MusicProjectData --separate_stems

  # Process a local WAV with explicit metadata:
  python process_song_offline.py --artist "Arik_Einstein" \
      --song "Atur_Mitzchek" --source_wav downloads/song.wav \
      --out_root /data/MusicProjectData
""",
    )
    parser.add_argument("--out_root", required=True,
                        help="Absolute path to MusicProjectData/ output root")
    parser.add_argument("--url",        default=None, help="YouTube URL")
    parser.add_argument("--source_wav", default=None, help="Path to a pre-downloaded WAV")
    parser.add_argument("--artist",     default=None)
    parser.add_argument("--album",      default=None)
    parser.add_argument("--song",       default=None,
                        help="Song name (auto-detected from YouTube if omitted)")
    parser.add_argument("--version_id", type=int, default=0,
                        help="Style/version ID for this song. Set per invocation; "
                             "the CSV no longer carries a version_id column. "
                             "Run one batch per style (e.g. 0 = Slakh, 1 = Israeli).")
    parser.add_argument("--separate_stems", action="store_true",
                        help="Run Demucs before MIDI transcription (default: off)")
    parser.add_argument("--source-pool-mode", "--source_pool_mode",
                        dest="source_pool_mode", action="store_true",
                        help="Stop after Basic-Pitch; write metadata.json + "
                             "augmented/ pairs only (no DSP). Use this when "
                             "out_root is the immutable source_pool/ root.")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-process even if completion marker already exists")
    args = parser.parse_args()

    row = {
        "url":        args.url        or "",
        "version_id": args.version_id,
        "artist":     args.artist     or "",
        "album":      args.album      or "",
        "song_name":  args.song       or "",
        "source_wav": args.source_wav or "",
    }
    result = process_song(
        row=row,
        out_root=Path(args.out_root),
        skip_if_exists=not args.no_skip,
        separate_stems=args.separate_stems,
        source_pool_mode=args.source_pool_mode,
    )
    print(f"\nStatus   : {result['status']}")
    print(f"Song dir : {result['song_dir']}")
    if result["status"] == "ok":
        print(f"Segments : {result['n_segments']}")
        print(f"Manifest : {result['manifest_path']}")
    elif result["status"] == "failed":
        print(f"Error    : {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
