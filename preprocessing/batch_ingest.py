"""
Batch Ingest — process every song in batch_songs.csv
=====================================================

Reads batch_songs.csv (5-column schema: artist, album, song, url, notes),
calls process_song() per row, and writes a consolidated dataset_manifest.csv.
All rows present in the CSV are processed (no per-row enable flag). Per-batch
style is set via ``--version_id``; per-batch augmentation via ``--augment``.
Validation fails loudly on any row missing required fields.

Supports both execution paths. The current Israeli workflow is Path B: process
one URL locally, upload WAV/MIDI/tensors/plots/manifests to Drive, then continue.
    Path A (Colab):  out_root = Drive path  ->  data lands directly on Drive
    Path B (local):  out_root = local dir   ->  then --upload_to_drive + --clean_after_upload

Usage examples
--------------
# Path A — all on Colab, data goes straight to Drive
python preprocessing/batch_ingest.py \
    --csv            /content/drive/MyDrive/MusicProject/batch_songs.csv \
    --out_root       /content/drive/MyDrive/MusicProject/MusicProjectData \
    --manifest_out   /content/drive/MyDrive/MusicProject/data/dataset_manifest.csv \
    --version_id     1

# Path B — local preprocessing, then upload & clean
python preprocessing/batch_ingest.py \
    --csv            batch_songs.csv \
    --out_root       C:/tmp/MusicProjectData \
    --manifest_out   C:/tmp/dataset_manifest.csv \
    --version_id     1 \
    --augment \
    --upload_to_drive  <Drive folder ID for MusicProjectData> \
    --clean_after_upload

# Resume (already-processed songs are skipped automatically)
# Just re-run the same command — skip_if_exists=True by default.
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow running from repo root or from preprocessing/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from process_song_offline import process_song, _MANIFEST_FIELDS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_batch_csv(path: Path) -> list[dict]:
    """Read batch_songs.csv (5-column schema: artist, album, song, url, notes).

    Validates each row has non-empty ``artist``, ``album``, ``song``, ``url``.
    Raises ``ValueError`` with the 1-based row number on the first bad row
    (fail loudly so users fix their CSV before a long batch runs).
    """
    required = ("artist", "album", "song", "url")
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing_cols = [c for c in required if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(
                f"{path}: missing required column(s) {missing_cols}. "
                f"Expected header: artist,album,song,url,notes"
            )
        for i, row in enumerate(reader, start=2):  # +1 for header, +1 for 1-indexed
            blanks = [c for c in required if not (row.get(c) or "").strip()]
            if blanks:
                raise ValueError(
                    f"{path} row {i}: missing required field(s) {blanks}. "
                    f"Every row must have artist, album, song, url."
                )
            rows.append(row)
    return rows


def _append_to_manifest(manifest_path: Path, song_manifest_path: Path) -> int:
    """Append rows from a per-song manifest_song.csv into the consolidated manifest.
    Returns number of rows appended."""
    if not song_manifest_path.exists():
        return 0

    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

    with open(song_manifest_path, newline="", encoding="utf-8") as src, \
         open(manifest_path, "a", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=_MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        n = 0
        for row in reader:
            writer.writerow({k: row.get(k, "") for k in _MANIFEST_FIELDS})
            n += 1
    return n


# Source-pool consolidated index: one row per ingested song. DSP/mels are
# deferred to derive_version.py on Colab, so this index — not a tensor manifest
# — is what curation and downstream version-derivation read.
_SOURCE_POOL_INDEX_FIELDS = [
    "artist", "album", "song", "url",
    "wav", "midi",
    "duration_s", "native_sr",
    "augmented_count",
    "ingest_status",
]


def _append_to_source_pool_index(
    index_path: Path,
    song_dir: Path,
    row: dict,
    status: str,
) -> int:
    """Append one row describing this song's source-pool artifacts.

    Reads ``metadata.json`` for duration/SR and counts ``augmented/*.wav`` files.
    Returns 1 if a row was written, 0 otherwise (e.g. metadata missing).
    """
    metadata_path = song_dir / "metadata.json"
    if not metadata_path.exists():
        return 0

    try:
        import json as _json
        with open(metadata_path, "r", encoding="utf-8") as fh:
            meta = _json.load(fh)
    except Exception:
        return 0

    aug_dir = song_dir / "augmented"
    aug_count = len(list(aug_dir.glob("*.wav"))) if aug_dir.exists() else 0

    write_header = not index_path.exists() or index_path.stat().st_size == 0
    out_row = {
        "artist":          row.get("artist", "")    or meta.get("artist", ""),
        "album":           row.get("album", "")     or meta.get("album", ""),
        "song":            row.get("song") or row.get("song_name", "") or meta.get("song_name", ""),
        "url":             row.get("url", "")       or (meta.get("url") or ""),
        "wav":             meta.get("wav", ""),
        "midi":            meta.get("midi", ""),
        "duration_s":      f"{float(meta.get('duration_s', 0.0)):.3f}",
        "native_sr":       meta.get("native_sr", ""),
        "augmented_count": aug_count,
        "ingest_status":   status,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SOURCE_POOL_INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(out_row)
    return 1


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

_AUG_CONFIGS = {
    "aug_pitch": {
        "enabled": True,
        "pitch_shift":  {"p": 1.0, "max_semitones": 2},
        "time_stretch": {"p": 0.0},
        "spec_augment": {"p": 0.0},
    },
    "aug_time": {
        "enabled": True,
        "pitch_shift":  {"p": 0.0},
        "time_stretch": {"p": 1.0, "max_pct": 0.10},
        "spec_augment": {"p": 0.0},
    },
    "aug_combined": {
        "enabled": True,
        "pitch_shift":  {"p": 1.0, "max_semitones": 2},
        "time_stretch": {"p": 1.0, "max_pct": 0.10},
        "spec_augment": {"p": 1.0, "time_mask_max": 20, "freq_mask_max": 6,
                         "n_time": 1, "n_freq": 1},
    },
}


def augment_song(
    song_dir: Path,
    out_root: Path,
    row: dict,
    manifest_out: Path,
) -> int:
    """Generate augmented .pt variants for every segment in song_dir.

    For each ``segment_NNNN.pt`` in ``mels/`` / ``piano_rolls/``, produces
    three new pairs named ``segment_NNNN_aug_pitch.pt``,
    ``segment_NNNN_aug_time.pt``, ``segment_NNNN_aug_combined.pt`` in the
    same directories.  Appends the new rows directly to manifest_out.
    Skips a variant file if it already exists.

    Returns number of new segment files written.
    """
    import random
    import torch
    from preprocessing.augmentation import JointAugment

    processed_dir   = song_dir / "processed_data"
    mels_dir        = processed_dir / "mels"
    piano_rolls_dir = processed_dir / "piano_rolls"

    if not mels_dir.exists():
        print(f"  [augment] mels dir not found: {mels_dir}")
        return 0

    mel_files = sorted(mels_dir.glob("segment_????.pt"))
    if not mel_files:
        print(f"  [augment] no base segments found in {mels_dir}")
        return 0

    version_id = row.get("version_id", "1")
    artist     = row.get("artist", "")
    album      = row.get("album", "")
    song_name  = row.get("song") or row.get("song_name", "")
    song_id    = f"{artist}__{album}__{song_name}"

    augmenters = {name: JointAugment(cfg) for name, cfg in _AUG_CONFIGS.items()}

    write_header = not manifest_out.exists() or manifest_out.stat().st_size == 0
    new_rows = []

    for mel_file in mel_files:
        stem = mel_file.stem          # e.g. "segment_0000"
        piano_file = piano_rolls_dir / f"{stem}.pt"
        if not piano_file.exists():
            continue

        # Load the paired target/conditioning tensors. Augmentation must touch
        # both together: changing pitch/time in the mel without applying the
        # matching transform to the piano roll would teach the model bad labels.
        mel_norm   = torch.load(mel_file,   weights_only=True)
        piano_roll = torch.load(piano_file, weights_only=True)

        for aug_name, aug in augmenters.items():
            out_mel   = mels_dir        / f"{stem}_{aug_name}.pt"
            out_piano = piano_rolls_dir / f"{stem}_{aug_name}.pt"
            if out_mel.exists() and out_piano.exists():
                continue   # already done — idempotent

            # Seed per song/segment/augmentation so re-runs produce identical
            # variants. That makes Drive uploads resumable and comparisons fair.
            random.seed(hash(f"{song_id}_{stem}_{aug_name}") & 0xFFFFFFFF)
            torch.manual_seed(hash(f"{song_id}_{stem}_{aug_name}") & 0xFFFFFFFF)

            mel_aug, pr_aug = aug(mel_norm, piano_roll)
            torch.save(mel_aug.clone(),  out_mel)
            torch.save(pr_aug.clone(),   out_piano)

            seg_idx_str = stem.replace("segment_", "")
            new_rows.append({
                "segment_path": str(out_mel.relative_to(out_root)),
                "score_path":   str(out_piano.relative_to(out_root)),
                "version_id":   version_id,
                "song_id":      song_id,
                "artist":       artist,
                "album":        album,
                "song_name":    song_name,
                "segment_idx":  f"{seg_idx_str}_{aug_name}",
                "duration_s":   "",
            })

    if new_rows:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_out, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
        print(f"  [augment] +{len(new_rows)} augmented segments written")
    else:
        print(f"  [augment] all variants already exist — skipped")

    return len(new_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_batch(
    csv_path: Path,
    out_root: Path,
    manifest_out: Path,
    resolved_csv: Path | None,
    log_path: Path | None,
    separate_stems: bool,
    no_skip: bool,
    upload_to_drive: str | None,
    clean_after_upload: bool,
    start_index: int,
    version_id: int,
    augment: bool,
    source_pool_mode: bool = False,
) -> None:
    """Run the batch ingest pipeline over every CSV row.

    This is the orchestration layer above process_song(): it handles resume,
    consolidated manifest writing, optional stored augmentation, optional Drive
    upload, local cleanup, and per-row logging. It intentionally processes one
    song at a time so a large Israeli ingest does not accumulate a full local
    copy of the dataset.

    ``version_id`` is injected into every row before dispatch (the CSV no
    longer carries a per-row version). ``augment`` applies to every row.

    When ``source_pool_mode`` is True:
      * ``out_root`` is treated as the immutable ``source_pool/`` root.
      * Each song writes WAV/MIDI/metadata.json + augmented WAV+MIDI pairs only
        (no mels, no ``.pt`` tensors). DSP is deferred to derive_version.py.
      * The in-batch tensor augmenter is skipped; per-song augmentation runs
        inside process_song() on raw WAV+MIDI.
      * ``manifest_out`` collects a source-pool index (one row per song with
        artist/album/song/url/wav/midi/duration/SR/aug-count/status) instead of
        a per-segment tensor manifest.
    """
    rows = _load_batch_csv(csv_path)
    total = len(rows)
    print(
        f"Loaded {total} songs from {csv_path}  "
        f"(version_id={version_id}, augment={augment}, source_pool_mode={source_pool_mode})"
    )

    # Drive upload helper — only imported when needed
    if upload_to_drive:
        from preprocessing.drive_sync import upload_song_to_drive, clean_local_song

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    log_rows = []

    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(rows):
        if i < start_index:
            print(f"[{i+1}/{total}] Skipping (before --start_index {start_index})")
            continue

        song_label = f"{row.get('artist','?')} / {row.get('song') or row.get('song_name','?')}"
        print(f"\n[{i+1}/{total}] {song_label}")

        # Inject batch-level version_id into the row (CSV no longer carries it).
        # In source-pool mode also inject the augment toggle, because the WAV+MIDI
        # augmenter inside process_song() reads row["augment"] (skips only on
        # "0"/"false"/"no"/"off"); without this, --augment would have no effect.
        row = {**row, "version_id": version_id, "song_name": row.get("song") or row.get("song_name", "")}
        if source_pool_mode:
            row["augment"] = "1" if augment else "0"

        result = process_song(
            row=row,
            out_root=out_root,
            skip_if_exists=not no_skip,
            separate_stems=separate_stems,
            source_pool_mode=source_pool_mode,
        )

        status = result["status"]
        stats[status] += 1

        # Consolidated index:
        #   * source-pool mode  → one row per song into source_pool_index.csv
        #     (read from metadata.json; tensor manifest does not exist here)
        #   * legacy DSP mode   → append per-segment rows from manifest_song.csv
        n_appended = 0
        if status in ("ok", "skipped"):
            if source_pool_mode and result.get("song_dir"):
                n_appended = _append_to_source_pool_index(
                    manifest_out, Path(result["song_dir"]), row, status,
                )
                print(f"  source_pool_index: +{n_appended} row")
            elif result.get("manifest_path"):
                n_appended = _append_to_manifest(manifest_out, Path(result["manifest_path"]))
                print(f"  manifest: +{n_appended} segments")

        # Augmentation (generate _aug_pitch / _aug_time / _aug_combined variants).
        # In source-pool mode, per-song WAV+MIDI augmentation already ran inside
        # process_song(); the tensor augmenter here is the wrong primitive
        # (it operates on mel/PR .pt files that don't exist in source-pool mode).
        n_aug = 0
        if (not source_pool_mode) \
                and augment \
                and status in ("ok", "skipped") \
                and result.get("song_dir"):
            n_aug = augment_song(
                song_dir=Path(result["song_dir"]),
                out_root=out_root,
                row=row,
                manifest_out=manifest_out,
            )
            # Augmented variants count toward the *current* row's manifest
            # contribution. We accumulate `n_aug` into `n_appended` *before*
            # appending this row's log dict (previously this indexed
            # log_rows[-1], which was either the prior row or, on i=0, a crash).

        # Path B: upload → clean
        # This is the canonical Israeli-data route: local disk holds at most the
        # current song, while Drive becomes the durable shared dataset store.
        # Upload for both "ok" (freshly processed) and "skipped" (already processed);
        # drive_sync skips files already present on Drive so this is idempotent.
        if status in ("ok", "skipped") and upload_to_drive and result.get("song_dir"):
            song_dir = Path(result["song_dir"])
            print(f"  Uploading {song_dir.name} to Drive …")
            upload_song_to_drive(song_dir, upload_to_drive)
            if clean_after_upload:
                print(f"  Cleaning local copy …")
                clean_local_song(song_dir)

        log_rows.append({
            "index": i,
            "artist": row.get("artist", ""),
            "song_name": row.get("song") or row.get("song_name", ""),
            "status": status,
            "n_segments": result.get("n_segments", 0),
            "n_appended": n_appended + n_aug,
            "error": result.get("error", ""),
        })

    # Write log
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["index", "artist", "song_name",
                                                     "status", "n_segments",
                                                     "n_appended", "error"])
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nLog written to {log_path}")

    # Write resolved CSV (same as input but with status column added)
    if resolved_csv:
        resolved_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, newline="", encoding="utf-8") as src, \
             open(resolved_csv, "w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            fieldnames = (reader.fieldnames or []) + ["ingest_status"]
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for j, row in enumerate(reader):
                row["ingest_status"] = log_rows[j]["status"] if j < len(log_rows) else ""
                writer.writerow(row)

    print(f"\n{'='*50}")
    print(f"Batch complete — ok={stats['ok']}  skipped={stats['skipped']}  "
          f"failed={stats['failed']}  total={total}")
    print(f"Manifest: {manifest_out}")

    # Upload consolidated manifest + log to Drive root folder
    if upload_to_drive:
        from preprocessing.drive_sync import _upload_file, _get_drive_service
        svc = _get_drive_service()
        for fpath in [manifest_out, log_path]:
            if fpath and fpath.exists():
                print(f"  Uploading {fpath.name} to Drive …")
                _upload_file(svc, fpath, upload_to_drive)
        print("  Drive upload complete.")


def main() -> None:
    """Parse batch-ingest CLI flags and run the CSV-driven pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch-process all enabled songs in batch_songs.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",      required=True, help="Path to batch_songs.csv")
    parser.add_argument("--out_root", required=True,
                        help="Root output directory for processed data")
    parser.add_argument("--manifest_out", required=True,
                        help="Path to write consolidated dataset_manifest.csv")
    parser.add_argument("--resolved_csv", default=None,
                        help="Optional: write a copy of batch_songs.csv with ingest_status column")
    parser.add_argument("--log", default=None,
                        help="Optional: path to write per-song log CSV")
    parser.add_argument("--separate_stems", action="store_true",
                        help="Run Demucs before MIDI transcription (default: off)")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-process even if a song was already processed")
    parser.add_argument("--upload_to_drive", default=None, metavar="FOLDER_ID",
                        help="(Path B) Upload each processed song dir to this Drive folder ID")
    parser.add_argument("--clean_after_upload", action="store_true",
                        help="(Path B) Delete local song dir after uploading to Drive")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Skip first N rows (for resuming mid-batch)")
    parser.add_argument("--version_id", type=int, default=1,
                        help="Style/version ID applied to every row in this batch "
                             "(default: 1 = Israeli). The CSV no longer carries "
                             "a per-row version_id; run one batch per style.")
    parser.add_argument("--augment", action="store_true",
                        help="Generate aug_pitch / aug_time / aug_combined "
                             "variants for every processed song (default: off).")
    parser.add_argument("--source_pool_mode", action="store_true",
                        help="Treat out_root as the immutable source_pool/ root: "
                             "write only WAV/MIDI/metadata.json + augmented WAV+MIDI "
                             "(no mels, no .pt). DSP is deferred to derive_version.py. "
                             "The consolidated --manifest_out becomes a source-pool "
                             "index with one row per song (Israeli ingest path).")
    args = parser.parse_args()

    run_batch(
        csv_path=Path(args.csv),
        out_root=Path(args.out_root),
        manifest_out=Path(args.manifest_out),
        resolved_csv=Path(args.resolved_csv) if args.resolved_csv else None,
        log_path=Path(args.log) if args.log else None,
        separate_stems=args.separate_stems,
        no_skip=args.no_skip,
        upload_to_drive=args.upload_to_drive,
        clean_after_upload=args.clean_after_upload,
        start_index=args.start_index,
        version_id=args.version_id,
        augment=args.augment,
        source_pool_mode=args.source_pool_mode,
    )


if __name__ == "__main__":
    main()
