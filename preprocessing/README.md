# `preprocessing/` — Input data preparation

Functions that turn raw audio into model-ready tensors.

## Architecture: `source_pool/` vs `versions/<v>/`

The pipeline is split into an **immutable source pool** (one WAV+MIDI per song,
plus pre-computed augmented WAV+MIDI pairs) and **cheap, regenerable versions**
(subset of songs + frozen DSPConfig + frozen aug policy). Basic-Pitch runs
**local-only on Windows** (`basic_pitch_env`). DSP + training run on Colab.
See main [../README.md](../README.md) section **Israeli Data Pipeline** for the
full Drive layout, CSV schemas, and the one-pass workflow.

## Active modules (used by current pipeline)

| File | What |
|---|---|
| `dsp_preprocessor.py` | Core DSP: mel extraction, normalization, segmentation, `dsp_config.json` persistence. Shared by `batch_ingest`, `process_song_offline`, and `derive_version`. |
| `audio_tp_midi_poc.py` | Basic-Pitch subprocess bridge (Python 3.10 env) for MIDI transcription. **Local-only** — never runs on Colab. |
| `youtube_downloader.py` | yt-dlp wrapper with auto-metadata extraction. |
| `source_separator.py` | Demucs v4 wrapper (off by default for Israeli pipeline; on with `--separate_stems`). |
| `augmentation.py` | Two surfaces: **(a)** `JointAugment` (legacy in-memory mel+piano-roll augment, used by `batch_ingest`); **(b)** offline WAV+MIDI augmentation: `pitch_shift_wav`, `time_stretch_wav`, `pitch_shift_midi`, `time_scale_midi`, `augment_song()` driven by `DEFAULT_AUGMENTATIONS` (ps±2, ts0.9/1.1). Used by `process_song_offline.py --source-pool-mode` to populate `source_pool/<song>/augmented/`. |
| `batch_ingest.py` | Batch driver: runs `process_song()` over `batch_songs.csv`, writes manifests, uploads to Drive via `--upload_to_drive`. Original local data pipeline entry point (kept for compatibility). |
| `process_song_offline.py` *(at project root)* | Per-song local entry point. With `--source-pool-mode` it writes raw `<song>.wav` + Basic-Pitch `<song>.mid` + `metadata.json` + `augmented/*.{wav,mid}` into `source_pool/<artist>/<album>/<song>/` and **skips DSP** (Colab does that). |
| `derive_version.py` | **Colab-friendly** DSP-only driver. Reads `source_pool/` + a `version_spec.yaml`, runs `DSPPreprocessor` on each (song, augmentation) pair, emits `versions/<v>/processed_data/<.../>{mels,piano_rolls,manifest_song.csv,dsp_config.json,preprocessing_demo.png}`. No Basic-Pitch dependency. |
| `dataset_visualizations.py` | Plot utilities + a `plot_preprocessing_demo()` 6-panel walkthrough (raw → resampled → mel filterbank → log-mel → normalised mel + segment boundaries + piano-roll inset). CLI: `python -m preprocessing.dataset_visualizations --demo <wav> [--midi <mid>] --out <png>`. Called best-effort from `derive_version` per song. |
| `split_dataset.py` | Song-grouped, deterministic train/val/test split (no song leakage). Held-out songs listed in `run_spec.yaml` should be excluded here. |
| `slakh_adapter.py` | Adapt Slakh2100 tracks to pipeline format (for sanity training). |
| `drive_sync.py` | Google Drive upload with retry (5× backoff) + skip-existing resume. |
| `midi_diagnostic.py`, `midi_visualizer.py` | Debugging / inspection helpers. |

## Stored augmentation behavior

| Path | Current behavior |
|---|---|
| `process_song_offline.py --source-pool-mode` | Writes 4 WAV+MIDI augmented pairs per song (ps+2, ps-2, ts0.9, ts1.1) into `source_pool/<song>/augmented/`. MIDI is derived deterministically: pretty_midi transpose for pitch shift, time-scale for time stretch. **Basic-Pitch is never re-run per augmentation.** |
| `derive_version.py` | Reads each augmented pair and produces a parallel `processed_data/<song>/<aug_tag>/` sub-tree of segmented tensors. The `aug_tag` column appears on every manifest row. |
| `batch_ingest.py` *(legacy path)* | Reads the `augment` flag; produces 3 in-memory augmented variant tensor sets (`_aug_pitch`, `_aug_time`, `_aug_combined`). Together with the original, that gives 4× data per song. |
| `batch_songs.csv` | 5 columns: `artist, album, song, url, notes`. Style and augmentation are set per batch via `--version_id` / `--augment` on `batch_ingest.py`. Used by both `batch_ingest.py` and `process_song_offline.py --source-pool-mode`. |

## Resolved: `batch_ingest.py` augmentation-count logging

Previously `batch_ingest.py` updated the augmentation count in `log_rows[-1]` before appending the current row, which could update the wrong log row (or crash on the first augmented row). **FIXED** in the Israeli readiness work: `n_aug` is accumulated into the current row before `log_rows.append(...)`. See [../docs/KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md).

## Deprecated / superseded

| File | What |
|---|---|
| `gdrive_uploader.py` | Older OAuth-based Drive uploader. Mostly superseded by `drive_sync.py` for the Israeli pipeline. Keep for compatibility/reference unless the Drive flow is consolidated later. |
