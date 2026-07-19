# Tests

This folder holds focused unit and integration tests. The canonical local test entry point is [../smoke_test_local.py](../smoke_test_local.py), which orchestrates a flat list of `T_*` checks. Cloud equivalents live in [../colab/smoke_test.ipynb](../colab/smoke_test.ipynb).

## Test files

| File | Purpose | Status |
|---|---|---|
| `test_augmentation.py` | `T_aug` — feeds a real WAV through `JointAugment`, vocodes, and verifies pitch/time behavior numerically | Implemented |
| `aug_hearing_test.py` | Hearing-oriented augmentation check that writes artifacts for manual listening | Implemented |
| `pipeline_full_test.py` | End-to-end 4-config integration test for ingest, optional Demucs, Drive upload, vocoder, F1, FAD, and visualization | Implemented |
| `test_metrics.py` | Future unit tests for `f1_eval.compute_f1()` and `fad_eval.compute_group_fad()` on synthetic inputs | Planned |
| `test_dsp.py` | Future round-trip mel/audio checks and segment-boundary tests | Planned |

## Running tests today

```powershell
# from MusicProject/
.\ml_env\Scripts\python.exe smoke_test_local.py            # all local tests
.\ml_env\Scripts\python.exe smoke_test_local.py --skip-t1  # skip the YouTube + Drive test
```

## Cloud tests

See [../colab/smoke_test.ipynb](../colab/smoke_test.ipynb) — CT1 (Colab env setup) → CT5 (split_dataset).

## Pipeline integration test (`pipeline_full_test.py`)

End-to-end 4-config integration test exercising `process_song` × {Demucs on/off} × {local, Drive upload}, plus a postprocessing block (BigVGAN round-trip + F1/FAD on shipped fixtures + FAD PCA visualization).

```powershell
# from MusicProject/
$env:PYTHONIOENCODING='utf-8'; .\ml_env\Scripts\python.exe tests\pipeline_full_test.py 2>&1 | Tee-Object -FilePath tests\_pipeline_full_test.log
```

**Outputs** (gitignored):
- `tests/_pipeline_full_test.log` — full stdout/stderr.
- `tests/_pipeline_full_test_out/` — 4 config dirs (`A_local_nosep`, `B_local_sep`, `C_drive_nosep`, `D_drive_sep`) + `_download/` cache. Configs C and D delete their local copies after Drive upload; C/D dirs end up empty (verify on Drive under `MusicProject/MusicProjectData/`).

**SUMMARY block** at the end of the log shows pass/fail per row.
