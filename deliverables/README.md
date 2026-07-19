# TA-grading deliverables

This folder is auto-populated by `build_deliverables.py`. It indexes every
artifact a grader needs to evaluate a training run, and it is intentionally
the **same layout for every style version** (Slakh v0, Israeli v1, future
v2+) so reviewers always know where to look.

## Quick start

```powershell
# from MusicProject/
.\ml_env\Scripts\python.exe .\build_deliverables.py --config .\deliverables\config.example.yaml
# then open deliverables/00_overview/index.html
```

### Real Israeli_3style bundle (reuse mode)

The graded bundle for the actual model reuses already-built figures instead of
recomputing (FAD/F1 come straight from the finalist-metrics run; latency and the
transfer grid come from the demo gallery), so it builds in seconds:

```powershell
# from MusicProject/
.\ml_env\Scripts\python.exe .\build_deliverables.py `
    --config .\deliverables\config_israeli_3style.yaml `
    --out_dir .\deliverables\israeli_3style
# then open deliverables/israeli_3style/00_overview/index.html
```

Any result section (`fad`, `f1`, `latency`, `transfer`) may carry a `reuse:`
block that points at pre-built assets (`source_dir` + `files` / `dirs` / `globs`
+ optional `source_index`) instead of the recompute inputs. See
`config_israeli_3style.yaml` for the full example.

## Folder map

```
deliverables/
├─ 00_overview/index.html         <- start here; links every stage
├─ 01_data_quality/
│  ├─ gate_report.html            <- pre-training data-quality gate
│  └─ _assets/                    <- standalone PNGs, gate_report.json, mel/PR grids
├─ 01b_preprocessing/
│  ├─ index.html                  <- per-song 6-panel preprocessing walkthrough + explainer
│  └─ _assets/                    <- <name>__preprocessing_demo.png per featured song, download_all.zip
├─ 02_fad/
│  ├─ fad_table.html              <- All-FAD + Group-FAD per style
│  └─ _assets/                    <- fad_per_style.png, fad_summary.json
├─ 03_f1/
│  ├─ f1_table.html               <- transcription F1 per benchmark
│  └─ _assets/                    <- f1_sensitivity.png, *.wav copies
├─ 04_latency/
│  ├─ latency_table.html          <- per-config latency + RTF
│  └─ _assets/                    <- latency_<config>.png, latency_summary.json
└─ 05_transfer_grid/
   ├─ transfer_grid.html          <- qualitative examples (mel + audio)
   └─ _assets/                    <- standalone PNGs and WAVs per example
```

## PowerPoint workflow

Every HTML page begins with a **Download** bar:

- `download_all.zip` — every asset for the stage in one archive.
- individual `.png` / `.wav` links — right-click → "Save link as…" for slides.

All PNGs are rendered at `dpi=200`, ~1600×900, with `bbox_inches='tight'`.
All WAVs are copied (not embedded) so you can drop them straight into a
slide's audio control.

## Config

See `config.example.yaml` for the schema. Every section except
`run_name` / `style_name` / `version_id` is optional — missing sections are
reported as `[skipped]` in the overview, so this same script works for an
input-only data-quality check, a full results pass, or anything in between.

## What goes where (gate decision matrix)

| Question | Look at |
|---|---|
| Is the dataset healthy before training? | `01_data_quality/gate_report.html` |
| What does our preprocessing actually do to one song? | `01b_preprocessing/index.html` |
| How close is the generated style to real? | `02_fad/fad_table.html` |
| Does the model still play the right notes? | `03_f1/f1_table.html` |
| Can we run it in real time? | `04_latency/latency_table.html` |
| What does a successful transfer sound like? | `05_transfer_grid/transfer_grid.html` |
