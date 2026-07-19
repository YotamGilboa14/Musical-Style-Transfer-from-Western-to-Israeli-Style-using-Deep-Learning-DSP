# `colab/` — Colab Pro notebooks

Cloud **training** and some verification workflows run from these notebooks. The preferred Israeli data ingestion path is local preprocessing through `batch_ingest.py --upload_to_drive`, followed by Drive-based verification in Colab.

## Active notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `train_sanity.ipynb` | Slakh2100 sanity/extended training and post-training evaluation | Active; 30k/100k/150k Slakh sanity runs complete and accepted (architecture validation done) |
| `train_israeli.ipynb` | Multi-version Slakh + Israeli training notebook | Used for the final 3-version run. `Israeli_3style` trained from scratch as a clean multi-version run (`n_versions=3`); no checkpoint embedding migration is needed |
| `data_ingest_israeli.ipynb` | Drive-side verification and split creation for Israeli tensors | Updated to the current per-song `processed_data/` layout and `split_dataset(manifest_path=..., out_dir=...)` API (see KNOWN_ISSUES) |
| `smoke_test.ipynb` | Cloud smoke tests CT1–CT8 (env, data validation, aug, eval) | Deferred — local pipeline proven; run before cloud training if needed |
| `data_pipeline.ipynb` | Mount Drive → clone repo → run `batch_ingest` + `split_dataset` end-to-end on Colab | Available, but not the preferred path for Israeli ingest |
| `postprocessing.ipynb` | Step search → in-Colab FAD + latency → composite-z best-step selection. Calls out the local-only F1 step explicitly (see §37). | Active; the supported postprocessing entry point |

## Deprecated (kept for defense reference)

| Notebook | Why kept |
|---|---|
| `tt_vae_gan_transfer.ipynb` | Phase 4B POC — trained the trumpet→violin VAE-GAN model. Referenced in `POC_EXPERIMENT.md`. **Not** part of the current diffusion pipeline. |

## Convention

Every notebook starts with Drive-mount + repo-clone cells so you can launch into a fresh Colab session without worrying about state.

Use `requirements_colab.txt` in Colab. Do not install `requirements_ml.txt` on Colab unless you intentionally want CPU PyTorch wheels to replace the CUDA build.

### Cell-header documentation template (project standard, ENGINEERING_DECISIONS §37)

Every newly added or edited code cell in this folder is preceded by a markdown header
that follows this exact shape:

```markdown
## Cell N — <Short title>

**What this does.** One or two sentences in plain English.

**Inputs.** Files / variables / env this cell reads.

**Outputs.** Files / variables / state this cell writes.

**Action required.** Anything the user must edit, click, approve, or run
elsewhere (e.g. an OAuth prompt, a local Basic-Pitch step). Use **⚠ Cell N — …**
in the heading when the action requires switching machines (e.g. F1 backfill).

**Runtime.** Order-of-magnitude (seconds / minutes / hours).
```

`colab/postprocessing.ipynb` is the reference implementation of this
template. Older notebooks (`train_sanity`, `train_israeli`, `data_ingest_israeli`,
`smoke_test`, `data_pipeline`, `tt_vae_gan_transfer`) predate the convention;
when a cell in any of them is touched, update its header to this shape in
the same edit.

## Data flow

```
Local machine:
  batch_ingest.py --upload_to_drive   →  uploads .pt tensors + manifests to Drive

Colab:
  mounts Drive  →  reads manifests + tensors  →  trains model  →  writes checkpoints to Drive
```
