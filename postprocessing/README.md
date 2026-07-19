# `postprocessing/` — Vocoding + evaluation

Everything that turns a mel back into audio and scores the result.

## Vocoder

| File | What |
|---|---|
| `vocoder_factory.py` | Unified factory — `create_vocoder(name)` returns one of `hifi_gan`, `bigvgan_22k`, `bigvgan_24k`. |
| `bigvgan_vocoder.py` | BigVGAN v2 22 kHz wrapper (primary vocoder). |
| `hifigan_model.py` | HiFi-GAN UNIVERSAL_V1 architecture (baseline for comparison). |
| `vocoder_inference.py` | High-level wav→mel→wav round-trip utilities. |
| `hifigan_checkpoints/` | HiFi-GAN UNIVERSAL_V1 weights. |

## Evaluation metrics

| File | What |
|---|---|
| `f1_eval.py` | Note-level F1 on Basic-Pitch transcription of generated WAV vs. reference MIDI (pitch + onset ± 50 ms). Two CLI modes: **(1)** single pair — `--generated_wav --reference_midi`; **(2)** batch run-dir — `--run-dir versions/<v>/inference_runs/<run_id>/` scores every `audio/*.wav` against the matching reference from `run_spec.copy.yaml`, writes `metrics/f1_per_pair.json`, and back-fills the `f1` column of `inference_runs/_index.csv`. Optional `--song <name>` filter. Requires Basic-Pitch → **local-only**. |
| `fad_eval.py` | Fréchet Audio Distance on VGGish embeddings; All-FAD and Group-FAD (per-version). The script attempts to load pretrained VGGish and can fall back to a deterministic random VGGish-like embedder, so reports must state which embedder was used. Runs on Colab and locally. |
| `fad_visualize.py` | PCA scatter + 2σ Gaussian ellipses for real vs. generated embeddings, with FAD score overlay. |
| `latency_eval.py` | Real-Time Factor (RTF) measurement. |

## Batch inference + best-step selection (Israeli pipeline)

| File | What |
|---|---|
| `run_inference_batch.py` | Cross-product runner: iterates `(song × step × style × role)` from a `run_spec.yaml`, calls `inference.synthesize`, writes per-output `audio/`, `mels/`, `midi/` files under `versions/<v>/inference_runs/<run_id>/`, computes per-run FAD (and F1 when Basic-Pitch is available), appends rows to `inference_runs/_index.csv`. Output stems follow `{song}__step_{N}__style_{target}__role_{role}` so the index is greppable. |
| `select_best_step.py` | Reads `_index.csv` for one run, computes composite `z(FAD) + z(1-F1)`, promotes the top-K (default 3) finalists into `versions/<v>/step_selection/step_<N>/` (audio + mel preview + `metrics.json`) and writes a blind-listening `index.html`. Tiebreak: prefer the earlier step. |

See [../ENGINEERING_DECISIONS.md](../ENGINEERING_DECISIONS.md) §6 (vocoder choice), §17 (benchmark infra), §27 (metrics design), §32 (data architecture), §35 (best-step protocol).

## Metric status note

Historical benchmark/presentation artifacts may still show VAE-GAN/URMP POC numbers; those are labeled and are **not** diffusion results. **The Israeli diffusion metrics are now delivered** on the 3-version `Israeli_3style` model: All-FAD + per-version Group-FAD **2.20-2.49** (pretrained VGGish), note-level **F1 3-5%** (low by design — noisy Basic-Pitch reference + pitch-altering transfer; used as a comparative ddim/step tie-breaker, not an absolute bar), and latency **RTF 0.133 (ddim100) / 0.202 (ddim200)** on an L4 GPU. Hearing tests passed. See [../ENGINEERING_DECISIONS.md](../ENGINEERING_DECISIONS.md) §38.

## Known warnings

- **`scipy LinAlgWarning: Matrix is singular`** from `fad_eval.frechet_distance`
  — emitted by `scipy.linalg.sqrtm(Σ_real · Σ_gen)` when the covariance product
  is rank-deficient. Common with small evaluation sets (group-FAD with few clips
  per composer group, or self-FAD where real == generated). **Benign**: the code
  detects non-finite output and falls back to epsilon-regularised sqrtm
  (`Σ + 1e-6·I`). The warning is now suppressed at the call site so logs stay
  clean; the epsilon fallback still prints if it actually triggers.
