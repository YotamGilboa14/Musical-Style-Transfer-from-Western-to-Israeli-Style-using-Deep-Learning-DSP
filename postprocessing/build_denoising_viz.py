"""Generate diffusion denoising-progression strips for the project showcase.

For each training campaign this loads the final checkpoint, conditions the
model on one real 5-second piano-roll segment, runs DDIM sampling while
capturing the running clean-mel estimate at every step
(``ddim_sample(return_intermediates=True)``), and saves a horizontal strip of
mel spectrograms from pure noise to the final generated mel.

Output: ``<training.denoising_dir>/<campaign_id>_denoising.png`` (local), which
``build_showcase.py`` then copies into ``assets/02_training/<campaign_id>/``.

Run (from ``MusicProject/``)::

    .\\ml_env\\Scripts\\python.exe .\\postprocessing\\build_denoising_viz.py \\
        --config .\\deliverables\\showcase_config.yaml

This is separate from ``build_showcase.py`` because it is the only step that
runs the model (a few seconds of GPU/CPU compute per campaign).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference import load_checkpoint, segment_piano_roll, denormalize_mel  # noqa: E402
from preprocessing.dsp_preprocessor import load_midi_to_piano_roll, DSPConfig  # noqa: E402

MEL_MIN, MEL_MAX = -80.0, 0.0
SEGMENT_SECONDS = 5.0


def _build_score(midi_path: Path, cfg: dict, device: torch.device) -> torch.Tensor:
    """First 5-second piano-roll segment as a [1, 256, T_seg] score tensor."""
    seg_frames = cfg["data"]["segment_frames"]
    overlap = cfg["sampling"]["overlap_frames"]
    pr = load_midi_to_piano_roll(str(midi_path), DSPConfig(), SEGMENT_SECONDS)
    prt = torch.from_numpy(pr).float()
    segs = segment_piano_roll(prt, seg_frames, overlap)
    return segs[0].view(1, -1, seg_frames).to(device)


def _plot_strip(intermediates: List[torch.Tensor], panels: int,
                title: str, out_png: Path) -> None:
    """Render ``panels`` mel snapshots (noise -> clean) as a horizontal strip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(intermediates)
    idx = np.linspace(0, n - 1, panels).round().astype(int)
    fig, axes = plt.subplots(1, panels, figsize=(2.05 * panels, 2.6))
    if panels == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        mel_db = denormalize_mel(intermediates[i][0], MEL_MIN, MEL_MAX).cpu().numpy()
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
                  vmin=MEL_MIN, vmax=MEL_MAX)
        pct = int(round(100 * i / max(1, n - 1)))
        ax.set_title(f"step {i}/{n - 1}\n({pct}% denoised)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("mel bins", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate denoising-progression strips.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--device", default=None,
                    help="cpu / cuda (default: cuda if available).")
    args = ap.parse_args(argv)

    cfg_all = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    tcfg = cfg_all.get("training", {})
    dcfg = tcfg.get("denoising", {})
    if not dcfg:
        print("No training.denoising block in config; nothing to do.")
        return 0

    out_dir = _ROOT / tcfg.get("denoising_dir", "deliverables/_denoising")
    midi = Path(dcfg["conditioning_midi"])
    n_steps = int(dcfg.get("n_ddim_steps", 100))
    panels = int(dcfg.get("panels", 8))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  steps={n_steps}  panels={panels}")

    ok = 0
    for camp in dcfg.get("campaigns", []):
        cid = camp["id"]
        ckpt = Path(camp["checkpoint"])
        want_ver = int(camp.get("version_id", 0))
        if not ckpt.exists():
            print(f"[{cid}] checkpoint missing: {ckpt}")
            continue
        try:
            print(f"[{cid}] loading {ckpt.name} ...")
            model, diffusion, cfg = load_checkpoint(str(ckpt), device)
            n_versions = cfg["conditioning"]["n_versions"]
            ver = max(0, min(want_ver, n_versions - 1))
            score = _build_score(midi, cfg, device)
            ver_t = torch.full((1,), ver, dtype=torch.long, device=device)
            print(f"[{cid}] sampling (version_id={ver}/{n_versions - 1}) ...")
            with torch.no_grad():
                _, inter = diffusion.ddim_sample(
                    score, ver_t, N=n_steps, return_intermediates=True)
            title = (f"{cid}: reverse-diffusion denoising  "
                     f"(version {ver}, {n_steps} DDIM steps)")
            out_png = out_dir / f"{cid}_denoising.png"
            _plot_strip(inter, panels, title, out_png)
            print(f"[{cid}] wrote {out_png}")
            ok += 1
        except Exception as e:
            print(f"[{cid}] FAILED: {e}")

    print(f"Done. {ok}/{len(dcfg.get('campaigns', []))} strips generated -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
