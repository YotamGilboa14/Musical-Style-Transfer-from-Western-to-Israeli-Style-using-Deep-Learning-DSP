"""
train.py — Training loop for the style-transfer diffusion model.

Usage:
    python train.py --config configs/default.yaml [options]

Key options:
    --ckpt_dir DIR         where to save checkpoints (default: runs/<timestamp>)
    --log_dir DIR          TensorBoard log directory (default: <ckpt_dir>/logs)
    --resume_from PATH     resume from checkpoint; use 'auto' to find latest in ckpt_dir
    --max_steps N          override total_steps from config
    --train_manifest PATH  override data.train_manifest from config
    --val_manifest PATH    override data.val_manifest from config
    --overfit-one-batch    quick sanity check: overfit a single batch (target loss < 0.05)
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model.unet import UNet1D
from model.diffusion import GaussianDiffusion
from data.dataset import MelPianoRollDataset
from preprocessing.augmentation import JointAugment


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Create a floating-point shadow copy of every model parameter."""
        self.decay = decay
        self.shadow = {k: v.clone().detach().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Move the shadow weights a little toward the latest model weights."""
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        """Load EMA weights into model (for sampling/eval)."""
        state = {k: v.to(model.state_dict()[k].dtype) for k, v in self.shadow.items()}
        model.load_state_dict(state, strict=True)

    def state_dict(self):
        """Return EMA weights in the same style as a PyTorch state_dict."""
        return self.shadow

    def load_state_dict(self, state):
        """Restore EMA weights from a checkpoint dictionary."""
        self.shadow = {k: v.clone().detach().float() for k, v in state.items()}


# ──────────────────────────────────────────────────────────────────────────────
# LR Schedule: warmup -> flat -> linear decay
# ──────────────────────────────────────────────────────────────────────────────

class WarmupFlatLinearScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Three-phase LR schedule:
      0            -> warmup_steps   linear ramp 0 -> lr_max
      warmup_steps -> decay_start    constant lr_max  (exploration)
      decay_start  -> total_steps    linear decay -> lr_min  (polish)

    decay_start = int(total_steps * decay_start_frac)

    Safe to extend: because the flat phase is constant lr_max, resuming with
    a larger total_steps causes no LR jump at the resume point -- the
    optimizer simply stays in the flat zone longer before polish fires.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, decay_start_frac,
                 lr_min, last_epoch=-1):
        """Store schedule boundaries before PyTorch initializes the scheduler."""
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_start_frac = decay_start_frac
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Return the learning rate for each optimizer parameter group."""
        step        = self.last_epoch
        decay_start = int(self.total_steps * self.decay_start_frac)
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * step / max(1, self.warmup_steps)
            elif step < decay_start:
                lr = base_lr                                       # flat
            else:
                t    = step - decay_start
                span = max(1, self.total_steps - decay_start)
                lr   = base_lr + (self.lr_min - base_lr) * min(1.0, t / span)
            lrs.append(max(lr, self.lr_min))
        return lrs


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load the YAML training config into a plain Python dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> tuple:
    """Construct UNet1D plus GaussianDiffusion from the config sections."""
    mc = cfg["model"]
    cc = cfg["conditioning"]
    dc = cfg["diffusion"]
    tc = cfg["training"]
    sc = cfg["sampling"]

    model = UNet1D(
        mel_channels=mc["mel_channels"],
        score_channels=mc["score_channels"],
        base_channels=mc["base_channels"],
        channel_mults=mc["channel_mults"],
        num_res_blocks_enc=mc["num_res_blocks_enc"],
        num_res_blocks_dec=mc["num_res_blocks_dec"],
        attention_levels=mc["attention_levels"],
        attn_heads=mc["attention_heads"],
        n_groups=mc["n_groups"],
        dropout=mc["dropout"],
        n_versions=cc["n_versions"],
        version_emb_dim=cc["version_emb_dim"],
        time_emb_dim=cc["time_emb_dim"],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        T=dc["T_train"],
        n_versions=cc["n_versions"],
        cfg_score=sc["cfg_score"],
        cfg_version=sc["cfg_version"],
        cfg_drop_score=tc["cfg_drop_score"],
        cfg_drop_version=tc["cfg_drop_version"],
        cfg_drop_both=tc["cfg_drop_both"],
    ).to(device)

    return model, diffusion


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _capture_rng_state() -> dict:
    """Capture random-number-generator state for reproducible resume."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: dict):
    """Restore RNG state so resumed training continues the same random stream."""
    rng = state["torch"]
    if not isinstance(rng, torch.Tensor):
        rng = torch.ByteTensor(rng)
    torch.set_rng_state(rng.cpu().to(torch.uint8))
    if state["cuda"] is not None and torch.cuda.is_available():
        cuda_states = [s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else torch.ByteTensor(s)
                       for s in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _find_latest_checkpoint(ckpt_dir: Path):
    """Return (step, path) of the highest step_*.pt in ckpt_dir, or (0, None)."""
    ckpts = sorted(
        [p for p in ckpt_dir.glob("step_*.pt") if p.stem.split("_")[1].isdigit()],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if ckpts:
        best = ckpts[-1]
        return int(best.stem.split("_")[1]), best
    return 0, None


def _prune_checkpoints(ckpt_dir: Path, keep_last: int = 3):
    """Delete old step_*.pt files, keeping the last keep_last, step_0.pt, and best_val.pt.

    A negative keep_last disables pruning entirely (keep all step checkpoints).
    """
    if keep_last < 0:
        return
    ckpts = sorted(
        [p for p in ckpt_dir.glob("step_*.pt") if p.stem.split("_")[1].isdigit()],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    protected = {"step_0.pt", "best_val.pt"}
    for p in ckpts[:-keep_last]:
        if p.name not in protected:
            p.unlink(missing_ok=True)


def save_checkpoint(ckpt_dir: Path, step: int, model, ema, optimizer, scheduler, cfg,
                    rng_state=None):
    """Write a full resume checkpoint and prune older step checkpoints."""
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg,
        "rng_state": rng_state,
    }
    path = ckpt_dir / f"step_{step}.pt"
    torch.save(ckpt, path)
    print(f"  [ckpt] Saved -> {path.name}")
    _prune_checkpoints(ckpt_dir, cfg["training"].get("checkpoint_keep_last", 3))
    return path


def generate_samples(diffusion, val_batch, ckpt_dir: Path, step: int, device, cfg):
    """Generate a few mel samples from the EMA model and save them as .pt files."""
    sc = cfg["sampling"]
    mel = val_batch["mel"][:4].to(device)
    score = val_batch["piano_roll"][:4].to(device)
    score_flat = score.view(score.shape[0], -1, score.shape[-1])
    ver = val_batch["version_id"][:4].to(device)

    sample_dir = ckpt_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        samples = diffusion.ddim_sample(
            score_flat, ver,
            N=sc["N_ddim"],
            cfg_score=sc["cfg_score"],
            cfg_version=sc["cfg_version"],
            overlap_frames=sc["overlap_frames"],
        )

    torch.save(samples.cpu(), sample_dir / f"step_{step}_mels.pt")
    torch.save(mel.cpu(), sample_dir / f"step_{step}_targets.pt")
    torch.save(score_flat.cpu(), sample_dir / f"step_{step}_scores.pt")
    print(f"  [sample] Saved {samples.shape[0]} mels -> samples/step_{step}_mels.pt")
    return samples.cpu(), mel.cpu()


def save_mel_png(gen_mel: torch.Tensor, tgt_mel: torch.Tensor, step: int, viz_dir: Path):
    """Save a 3-panel mel comparison PNG: Target | Generated | Difference.

    Mels are in the model's [-1, 1] normalised range.
    Saved as viz_dir/step_{step:06d}.png at DPI=100.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    tgt  = tgt_mel.float().numpy()   # [80, 430]
    gen  = gen_mel.float().numpy()   # [80, 430]
    diff = gen - tgt

    vmin, vmax = -1.0, 1.0
    diff_scale = max(abs(diff.min()), abs(diff.max())) + 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(15, 3))

    axes[0].imshow(tgt, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title("Target mel")
    axes[0].set_xlabel("time frame")
    axes[0].set_ylabel("mel bin")

    axes[1].imshow(gen, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Generated (step {step:,})")
    axes[1].set_xlabel("time frame")

    axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-diff_scale, vmax=diff_scale)
    axes[2].set_title("Difference (gen − tgt)")
    axes[2].set_xlabel("time frame")

    plt.suptitle(f"Mel snapshot — step {step:,}", fontsize=10)
    plt.tight_layout()

    out_path = viz_dir / f"step_{step:06d}.png"
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [mel_viz] Saved -> {out_path.name}")


def compute_val_loss(diffusion, val_loader, device, cfg, max_batches: int = 32) -> float:
    """Compute mean L1 diffusion loss over up to max_batches of the val set."""
    dc = cfg["diffusion"]
    diffusion.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            mel = batch["mel"].to(device)
            score = batch["piano_roll"].to(device)
            ver = batch["version_id"].to(device)
            B = mel.shape[0]
            score_flat = score.view(B, -1, mel.shape[-1])
            t = torch.randint(0, dc["T_train"], (B,), device=device)
            loss = diffusion.p_losses(mel, score_flat, ver, t)
            total += loss.item() * B
            count += B
    diffusion.train()
    return total / max(count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    """Run the full diffusion training loop from parsed CLI arguments.

    The loop owns model construction, data loading, optimizer/scheduler setup,
    checkpoint resume, optional one-batch overfit sanity testing, periodic
    validation, EMA sampling snapshots, TensorBoard logging, and final save.
    """
    cfg = load_config(args.config)
    tc = cfg["training"]
    dc = cfg["data"]

    # ── Device ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    use_amp = tc["precision"] == "bf16" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # ── Directories ──────────────────────────────────────────────────────
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = Path("runs") / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) if args.log_dir else ckpt_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoints: {ckpt_dir}")
    print(f"TensorBoard: {log_dir}")
    writer = SummaryWriter(log_dir=str(log_dir))

    # CSV loss log — readable without TensorBoard
    loss_csv_path = ckpt_dir / "loss_log.csv"
    _csv_is_new = not loss_csv_path.exists()
    _loss_csv = open(loss_csv_path, "a", newline="", encoding="utf-8")
    _csv_writer = csv.writer(_loss_csv)
    if _csv_is_new:
        _csv_writer.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm"])
        _loss_csv.flush()

    # ── max_steps ────────────────────────────────────────────────────────
    max_steps = args.max_steps if args.max_steps is not None else tc["total_steps"]
    print(f"Training for {max_steps} steps")

    # ── Model ────────────────────────────────────────────────────────────
    model, diffusion = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    ema = EMA(model, decay=tc["ema_decay"])

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = WarmupFlatLinearScheduler(
        optimizer,
        warmup_steps=tc["warmup_steps"],
        total_steps=max_steps,
        decay_start_frac=tc.get("decay_start_frac", 0.9),
        lr_min=tc["lr_min"],
    )
    scaler = GradScaler('cuda', enabled=use_amp)

    # ── Dataset ──────────────────────────────────────────────────────────
    # Manifests contain relative paths. On Colab, args.data_root can point at
    # the Drive dataset root; locally it can point at Google Drive for Desktop
    # or a temporary ingest folder without changing the CSV contents.
    project_root = Path(__file__).parent
    train_manifest_path = (
        Path(args.train_manifest) if args.train_manifest
        else project_root / dc["train_manifest"]
    )
    val_manifest_path = (
        Path(args.val_manifest) if args.val_manifest
        else project_root / dc["val_manifest"]
    )

    # ── Augmentation ─────────────────────────────────────────────────────
    aug_cfg = cfg.get("augmentation", {"enabled": False})
    train_augment = JointAugment(aug_cfg) if aug_cfg.get("enabled", False) else None
    if train_augment is not None:
        print("Augmentation: enabled")

    train_ds = MelPianoRollDataset(str(train_manifest_path), manifest_root=args.data_root, augment=train_augment)
    val_ds = MelPianoRollDataset(str(val_manifest_path), manifest_root=args.data_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=dc["num_workers"],
        pin_memory=dc["pin_memory"] and device.type == "cuda",
        persistent_workers=dc["persistent_workers"] and dc["num_workers"] > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ── Resume ───────────────────────────────────────────────────────────
    # Full step_*.pt checkpoints include optimizer, scheduler, EMA, and RNG
    # state. best_val.pt is smaller and should be treated as an evaluation
    # artifact unless a later code-fix phase changes that contract.
    start_step = 0
    resume_path = None
    if args.resume_from == "auto":
        _, resume_path = _find_latest_checkpoint(ckpt_dir)
        if resume_path is None:
            print("No checkpoint found in ckpt_dir — starting from scratch.")
        else:
            print(f"Auto-resume: found {resume_path.name}")
    elif args.resume_from:
        resume_path = Path(args.resume_from)

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        if ckpt.get("rng_state"):
            _restore_rng_state(ckpt["rng_state"])
        print(f"Resumed from step {start_step}")

    # ── Overfit-one-batch sanity check ────────────────────────────────────
    if args.overfit_one_batch:
        print("\n=== OVERFIT-ONE-BATCH SANITY CHECK ===")
        # The warmup scheduler is never stepped in this sanity loop, so the
        # optimizer would otherwise sit at lr_min (warmup step 0 -> lr≈0,
        # clamped to lr_min). Pin the LR to the configured peak so the single
        # batch can actually be memorised within the step budget.
        for g in optimizer.param_groups:
            g["lr"] = tc["lr"]
        print(f"Overfit LR pinned to {tc['lr']:.1e} (bypassing warmup schedule)")
        # Budget is configurable via --max_steps so a healthy model has room to
        # actually cross the target (CFG dropout + random-t make single-batch
        # memorisation slower than a clean overfit). Defaults to 1000.
        overfit_steps = args.max_steps if args.max_steps else 1000
        print(f"Overfit budget: {overfit_steps} steps")
        batch = next(iter(train_loader))
        mel = batch["mel"].to(device)
        score = batch["piano_roll"].to(device).view(mel.shape[0], -1, mel.shape[-1])
        ver = batch["version_id"].to(device)
        model.train()
        recent = deque(maxlen=50)
        for i in range(overfit_steps):
            t = torch.randint(0, cfg["diffusion"]["T_train"], (mel.shape[0],), device=device)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = diffusion.p_losses(mel, score, ver, t)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            recent.append(loss.item())
            if i % 100 == 0:
                print(f"  step {i:4d}  loss={loss.item():.4f}")
        # Per-step loss is noisy (random t each step), so judge convergence on a
        # trailing average rather than a single final sample.
        avg_recent = sum(recent) / len(recent)
        print(f"\nFinal loss (last {len(recent)}-step avg): {avg_recent:.4f}  "
              f"(target < 0.05)")
        if avg_recent > 0.05:
            print("WARNING: loss did not converge. Check model for bugs before full training.")
        else:
            print("PASS: ready for full training.")
        writer.close()
        _loss_csv.close()
        return

    # ── Val batch (fixed, for sample generation) ─────────────────────────
    val_batch = next(iter(val_loader))

    # ── Training ─────────────────────────────────────────────────────────
    train_iter = iter(train_loader)
    step = start_step
    best_val_loss = float("inf")

    model.train()
    t_start = time.time()

    while step < max_steps:
        # Infinite data iterator. Training is step-based rather than epoch-based,
        # so when DataLoader finishes one pass over the manifest we immediately
        # restart it and keep counting global optimization steps.
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        mel = batch["mel"].to(device)                              # [B, 80, 430]
        score = batch["piano_roll"].to(device)                     # [B, 2, 128, 430]
        ver = batch["version_id"].to(device)                       # [B]
        B = mel.shape[0]
        score_flat = score.view(B, -1, mel.shape[-1])              # [B, 256, 430]

        t_diff = torch.randint(0, cfg["diffusion"]["T_train"], (B,), device=device)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss = diffusion.p_losses(mel, score_flat, ver, t_diff)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        scheduler.step()

        # ── Train loss logging ───────────────────────────────────────────
        if step % tc["log_every"] == 0:
            elapsed = time.time() - t_start
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"step {step:7d} | loss {loss.item():.4f} | "
                f"grad {grad_norm:.3f} | lr {lr_now:.2e} | {elapsed:.0f}s"
            )
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr_now, step)
            writer.add_scalar("train/grad_norm", grad_norm, step)
            _csv_writer.writerow([step, f"{loss.item():.6f}", "", f"{lr_now:.2e}", f"{grad_norm:.4f}"])
            _loss_csv.flush()

        # ── Val loss + samples + checkpoint ─────────────────────────────
        if step % tc["save_every"] == 0 and step > 0:
            # Val loss
            val_loss = compute_val_loss(diffusion, val_loader, device, cfg)
            writer.add_scalar("val/loss", val_loss, step)
            print(f"  [val]  loss {val_loss:.4f}")
            _csv_writer.writerow([step, "", f"{val_loss:.6f}", "", ""])
            _loss_csv.flush()

            # Best val checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = ckpt_dir / "best_val.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "val_loss": val_loss,
                }, best_path)
                print(f"  [ckpt] New best val -> {best_path.name}")

            # Sample generation (EMA model). EMA weights usually sound smoother
            # for diffusion sampling than the raw just-updated weights. We
            # snapshot the raw weights, swap EMA in for sampling, then restore
            # the raw weights so training continues from the actual optimizer
            # state and the full-step checkpoint saved below contains raw (not
            # EMA) weights under `model` (EMA is saved separately via
            # `ema.state_dict()` inside `save_checkpoint`).
            model.eval()
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            gen_samples, tgt_mels = generate_samples(diffusion, val_batch, ckpt_dir, step, device, cfg)
            # Mel viz PNG every checkpoint (we are already inside the save_every block).
            save_mel_png(gen_samples[0], tgt_mels[0], step, ckpt_dir / "mel_viz")
            # Restore raw weights before continuing training / saving ckpt.
            model.load_state_dict(raw_state)
            del raw_state
            model.train()

            # Checkpoint
            save_checkpoint(ckpt_dir, step, model, ema, optimizer, scheduler, cfg,
                            rng_state=_capture_rng_state())

        step += 1

    # Final checkpoint + val loss
    val_loss = compute_val_loss(diffusion, val_loader, device, cfg)
    writer.add_scalar("val/loss", val_loss, step)
    _csv_writer.writerow([step, "", f"{val_loss:.6f}", "", ""])
    save_checkpoint(ckpt_dir, step, model, ema, optimizer, scheduler, cfg,
                    rng_state=_capture_rng_state())
    writer.close()
    _loss_csv.close()
    print(f"Training complete. Final val loss: {val_loss:.4f}")


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Build the training CLI parser and return parsed arguments."""
    p = argparse.ArgumentParser(description="Train diffusion style-transfer model")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt_dir", default=None,
                   help="Checkpoint directory (default: runs/<timestamp>)")
    p.add_argument("--log_dir", default=None,
                   help="TensorBoard log directory (default: <ckpt_dir>/logs)")
    p.add_argument("--resume_from", default=None,
                   help="Path to checkpoint, or 'auto' for latest in ckpt_dir")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override total_steps from config")
    p.add_argument("--train_manifest", default=None,
                   help="Override data.train_manifest from config")
    p.add_argument("--val_manifest", default=None,
                   help="Override data.val_manifest from config")
    p.add_argument("--data_root", default=None,
                   help="Root directory for resolving relative paths in the manifest "
                        "(default: directory containing the manifest CSV)")
    p.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="Run 1000-step single-batch overfit test (sanity check)",
    )
    # Deprecated aliases kept for backwards compatibility
    p.add_argument("--output-dir", default=None, help="[deprecated] Use --ckpt_dir")
    p.add_argument("--resume", default=None, help="[deprecated] Use --resume_from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Handle deprecated aliases
    if args.output_dir and not args.ckpt_dir:
        args.ckpt_dir = args.output_dir
    if args.resume and not args.resume_from:
        args.resume_from = args.resume
    train(args)
