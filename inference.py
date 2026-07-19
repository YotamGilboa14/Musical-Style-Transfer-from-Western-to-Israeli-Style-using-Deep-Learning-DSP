"""
inference.py — MIDI + version ID → WAV via DDIM sampling.

Usage:
    python inference.py \\
        --midi path/to/score.mid \\
        --version 3 \\
        --checkpoint runs/20260422_120000/checkpoint_latest.pt \\
        --output output.wav \\
        [--duration 30.0] [--cfg 1.25] [--ddim-steps 100]

The script:
  1. Converts MIDI → piano roll [2, 128, T_total] using the project preprocessor.
  2. Segments the piano roll into overlapping 5s chunks.
  3. Runs DDIM sampling with compound CFG (3 forward passes per step).
  4. Concatenates segments with overlap blending.
  5. Denormalises the mel (min/max stored in checkpoint or given as args).
  6. Vocodes with BigVGAN 22kHz.
  7. Saves the WAV.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import yaml

# ── Project path ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from model.unet import UNet1D
from model.diffusion import GaussianDiffusion
from preprocessing.dsp_preprocessor import load_midi_to_piano_roll, DSPConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Recreate the trained model/diffusion objects and load checkpoint weights.

    The checkpoint stores the config used during training, so inference does
    not need the user to manually re-enter channel sizes, number of versions,
    or CFG defaults. EMA weights are preferred because they are smoother for
    sampling than the raw training weights.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    mc = cfg["model"]
    cc = cfg["conditioning"]
    dc = cfg["diffusion"]
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
        dropout=0.0,  # no dropout at inference
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
    ).to(device)

    # Load EMA weights if present, otherwise raw model weights
    if "ema" in ckpt:
        state = {k: v.to(next(model.parameters()).dtype) for k, v in ckpt["ema"].items()}
        model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=True)

    model.eval()
    return model, diffusion, cfg


def segment_piano_roll(piano_roll: torch.Tensor, segment_frames: int, overlap_frames: int):
    """
    Chop a full-length piano roll into overlapping segments.

    Args:
        piano_roll:     [2, 128, T_total]
        segment_frames: T_seg (= 430)
        overlap_frames: number of frames to overlap between segments
    Returns:
        list of [2, 128, T_seg] tensors (padded with zeros if necessary)
    """
    stride = segment_frames - overlap_frames
    T = piano_roll.shape[-1]
    segments = []
    start = 0
    while start < T:
        end = start + segment_frames
        chunk = piano_roll[..., start:end]
        if chunk.shape[-1] < segment_frames:
            # Zero-pad the last segment
            pad = segment_frames - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        segments.append(chunk)
        if end >= T:
            break
        start += stride
    return segments


def denormalize_mel(mel_norm: torch.Tensor, mel_min: float = -80.0, mel_max: float = 0.0) -> torch.Tensor:
    """
    Reverse the [-1, 1] normalization applied during preprocessing.
    mel_norm ∈ [-1, 1]  →  mel_db ∈ [mel_min, mel_max]
    """
    return (mel_norm + 1.0) / 2.0 * (mel_max - mel_min) + mel_min


# ──────────────────────────────────────────────────────────────────────────────
# Main synthesis function
# ──────────────────────────────────────────────────────────────────────────────

def synthesize(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    cfg: dict,
    midi_path: str,
    version_id: int,
    duration_s: float = 30.0,
    cfg_score: float = 1.25,
    cfg_version: float = 1.25,
    n_ddim_steps: int = 100,
    mel_min: float = -80.0,
    mel_max: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Full MIDI → WAV pipeline.

    Returns:
        audio: float32 numpy array at 22050 Hz
    """
    sc_cfg = cfg["sampling"]
    segment_frames = cfg["data"]["segment_frames"]
    overlap_frames = sc_cfg["overlap_frames"]

    # ── 1. MIDI → piano roll ──────────────────────────────────────────────
    # Inference starts from symbolic score, not source audio. The score says
    # what notes to play; version_id says which learned style to render them in.
    dsp_config = DSPConfig()
    piano_roll = load_midi_to_piano_roll(midi_path, dsp_config, duration_s)  # [2, 128, T_total]
    piano_roll_t = torch.from_numpy(piano_roll).float()

    # ── 2. Segment ───────────────────────────────────────────────────────
    # The U-Net was trained on fixed 5-second windows, so longer MIDI files are
    # split into overlapping windows and stitched back together after sampling.
    segments = segment_piano_roll(piano_roll_t, segment_frames, overlap_frames)
    n_seg = len(segments)

    score_batch = torch.stack(segments, dim=0).to(device)          # [N_seg, 2, 128, T_seg]
    score_flat = score_batch.view(n_seg, -1, segment_frames)        # [N_seg, 256, T_seg]
    ver_batch = torch.full((n_seg,), version_id, dtype=torch.long, device=device)

    # ── 3. DDIM sampling ─────────────────────────────────────────────────
    # Each segment is generated from noise using the compound CFG sampler in
    # model/diffusion.py: unconditional + score direction + version direction.
    with torch.no_grad():
        mels = diffusion.ddim_sample(
            score_flat,
            ver_batch,
            N=n_ddim_steps,
            cfg_score=cfg_score,
            cfg_version=cfg_version,
            overlap_frames=overlap_frames,
        )  # [N_seg, 80, T_seg]

    # ── 4. Concatenate with overlap ───────────────────────────────────────
    full_mel = GaussianDiffusion.concat_with_overlap(mels, overlap_frames)  # [80, T_total]

    # ── 5. Denormalise ────────────────────────────────────────────────────
    # The network outputs the training range [-1, 1]. First map back to the dB
    # scale used in DSP, then convert dB → BigVGAN's log-magnitude convention.
    # BigVGAN's mel_to_audio() feeds the tensor straight to the generator and
    # expects log-magnitude mels (natural log of amplitude), NOT dB. Skipping
    # this conversion makes the mel ~8.7x too large in magnitude and produces
    # broken "water-bubbling" audio. This matches the working training-notebook
    # demo path (train_israeli.ipynb Cell 7): mel_db * (ln(10) / 20).
    full_mel_db = denormalize_mel(full_mel, mel_min, mel_max)
    full_mel_logmag = full_mel_db * (np.log(10.0) / 20.0)

    # ── 6. Vocode ─────────────────────────────────────────────────────────
    from postprocessing.vocoder_factory import create_vocoder
    vocoder = create_vocoder("bigvgan_22k")
    audio = vocoder.mel_to_audio(full_mel_logmag.cpu())  # numpy float32

    return audio


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Build the inference CLI parser and return parsed arguments."""
    p = argparse.ArgumentParser(description="Diffusion style transfer: MIDI + version → WAV")
    p.add_argument("--midi", required=True, help="Path to input MIDI file")
    p.add_argument("--version", type=int, required=True, help="Target version ID (integer)")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--output", required=True, help="Output WAV path")
    p.add_argument("--duration", type=float, default=30.0, help="Duration in seconds (default 30)")
    p.add_argument("--cfg", type=float, default=1.25, help="CFG guidance weight (default 1.25, same for score+version)")
    p.add_argument("--cfg-score", type=float, default=None, help="Override score guidance weight")
    p.add_argument("--cfg-version", type=float, default=None, help="Override version guidance weight")
    p.add_argument("--ddim-steps", type=int, default=100, help="Number of DDIM sampling steps")
    p.add_argument("--mel-min", type=float, default=-80.0, help="Mel normalization min (dB)")
    p.add_argument("--mel-max", type=float, default=0.0, help="Mel normalization max (dB)")
    p.add_argument("--device", default=None, help="Device: cuda / cpu (auto-detected if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, diffusion, cfg = load_checkpoint(args.checkpoint, device)

    print(f"Synthesising {args.duration}s | version={args.version} | DDIM steps={args.ddim_steps}")
    audio = synthesize(
        model, diffusion, cfg,
        midi_path=args.midi,
        version_id=args.version,
        duration_s=args.duration,
        cfg_score=args.cfg_score or args.cfg,
        cfg_version=args.cfg_version or args.cfg,
        n_ddim_steps=args.ddim_steps,
        mel_min=args.mel_min,
        mel_max=args.mel_max,
        device=device,
    )

    sf.write(args.output, audio, samplerate=22050)
    print(f"Saved → {args.output}  ({len(audio)/22050:.1f}s)")
