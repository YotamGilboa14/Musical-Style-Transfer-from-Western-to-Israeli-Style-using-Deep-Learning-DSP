#!/usr/bin/env python3
"""
End-to-End POC: Timbre Transfer Pipeline (Trumpet → Violin)
=============================================================

.. note::
   **DEPRECATED FOR ACTIVE TRAINING — KEPT FOR DEFENSE REFERENCE.**

   This script demonstrates the **VAE-GAN POC** (Phase 4B). It is **not** part
   of the active Israeli single-style diffusion pipeline. We keep it because:

     1. It is the canonical record of the trumpet → violin POC referenced in
        ``POC_EXPERIMENT.md`` and the defense materials.
     2. ``run_inference_benchmark.py`` reuses the same imports for latency
        benchmarking against the diffusion model.

   For the active pipeline, see ``train.py`` and ``inference.py``.

Demonstrates the full pipeline:
  WAV → mel spectrogram → VAE-GAN (trumpet→violin) → vocoder → playable WAV

Runs two mel computation paths and three vocoding strategies to compare:

  Path A — Model mel:    utils.melspectrogram(wav) → [0,1] directly
  Path B — Pipeline mel: dsp_preprocessor → dB → normalize_mel → [-1,1] → convert → [0,1]

  Vocoding strategies:
    - Griffin-Lim:       mel [0,1] → phase estimation → WAV (baseline)
    - Direct BigVGAN:    mel [0,1] → dB → linear → log → BigVGAN → WAV
    - Double BigVGAN:    mel [0,1] → Griffin-Lim → WAV → BigVGAN native mel → BigVGAN → WAV

Output (7 files in poc_output/):
    original.wav                    Source trumpet
    A_griffinlim.wav                Model mel → Griffin-Lim
    A_direct_bigvgan.wav            Model mel → direct BigVGAN
    A_double_bigvgan.wav            Model mel → Griffin-Lim → BigVGAN
    B_griffinlim.wav                Pipeline mel → Griffin-Lim
    B_direct_bigvgan.wav            Pipeline mel → direct BigVGAN
    B_double_bigvgan.wav            Pipeline mel → Griffin-Lim → BigVGAN

Usage:
    # Activate ml_env first
    python run_poc_transfer.py

    # Or specify a different input WAV:
    python run_poc_transfer.py --input path/to/trumpet.wav

    # Skip BigVGAN (Griffin-Lim only, faster):
    python run_poc_transfer.py --no-bigvgan

Requirements:
    - Trained weights in models/tt_vae_gan/saved_models/pipeline_urmp/
      (encoder_500.pth, G2_500.pth)
    - Set TT_VAE_GAN_USE_PIPELINE=1 env var (auto-set by this script)

Author: Yotam & Gal — StyleTransfer Music Project
Date: February 2026
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Force pipeline params before any model imports
os.environ['TT_VAE_GAN_USE_PIPELINE'] = '1'

import torch
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

# ─── Project imports ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.tt_vae_gan.pipeline_inference import (
    load_model, infer_mel, pipeline_to_model, model_to_pipeline,
    reconstruct_with_griffinlim, model_to_db, db_to_linear,
)
from models.tt_vae_gan.utils import (
    melspectrogram as model_melspectrogram,
    preprocess_wav, sample_rate as MODEL_SR,
)
from preprocessing.dsp_preprocessor import (
    DSPConfig, load_and_resample_audio, extract_mel_spectrogram, normalize_mel,
)


# ─── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT = (
    PROJECT_ROOT / "models" / "tt_vae_gan" / "data" / "data_urmp"
    / "spkr_1" / "AuSep_1_tpt_05_Entertainer.wav"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "poc_output"
MODEL_NAME = "pipeline_urmp"
EPOCH = 500
TRG_ID = "2"          # G2 = violin
IMG_HEIGHT = 80        # pipeline mel bins
SAMPLE_RATE = 22050


def compute_model_mel(wav_path: str) -> np.ndarray:
    """Path A: Compute mel using the VAE-GAN model's own mel pipeline.
    
    Chain: WAV → preprocess_wav() → melspectrogram() → [0, 1]
    This is the exact same path used during training and Colab inference.
    """
    print("  [Path A] Computing mel via model's melspectrogram()...")
    wav = preprocess_wav(wav_path)
    mel_01 = model_melspectrogram(wav)
    print(f"  Mel shape: {mel_01.shape}, range: [{mel_01.min():.4f}, {mel_01.max():.4f}]")
    return mel_01


def compute_pipeline_mel(wav_path: str) -> np.ndarray:
    """Path B: Compute mel using our DSP preprocessing pipeline.
    
    Chain: WAV → load_and_resample → extract_mel_spectrogram (librosa power_to_db)
           → normalize_mel → [-1, 1] → pipeline_to_model → [0, 1]
    This is the path real songs go through in process_song_offline.py.
    """
    print("  [Path B] Computing mel via pipeline's dsp_preprocessor...")
    config = DSPConfig()
    y = load_and_resample_audio(Path(wav_path), target_sr=config.sample_rate)
    mel_db = extract_mel_spectrogram(y, config)
    mel_pipeline, mel_min, mel_max = normalize_mel(mel_db)
    mel_01 = pipeline_to_model(mel_pipeline)
    print(f"  Mel shape: {mel_01.shape}, range: [{mel_01.min():.4f}, {mel_01.max():.4f}]")
    print(f"  dB range: [{mel_min:.1f}, {mel_max:.1f}]")
    return mel_01


def vocode_griffinlim(mel_01: np.ndarray) -> np.ndarray:
    """Griffin-Lim phase reconstruction (baseline, no neural vocoder)."""
    return reconstruct_with_griffinlim(mel_01, n_iter=64)


def vocode_direct_bigvgan(mel_01: np.ndarray, vocoder) -> np.ndarray:
    """Direct conversion: mel [0,1] → dB → linear → log → BigVGAN → WAV.
    
    Converts through dB scale to BigVGAN's native log-magnitude format.
    """
    S_db = model_to_db(mel_01)
    S_linear = db_to_linear(S_db)
    log_mel = np.log(np.maximum(S_linear, 1e-5))
    mel_tensor = torch.from_numpy(log_mel).float().unsqueeze(0)
    audio = vocoder.mel_to_audio(mel_tensor)
    return audio


def vocode_double_bigvgan(mel_01: np.ndarray, vocoder) -> np.ndarray:
    """Double conversion: mel [0,1] → Griffin-Lim → WAV → BigVGAN native mel → BigVGAN → WAV.
    
    Proven approach from Colab. Griffin-Lim provides a rough waveform,
    then BigVGAN re-analyzes and re-synthesizes with neural quality.
    """
    # Step 1: Griffin-Lim to get a rough waveform
    wav_gl = vocode_griffinlim(mel_01)
    
    # Step 2: Re-analyze with BigVGAN's native mel computation
    wav_tensor = torch.from_numpy(wav_gl).float().unsqueeze(0)
    if vocoder.device.type == 'cuda':
        wav_tensor = wav_tensor.cuda()
    native_mel = vocoder._get_mel_spectrogram(wav_tensor, vocoder.h)
    
    # Step 3: BigVGAN synthesis
    audio = vocoder.mel_to_audio(native_mel)
    return audio


def save_wav(audio: np.ndarray, path: Path, sr: int = SAMPLE_RATE):
    """Save audio to WAV file, clipping to [-1, 1]."""
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sr)
    duration = len(audio) / sr
    size_kb = path.stat().st_size / 1024
    print(f"    Saved: {path.name} ({duration:.1f}s, {size_kb:.0f} KB)")


def plot_mel_comparison(mel_A_src: np.ndarray, mel_A_trg: np.ndarray,
                        mel_B_src: np.ndarray, mel_B_trg: np.ndarray,
                        output_dir: Path, sr: int = SAMPLE_RATE,
                        hop_length: int = 256):
    """Plot mel spectrogram comparison grid: source vs transferred for both paths.
    
    Generates two figures:
      1. mel_comparison.png — 2x2 grid (Path A src/trg, Path B src/trg)
      2. mel_difference.png — difference maps showing what the model changed
    """
    # Time axis in seconds
    n_frames = mel_A_src.shape[1]
    duration = n_frames * hop_length / sr

    # ─── Figure 1: 2x2 comparison grid ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), sharex=True, sharey=True)
    fig.suptitle('POC Mel Spectrogram Comparison: Trumpet -> Violin',
                 fontsize=14, fontweight='bold')

    titles = [
        ('Path A: Source (Model Mel)', mel_A_src),
        ('Path A: Transferred (Violin)', mel_A_trg),
        ('Path B: Source (Pipeline Mel)', mel_B_src),
        ('Path B: Transferred (Violin)', mel_B_trg),
    ]
    for ax, (title, mel) in zip(axes.flat, titles):
        im = ax.imshow(mel, aspect='auto', origin='lower', cmap='magma',
                       extent=[0, duration, 0, mel.shape[0]])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Mel Bin')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 1].set_xlabel('Time (s)')

    # Shared colorbar — placed well outside the plot area
    fig.subplots_adjust(hspace=0.3, wspace=0.15, right=0.82)
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02,
                 pad=0.08, label='Normalized [0, 1]')
    path_cmp = output_dir / 'mel_comparison.png'
    fig.savefig(str(path_cmp), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path_cmp.name}")

    # ─── Figure 2: Difference maps ─────────────────────────────────────
    diff_A = mel_A_trg - mel_A_src
    diff_B = mel_B_trg - mel_B_src
    vmax = max(np.abs(diff_A).max(), np.abs(diff_B).max())

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 5), sharex=True, sharey=True)
    fig2.suptitle('Mel Difference Maps (Transferred - Source)',
                  fontsize=14, fontweight='bold')

    for ax, (title, diff) in zip(axes2, [
        ('Path A: Model Mel Difference', diff_A),
        ('Path B: Pipeline Mel Difference', diff_B),
    ]):
        im2 = ax.imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax,
                        extent=[0, duration, 0, diff.shape[0]])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mel Bin')

    fig2.subplots_adjust(wspace=0.15, right=0.82)
    fig2.colorbar(im2, ax=axes2, orientation='vertical', fraction=0.02,
                  pad=0.08, label='Delta (blue=reduced, red=boosted)')
    path_diff = output_dir / 'mel_difference.png'
    fig2.savefig(str(path_diff), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"    Saved: {path_diff.name}")

    # ─── Figure 3: Zoomed comparison (first 5 seconds) ─────────────────
    zoom_frames = min(int(5.0 * sr / hop_length), n_frames)
    fig3, axes3 = plt.subplots(2, 2, figsize=(18, 9), sharex=True, sharey=True)
    fig3.suptitle('Zoomed: First 5 Seconds', fontsize=14, fontweight='bold')

    zoom_data = [
        ('A Source', mel_A_src[:, :zoom_frames]),
        ('A Transferred', mel_A_trg[:, :zoom_frames]),
        ('B Source', mel_B_src[:, :zoom_frames]),
        ('B Transferred', mel_B_trg[:, :zoom_frames]),
    ]
    zoom_dur = zoom_frames * hop_length / sr
    for ax, (title, mel) in zip(axes3.flat, zoom_data):
        im3 = ax.imshow(mel, aspect='auto', origin='lower', cmap='magma',
                        extent=[0, zoom_dur, 0, mel.shape[0]])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Mel Bin')
    axes3[1, 0].set_xlabel('Time (s)')
    axes3[1, 1].set_xlabel('Time (s)')

    fig3.subplots_adjust(hspace=0.3, wspace=0.15, right=0.82)
    fig3.colorbar(im3, ax=axes3, orientation='vertical', fraction=0.02,
                  pad=0.08, label='Normalized [0, 1]')
    path_zoom = output_dir / 'mel_zoomed_5s.png'
    fig3.savefig(str(path_zoom), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"    Saved: {path_zoom.name}")

    # Print stats
    print()
    print("  Mel Statistics:")
    for name, mel in [('A src', mel_A_src), ('A trg', mel_A_trg),
                      ('B src', mel_B_src), ('B trg', mel_B_trg)]:
        print(f"    {name:6s}: mean={mel.mean():.4f}, std={mel.std():.4f}, "
              f"min={mel.min():.4f}, max={mel.max():.4f}")
    print(f"    A diff: mean={diff_A.mean():.4f}, std={diff_A.std():.4f}")
    print(f"    B diff: mean={diff_B.mean():.4f}, std={diff_B.std():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End POC: Trumpet → Violin Timbre Transfer")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Input trumpet WAV file")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for WAV results")
    parser.add_argument("--no-bigvgan", action="store_true",
                        help="Skip BigVGAN (Griffin-Lim only, much faster)")
    parser.add_argument("--epoch", type=int, default=EPOCH,
                        help="Checkpoint epoch to load")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    print("=" * 70)
    print("  END-TO-END POC: Trumpet -> Violin Timbre Transfer")
    print("=" * 70)
    print(f"  Input:      {input_path.name}")
    print(f"  Output dir: {output_dir}")
    print(f"  Model:      {MODEL_NAME} (epoch {args.epoch}, G{TRG_ID}=violin)")
    print(f"  BigVGAN:    {'DISABLED' if args.no_bigvgan else 'ENABLED'}")
    print()

    # ─── Verify input exists ────────────────────────────────────────────
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Download a trumpet WAV from URMP or specify --input")
        sys.exit(1)

    # ─── Verify checkpoint exists ───────────────────────────────────────
    ckpt_dir = PROJECT_ROOT / "models" / "tt_vae_gan" / "saved_models" / MODEL_NAME
    encoder_path = ckpt_dir / f"encoder_{args.epoch}.pth"
    gen_path = ckpt_dir / f"G{TRG_ID}_{args.epoch}.pth"
    for p, label in [(encoder_path, "Encoder"), (gen_path, "Generator G2")]:
        if not p.exists():
            print(f"ERROR: {label} checkpoint not found: {p}")
            print(f"Download from Google Drive -> MusicProject_Colab/tt_vae_gan/"
                  f"checkpoints/pipeline_urmp/")
            print(f"Place in: {ckpt_dir}/")
            sys.exit(1)

    # ─── Step 0: Save original (resampled) ──────────────────────────────
    print("[Step 0] Loading original audio...")
    original_wav, _ = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)
    save_wav(original_wav, output_dir / "original.wav")
    print()

    # ─── Step 1: Load VAE-GAN model ─────────────────────────────────────
    print("[Step 1] Loading VAE-GAN model...")
    t0 = time.time()
    model_dict = load_model(
        model_name=MODEL_NAME,
        epoch=args.epoch,
        trg_id=TRG_ID,
        img_height=IMG_HEIGHT,
        img_width=128,
        dim=32,
    )
    print(f"  Model loaded on {model_dict['device']} ({time.time()-t0:.1f}s)")
    print()

    # ─── Step 2: Compute mels (both paths) ──────────────────────────────
    print("[Step 2] Computing mel spectrograms...")
    mel_A = compute_model_mel(str(input_path))
    mel_B = compute_pipeline_mel(str(input_path))
    print()

    # ─── Step 3: Run inference (both paths) ─────────────────────────────
    print("[Step 3] Running VAE-GAN inference (trumpet -> violin)...")
    
    print("  [Path A] Model mel inference...")
    t0 = time.time()
    transferred_A = infer_mel(mel_A, model_dict, n_overlap=4)
    print(f"  Done ({time.time()-t0:.1f}s). Output range: "
          f"[{transferred_A.min():.4f}, {transferred_A.max():.4f}]")

    print("  [Path B] Pipeline mel inference...")
    t0 = time.time()
    transferred_B = infer_mel(mel_B, model_dict, n_overlap=4)
    print(f"  Done ({time.time()-t0:.1f}s). Output range: "
          f"[{transferred_B.min():.4f}, {transferred_B.max():.4f}]")
    print()

    # ─── Step 3b: Plot mel spectrograms ─────────────────────────────────
    print("[Step 3b] Plotting mel spectrogram comparisons...")
    plot_mel_comparison(mel_A, transferred_A, mel_B, transferred_B, output_dir)
    print()

    # ─── Step 4: Vocoding — Griffin-Lim (both paths) ────────────────────
    print("[Step 4] Vocoding with Griffin-Lim...")
    
    print("  [A] Griffin-Lim...")
    wav_A_gl = vocode_griffinlim(transferred_A)
    save_wav(wav_A_gl, output_dir / "A_griffinlim.wav")

    print("  [B] Griffin-Lim...")
    wav_B_gl = vocode_griffinlim(transferred_B)
    save_wav(wav_B_gl, output_dir / "B_griffinlim.wav")
    print()

    # ─── Step 5: Vocoding — BigVGAN (both paths, both strategies) ───────
    if not args.no_bigvgan:
        print("[Step 5] Loading BigVGAN vocoder...")
        t0 = time.time()
        from postprocessing.bigvgan_vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(model_name="bigvgan_22k")
        print(f"  Loaded ({time.time()-t0:.1f}s)")
        print()

        # Direct BigVGAN
        print("[Step 5a] Direct BigVGAN vocoding...")
        print("  [A] Direct BigVGAN...")
        wav_A_direct = vocode_direct_bigvgan(transferred_A, vocoder)
        save_wav(wav_A_direct, output_dir / "A_direct_bigvgan.wav")

        print("  [B] Direct BigVGAN...")
        wav_B_direct = vocode_direct_bigvgan(transferred_B, vocoder)
        save_wav(wav_B_direct, output_dir / "B_direct_bigvgan.wav")
        print()

        # Double-conversion BigVGAN
        print("[Step 5b] Double-conversion BigVGAN vocoding...")
        print("  [A] Double BigVGAN (Griffin-Lim → BigVGAN)...")
        wav_A_double = vocode_double_bigvgan(transferred_A, vocoder)
        save_wav(wav_A_double, output_dir / "A_double_bigvgan.wav")

        print("  [B] Double BigVGAN (Griffin-Lim → BigVGAN)...")
        wav_B_double = vocode_double_bigvgan(transferred_B, vocoder)
        save_wav(wav_B_double, output_dir / "B_double_bigvgan.wav")
        print()

    # ─── Summary ────────────────────────────────────────────────────────
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"  Input: {input_path.name}")
    print(f"  Mel A (model):    shape {mel_A.shape}")
    print(f"  Mel B (pipeline): shape {mel_B.shape}")
    print()
    print(f"  Output files in {output_dir}/:")
    for f in sorted(output_dir.glob("*.wav")):
        size_kb = f.stat().st_size / 1024
        audio, _ = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
        print(f"    {f.name:30s}  {len(audio)/SAMPLE_RATE:5.1f}s  {size_kb:6.0f} KB")
    print()
    print("  Comparison guide:")
    print("    A vs B:          Model mel vs Pipeline mel (same model, different preprocessing)")
    print("    griffinlim:      Baseline (no neural vocoder, metallic quality)")
    print("    direct_bigvgan:  Mel -> dB -> linear -> log -> BigVGAN (theoretical best)")
    print("    double_bigvgan:  Mel -> Griffin-Lim -> BigVGAN (proven on Colab)")
    print()
    print("  Listen to all files and compare quality!")
    print("=" * 70)


if __name__ == "__main__":
    main()
