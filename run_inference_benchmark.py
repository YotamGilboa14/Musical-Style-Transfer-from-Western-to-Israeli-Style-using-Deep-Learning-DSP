#!/usr/bin/env python3
"""
Inference Benchmark: Latency Evaluation on Test-Split Trumpet Files
=====================================================================

Runs the full pipeline (pipeline mel + VAE-GAN + BigVGAN vocoder) on
URMP test-split trumpet recordings and measures per-patch and per-segment
latency. Produces:
  - Per-file latency reports (stdout + JSON)
  - Latency visualization plots (PNG)
  - Transferred audio via selected vocoder (default: BigVGAN 22kHz)

This uses our pipeline mel (dsp_preprocessor) — NOT the model's internal
mel — because from this point on all evaluation is done through our
pipeline, as established after the POC validation (see ENGINEERING_DECISIONS.md §15).

Test-split trumpet files (determined by refs_test.pickle):
  - AuSep_2_tpt_15_Surprise.wav
  - AuSep_2_tpt_31_Slavonic.wav
  - AuSep_1_tpt_33_Elise.wav

Usage:
    python run_inference_benchmark.py                         # all test files, BigVGAN
    python run_inference_benchmark.py --file AuSep_2_tpt_15_Surprise.wav
    python run_inference_benchmark.py --vocoder griffinlim    # lightweight, no GPU
    python run_inference_benchmark.py --vocoder hifigan       # HiFi-GAN baseline
    python run_inference_benchmark.py --output_dir bench_output

Author: Yotam & Gal — StyleTransfer Music Project
Date: February 2026
"""

import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
from pathlib import Path

# Force pipeline params
os.environ['TT_VAE_GAN_USE_PIPELINE'] = '1'

import torch
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.tt_vae_gan.pipeline_inference import (
    load_model, infer_mel, pipeline_to_model, model_to_pipeline,
    reconstruct_with_griffinlim, model_to_db, db_to_linear,
)
from preprocessing.dsp_preprocessor import (
    DSPConfig, extract_mel_spectrogram, normalize_mel,
    load_and_resample_audio,
)
from postprocessing.latency_eval import (
    evaluate_latency, plot_latency, print_latency_report,
)


# --- Constants ---------------------------------------------------------------
URMP_DIR = PROJECT_ROOT / "models" / "tt_vae_gan" / "data" / "data_urmp"
SPKR_1_DIR = URMP_DIR / "spkr_1"  # trumpet
REFS_TEST_PATH = URMP_DIR / "refs_test.pickle"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark_output"
MODEL_NAME = "pipeline_urmp"
EPOCH = 500
TRG_ID = "2"          # G2 = violin
IMG_HEIGHT = 80
SAMPLE_RATE = 22050

# Pipeline DSP config (same params used throughout the project)
DSP_CFG = DSPConfig()

VOCODER_CHOICES = ['bigvgan', 'hifigan', 'griffinlim']


def get_test_trumpet_files() -> list:
    """Load test-split trumpet filenames from refs_test.pickle."""
    if not REFS_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test refs not found: {REFS_TEST_PATH}\n"
            "Run data_prep_urmp.py first to create train/eval/test splits.")
    
    with open(REFS_TEST_PATH, 'rb') as f:
        refs = pickle.load(f)
    
    # Speaker 0 = trumpet
    trumpet_files = refs.get(0, [])
    return sorted(trumpet_files)


def run_benchmark_on_file(wav_path: Path, model_dict: dict,
                          output_dir: Path, vocoder=None,
                          vocoder_name: str = 'bigvgan') -> dict:
    """
    Run inference benchmark on a single trumpet WAV file using pipeline mel.
    
    Flow: WAV -> pipeline mel [-1,1] -> [0,1] -> VAE-GAN -> [0,1] -> vocoder -> WAV
    
    Returns:
        dict with benchmark results including timing report
    """
    file_name = wav_path.stem
    file_output_dir = output_dir / file_name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'-' * 60}")
    print(f"  Benchmarking: {wav_path.name}")
    print(f"{'-' * 60}")
    
    # Load audio using pipeline preprocessor
    print("  Computing mel spectrogram (pipeline path)...")
    wav = load_and_resample_audio(wav_path, target_sr=SAMPLE_RATE)
    audio_duration = len(wav) / SAMPLE_RATE
    
    # Pipeline mel: WAV -> dB mel -> normalize to [-1,1]
    mel_db = extract_mel_spectrogram(wav, DSP_CFG)
    mel_pipeline, mel_min, mel_max = normalize_mel(mel_db)
    
    # Convert pipeline [-1,1] -> model [0,1]
    mel_01 = pipeline_to_model(mel_pipeline)
    print(f"  Mel shape: {mel_01.shape}, duration: {audio_duration:.1f}s")
    
    # Run inference with timing
    print("  Running VAE-GAN inference with timing...")
    transferred_01, timing_info = infer_mel(mel_01, model_dict,
                                            n_overlap=4,
                                            return_timing=True)
    
    # Evaluate latency
    report = evaluate_latency(timing_info, sample_rate=SAMPLE_RATE,
                              hop_length=256, segment_duration=5.0)
    
    # Print report
    print_latency_report(report, file_name=file_name)
    
    # Save latency plot
    plot_path = file_output_dir / f"{file_name}_latency.png"
    plot_latency(report, output_path=str(plot_path),
                 title_suffix=file_name)
    
    # Save timing data as JSON (exclude numpy arrays)
    json_report = {k: v for k, v in report.items() 
                   if not isinstance(v, np.ndarray)}
    json_report['patch_latencies'] = timing_info['patch_latencies']
    json_report['mel_path'] = 'pipeline'
    json_report['vocoder'] = vocoder_name
    json_path = file_output_dir / f"{file_name}_timing.json"
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"  Timing data saved: {json_path.name}")
    
    # Save original (resampled)
    orig_path = file_output_dir / f"{file_name}_original.wav"
    sf.write(str(orig_path), wav, SAMPLE_RATE)
    print(f"  Original saved: {orig_path.name}")
    
    # Save transferred mel
    mel_path = file_output_dir / f"{file_name}_transferred_mel.npy"
    np.save(str(mel_path), transferred_01)
    
    # Vocoding
    if vocoder_name == 'griffinlim':
        wav_out = reconstruct_with_griffinlim(transferred_01, n_iter=64)
        wav_out = np.clip(wav_out, -1.0, 1.0)
        out_path = file_output_dir / f"{file_name}_transferred_violin.wav"
        sf.write(str(out_path), wav_out, SAMPLE_RATE)
        dur = len(wav_out) / SAMPLE_RATE
        print(f"  Transferred audio (Griffin-Lim) saved: {out_path.name} ({dur:.1f}s)")
    elif vocoder is not None:
        # Direct vocoding: model [0,1] -> dB -> linear -> log -> BigVGAN -> WAV
        voc_label = 'BigVGAN' if 'bigvgan' in vocoder_name else 'HiFi-GAN'
        print(f"  Vocoding (Direct {voc_label})...")
        S_db = model_to_db(transferred_01)
        S_linear = db_to_linear(S_db)
        log_mel = np.log(np.maximum(S_linear, 1e-5))
        mel_tensor = torch.from_numpy(log_mel).float().unsqueeze(0)
        if vocoder.device.type == 'cuda':
            mel_tensor = mel_tensor.cuda()
        audio_out = vocoder.mel_to_audio(mel_tensor)
        audio_out = np.clip(audio_out, -1.0, 1.0)
        
        out_path = file_output_dir / f"{file_name}_transferred_violin.wav"
        sf.write(str(out_path), audio_out, SAMPLE_RATE)
        dur = len(audio_out) / SAMPLE_RATE
        print(f"  Transferred audio ({voc_label}) saved: {out_path.name} ({dur:.1f}s)")
    
    return report


def plot_summary(all_reports: dict, output_dir: Path):
    """
    Create a summary plot comparing latency across all test files.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Inference Latency Summary — All Test Files',
                 fontsize=14, fontweight='bold')
    
    file_names = list(all_reports.keys())
    n_files = len(file_names)
    
    # ─── Left: Box plot of per-patch latencies ──────────────────────────
    all_latencies = [all_reports[f]['latencies_ms'] for f in file_names]
    short_names = [f.replace('AuSep_', '').replace('_tpt_', '\ntpt_') 
                   for f in file_names]
    
    bp = ax1.boxplot(all_latencies, tick_labels=short_names, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2))
    colors_box = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for patch, color in zip(bp['boxes'], colors_box[:n_files]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_xlabel('Test File', fontsize=11)
    ax1.set_ylabel('Patch Latency (ms)', fontsize=11)
    ax1.set_title('Per-Patch Latency Distribution', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # ─── Right: Segment latencies comparison ────────────────────────────
    bar_width = 0.25
    max_segments = max(r['n_segments'] for r in all_reports.values())
    x = np.arange(max_segments)
    
    for idx, fname in enumerate(file_names):
        r = all_reports[fname]
        seg_s = r['segment_latencies_s']
        short = fname.replace('AuSep_', '').replace('_tpt_', ' tpt ')
        ax2.bar(x[:len(seg_s)] + idx * bar_width, seg_s,
                width=bar_width, alpha=0.8, label=short)
    
    ax2.axhline(y=5.0, color='#F44336', linewidth=2, linestyle='--',
                label='5s real-time target')
    ax2.set_xlabel('Segment Index (5s segments)', fontsize=11)
    ax2.set_ylabel('Inference Time (seconds)', fontsize=11)
    ax2.set_title('Per-Segment Latency vs Target', fontsize=11)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    summary_path = output_dir / 'latency_summary.png'
    fig.savefig(str(summary_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Summary plot saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference Benchmark: Latency on Test-Split Trumpet Files (Pipeline Mel)")
    parser.add_argument("--file", type=str, default=None,
                        help="Benchmark a specific file (e.g., AuSep_2_tpt_15_Surprise.wav)")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for results")
    parser.add_argument("--vocoder", type=str, default="bigvgan",
                        choices=VOCODER_CHOICES,
                        help="Vocoder for waveform reconstruction (default: bigvgan)")
    parser.add_argument("--epoch", type=int, default=EPOCH,
                        help="Checkpoint epoch to load")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  INFERENCE BENCHMARK: Latency Evaluation (Pipeline Mel)")
    print("=" * 70)
    print(f"  Vocoder: {args.vocoder}")
    
    # ─── Determine test files ───────────────────────────────────────────
    test_files = get_test_trumpet_files()
    print(f"\n  Test-split trumpet files ({len(test_files)}):")
    for f in test_files:
        print(f"    {f}")
    
    if args.file:
        if args.file not in test_files:
            # Check if it exists anyway
            wav_path = SPKR_1_DIR / args.file
            if wav_path.exists():
                print(f"\n  NOTE: {args.file} is NOT in test split, "
                      f"running anyway.")
                test_files = [args.file]
            else:
                print(f"\n  ERROR: File not found: {wav_path}")
                sys.exit(1)
        else:
            test_files = [args.file]
    
    # ─── Verify files exist ─────────────────────────────────────────────
    for f in test_files:
        wav_path = SPKR_1_DIR / f
        if not wav_path.exists():
            print(f"  ERROR: WAV not found: {wav_path}")
            sys.exit(1)
    
    # ─── Load model ─────────────────────────────────────────────────────
    print(f"\n  Loading model: {MODEL_NAME} (epoch {args.epoch})...")
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
    
    # ─── Load vocoder ───────────────────────────────────────────────────
    vocoder = None
    vocoder_name = args.vocoder
    if vocoder_name == 'bigvgan':
        print("  Loading BigVGAN vocoder...")
        t0 = time.time()
        from postprocessing.bigvgan_vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(model_name="bigvgan_22k")
        print(f"  BigVGAN loaded ({time.time()-t0:.1f}s)")
    elif vocoder_name == 'hifigan':
        print("  Loading HiFi-GAN vocoder...")
        t0 = time.time()
        from postprocessing.vocoder_factory import create_vocoder
        vocoder = create_vocoder('hifigan')
        print(f"  HiFi-GAN loaded ({time.time()-t0:.1f}s)")
    else:
        print("  Using Griffin-Lim (no neural vocoder)")
    
    # ─── Run benchmarks ─────────────────────────────────────────────────
    all_reports = {}
    
    for f in test_files:
        wav_path = SPKR_1_DIR / f
        report = run_benchmark_on_file(wav_path, model_dict, output_dir,
                                       vocoder=vocoder,
                                       vocoder_name=vocoder_name)
        all_reports[wav_path.stem] = report
    
    # ─── Summary ────────────────────────────────────────────────────────
    if len(all_reports) > 1:
        plot_summary(all_reports, output_dir)
    
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\n  Results in: {output_dir}/")
    print(f"  Files benchmarked: {len(all_reports)}")
    
    # Overall stats
    all_means = [r['mean_ms'] for r in all_reports.values()]
    all_rtfs = [r['rtf'] for r in all_reports.values()]
    all_targets = [r['meets_target'] for r in all_reports.values()]
    
    print(f"\n  Aggregate Statistics:")
    print(f"    Mean patch latency:  {np.mean(all_means):.2f} ms")
    print(f"    Mean RTF:            {np.mean(all_rtfs):.4f}")
    print(f"    Meets 5s target:     {sum(all_targets)}/{len(all_targets)} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
