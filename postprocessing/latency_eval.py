"""
Latency Evaluation Module
==========================
Measures and visualizes per-patch inference latency for the VAE-GAN
timbre transfer model.

Key metrics:
  - Per-patch latency (ms)
  - Mean / median / p95 / max latency
  - Real-time factor (RTF): latency vs audio duration per patch
  - 5-second segment latency (target: ≤5s per 5s of audio)

Usage:
    from postprocessing.latency_eval import evaluate_latency, plot_latency

    timing_info = {...}  # from infer_mel(return_timing=True)
    report = evaluate_latency(timing_info, sample_rate=22050, hop_length=256)
    plot_latency(report, output_path='latency_report.png')

Author: Yotam & Gal — StyleTransfer Music Project
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional


def evaluate_latency(timing_info: Dict[str, Any],
                     sample_rate: int = 22050,
                     hop_length: int = 256,
                     segment_duration: float = 5.0) -> Dict[str, Any]:
    """
    Analyze inference timing data and compute latency metrics.
    
    Args:
        timing_info: dict from infer_mel(return_timing=True) containing:
            - patch_latencies: list of per-patch times (seconds)
            - total_time: total inference time (seconds)
            - n_patches: number of patches processed
            - patch_width_frames: frames per patch (e.g., 128)
            - hop_frames: hop between patches (e.g., 32)
            - input_frames: total input frames
        sample_rate: audio sample rate (Hz)
        hop_length: mel hop length (samples)
        segment_duration: target segment duration (seconds) for RTF calc
    
    Returns:
        dict with:
            - All original timing_info fields
            - latencies_ms: per-patch latencies in milliseconds
            - mean_ms, median_ms, p95_ms, max_ms, min_ms: statistics
            - patch_duration_s: audio duration per patch (seconds)
            - rtf: real-time factor (inference_time / audio_duration)
            - total_audio_duration_s: total audio duration
            - segment_frames: frames per segment_duration
            - segment_latencies_ms: estimated latency per 5s segment
            - meets_realtime: whether mean patch latency is below real-time
            - meets_target: whether 5s segments process in ≤5s
    """
    latencies = np.array(timing_info['patch_latencies'])
    latencies_ms = latencies * 1000.0
    
    patch_width = timing_info['patch_width_frames']
    hop = timing_info['hop_frames']
    n_input_frames = timing_info['input_frames']
    
    # Audio duration per patch
    patch_audio_s = patch_width * hop_length / sample_rate
    
    # Total audio duration
    total_audio_s = n_input_frames * hop_length / sample_rate
    
    # Real-time factor: total_inference / total_audio
    rtf = timing_info['total_time'] / total_audio_s if total_audio_s > 0 else float('inf')
    
    # Segment-level latency estimation
    # How many patches cover one 5s segment?
    segment_frames = int(segment_duration * sample_rate / hop_length)
    patches_per_segment = max(1, segment_frames // hop)
    
    # Compute per-segment latencies by grouping patches
    n_segments = max(1, len(latencies) // patches_per_segment)
    segment_latencies = []
    for i in range(n_segments):
        start = i * patches_per_segment
        end = min(start + patches_per_segment, len(latencies))
        seg_time = latencies[start:end].sum()
        segment_latencies.append(seg_time)
    segment_latencies_ms = np.array(segment_latencies) * 1000.0
    
    # Target check: 5s of audio should process in ≤5s
    meets_target = all(s <= segment_duration for s in segment_latencies)
    
    report = {
        **timing_info,
        'latencies_ms': latencies_ms,
        'mean_ms': float(np.mean(latencies_ms)),
        'median_ms': float(np.median(latencies_ms)),
        'p95_ms': float(np.percentile(latencies_ms, 95)),
        'max_ms': float(np.max(latencies_ms)),
        'min_ms': float(np.min(latencies_ms)),
        'std_ms': float(np.std(latencies_ms)),
        'patch_duration_s': patch_audio_s,
        'rtf': rtf,
        'total_audio_duration_s': total_audio_s,
        'segment_duration_s': segment_duration,
        'segment_frames': segment_frames,
        'patches_per_segment': patches_per_segment,
        'n_segments': n_segments,
        'segment_latencies_s': np.array(segment_latencies),
        'segment_latencies_ms': segment_latencies_ms,
        'meets_realtime': rtf <= 1.0,
        'meets_target': meets_target,
    }
    return report


def plot_latency(report: Dict[str, Any],
                 output_path: Optional[str] = None,
                 title_suffix: str = '') -> plt.Figure:
    """
    Create a segment-wise latency visualization.
    
    Generates a single figure with per-segment (5s) latency bars
    and a real-time target threshold line.
    
    Args:
        report: dict from evaluate_latency()
        output_path: path to save PNG (or None to return figure)
        title_suffix: optional suffix for the main title
    
    Returns:
        matplotlib Figure
    """
    segment_latencies_ms = report['segment_latencies_ms']
    segment_duration = report['segment_duration_s']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    suptitle = 'VAE-GAN Inference Latency — Segment-Wise'
    if title_suffix:
        suptitle += f' — {title_suffix}'
    
    # ─── Per-segment (5s) latency ──────────────────────────────────────
    n_segments = len(segment_latencies_ms)
    seg_indices = np.arange(n_segments)
    seg_latencies_s = segment_latencies_ms / 1000.0
    
    # Color by target (5s threshold)
    seg_colors = ['#4CAF50' if lat <= segment_duration else '#F44336'
                  for lat in report['segment_latencies_s']]
    
    bars = ax.bar(seg_indices, seg_latencies_s, color=seg_colors, alpha=0.8,
                  width=0.6, edgecolor='none', label='Segment latency')
    
    # Target threshold line
    ax.axhline(y=segment_duration, color='#F44336', linewidth=2,
               linestyle='--',
               label=f'Real-time target: {segment_duration:.0f}s')
    
    # Add value labels on bars
    for idx, (bar, val) in enumerate(zip(bars, seg_latencies_s)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    
    ax.set_xlabel('Segment Index (5-second audio segments)', fontsize=12)
    ax.set_ylabel('Inference Time (seconds)', fontsize=12)
    
    target_str = 'MEETS TARGET' if report['meets_target'] else 'EXCEEDS TARGET'
    rt_str = 'Real-time' if report['meets_realtime'] else f'RTF={report["rtf"]:.2f}'
    title_color = 'green' if report['meets_target'] else 'red'
    ax.set_title(suptitle, fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_xlim(-0.5, n_segments - 0.5)
    ax.set_ylim(0, max(max(seg_latencies_s) * 1.3, segment_duration * 1.2))
    ax.grid(axis='y', alpha=0.3)
    
    # Stats text box
    stats_text = (
        f'Total inference: {report["total_time"]:.2f}s\n'
        f'Audio duration:  {report["total_audio_duration_s"]:.1f}s\n'
        f'RTF: {report["rtf"]:.4f} ({1/report["rtf"]:.0f}× real-time)\n'
        f'Mean patch: {report["mean_ms"]:.1f} ms  |  P95: {report["p95_ms"]:.1f} ms\n'
        f'Result: {"✓" if report["meets_target"] else "✗"} {target_str}'
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      alpha=0.9, edgecolor='#333'))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"  Latency plot saved: {output_path}")
    
    return fig


def print_latency_report(report: Dict[str, Any], file_name: str = ''):
    """Print a formatted latency report to stdout."""
    print()
    print("=" * 60)
    title = "LATENCY EVALUATION REPORT"
    if file_name:
        title += f" — {file_name}"
    print(f"  {title}")
    print("=" * 60)
    print()
    print(f"  Input:  {report['input_frames']} frames "
          f"({report['total_audio_duration_s']:.1f}s audio)")
    print(f"  Patches: {report['n_patches']} "
          f"({report['patch_width_frames']}-frame windows, "
          f"{report['hop_frames']}-frame hop)")
    print()
    print("  Per-Patch Latency:")
    print(f"    Mean:   {report['mean_ms']:7.2f} ms")
    print(f"    Median: {report['median_ms']:7.2f} ms")
    print(f"    P95:    {report['p95_ms']:7.2f} ms")
    print(f"    Max:    {report['max_ms']:7.2f} ms")
    print(f"    Min:    {report['min_ms']:7.2f} ms")
    print(f"    Std:    {report['std_ms']:7.2f} ms")
    print()
    print(f"  Total inference time: {report['total_time']:.3f}s")
    print(f"  Real-Time Factor (RTF): {report['rtf']:.4f}")
    rtf_str = "✓ Faster than real-time" if report['meets_realtime'] else "✗ Slower than real-time"
    print(f"    {rtf_str}")
    print()
    print(f"  Per-{report['segment_duration_s']:.0f}s Segment Latency "
          f"({report['n_segments']} segments, "
          f"~{report['patches_per_segment']} patches each):")
    for i, seg_s in enumerate(report['segment_latencies_s']):
        flag = "✓" if seg_s <= report['segment_duration_s'] else "✗"
        print(f"    Segment {i:3d}: {seg_s:6.3f}s  {flag}")
    
    target_str = ("✓ ALL segments meet target" if report['meets_target']
                  else "✗ Some segments EXCEED target")
    print(f"\n  Target ({report['segment_duration_s']:.0f}s per "
          f"{report['segment_duration_s']:.0f}s audio): {target_str}")
    print("=" * 60)
