"""
Source Separation Module (Demucs v4)
=====================================

Separates a mixed audio track into individual instrument stems using
Meta's Hybrid Transformer Demucs (htdemucs) model.

Output stems (4 default): vocals, drums, bass, other
Output format: WAV files at Demucs native sample rate (44,100 Hz stereo)

Downstream processing (e.g., dsp_preprocessor.load_and_resample_audio)
handles resampling to 22,050 Hz as needed.

Usage as CLI:
    python source_separator.py path/to/song.wav
    python source_separator.py path/to/song.wav --output-dir path/to/output
    python source_separator.py path/to/song.wav --model htdemucs --stem vocals

Usage from Python:
    from preprocessing.source_separator import separate_stems
    stems = separate_stems("path/to/song.wav", output_dir="path/to/output")
    # stems = {"vocals": Path("..."), "drums": Path("..."), ...}

Author: Yotam & Gal - StyleTransfer Music Project
Date: February 2026
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Demucs imports (deferred to functions for fast CLI --help)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default model: htdemucs_ft is the fine-tuned Hybrid Transformer Demucs v4.
# It achieves state-of-the-art 9.0 dB SDR on MUSDB-HQ.
# Takes ~4x longer than htdemucs but produces noticeably cleaner stems.
# On CPU, expect ~6 min for a 4-minute song with htdemucs_ft.
DEFAULT_MODEL = "htdemucs_ft"

# Default stems produced by htdemucs / htdemucs_ft
STEM_NAMES = ["vocals", "drums", "bass", "other"]

# 6-source model (htdemucs_6s) additionally produces "guitar" and "piano"
STEM_NAMES_6S = ["vocals", "drums", "bass", "other", "guitar", "piano"]


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def separate_stems(
    audio_path: str | Path,
    output_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    progress: bool = True,
) -> dict[str, Path]:
    """
    Separate an audio file into instrument stems using Demucs.

    Args:
        audio_path: Path to the input audio file (WAV, MP3, FLAC, etc.)
        output_dir: Directory to save separated stems. If None, creates
                     a 'stems' subdirectory next to the input file.
        model_name: Demucs model to use. Options:
                     - 'htdemucs_ft': fine-tuned, best quality (default)
                     - 'htdemucs': faster, slightly lower quality
                     - 'htdemucs_6s': 6 sources (adds guitar + piano)
                     - 'hdemucs_mmi': Hybrid Demucs v3
                     - 'mdx_extra': MDX challenge model
        device: 'cpu' or 'cuda' for GPU acceleration.
        progress: Show progress bar during separation.

    Returns:
        Dictionary mapping stem names to their saved file paths.
        e.g. {"vocals": Path("stems/vocals.wav"), "drums": Path("stems/drums.wav"), ...}

    Raises:
        FileNotFoundError: If audio_path does not exist.
        RuntimeError: If separation fails.
    """
    import torch
    import numpy as np
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.separate import load_track

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Set up output directory
    if output_dir is None:
        output_dir = audio_path.parent / "stems"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Input: {audio_path.name}")
    print(f"  Output dir: {output_dir}")

    # Load model
    print(f"  Loading model '{model_name}'...")
    start_time = time.time()

    model = get_model(name=model_name)
    model.to(device)
    model.eval()

    load_time = time.time() - start_time
    print(f"  Model loaded in {load_time:.1f}s")
    print(f"  Model sample rate: {model.samplerate} Hz")
    print(f"  Model sources: {model.sources}")

    # Load and resample audio to model's expected sample rate
    print(f"  Loading audio...")
    wav = load_track(audio_path, model.audio_channels, model.samplerate)

    # Normalize (same as demucs.separate.main does)
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()

    # Run separation
    print(f"  Separating stems...")
    sep_start = time.time()

    with torch.no_grad():
        sources = apply_model(
            model, wav[None],
            device=device,
            progress=progress,
            split=True,
            overlap=0.25,
        )[0]  # Remove batch dimension

    # Denormalize
    sources *= ref.std()
    sources += ref.mean()

    sep_time = time.time() - sep_start
    print(f"  Separation completed in {sep_time:.1f}s")

    # Save each stem using soundfile (avoids torchaudio/torchcodec issues)
    stem_paths = {}
    for source_tensor, stem_name in zip(sources, model.sources):
        stem_file = output_dir / f"{stem_name}.wav"
        # Convert from (channels, samples) tensor to (samples, channels) numpy
        audio_np = source_tensor.cpu().numpy().T  # (samples, channels)
        # Clip to prevent clipping artifacts
        audio_np = np.clip(audio_np, -1.0, 1.0)
        sf.write(str(stem_file), audio_np, model.samplerate, subtype='PCM_16')
        # Report file size
        size_mb = stem_file.stat().st_size / (1024 * 1024)
        print(f"    ✓ {stem_name}.wav ({size_mb:.1f} MB)")
        stem_paths[stem_name] = stem_file

    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.1f}s")

    return stem_paths


def check_stems_exist(stems_dir: str | Path, expected_stems: list[str] | None = None) -> bool:
    """
    Check if all expected stem files already exist in the given directory.

    Args:
        stems_dir: Directory to check for stem files.
        expected_stems: List of stem names to check. Defaults to STEM_NAMES.

    Returns:
        True if all expected stems exist as WAV files.
    """
    stems_dir = Path(stems_dir)
    if expected_stems is None:
        expected_stems = STEM_NAMES

    if not stems_dir.exists():
        return False

    for stem_name in expected_stems:
        stem_file = stems_dir / f"{stem_name}.wav"
        if not stem_file.exists():
            return False

    return True


def get_stem_path(stems_dir: str | Path, stem_name: str = "vocals") -> Path:
    """
    Get the path to a specific stem file. Useful for selecting which
    stem to feed into DDSP timbre transfer (Phase 4).

    Args:
        stems_dir: Directory containing separated stems.
        stem_name: Name of the stem ('vocals', 'drums', 'bass', 'other').

    Returns:
        Path to the stem WAV file.

    Raises:
        FileNotFoundError: If the stem file does not exist.
    """
    stems_dir = Path(stems_dir)
    stem_file = stems_dir / f"{stem_name}.wav"
    if not stem_file.exists():
        available = [f.stem for f in stems_dir.glob("*.wav")]
        raise FileNotFoundError(
            f"Stem '{stem_name}' not found in {stems_dir}. "
            f"Available stems: {available}"
        )
    return stem_file


def list_available_models() -> list[str]:
    """
    List well-known Demucs model names.

    Returns:
        List of model name strings.
    """
    return [
        "htdemucs_ft",   # Fine-tuned Hybrid Transformer, best quality
        "htdemucs",      # Hybrid Transformer, faster
        "htdemucs_6s",   # 6 sources (adds guitar + piano)
        "hdemucs_mmi",   # Hybrid Demucs v3
        "mdx_extra",     # MDX challenge, extra training data
        "mdx_extra_q",   # Quantized MDX extra
    ]


# ============================================================================
# CLI
# ============================================================================

def main():
    """Parse the Demucs CLI wrapper arguments and run/source-check separation."""
    parser = argparse.ArgumentParser(
        description="Separate an audio file into instrument stems using Demucs v4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.wav
  %(prog)s song.wav --output-dir stems/
  %(prog)s song.wav --model htdemucs --device cuda
  %(prog)s song.wav --stem vocals    (only print path to vocals stem)
  %(prog)s --list-models             (show available models)
        """,
    )
    parser.add_argument(
        "audio", nargs="?",
        help="Path to the audio file to separate.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory for stems. Default: <audio_dir>/stems/",
    )
    parser.add_argument(
        "--model", "-n", type=str, default=DEFAULT_MODEL,
        help=f"Demucs model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cpu",
        help="Device to run on: 'cpu' or 'cuda'. Default: cpu",
    )
    parser.add_argument(
        "--stem", type=str, default=None,
        help="After separation, print only the path to this stem.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available Demucs models and exit.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bar.",
    )

    args = parser.parse_args()

    if args.list_models:
        models = list_available_models()
        print("Available Demucs models:")
        for m in models:
            marker = " (default)" if m == DEFAULT_MODEL else ""
            print(f"  - {m}{marker}")
        return

    if args.audio is None:
        parser.error("Audio file path is required (or use --list-models).")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    print("=" * 60)
    print("Demucs Source Separation")
    print("=" * 60)

    stems = separate_stems(
        audio_path=audio_path,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        progress=not args.no_progress,
    )

    print(f"\n  Stems saved to: {list(stems.values())[0].parent}")

    if args.stem:
        if args.stem in stems:
            print(f"\n  Requested stem path: {stems[args.stem]}")
        else:
            print(f"\n  WARNING: Stem '{args.stem}' not found. Available: {list(stems.keys())}")


if __name__ == "__main__":
    main()
