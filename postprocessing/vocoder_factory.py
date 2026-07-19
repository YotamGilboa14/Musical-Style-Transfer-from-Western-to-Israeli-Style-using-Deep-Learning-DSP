"""
Unified Vocoder Factory
========================

Provides a single entry point for creating any of our vocoder models:

  1. "hifigan"     - HiFi-GAN UNIVERSAL_V1 (22kHz, 80 bands, fmax=8000)
                     Original speech-trained model. Fastest, lowest memory.
                     
  2. "bigvgan_22k" - BigVGAN v2 22kHz (80 bands, fmax=8000)
                     Drop-in upgrade. Same mel config, music-trained.
                     
  3. "bigvgan_24k" - BigVGAN v2 24kHz (100 bands, fmax=12000)
                     Highest fidelity. Wider frequency range.
                     Requires its own mel computation (different config).

Usage:
    from postprocessing.vocoder_factory import create_vocoder, AVAILABLE_VOCODERS
    
    # List available vocoders
    for name, info in AVAILABLE_VOCODERS.items():
        print(f"  {name}: {info}")
    
    # Create any vocoder by name
    vocoder = create_vocoder("bigvgan_22k")
    vocoder.wav_to_wav("input.wav", "output.wav")

All vocoders share the same interface:
    - .wav_to_wav(input_path, output_path) → np.ndarray
    - .wav_to_wav_segmented(input_path, output_path) → np.ndarray
    - .mel_to_audio(mel_tensor) → np.ndarray
    - .segments_to_wav(mel_dir, output_path, mel_min, mel_max) → np.ndarray

Author: Yotam & Gal - StyleTransfer Music Project
Date: February 2026
"""

from typing import Optional


# ============================================================================
# AVAILABLE VOCODERS
# ============================================================================

AVAILABLE_VOCODERS = {
    "hifigan": {
        "description": "HiFi-GAN UNIVERSAL_V1 — 22kHz, 80 bands, fmax=8000 (speech-trained)",
        "sample_rate": 22050,
        "n_mels": 80,
        "fmax": 8000,
        "params": "14M",
        "training_data": "Speech (LibriTTS/VCTK/LJSpeech)",
    },
    "bigvgan_22k": {
        "description": "BigVGAN v2 — 22kHz, 80 bands, fmax=8000 (music-trained, drop-in upgrade)",
        "sample_rate": 22050,
        "n_mels": 80,
        "fmax": 8000,
        "params": "112M",
        "training_data": "Diverse (speech + instruments + environmental)",
    },
    "bigvgan_24k": {
        "description": "BigVGAN v2 — 24kHz, 100 bands, fmax=12000 (highest fidelity)",
        "sample_rate": 24000,
        "n_mels": 100,
        "fmax": 12000,
        "params": "112M",
        "training_data": "Diverse (speech + instruments + environmental)",
    },
}


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_vocoder(name: str, device: Optional[str] = None):
    """Create a vocoder instance by name.
    
    Args:
        name: One of "hifigan", "bigvgan_22k", "bigvgan_24k"
        device: 'cuda' or 'cpu'. Auto-detected if None.
    
    Returns:
        Vocoder instance with .wav_to_wav(), .mel_to_audio(), etc.
    
    Raises:
        ValueError: If name is not recognized
    """
    if name not in AVAILABLE_VOCODERS:
        valid = ", ".join(AVAILABLE_VOCODERS.keys())
        raise ValueError(
            f"Unknown vocoder '{name}'. Available: {valid}"
        )
    
    if name == "hifigan":
        from postprocessing.vocoder_inference import HiFiGANVocoder
        return HiFiGANVocoder(device=device)
    
    elif name in ("bigvgan_22k", "bigvgan_24k"):
        from postprocessing.bigvgan_vocoder import BigVGANVocoder
        return BigVGANVocoder(model_name=name, device=device)
    
    else:
        raise ValueError(f"No implementation for vocoder '{name}'")


def list_vocoders():
    """Print a table of all available vocoders."""
    print("\nAvailable Vocoders:")
    print("-" * 80)
    for name, info in AVAILABLE_VOCODERS.items():
        print(f"  {name:15s} | {info['description']}")
        print(f"  {'':15s} | SR={info['sample_rate']}, mels={info['n_mels']}, "
              f"fmax={info['fmax']}, params={info['params']}")
        print(f"  {'':15s} | Training: {info['training_data']}")
        print()
    print("-" * 80)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Vocoder Interface")
    parser.add_argument('--list', action='store_true', help='List available vocoders')
    parser.add_argument('--vocoder', type=str, default='hifigan',
                       choices=list(AVAILABLE_VOCODERS.keys()),
                       help='Vocoder to use')
    parser.add_argument('--input', type=str, help='Input WAV file')
    parser.add_argument('--output', type=str, help='Output WAV file')
    parser.add_argument('--segmented', action='store_true',
                       help='Process in 5s segments')
    
    args = parser.parse_args()
    
    if args.list:
        list_vocoders()
    elif args.input:
        from pathlib import Path
        
        if args.output is None:
            p = Path(args.input)
            args.output = str(p.parent / f"{p.stem}_{args.vocoder}_reconstructed.wav")
        
        vocoder = create_vocoder(args.vocoder)
        
        if args.segmented:
            vocoder.wav_to_wav_segmented(args.input, args.output)
        else:
            vocoder.wav_to_wav(args.input, args.output)
    else:
        parser.print_help()
