"""
HiFi-GAN Vocoder Inference Module
===================================

Converts mel-spectrograms back to audio waveforms using the pre-trained
HiFi-GAN UNIVERSAL_V1 generator. This is the post-processing counterpart
to our DSP preprocessing pipeline.

Pipeline position:
  [Audio] → preprocessing → [Mel Tensors] → (model) → [Mel Tensors] → vocoder → [Audio]
                                                                        ^^^^^^^^
                                                                     THIS MODULE

Key design decisions:
  - Uses HiFi-GAN's own mel computation for round-trip tests (ensures fidelity)
  - Provides conversion from our librosa-based mels to HiFi-GAN format
  - Supports both segment-by-segment and full-song reconstruction

Mel Format Differences (IMPORTANT):
  Our preprocessing: librosa.power_to_db → dB scale → normalized to [-1, 1]
  HiFi-GAN internal: torch.log(clamp(magnitude_mel, min=1e-5)) → log-magnitude
  These are different representations! This module handles the conversion.

Author: Yotam & Gal - StyleTransfer Music Project
Date: February 2026
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import torch
import librosa
import soundfile as sf

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from postprocessing.hifigan_model import Generator, AttrDict


# ============================================================================
# CONSTANTS
# ============================================================================

# Maximum WAV value for int16 format (HiFi-GAN convention)
MAX_WAV_VALUE = 32768.0

# Default checkpoint directory (relative to project root)
DEFAULT_CHECKPOINT_DIR = "postprocessing/hifigan_checkpoints/UNIVERSAL_V1"


# ============================================================================
# MEL SPECTROGRAM (HiFi-GAN NATIVE FORMAT)
# ============================================================================

# Global caches for mel basis and hann window (following HiFi-GAN convention)
_mel_basis = {}
_hann_window = {}


def hifigan_mel_spectrogram(y: torch.Tensor, n_fft: int = 1024, num_mels: int = 80,
                             sampling_rate: int = 22050, hop_size: int = 256,
                             win_size: int = 1024, fmin: float = 0.0,
                             fmax: float = 8000.0, center: bool = False) -> torch.Tensor:
    """Compute mel-spectrogram using HiFi-GAN's exact method.
    
    This is the NATIVE mel format that the HiFi-GAN generator was trained on.
    It differs from librosa's power_to_db:
      - Uses magnitude (not power) spectrogram
      - Applies natural log (not 10*log10)
      - Uses center=False padding
    
    Args:
        y: (batch, audio_samples) or (audio_samples,) waveform tensor
        Other params: match HiFi-GAN config exactly
    
    Returns:
        mel: (batch, num_mels, time_frames) log-magnitude mel spectrogram
    """
    from librosa.filters import mel as librosa_mel_fn
    
    global _mel_basis, _hann_window
    
    # Ensure y is at least 2D
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    # Build mel filterbank (cached)
    key = f"{fmax}_{y.device}"
    if key not in _mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
    
    # Build Hann window (cached)
    win_key = str(y.device)
    if win_key not in _hann_window:
        _hann_window[win_key] = torch.hann_window(win_size).to(y.device)
    
    # Pad signal for center=False STFT (HiFi-GAN convention)
    # This ensures the first frame starts at the beginning of the signal
    pad_amount = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_amount, pad_amount), mode='reflect')
    y = y.squeeze(1)
    
    # Compute STFT → magnitude spectrogram
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                      window=_hann_window[win_key], center=center,
                      pad_mode='reflect', normalized=False, onesided=True,
                      return_complex=True)
    spec = torch.abs(spec)  # Complex → magnitude
    
    # Apply mel filterbank
    spec = torch.matmul(_mel_basis[key], spec)
    
    # Log compression (HiFi-GAN's spectral_normalize_torch)
    # Using natural log, not dB. Clamp at 1e-5 to avoid log(0)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    
    return spec


# ============================================================================
# VOCODER CLASS
# ============================================================================

class HiFiGANVocoder:
    """HiFi-GAN vocoder wrapper for mel-to-waveform conversion.
    
    Usage:
        vocoder = HiFiGANVocoder("path/to/checkpoint_dir")
        
        # Round-trip: WAV → mel → WAV (uses HiFi-GAN's native mel format)
        audio = vocoder.wav_to_wav("input.wav", "output.wav")
        
        # From our preprocessed mels: tensor → WAV
        vocoder.mel_tensor_to_wav(mel_tensor, mel_min, mel_max, "output.wav")
    """
    
    def __init__(self, checkpoint_dir: str = None, device: str = None):
        """Initialize vocoder with pre-trained checkpoint.
        
        Args:
            checkpoint_dir: Directory containing config.json and generator checkpoint.
                          If None, uses default location.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        # Determine device (prefer GPU if available)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Resolve checkpoint directory
        if checkpoint_dir is None:
            project_root = Path(__file__).parent.parent
            checkpoint_dir = str(project_root / DEFAULT_CHECKPOINT_DIR)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Load config (contains all mel/model hyperparameters)
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"HiFi-GAN config not found at {config_path}. "
                f"Please download the UNIVERSAL_V1 checkpoint first."
            )
        
        with open(config_path) as f:
            config_data = json.load(f)
        self.h = AttrDict(config_data)
        
        # Build generator model
        self.generator = Generator(self.h).to(self.device)
        
        # Load pre-trained weights
        checkpoint_file = self._find_checkpoint()
        state_dict = torch.load(checkpoint_file, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(state_dict['generator'])
        
        # Prepare for inference: eval mode + remove weight norm
        self.generator.eval()
        self.generator.remove_weight_norm()
        
        print(f"✓ HiFi-GAN vocoder loaded from {self.checkpoint_dir.name}")
        print(f"  Device: {self.device}")
        print(f"  Config: sr={self.h.sampling_rate}, hop={self.h.hop_size}, "
              f"mels={self.h.num_mels}, fmax={self.h.fmax}")
    
    def _find_checkpoint(self) -> Path:
        """Find the generator checkpoint file in the directory.
        
        HiFi-GAN checkpoints are named like 'g_02500000'.
        We look for files starting with 'g_' and pick the latest one.
        """
        # Look for generator checkpoint files (named g_XXXXXXXX)
        g_files = sorted(self.checkpoint_dir.glob("g_*"))
        if g_files:
            return g_files[-1]  # Latest checkpoint
        
        # Also check for 'generator' or 'generator.pth'
        for name in ['generator', 'generator.pth', 'generator.pt']:
            path = self.checkpoint_dir / name
            if path.exists():
                return path
        
        raise FileNotFoundError(
            f"No generator checkpoint found in {self.checkpoint_dir}. "
            f"Expected files matching 'g_*' pattern."
        )
    
    def _compute_native_mel(self, audio: np.ndarray, sr: int = None) -> torch.Tensor:
        """Compute mel-spectrogram using HiFi-GAN's native format.
        
        This ensures exact compatibility with the pre-trained generator.
        
        Args:
            audio: (samples,) numpy array, expected in [-1, 1] range
            sr: Sample rate (must match config, default 22050)
        
        Returns:
            mel: (1, 80, frames) tensor in HiFi-GAN's log-magnitude format
        """
        if sr is not None and sr != self.h.sampling_rate:
            raise ValueError(
                f"Sample rate mismatch: got {sr}, expected {self.h.sampling_rate}"
            )
        
        # Convert numpy audio to torch tensor
        wav_tensor = torch.FloatTensor(audio).to(self.device)
        
        # Compute mel using HiFi-GAN's exact method
        mel = hifigan_mel_spectrogram(
            wav_tensor,
            n_fft=self.h.n_fft,
            num_mels=self.h.num_mels,
            sampling_rate=self.h.sampling_rate,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            fmin=self.h.fmin,
            fmax=self.h.fmax
        )
        
        return mel
    
    @torch.no_grad()
    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """Convert a mel-spectrogram tensor to audio waveform.
        
        Args:
            mel: (batch, 80, frames) or (80, frames) mel tensor in HiFi-GAN format
        
        Returns:
            audio: (samples,) numpy array, float32 in [-1, 1]
        """
        # Ensure correct shape: (batch, 80, frames)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        
        mel = mel.to(self.device)
        
        # Generate audio through the generator
        y_g_hat = self.generator(mel)
        
        # Squeeze batch and channel dimensions, move to CPU
        audio = y_g_hat.squeeze().cpu().numpy()
        
        return audio
    
    def wav_to_wav(self, input_path: str, output_path: str) -> np.ndarray:
        """Round-trip test: WAV → native mel → generator → WAV.
        
        Uses HiFi-GAN's own mel computation to ensure maximum fidelity.
        The output should sound like a slightly degraded version of the input.
        This validates that the vocoder pipeline is working correctly.
        
        Args:
            input_path: Path to input WAV file
            output_path: Path to save reconstructed WAV
        
        Returns:
            audio: reconstructed audio as numpy array
        """
        print(f"  Loading audio: {Path(input_path).name}")
        
        # Load audio at HiFi-GAN's expected sample rate
        audio, sr = librosa.load(input_path, sr=self.h.sampling_rate, mono=True)
        print(f"  Duration: {len(audio)/sr:.2f}s, SR: {sr}")
        
        # Compute mel using HiFi-GAN's native format
        print(f"  Computing mel-spectrogram (HiFi-GAN native format)...")
        mel = self._compute_native_mel(audio, sr)
        print(f"  Mel shape: {mel.shape}")
        
        # Generate audio from mel
        print(f"  Running HiFi-GAN generator...")
        reconstructed = self.mel_to_audio(mel)
        
        # Trim to original length (generator may produce slightly different length)
        if len(reconstructed) > len(audio):
            reconstructed = reconstructed[:len(audio)]
        
        # Save as WAV
        sf.write(output_path, reconstructed, sr)
        print(f"  ✓ Saved: {output_path}")
        print(f"  Output duration: {len(reconstructed)/sr:.2f}s")
        
        return reconstructed
    
    def segments_to_wav(self, mel_dir: str, output_path: str,
                        mel_min: float = -80.0, mel_max: float = 0.0) -> np.ndarray:
        """Reconstruct a full song from segmented mel tensors.
        
        Loads all mel segment .pt files from a directory, converts each
        from our [-1,1] normalized format to HiFi-GAN format, runs the
        generator, and concatenates the audio segments.
        
        IMPORTANT: This requires converting from our mel format to HiFi-GAN's.
        The conversion is approximate since they use different log scales.
        For best results, use wav_to_wav() which uses native mel computation.
        
        Args:
            mel_dir: Directory containing *_mel.pt files
            output_path: Where to save the reconstructed WAV
            mel_min: The dB minimum used during normalization 
            mel_max: The dB maximum used during normalization
        
        Returns:
            audio: full reconstructed audio as numpy array
        """
        mel_dir = Path(mel_dir)
        mel_files = sorted(mel_dir.glob("*_mel.pt"))
        
        if not mel_files:
            raise FileNotFoundError(f"No mel tensor files found in {mel_dir}")
        
        print(f"  Found {len(mel_files)} mel segments in {mel_dir.name}")
        
        all_audio = []
        for i, mel_file in enumerate(mel_files):
            # Load our normalized mel tensor: (80, 430) in [-1, 1]
            mel_norm = torch.load(mel_file, map_location=self.device, weights_only=True)
            
            # Denormalize: [-1, 1] → dB scale
            mel_db = (mel_norm + 1.0) / 2.0 * (mel_max - mel_min) + mel_min
            
            # Convert from librosa power_to_db to HiFi-GAN log-magnitude format:
            # power_to_db: val_db = 10 * log10(power / ref) 
            # We need: log(magnitude) = log(sqrt(power)) = 0.5 * ln(power)
            # power = ref * 10^(val_db / 10)    (ref=1.0 from np.max normalization)
            # But ref=np.max which we don't know... Using ref=1.0 as approximation
            # log(magnitude) = 0.5 * (val_db / 10) * ln(10) = val_db * 0.1151
            mel_hifigan = mel_db * (np.log(10) / 20.0)  # dB to log-magnitude
            
            # Ensure correct shape for generator: (1, 80, frames)
            if mel_hifigan.dim() == 2:
                mel_hifigan = mel_hifigan.unsqueeze(0)
            
            # Generate audio for this segment
            segment_audio = self.mel_to_audio(mel_hifigan)
            all_audio.append(segment_audio)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processed segment {i+1}/{len(mel_files)}")
        
        # Concatenate all segments
        full_audio = np.concatenate(all_audio)
        
        # Save as WAV
        sf.write(output_path, full_audio, self.h.sampling_rate)
        print(f"  ✓ Full song saved: {output_path}")
        print(f"  Duration: {len(full_audio)/self.h.sampling_rate:.2f}s")
        
        return full_audio
    
    def wav_to_wav_segmented(self, input_path: str, output_path: str,
                              segment_seconds: float = 5.0) -> np.ndarray:
        """Round-trip with segmentation (matches our preprocessing pipeline).
        
        Processes audio in 5-second segments (same as training data),
        which better simulates what the full pipeline will do.
        
        Args:
            input_path: Path to input WAV
            output_path: Path to save output WAV
            segment_seconds: Segment duration (default 5.0s, matches DSPConfig)
        
        Returns:
            audio: reconstructed audio
        """
        print(f"  Loading: {Path(input_path).name}")
        audio, sr = librosa.load(input_path, sr=self.h.sampling_rate, mono=True)
        
        segment_samples = int(segment_seconds * sr)
        n_segments = len(audio) // segment_samples
        
        print(f"  Processing {n_segments} segments of {segment_seconds}s each...")
        
        all_audio = []
        for i in range(n_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = audio[start:end]
            
            # Native mel → generator → audio
            mel = self._compute_native_mel(segment, sr)
            reconstructed = self.mel_to_audio(mel)
            
            # Trim to exact segment length
            reconstructed = reconstructed[:segment_samples]
            all_audio.append(reconstructed)
        
        full_audio = np.concatenate(all_audio)
        sf.write(output_path, full_audio, sr)
        
        print(f"  ✓ Saved: {output_path} ({len(full_audio)/sr:.2f}s)")
        return full_audio


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface for vocoder inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HiFi-GAN Vocoder Inference")
    parser.add_argument('input', type=str, help='Input WAV file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output WAV path (default: input_reconstructed.wav)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='HiFi-GAN checkpoint directory')
    parser.add_argument('--segmented', action='store_true',
                       help='Process in 5s segments (simulates training pipeline)')
    
    args = parser.parse_args()
    
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_reconstructed.wav")
    
    vocoder = HiFiGANVocoder(args.checkpoint_dir)
    
    if args.segmented:
        vocoder.wav_to_wav_segmented(args.input, args.output)
    else:
        vocoder.wav_to_wav(args.input, args.output)


if __name__ == "__main__":
    main()
