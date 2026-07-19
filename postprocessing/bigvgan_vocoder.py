"""
BigVGAN v2 Vocoder Inference Module
=====================================

Converts mel-spectrograms back to audio waveforms using pre-trained
BigVGAN v2 generators from NVIDIA. Supports two model variants:

  1. bigvgan_v2_22khz_80band_fmax8k_256x  (drop-in HiFi-GAN replacement)
     - 22050 Hz, 80 mel bins, fmax=8000, hop=256
     - Same mel config as our DSP pipeline → direct compatibility

  2. bigvgan_v2_24khz_100band_256x  (upgraded fidelity)
     - 24000 Hz, 100 mel bins, fmax=12000, hop=256
     - Wider frequency range → better music reconstruction

Both models trained on diverse audio (speech + instruments + environmental),
which is a significant upgrade over HiFi-GAN UNIVERSAL_V1 (speech-only).

Models downloaded automatically from HuggingFace Hub on first use.

Pipeline position:
  [Audio] → preprocessing → [Mel Tensors] → (model) → [Mel Tensors] → vocoder → [Audio]
                                                                        ^^^^^^^^
                                                                     THIS MODULE

Author: Yotam & Gal - StyleTransfer Music Project
Date: February 2026
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import time

import torch
import librosa
import soundfile as sf

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_WAV_VALUE = 32768.0

# Available BigVGAN v2 models (HuggingFace model IDs)
BIGVGAN_MODELS = {
    "bigvgan_22k": {
        "hf_id": "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x",
        "description": "22kHz, 80 mel bins, fmax=8000 (drop-in HiFi-GAN replacement)",
        "sample_rate": 22050,
        "n_mels": 80,
        "fmax": 8000,
        "hop_size": 256,
    },
    "bigvgan_24k": {
        "hf_id": "nvidia/bigvgan_v2_24khz_100band_256x",
        "description": "24kHz, 100 mel bins, fmax=12000 (upgraded fidelity)",
        "sample_rate": 24000,
        "n_mels": 100,
        "fmax": 12000,
        "hop_size": 256,
    },
}


# ============================================================================
# VOCODER CLASS
# ============================================================================

class BigVGANVocoder:
    """BigVGAN v2 vocoder wrapper for mel-to-waveform conversion.
    
    Downloads the model from HuggingFace Hub on first use, then caches locally.
    
    Usage:
        # Drop-in replacement for HiFi-GAN (same mel params)
        vocoder = BigVGANVocoder(model_name="bigvgan_22k")
        
        # Or upgraded fidelity version
        vocoder = BigVGANVocoder(model_name="bigvgan_24k")
        
        # Round-trip test: WAV → mel → generator → WAV
        audio = vocoder.wav_to_wav("input.wav", "output.wav")
    """
    
    def __init__(self, model_name: str = "bigvgan_22k", device: str = None):
        """Initialize BigVGAN vocoder with pre-trained checkpoint from HuggingFace.
        
        Uses manual loading to avoid huggingface_hub API incompatibility
        with BigVGAN's _from_pretrained() method (see §14 in ENGINEERING_DECISIONS.md).
        
        Args:
            model_name: One of "bigvgan_22k" or "bigvgan_24k"
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        import bigvgan as bigvgan_module
        from bigvgan.env import AttrDict
        from bigvgan.meldataset import get_mel_spectrogram as _get_mel
        from huggingface_hub import hf_hub_download
        import json as _json
        self._get_mel_spectrogram = _get_mel
        
        if model_name not in BIGVGAN_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(BIGVGAN_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.model_info = BIGVGAN_MODELS[model_name]
        
        # Determine device. BigVGAN is much faster on CUDA, but CPU remains
        # useful for short smoke tests and environments without a GPU.
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Manual loading: download config + weights, then construct model
        # Bypasses BigVGAN.from_pretrained() which relies on huggingface_hub
        # kwargs (proxies, resume_download) that newer hub versions removed.
        hf_id = self.model_info["hf_id"]
        print(f"  Loading BigVGAN model: {hf_id}")
        print(f"  (First run will download ~450MB from HuggingFace Hub)")
        
        t0 = time.time()
        config_path = hf_hub_download(repo_id=hf_id, filename='config.json')
        ckpt_path = hf_hub_download(repo_id=hf_id, filename='bigvgan_generator.pt')
        
        with open(config_path) as f:
            h = AttrDict(_json.load(f))
        
        self.generator = bigvgan_module.BigVGAN(h, use_cuda_kernel=False)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.generator.load_state_dict(ckpt['generator'])
        
        # Prepare for inference. remove_weight_norm() is a standard BigVGAN
        # inference step: it folds training-time normalization into the weights
        # and speeds up generation.
        self.generator.remove_weight_norm()
        self.generator = self.generator.eval().to(self.device)
        
        # Access config (same AttrDict-like interface as HiFi-GAN)
        self.h = self.generator.h
        
        elapsed = time.time() - t0
        print(f"  ✓ BigVGAN [{model_name}] loaded in {elapsed:.1f}s")
        print(f"    Device: {self.device}")
        print(f"    Config: sr={self.h.sampling_rate}, hop={self.h.hop_size}, "
              f"mels={self.h.num_mels}, fmax={self.h.fmax}")
    
    def _compute_native_mel(self, audio: np.ndarray, sr: int = None) -> torch.Tensor:
        """Compute mel-spectrogram using BigVGAN's native format.
        
        Uses BigVGAN's own get_mel_spectrogram to ensure exact compatibility.
        
        Args:
            audio: (samples,) numpy array in [-1, 1]
            sr: Sample rate (must match model config)
        
        Returns:
            mel: (1, n_mels, frames) tensor in BigVGAN's log-magnitude format
        """
        if sr is not None and sr != self.h.sampling_rate:
            raise ValueError(
                f"Sample rate mismatch: got {sr}, expected {self.h.sampling_rate}"
            )
        
        # Convert numpy audio to torch tensor: (1, samples). BigVGAN expects a
        # batch dimension even for a single audio clip.
        wav_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        # Compute mel using BigVGAN's exact method
        mel = self._get_mel_spectrogram(wav_tensor, self.h)
        
        return mel
    
    @torch.no_grad()
    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """Convert a mel-spectrogram tensor to audio waveform.
        
        Args:
            mel: (batch, n_mels, frames) or (n_mels, frames) mel tensor
        
        Returns:
            audio: (samples,) numpy array, float32 in [-1, 1]
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        
        mel = mel.to(self.device)
        
        with torch.inference_mode():
            # BigVGAN output: (batch, 1, samples)
            wav_gen = self.generator(mel)
        
        # Squeeze to 1D and move to CPU
        audio = wav_gen.squeeze().cpu().numpy()
        
        # Clip to [-1, 1] range so soundfile writing cannot overflow or wrap.
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def wav_to_wav(self, input_path: str, output_path: str) -> np.ndarray:
        """Round-trip test: WAV → native mel → generator → WAV.
        
        Uses BigVGAN's own mel computation for maximum fidelity.
        
        Args:
            input_path: Path to input WAV file
            output_path: Path to save reconstructed WAV
        
        Returns:
            audio: reconstructed audio as numpy array
        """
        print(f"  Loading audio: {Path(input_path).name}")
        
        # Load at the model's expected sample rate
        audio, sr = librosa.load(input_path, sr=self.h.sampling_rate, mono=True)
        print(f"  Duration: {len(audio)/sr:.2f}s, SR: {sr}")
        
        # Compute mel using BigVGAN's native format
        print(f"  Computing mel-spectrogram (BigVGAN native format)...")
        mel = self._compute_native_mel(audio, sr)
        print(f"  Mel shape: {mel.shape}")
        
        # Generate audio from mel
        print(f"  Running BigVGAN [{self.model_name}] generator...")
        t0 = time.time()
        reconstructed = self.mel_to_audio(mel)
        elapsed = time.time() - t0
        print(f"  Generation took {elapsed:.1f}s")
        
        # Trim to original length
        if len(reconstructed) > len(audio):
            reconstructed = reconstructed[:len(audio)]
        
        # Save as WAV using soundfile (avoids torchaudio/torchcodec issues)
        sf.write(output_path, reconstructed, sr)
        print(f"  ✓ Saved: {output_path}")
        print(f"  Output duration: {len(reconstructed)/sr:.2f}s")
        
        return reconstructed
    
    def segments_to_wav(self, mel_dir: str, output_path: str,
                        mel_min: float = -80.0, mel_max: float = 0.0) -> np.ndarray:
        """Reconstruct a full song from segmented mel tensors.
        
        Loads all mel segment .pt files, converts from our [-1,1] normalized
        format to BigVGAN's log-magnitude format, and concatenates.
        
        IMPORTANT: This conversion is approximate. For best results,
        use wav_to_wav() which uses native mel computation.
        
        Args:
            mel_dir: Directory containing *_mel.pt files
            output_path: Where to save the reconstructed WAV
            mel_min: dB minimum used in normalization
            mel_max: dB maximum used in normalization
        
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
            # Load our normalized mel: (n_mels, frames) in [-1, 1]
            mel_norm = torch.load(mel_file, map_location=self.device, weights_only=True)
            
            # Denormalize: [-1, 1] → dB scale
            mel_db = (mel_norm + 1.0) / 2.0 * (mel_max - mel_min) + mel_min
            
            # Convert from librosa power_to_db to log-magnitude format:
            # Same conversion as HiFi-GAN: dB to log-magnitude
            mel_log = mel_db * (np.log(10) / 20.0)
            
            if mel_log.dim() == 2:
                mel_log = mel_log.unsqueeze(0)
            
            segment_audio = self.mel_to_audio(mel_log)
            all_audio.append(segment_audio)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processed segment {i+1}/{len(mel_files)}")
        
        full_audio = np.concatenate(all_audio)
        sf.write(output_path, full_audio, self.h.sampling_rate)
        print(f"  ✓ Full song saved: {output_path}")
        print(f"  Duration: {len(full_audio)/self.h.sampling_rate:.2f}s")
        
        return full_audio
    
    def wav_to_wav_segmented(self, input_path: str, output_path: str,
                              segment_seconds: float = 5.0) -> np.ndarray:
        """Round-trip with segmentation (matches preprocessing pipeline).
        
        Processes audio in segments, which better simulates the full pipeline.
        
        Args:
            input_path: Path to input WAV
            output_path: Path to save output WAV
            segment_seconds: Segment duration (default 5.0s)
        
        Returns:
            audio: reconstructed audio
        """
        print(f"  Loading: {Path(input_path).name}")
        audio, sr = librosa.load(input_path, sr=self.h.sampling_rate, mono=True)
        
        segment_samples = int(segment_seconds * sr)
        n_segments = len(audio) // segment_samples
        
        print(f"  Processing {n_segments} segments of {segment_seconds}s each...")
        
        all_audio = []
        t0 = time.time()
        for i in range(n_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = audio[start:end]
            
            mel = self._compute_native_mel(segment, sr)
            reconstructed = self.mel_to_audio(mel)
            reconstructed = reconstructed[:segment_samples]
            all_audio.append(reconstructed)
        
        full_audio = np.concatenate(all_audio)
        elapsed = time.time() - t0
        
        sf.write(output_path, full_audio, sr)
        print(f"  ✓ Saved: {output_path} ({len(full_audio)/sr:.2f}s, {elapsed:.1f}s)")
        return full_audio


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface for BigVGAN vocoder inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BigVGAN v2 Vocoder Inference")
    parser.add_argument('input', type=str, help='Input WAV file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output WAV path (default: input_bigvgan_<model>.wav)')
    parser.add_argument('--model', type=str, default='bigvgan_22k',
                       choices=list(BIGVGAN_MODELS.keys()),
                       help='BigVGAN model variant (default: bigvgan_22k)')
    parser.add_argument('--segmented', action='store_true',
                       help='Process in 5s segments')
    
    args = parser.parse_args()
    
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_{args.model}_reconstructed.wav")
    
    vocoder = BigVGANVocoder(model_name=args.model)
    
    if args.segmented:
        vocoder.wav_to_wav_segmented(args.input, args.output)
    else:
        vocoder.wav_to_wav(args.input, args.output)


if __name__ == "__main__":
    main()
