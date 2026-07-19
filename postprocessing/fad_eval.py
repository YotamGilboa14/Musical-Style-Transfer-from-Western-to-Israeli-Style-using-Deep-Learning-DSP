"""
Fréchet Audio Distance (FAD) Evaluation Module
================================================
Computes FAD between synthesized audio and real reference audio
to measure realism of style-transferred outputs.

FAD is the audio analog of FID (Fréchet Inception Distance).
It computes the Fréchet distance between two multivariate Gaussian
distributions fitted to audio embeddings from a VGGish-like model.

**Embedding model:** VGGish when pretrained weights can be loaded; otherwise a
deterministic randomly initialized VGGish-like network is used as a fallback.
  - Input: 16kHz mono audio → log-mel spectrograms (64 bins, 0.96s windows)
  - Output: 128-dimensional embedding per window
    - Reports should state whether pretrained or fallback embeddings were used

**FAD formula:**
  FAD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^½)
  
  where (μ_r, Σ_r) = statistics of real audio embeddings
        (μ_g, Σ_g) = statistics of generated audio embeddings

Lower FAD = more realistic generated audio. FAD=0 means identical distributions.

Usage:
    from postprocessing.fad_eval import compute_fad, evaluate_all_fad

    # Compare two directories of WAV files
    fad_score = compute_fad('real_wavs/', 'generated_wavs/')

    # Full evaluation: synthesized violin vs real URMP violin
    results = evaluate_all_fad(
        real_dir='models/tt_vae_gan/data/data_urmp/spkr_2/',
        generated_dir='benchmark_output/',
    )

Author: Yotam & Gal — StyleTransfer Music Project
Date: February 2026
"""

import os
import sys
import tempfile
import warnings
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import librosa
from scipy import linalg
from scipy.linalg import LinAlgWarning


# ─── VGGish-style Mel Spectrogram for Embeddings ────────────────────────────────

# VGGish parameters (from Google's VGGish)
VGGISH_SR = 16000
VGGISH_N_MELS = 64
VGGISH_WINDOW_S = 0.025   # 25ms
VGGISH_HOP_S = 0.010      # 10ms
VGGISH_SEGMENT_S = 0.96   # 0.96s per embedding window
VGGISH_SEGMENT_FRAMES = 96  # 96 frames × 10ms = 0.96s
VGGISH_EMBED_DIM = 128


def compute_vggish_mel(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Compute VGGish-compatible log-mel spectrograms from audio.
    
    Resamples to 16kHz, computes 64-bin log-mel, segments into
    0.96s windows (96 frames).
    
    Args:
        audio: mono audio, any sample rate
        sr: sample rate of input audio
    
    Returns:
        log_mel_segments: shape (n_segments, 96, 64), float32
    """
    # Resample to 16kHz because VGGish embeddings expect a fixed audio scale.
    # FAD compares embedding distributions, so real and generated audio must go
    # through exactly the same preprocessing path.
    if sr != VGGISH_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=VGGISH_SR)
    
    # Compute mel spectrogram
    n_fft = int(VGGISH_WINDOW_S * VGGISH_SR)  # 400
    hop_length = int(VGGISH_HOP_S * VGGISH_SR)  # 160
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=VGGISH_SR,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=VGGISH_N_MELS,
        fmin=125.0, fmax=7500.0,
    )
    
    # Log-mel (stabilized)
    log_mel = np.log(np.maximum(mel_spec, 1e-7)).T  # (time, n_mels)
    
    # Segment into 96-frame (0.96s) windows. Each window becomes one embedding;
    # the final FAD score compares the cloud of real embeddings with the cloud
    # of generated embeddings.
    n_frames = log_mel.shape[0]
    n_segments = n_frames // VGGISH_SEGMENT_FRAMES
    
    if n_segments == 0:
        # Audio too short — pad to at least one segment
        pad_len = VGGISH_SEGMENT_FRAMES - n_frames
        log_mel = np.pad(log_mel, ((0, pad_len), (0, 0)), mode='constant')
        n_segments = 1
    
    segments = []
    for i in range(n_segments):
        start = i * VGGISH_SEGMENT_FRAMES
        end = start + VGGISH_SEGMENT_FRAMES
        segments.append(log_mel[start:end])
    
    return np.array(segments, dtype=np.float32)  # (n_segments, 96, 64)


# ─── Lightweight VGGish-like Embedding Network ──────────────────────────────────

class VGGishEmbedder(nn.Module):
    """
    Lightweight VGGish-like network for audio embedding extraction.
    
    Architecture mirrors Google's VGGish but without pre-trained weights.
    Uses random initialization — embeddings are still meaningful for
    FAD computation because the convolutional features capture
    spectral structure regardless of specific training.
    
    For production use, load pre-trained VGGish weights from:
    https://github.com/harritaylor/torchvggish
    
    Input: (batch, 1, 96, 64) log-mel spectrogram
    Output: (batch, 128) embedding
    """
    
    def __init__(self):
        """Create the convolutional feature extractor and 128-D projection head."""
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.embedding = nn.Sequential(
            nn.Linear(512 * 6 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, VGGISH_EMBED_DIM),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 96, 64) log-mel spectrogram
        Returns:
            (batch, 128) embedding
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x


_embedder_cache = {}

def get_embedder(device: torch.device = None, use_pretrained: bool = True) -> VGGishEmbedder:
    """
    Get or create a VGGish embedder.
    
    Attempts to load torchvggish pre-trained weights first.
    Falls back to randomly initialized (still valid for FAD comparison
    as long as the same embedder is used for both real and generated).
    
    Args:
        device: torch device
        use_pretrained: try to load pre-trained VGGish weights
    
    Returns:
        VGGishEmbedder in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cache_key = str(device)
    if cache_key in _embedder_cache:
        return _embedder_cache[cache_key]
    
    model = None
    
    # Try loading pre-trained VGGish via torch.hub. If this fails, the fallback
    # is deterministic so repeated reports are at least internally comparable,
    # but final claims must state that fallback embeddings were used.
    if use_pretrained:
        try:
            model = torch.hub.load('harritaylor/torchvggish', 'vggish',
                                   trust_repo=True)
            model.postprocess = False  # Get raw embeddings, not PCA'd
            model.preprocess = False   # We do our own mel computation
            model._fad_is_pretrained = True
            model._fad_embedder_label = 'VGGish (pre-trained, torch.hub)'
            print("  [FAD] Loaded pre-trained VGGish from torch.hub")
        except Exception as e:
            print(f"  [FAD] Could not load pre-trained VGGish: {e}")
            print("  [FAD] Using randomly initialized VGGish-like network")
            model = None
    
    if model is None:
        model = VGGishEmbedder()
        # Set deterministic seed for reproducibility
        torch.manual_seed(42)
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        model._fad_is_pretrained = False
        model._fad_embedder_label = 'VGGish-like (random init, deterministic seed)'
        print("  [FAD] Using VGGish-like embedder (random init, deterministic seed)")
    
    model = model.to(device)
    model.eval()
    _embedder_cache[cache_key] = model
    return model


# ─── Embedding Extraction ───────────────────────────────────────────────────────

def extract_embeddings_from_audio(audio: np.ndarray, sr: int = 22050,
                                  embedder: nn.Module = None,
                                  device: torch.device = None,
                                  batch_size: int = 32) -> np.ndarray:
    """
    Extract VGGish embeddings from a single audio signal.
    
    Args:
        audio: mono audio array
        sr: sample rate
        embedder: embedding model (or None to auto-create)
        device: torch device
        batch_size: batch size for embedding extraction
    
    Returns:
        embeddings: shape (n_segments, 128), float32
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if embedder is None:
        embedder = get_embedder(device)
    
    # Compute log-mel segments
    mel_segments = compute_vggish_mel(audio, sr)  # (n_seg, 96, 64)
    
    # Extract embeddings in batches to avoid GPU/CPU memory spikes on long
    # evaluation folders.
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(mel_segments), batch_size):
            batch = mel_segments[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().unsqueeze(1)  # (B, 1, 96, 64)
            batch_tensor = batch_tensor.to(device)
            
            embeds = embedder(batch_tensor)
            all_embeds.append(embeds.cpu().numpy())
    
    return np.concatenate(all_embeds, axis=0)  # (n_segments, 128)


def extract_embeddings_from_directory(wav_dir: str,
                                      embedder: nn.Module = None,
                                      device: torch.device = None,
                                      sr: int = 22050,
                                      batch_size: int = 32) -> np.ndarray:
    """
    Extract VGGish embeddings from all WAV files in a directory.
    
    Args:
        wav_dir: path to directory containing WAV files
        embedder: embedding model
        device: torch device
        sr: sample rate to load audio at
        batch_size: batch size for embedding extraction
    
    Returns:
        embeddings: shape (total_segments, 128), float32
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if embedder is None:
        embedder = get_embedder(device)
    
    wav_dir = Path(wav_dir)
    wav_files = sorted(list(wav_dir.glob("*.wav")))
    
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {wav_dir}")
    
    all_embeddings = []
    for wav_path in wav_files:
        try:
            audio, file_sr = librosa.load(str(wav_path), sr=sr, mono=True)
        except Exception as e:
            print(f"  WARNING: Could not load {wav_path.name}: {e}")
            continue
        
        embeds = extract_embeddings_from_audio(
            audio, sr=sr, embedder=embedder, device=device,
            batch_size=batch_size)
        all_embeddings.append(embeds)
    
    if not all_embeddings:
        raise ValueError(f"No valid audio files found in {wav_dir}")
    
    return np.concatenate(all_embeddings, axis=0)


# ─── Fréchet Distance Computation ───────────────────────────────────────────────

def compute_statistics(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of embedding distribution.
    
    Args:
        embeddings: shape (n_samples, embed_dim)
    
    Returns:
        (mu, sigma): mean vector and covariance matrix
    """
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Compute the Fréchet distance between two multivariate Gaussians.
    
    FAD = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁·Σ₂)^½)
    
    Args:
        mu1, sigma1: statistics of distribution 1 (e.g., real)
        mu2, sigma2: statistics of distribution 2 (e.g., generated)
    
    Returns:
        Fréchet distance (float, ≥ 0)
    """
    diff = mu1 - mu2

    # Product of covariances.
    # NOTE: scipy emits `LinAlgWarning: Matrix is singular` whenever sigma1·sigma2
    # is rank-deficient — common with small evaluation sets (e.g. group-FAD where
    # each composer group has only a handful of clips, or when real == generated).
    # The warning is benign here: we detect non-finite output below and fall back
    # to an epsilon-regularised sqrtm. We suppress the warning so test logs stay
    # readable; the epsilon fallback still prints if it actually triggers.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=LinAlgWarning)
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Handle numerical instability
    if not np.isfinite(covmean).all():
        print("  WARNING: sqrtm produced non-finite values, adding epsilon")
        eps = np.eye(sigma1.shape[0]) * 1e-6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=LinAlgWarning)
            covmean, _ = linalg.sqrtm((sigma1 + eps) @ (sigma2 + eps), disp=False)
    
    # Remove imaginary components (numerical artifacts)
    if np.iscomplexobj(covmean):
        if np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            print("  WARNING: Large imaginary component in sqrtm result")
            covmean = covmean.real
    
    fad = (diff @ diff +
           np.trace(sigma1) + np.trace(sigma2) -
           2 * np.trace(covmean))
    
    return float(np.maximum(fad, 0))  # Clamp to ≥0 for numerical stability


# ─── High-Level API ─────────────────────────────────────────────────────────────

def compute_fad(real_dir: str, generated_dir: str,
                sr: int = 22050,
                use_pretrained: bool = True) -> float:
    """
    Compute FAD between real and generated audio directories.
    
    Args:
        real_dir: directory with real/reference WAV files
        generated_dir: directory with generated/synthesized WAV files
        sr: sample rate for loading audio
        use_pretrained: try to use pre-trained VGGish
    
    Returns:
        FAD score (float, lower = better)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = get_embedder(device, use_pretrained=use_pretrained)
    
    print(f"  Extracting embeddings from real audio ({real_dir})...")
    real_embeds = extract_embeddings_from_directory(
        real_dir, embedder=embedder, device=device, sr=sr)
    print(f"  → {real_embeds.shape[0]} embedding vectors from "
          f"{len(list(Path(real_dir).glob('*.wav')))} files")
    
    print(f"  Extracting embeddings from generated audio ({generated_dir})...")
    gen_embeds = extract_embeddings_from_directory(
        generated_dir, embedder=embedder, device=device, sr=sr)
    print(f"  → {gen_embeds.shape[0]} embedding vectors from "
          f"{len(list(Path(generated_dir).glob('*.wav')))} files")
    
    # Compute statistics
    mu_real, sigma_real = compute_statistics(real_embeds)
    mu_gen, sigma_gen = compute_statistics(gen_embeds)
    
    # Compute FAD
    fad = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    print(f"\n  FAD Score: {fad:.4f}")
    print(f"  (Lower = more realistic. FAD=0 means identical distributions)")
    
    return fad


def evaluate_all_fad(real_dir: str, generated_dir: str,
                     sr: int = 22050,
                     use_pretrained: bool = True) -> Dict[str, Any]:
    """
    Full FAD evaluation with detailed report.
    
    Args:
        real_dir: directory with real/reference WAV files
        generated_dir: directory with generated/synthesized WAV files  
        sr: sample rate
        use_pretrained: try pre-trained VGGish
    
    Returns:
        dict with:
            - fad_score: float
            - n_real_files: int
            - n_gen_files: int
            - n_real_embeddings: int
            - n_gen_embeddings: int
            - real_dir, generated_dir: paths
            - embedding_model: str
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = get_embedder(device, use_pretrained=use_pretrained)
    
    print()
    print("=" * 60)
    print("  FRÉCHET AUDIO DISTANCE (FAD) EVALUATION")
    print("=" * 60)
    
    real_files = sorted(Path(real_dir).glob("*.wav"))
    gen_files = sorted(Path(generated_dir).glob("*.wav"))
    
    print(f"\n  Real audio:      {real_dir}")
    print(f"    Files: {len(real_files)}")
    for f in real_files:
        print(f"      {f.name}")
    
    print(f"\n  Generated audio: {generated_dir}")
    print(f"    Files: {len(gen_files)}")
    for f in gen_files:
        print(f"      {f.name}")
    
    # Extract embeddings
    print("\n  Extracting embeddings...")
    real_embeds = extract_embeddings_from_directory(
        real_dir, embedder=embedder, device=device, sr=sr)
    gen_embeds = extract_embeddings_from_directory(
        generated_dir, embedder=embedder, device=device, sr=sr)
    
    print(f"  Real embeddings:      {real_embeds.shape}")
    print(f"  Generated embeddings: {gen_embeds.shape}")
    
    # Statistics
    mu_real, sigma_real = compute_statistics(real_embeds)
    mu_gen, sigma_gen = compute_statistics(gen_embeds)
    
    # FAD
    fad = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    # Embedding distance (cosine similarity of means)
    cos_sim = (mu_real @ mu_gen) / (np.linalg.norm(mu_real) * np.linalg.norm(mu_gen) + 1e-8)
    
    print(f"\n  ─── Results ───")
    print(f"  FAD Score:           {fad:.4f}")
    print(f"  Mean Cosine Sim:     {cos_sim:.4f}")
    print(f"  Real μ norm:         {np.linalg.norm(mu_real):.4f}")
    print(f"  Generated μ norm:    {np.linalg.norm(mu_gen):.4f}")
    print()
    
    # Interpretation
    if fad < 5:
        quality = "Excellent — near-indistinguishable from real"
    elif fad < 15:
        quality = "Good — realistic with minor differences"
    elif fad < 50:
        quality = "Moderate — audible differences from real"
    elif fad < 150:
        quality = "Poor — significant quality gap"
    else:
        quality = "Very poor — distribution far from real audio"
    print(f"  Interpretation: {quality}")
    print("=" * 60)
    
    # Determine embedding model name from the embedder's own flag (set in
    # `get_embedder`). This correctly reports "pre-trained" vs "random init
    # fallback" even when torch.hub silently failed and `use_pretrained=True`
    # was requested but not honored.
    embedder_label = getattr(embedder, '_fad_embedder_label', None)
    is_pretrained  = getattr(embedder, '_fad_is_pretrained', None)
    if embedder_label is None:
        # Conservative fallback: derive from the requested flag.
        embedder_label = 'VGGish (pre-trained)' if use_pretrained else 'VGGish-like (random init)'
        is_pretrained  = bool(use_pretrained)
    
    return {
        'fad_score': fad,
        'cosine_similarity': float(cos_sim),
        'n_real_files': len(real_files),
        'n_gen_files': len(gen_files),
        'n_real_embeddings': real_embeds.shape[0],
        'n_gen_embeddings': gen_embeds.shape[0],
        'real_dir': str(real_dir),
        'generated_dir': str(generated_dir),
        'embedding_model': embedder_label,
        'embedding_is_pretrained': bool(is_pretrained),
        'embed_dim': VGGISH_EMBED_DIM,
        'quality_interpretation': quality,
    }


# ─── Group-FAD ───────────────────────────────────────────────────────────────────

def compute_group_fad(
    real_dir: str,
    generated_dir: str,
    version_id: int = 0,
    version_manifest_csv: Optional[str] = None,
    sr: int = 22050,
    use_pretrained: bool = True,
) -> Dict[str, Any]:
    """
    Compute Group-FAD: FAD scoped to a single version/style group.

    For historical single-version runs this is identical to All-FAD because
    all audio belongs to one real version. For the active Slakh+Israeli setup,
    call this once per real version, such as Slakh ``version_id=0`` and Israeli
    ``version_id=1``. Pass ``version_manifest_csv`` to restrict both
    directories to WAV files whose ``segment_path`` stem matches rows with the
    requested ``version_id``.

    Args:
        real_dir:               directory with real/reference WAV files
        generated_dir:          directory with generated WAV files
        version_id:             version/style label to filter on (default: 0)
        version_manifest_csv:   optional path to a manifest CSV with a
                                ``version_id`` column; if provided, only WAV
                                files whose stem matches a manifest row with
                                the requested version_id are included.
                                If None, all files in both dirs are used.
        sr:                     sample rate for loading audio
        use_pretrained:         try pre-trained VGGish weights

    Returns:
        dict with keys:
            group_fad       (float) — the FAD score for this group
            version_id      (int)
            n_real_files    (int)
            n_gen_files     (int)
            fad_details     (dict) — full result from evaluate_all_fad()
    """
    import shutil

    real_dir = Path(real_dir)
    generated_dir = Path(generated_dir)

    if version_manifest_csv is None:
        # No filtering — use all files (correct for single-version model)
        details = evaluate_all_fad(
            str(real_dir), str(generated_dir),
            sr=sr, use_pretrained=use_pretrained,
        )
        return {
            "group_fad": details["fad_score"],
            "version_id": version_id,
            "n_real_files": details["n_real_files"],
            "n_gen_files": details["n_gen_files"],
            "fad_details": details,
        }

    # Build set of allowed WAV stems for the requested version_id
    import pandas as pd
    manifest = pd.read_csv(version_manifest_csv)
    if "version_id" not in manifest.columns or "segment_path" not in manifest.columns:
        raise ValueError(
            "version_manifest_csv must have 'version_id' and 'segment_path' columns"
        )
    allowed_stems = set(
        Path(p).stem
        for p in manifest.loc[manifest["version_id"] == version_id, "segment_path"]
    )

    if not allowed_stems:
        raise ValueError(
            f"No segments found for version_id={version_id} in {version_manifest_csv}"
        )

    # Copy matching WAV files to temporary directories, then compute FAD
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_real = Path(tmp_root) / "real"
        tmp_gen = Path(tmp_root) / "gen"
        tmp_real.mkdir()
        tmp_gen.mkdir()

        for wav in real_dir.glob("*.wav"):
            if wav.stem in allowed_stems:
                shutil.copy2(wav, tmp_real / wav.name)
        for wav in generated_dir.glob("*.wav"):
            if wav.stem in allowed_stems:
                shutil.copy2(wav, tmp_gen / wav.name)

        n_real = len(list(tmp_real.glob("*.wav")))
        n_gen = len(list(tmp_gen.glob("*.wav")))

        if n_real == 0:
            raise FileNotFoundError(
                f"No real WAV files matched version_id={version_id} in {real_dir}"
            )
        if n_gen == 0:
            raise FileNotFoundError(
                f"No generated WAV files matched version_id={version_id} in {generated_dir}"
            )

        details = evaluate_all_fad(
            str(tmp_real), str(tmp_gen),
            sr=sr, use_pretrained=use_pretrained,
        )

    return {
        "group_fad": details["fad_score"],
        "version_id": version_id,
        "n_real_files": n_real,
        "n_gen_files": n_gen,
        "fad_details": details,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute Fréchet Audio Distance between two sets of audio files")
    parser.add_argument('--real', type=str, required=True,
                        help='Directory with real/reference WAV files')
    parser.add_argument('--generated', type=str, required=True,
                        help='Directory with generated/synthesized WAV files')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sample rate for loading audio (default: 22050)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Skip loading pre-trained VGGish weights')
    args = parser.parse_args()
    
    import json
    results = evaluate_all_fad(
        real_dir=args.real,
        generated_dir=args.generated,
        sr=args.sr,
        use_pretrained=not args.no_pretrained,
    )
    
    # Save results to JSON
    output_path = Path(args.generated) / 'fad_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")
