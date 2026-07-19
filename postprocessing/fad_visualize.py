"""
FAD Visualization — Gaussian Distribution Plot
=================================================
Visualizes the Fréchet Audio Distance between real (violin) and
generated (trumpet→violin transfer) audio embeddings.

Produces a 2D PCA projection showing:
- Scatter of all real and generated embeddings
- Gaussian ellipses (2σ) for each distribution
- FAD score and cosine similarity displayed on the plot
- Per-sample legend

Usage:
    python -m postprocessing.fad_visualize \
        --real benchmark_output/fad_real \
        --generated benchmark_output/fad_generated \
        --output benchmark_output/fad_visualization.png

Author: Yotam & Gal — StyleTransfer Music Project
Date: February 2026
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from pathlib import Path
from sklearn.decomposition import PCA
from scipy import linalg

import torch
import librosa

# Import from sibling module
from postprocessing.fad_eval import (
    get_embedder,
    extract_embeddings_from_audio,
    compute_statistics,
    frechet_distance,
    VGGISH_EMBED_DIM,
)


def _draw_confidence_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Draw a 2D Gaussian confidence ellipse on ax."""
    # Eigenvalues and eigenvectors of the 2x2 covariance
    vals, vecs = np.linalg.eigh(cov)
    # Sort descending
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Angle of the major axis
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height = 2 * n_std * sqrt(eigenvalue)
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def visualize_fad(real_dir: str, generated_dir: str,
                  output_path: str = None,
                  sr: int = 22050,
                  title: str = None) -> str:
    """
    Create FAD visualization with Gaussian ellipses in PCA space.

    Args:
        real_dir: directory with real WAV files
        generated_dir: directory with generated WAV files
        output_path: where to save the PNG
        sr: sample rate
        title: custom plot title (defaults to a generic real-vs-generated label).
            Pass the song / experiment name to override the legacy
            "Trumpet→Violin" wording.

    Returns:
        path to saved PNG
    """
    real_dir = Path(real_dir)
    generated_dir = Path(generated_dir)

    if output_path is None:
        output_path = str(generated_dir / 'fad_visualization.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = get_embedder(device, use_pretrained=True)

    # ── Extract per-file embeddings ──────────────────────────────────────
    real_files = sorted(real_dir.glob('*.wav'))
    gen_files = sorted([f for f in generated_dir.glob('*.wav')])

    print(f"  Extracting embeddings from {len(real_files)} real files...")
    real_per_file = {}
    all_real = []
    for f in real_files:
        audio, _ = librosa.load(str(f), sr=sr, mono=True)
        emb = extract_embeddings_from_audio(audio, sr=sr, embedder=embedder, device=device)
        real_per_file[f.stem] = emb
        all_real.append(emb)
    all_real = np.concatenate(all_real, axis=0)

    print(f"  Extracting embeddings from {len(gen_files)} generated files...")
    gen_per_file = {}
    all_gen = []
    for f in gen_files:
        audio, _ = librosa.load(str(f), sr=sr, mono=True)
        emb = extract_embeddings_from_audio(audio, sr=sr, embedder=embedder, device=device)
        gen_per_file[f.stem] = emb
        all_gen.append(emb)
    all_gen = np.concatenate(all_gen, axis=0)

    print(f"  Real embeddings:      {all_real.shape}")
    print(f"  Generated embeddings: {all_gen.shape}")

    # ── Compute FAD in full 128-D space ──────────────────────────────────
    mu_real, sigma_real = compute_statistics(all_real)
    mu_gen, sigma_gen = compute_statistics(all_gen)
    fad_score = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    cos_sim = float((mu_real @ mu_gen) /
                    (np.linalg.norm(mu_real) * np.linalg.norm(mu_gen) + 1e-8))

    print(f"  FAD = {fad_score:.4f},  Cosine Sim = {cos_sim:.4f}")

    # ── PCA to 2D ────────────────────────────────────────────────────────
    combined = np.vstack([all_real, all_gen])
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    explained = pca.explained_variance_ratio_

    real_2d = combined_2d[:len(all_real)]
    gen_2d = combined_2d[len(all_real):]

    # ── Per-file 2D stats ────────────────────────────────────────────────
    # Split real_2d back into per-file
    idx = 0
    real_2d_per_file = {}
    for name, emb in real_per_file.items():
        n = emb.shape[0]
        real_2d_per_file[name] = real_2d[idx:idx+n]
        idx += n

    idx = 0
    gen_2d_per_file = {}
    for name, emb in gen_per_file.items():
        n = emb.shape[0]
        gen_2d_per_file[name] = gen_2d[idx:idx+n]
        idx += n

    # ── Gaussian stats in 2D ─────────────────────────────────────────────
    mu_real_2d = np.mean(real_2d, axis=0)
    cov_real_2d = np.cov(real_2d, rowvar=False)
    mu_gen_2d = np.mean(gen_2d, axis=0)
    cov_gen_2d = np.cov(gen_2d, rowvar=False)

    # ── PLOT ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))

    # Color palettes
    real_colors = ['#2196F3', '#1565C0', '#0D47A1', '#42A5F5']  # blues
    gen_colors = ['#F44336', '#E91E63', '#FF5722']  # reds

    # Scatter real per-file
    for i, (name, pts) in enumerate(real_2d_per_file.items()):
        short = name.replace('AuSep_', '').replace('_vn_', ' vn ')
        c = real_colors[i % len(real_colors)]
        ax.scatter(pts[:, 0], pts[:, 1], c=c, alpha=0.25, s=12,
                   label=f'Real: {short} ({len(pts)})')

    # Scatter generated per-file
    for i, (name, pts) in enumerate(gen_2d_per_file.items()):
        short = name.replace('AuSep_', '').replace('_tpt_', ' tpt→vn ').\
            replace('_transferred_violin', '')
        c = gen_colors[i % len(gen_colors)]
        ax.scatter(pts[:, 0], pts[:, 1], c=c, alpha=0.35, s=18, marker='D',
                   label=f'Gen: {short} ({len(pts)})')

    # Gaussian ellipses (2σ and 3σ)
    _draw_confidence_ellipse(ax, mu_real_2d, cov_real_2d, n_std=2.0,
                             edgecolor='#1565C0', linewidth=2.5,
                             facecolor='#2196F3', alpha=0.08,
                             linestyle='-', label='Real 2σ')
    _draw_confidence_ellipse(ax, mu_real_2d, cov_real_2d, n_std=3.0,
                             edgecolor='#1565C0', linewidth=1.5,
                             facecolor='none', alpha=0.3,
                             linestyle='--')
    _draw_confidence_ellipse(ax, mu_gen_2d, cov_gen_2d, n_std=2.0,
                             edgecolor='#C62828', linewidth=2.5,
                             facecolor='#F44336', alpha=0.08,
                             linestyle='-', label='Generated 2σ')
    _draw_confidence_ellipse(ax, mu_gen_2d, cov_gen_2d, n_std=3.0,
                             edgecolor='#C62828', linewidth=1.5,
                             facecolor='none', alpha=0.3,
                             linestyle='--')

    # Draw line between means
    ax.plot([mu_real_2d[0], mu_gen_2d[0]], [mu_real_2d[1], mu_gen_2d[1]],
            'k-', linewidth=1.5, alpha=0.6, zorder=5)
    ax.plot(*mu_real_2d, 'o', color='#1565C0', markersize=10, zorder=6,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(*mu_gen_2d, 'D', color='#C62828', markersize=10, zorder=6,
            markeredgecolor='white', markeredgewidth=1.5)

    # Midpoint annotation
    mid = (mu_real_2d + mu_gen_2d) / 2
    euclidean_2d = np.linalg.norm(mu_real_2d - mu_gen_2d)

    # ── Text block with metrics ──────────────────────────────────────────
    if fad_score < 5:
        quality = "Excellent"
    elif fad_score < 15:
        quality = "Good"
    elif fad_score < 50:
        quality = "Moderate"
    else:
        quality = "Poor"

    embedder_label = getattr(embedder, '_fad_embedder_label',
                             'VGGish (pre-trained)')
    textstr = (
        f"FAD Score: {fad_score:.4f}  ({quality})\n"
        f"Cosine Similarity: {cos_sim:.4f}\n"
        f"PCA Explained Var: {explained[0]:.1%} + {explained[1]:.1%}\n"
        f"────────────────────────\n"
        f"Real:  {all_real.shape[0]} embeddings from {len(real_files)} files\n"
        f"Gen:   {all_gen.shape[0]} embeddings from {len(gen_files)} files\n"
        f"Embed dim: {VGGISH_EMBED_DIM}  ({embedder_label})"
    )

    props = dict(boxstyle='round,pad=0.6', facecolor='white',
                 edgecolor='#333333', alpha=0.92)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    # FAD distance label along line
    ax.annotate(f'  FAD = {fad_score:.4f}', xy=mid,
                fontsize=11, fontweight='bold', color='#333',
                ha='left', va='bottom')

    # Legend & labels
    ax.set_xlabel(f'PC1 ({explained[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained[1]:.1%} variance)', fontsize=12)
    plot_title = title if title else (
        'Fréchet Audio Distance — Real vs Generated\n'
        'VGGish Embedding Space (PCA Projection)'
    )
    ax.set_title(plot_title, fontsize=14, fontweight='bold')

    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9,
              ncol=1, borderpad=0.8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ FAD visualization saved: {output_path}")
    return output_path


def visualize_fad_1d(real_dir: str, generated_dir: str,
                     output_path: str = None,
                     sr: int = 22050) -> str:
    """
    Create a simple 1D Gaussian bell-curve FAD visualization.

    Projects 128-D VGGish embeddings onto the Fisher discriminant axis
    (the line connecting the two distribution means) and plots
    overlapping Gaussian density curves with shaded overlap.

    Args:
        real_dir: directory with real violin WAV files
        generated_dir: directory with generated (transferred) WAV files
        output_path: where to save the PNG
        sr: sample rate

    Returns:
        path to saved PNG
    """
    from scipy.stats import norm

    real_dir = Path(real_dir)
    generated_dir = Path(generated_dir)

    if output_path is None:
        output_path = str(generated_dir / 'fad_visualization_1d.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = get_embedder(device, use_pretrained=True)

    # ── Extract embeddings ───────────────────────────────────────────────
    real_files = sorted(real_dir.glob('*.wav'))
    gen_files = sorted(generated_dir.glob('*.wav'))

    print(f"  [1D] Extracting embeddings from {len(real_files)} real files...")
    all_real = []
    for f in real_files:
        audio, _ = librosa.load(str(f), sr=sr, mono=True)
        emb = extract_embeddings_from_audio(audio, sr=sr, embedder=embedder, device=device)
        all_real.append(emb)
    all_real = np.concatenate(all_real, axis=0)

    print(f"  [1D] Extracting embeddings from {len(gen_files)} generated files...")
    all_gen = []
    for f in gen_files:
        audio, _ = librosa.load(str(f), sr=sr, mono=True)
        emb = extract_embeddings_from_audio(audio, sr=sr, embedder=embedder, device=device)
        all_gen.append(emb)
    all_gen = np.concatenate(all_gen, axis=0)

    # ── Full 128-D FAD ───────────────────────────────────────────────────
    mu_real, sigma_real = compute_statistics(all_real)
    mu_gen, sigma_gen = compute_statistics(all_gen)
    fad_score = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    cos_sim = float((mu_real @ mu_gen) /
                    (np.linalg.norm(mu_real) * np.linalg.norm(mu_gen) + 1e-8))

    print(f"  [1D] FAD = {fad_score:.4f},  Cosine Sim = {cos_sim:.4f}")

    # ── Project onto Fisher discriminant axis ────────────────────────────
    direction = mu_gen - mu_real
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-10:
        direction = np.random.randn(len(direction))
        dir_norm = np.linalg.norm(direction)
    direction = direction / dir_norm

    proj_real = all_real @ direction
    proj_gen = all_gen @ direction

    # ── Fit 1D Gaussians ─────────────────────────────────────────────────
    mu_r, std_r = np.mean(proj_real), np.std(proj_real)
    mu_g, std_g = np.mean(proj_gen), np.std(proj_gen)

    # X range covering both distributions
    lo = min(mu_r - 4 * std_r, mu_g - 4 * std_g)
    hi = max(mu_r + 4 * std_r, mu_g + 4 * std_g)
    x = np.linspace(lo, hi, 500)

    pdf_real = norm.pdf(x, mu_r, std_r)
    pdf_gen = norm.pdf(x, mu_g, std_g)

    # ── PLOT ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # Density curves
    ax.fill_between(x, pdf_real, alpha=0.25, color='#2196F3')
    ax.plot(x, pdf_real, color='#1565C0', linewidth=2.5,
            label=f'Real Violin  (n={len(proj_real)}, μ={mu_r:.2f}, σ={std_r:.2f})')

    ax.fill_between(x, pdf_gen, alpha=0.25, color='#F44336')
    ax.plot(x, pdf_gen, color='#C62828', linewidth=2.5,
            label=f'Generated  (n={len(proj_gen)}, μ={mu_g:.2f}, σ={std_g:.2f})')

    # Mean markers
    ax.axvline(mu_r, color='#1565C0', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.axvline(mu_g, color='#C62828', linewidth=1.5, linestyle='--', alpha=0.7)

    # Distance arrow between means
    y_arrow = max(max(pdf_real), max(pdf_gen)) * 0.85
    ax.annotate('', xy=(mu_g, y_arrow), xytext=(mu_r, y_arrow),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=2))
    mid_x = (mu_r + mu_g) / 2
    ax.text(mid_x, y_arrow * 1.05,
            f'Separation = {abs(mu_g - mu_r):.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333')

    # ── Metrics text box ─────────────────────────────────────────────────
    if fad_score < 5:
        quality = "Excellent"
    elif fad_score < 15:
        quality = "Good"
    elif fad_score < 50:
        quality = "Moderate"
    else:
        quality = "Poor"

    textstr = (
        f"FAD Score: {fad_score:.4f}  ({quality})\n"
        f"Cosine Similarity: {cos_sim:.4f}\n"
        f"────────────────────────\n"
        f"Real:  {all_real.shape[0]} embeddings\n"
        f"Gen:   {all_gen.shape[0]} embeddings\n"
        f"Projection: Fisher axis (128-D → 1-D)"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white',
                 edgecolor='#333333', alpha=0.92)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    ax.set_xlabel('Projection onto Fisher Discriminant Axis', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Fréchet Audio Distance — Real vs Generated Distribution\n'
                 'VGGish Embedding Space (1-D Fisher Projection)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ FAD 1D visualization saved: {output_path}")
    return output_path


# ─── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Visualize FAD as Gaussian distribution plot")
    parser.add_argument('--real', type=str, default='benchmark_output/fad_real',
                        help='Directory with real violin WAV files')
    parser.add_argument('--generated', type=str, default='benchmark_output/fad_generated',
                        help='Directory with generated (transferred) WAV files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path (default: <generated>/fad_visualization.png)')
    parser.add_argument('--mode', choices=['pca', '1d', 'both'], default='both',
                        help='Visualization mode: pca (2D ellipses), 1d (bell curves), or both')
    args = parser.parse_args()

    if args.mode in ('pca', 'both'):
        visualize_fad(
            real_dir=args.real,
            generated_dir=args.generated,
            output_path=args.output,
        )
    if args.mode in ('1d', 'both'):
        out_1d = args.output
        if out_1d:
            p = Path(out_1d)
            out_1d = str(p.with_name(p.stem + '_1d' + p.suffix))
        visualize_fad_1d(
            real_dir=args.real,
            generated_dir=args.generated,
            output_path=out_1d,
        )
