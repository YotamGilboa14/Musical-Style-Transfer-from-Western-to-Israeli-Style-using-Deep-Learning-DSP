"""
build_demo_gallery.py — Master "browse everything" HTML hub for demo renders.

This is the Phase-D triage hub for Mission E: it scans the whole-song demo
renders under ``demo_external/_renders/<run_id>/`` and builds a single HTML page
that groups every render by song, with:

  * the original download + its conditioning piano-roll (once per song),
  * per render: mel-spectrogram PNG, waveform PNG, an in-page audio player, and
    its latency (infer_s / RTF) pulled from ``timing_infos.json``,
  * a top-level latency comparison chart (ddim100 vs ddim200) built from the
    new whole-song timing format ``{song, step, ddim_steps, infer_s, audio_s,
    rtf}`` written by ``run_inference_batch`` (the stock
    ``build_latency_visualizations`` expects the old per-patch VAE-GAN format,
    so this module ships its own chart).

Design notes
------------
* Whole-song WAVs are ~50 MB each (70 renders ≈ 3.5 GB), so the hub **references**
  audio/PNG-source in place via relative links rather than copying — this is the
  browse/triage hub, distinct from the curated ``build_deliverables`` bundle
  (which copies slide-ready assets).
* Only small PNGs are generated, into ``<out_dir>/_assets/``.
* Filename stem convention (from ``run_inference_batch._stable_name``)::

      {song}__step_{N}__style_{target}__role_{role}

  parsed by splitting on the ``__`` separator.

CLI
---
    python -m postprocessing.build_demo_gallery \
        --renders-root "G:/My Drive/MusicProject/versions/Israeli_3style/demo_external/_renders" \
        --songs-root   "G:/My Drive/MusicProject/versions/Israeli_3style/demo_external" \
        --out-dir      "G:/My Drive/MusicProject/versions/Israeli_3style/demo_external/_gallery"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the shared HTML/asset helpers so the hub matches the deliverables style.
from postprocessing.results_visualizations import (
    _DEFAULT_DPI,
    _FIGSIZE_WIDE,
    _HTML_HEAD,
    _h,
    _rel,
)

_SR = 22050
_HOP = 256


# ---------------------------------------------------------------------------
# Stem parsing
# ---------------------------------------------------------------------------
def parse_stem(stem: str) -> dict[str, str] | None:
    """Parse ``{song}__step_{N}__style_{target}__role_{role}`` into parts.

    ``song`` may itself contain single underscores (e.g. ``Rihanna_Diamonds``);
    the fields are separated by the double-underscore delimiter.
    """
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    song, step_p, style_p, role_p = parts
    if not (step_p.startswith("step_") and style_p.startswith("style_")
            and role_p.startswith("role_")):
        return None
    return {
        "song": song,
        "step": step_p[len("step_"):],
        "style": style_p[len("style_"):],
        "role": role_p[len("role_"):],
    }


# ---------------------------------------------------------------------------
# PNG renderers (each writes one dpi=200 slide-ready PNG)
# ---------------------------------------------------------------------------
def render_mel_png(mel_pt: Path, out_png: Path, *, title: str,
                   sr: int = _SR, hop: int = _HOP) -> Path:
    """Render a mel tensor (``.mel.pt``, shape ``[n_mels, T]``) to a PNG."""
    import torch

    mel = torch.load(mel_pt, map_location="cpu")
    mel = mel.numpy() if hasattr(mel, "numpy") else np.asarray(mel)
    mel = np.squeeze(mel)
    n_frames = mel.shape[-1]
    duration = n_frames * hop / sr

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
    im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, duration, 0, mel.shape[0]])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=_DEFAULT_DPI)
    plt.close(fig)
    return out_png


def render_waveform_png(wav_path: Path, out_png: Path, *, title: str) -> Path:
    """Render a WAV amplitude envelope to a PNG."""
    import soundfile as sf

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t = np.arange(len(audio)) / float(sr)

    fig, ax = plt.subplots(figsize=(_FIGSIZE_WIDE[0], 3.0), dpi=_DEFAULT_DPI)
    ax.plot(t, audio, linewidth=0.4, color="#1f77b4")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(-1.02, 1.02)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=_DEFAULT_DPI)
    plt.close(fig)
    return out_png


def render_piano_roll_png(midi_path: Path, out_png: Path, *, title: str,
                          fs: int = 100) -> Path:
    """Render a MIDI file's piano roll (conditioning score) to a PNG."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    roll = pm.get_piano_roll(fs=fs)  # [128, T]
    duration = roll.shape[1] / float(fs)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
    im = ax.imshow(roll, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, duration, 0, 128])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI pitch")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="velocity")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=_DEFAULT_DPI)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Latency (whole-song RTF format)
# ---------------------------------------------------------------------------
def load_timing(renders_root: Path) -> dict[str, dict]:
    """Merge every ``timing_infos.json`` under ``renders_root`` into one map.

    Smoke / warm-up render dirs (name contains ``SMOKE``) are skipped: their
    first-call GPU-allocation overhead inflates RTF (~0.34 vs ~0.13 batch) and
    would contaminate the reported latency, so they are excluded from the
    latency aggregation.
    """
    merged: dict[str, dict] = {}
    for tj in sorted(renders_root.glob("*/timing_infos.json")):
        if "smoke" in tj.parent.name.lower():
            continue
        try:
            data = json.loads(tj.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict):
            merged.update(data)
    return merged


def build_latency_chart(timing_map: dict[str, dict], out_png: Path) -> dict:
    """RTF-by-ddim comparison chart from whole-song timing dicts.

    Returns an aggregate summary ``{ddim: {mean_rtf, mean_infer_s, n}}``.
    """
    by_ddim: dict[int, list[dict]] = {}
    for ti in timing_map.values():
        by_ddim.setdefault(int(ti.get("ddim_steps", 0)), []).append(ti)

    summary: dict[str, Any] = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)

    ddims = sorted(by_ddim)
    labels = [f"ddim{d}" for d in ddims]

    mean_rtf = [float(np.mean([t["rtf"] for t in by_ddim[d]])) for d in ddims]
    mean_inf = [float(np.mean([t["infer_s"] for t in by_ddim[d]])) for d in ddims]
    for d, r, s in zip(ddims, mean_rtf, mean_inf):
        summary[f"ddim{d}"] = {"mean_rtf": round(r, 4),
                               "mean_infer_s": round(s, 2),
                               "n": len(by_ddim[d])}

    bars1 = ax1.bar(labels, mean_rtf, color=["#1f77b4", "#ff7f0e"][:len(ddims)])
    for b, v in zip(bars1, mean_rtf):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.axhline(1.0, color="#888", linestyle="--", linewidth=1)
    ax1.text(0.02, 1.02, "realtime (RTF=1)", color="#888", fontsize=9)
    ax1.set_ylabel("Mean RTF (infer_s / audio_s; lower = faster)")
    ax1.set_title("Real-time factor by sampler", fontsize=13, fontweight="bold")

    # Per-render scatter to show spread
    for i, d in enumerate(ddims):
        xs = np.full(len(by_ddim[d]), i) + (np.random.rand(len(by_ddim[d])) - 0.5) * 0.2
        ys = [t["rtf"] for t in by_ddim[d]]
        ax2.scatter(xs, ys, alpha=0.6, s=24,
                    color=["#1f77b4", "#ff7f0e"][i % 2], label=labels[i])
    ax2.set_xticks(range(len(ddims)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("RTF per render")
    ax2.set_title("Per-render RTF spread", fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=_DEFAULT_DPI)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------
def _find_song_midi(songs_root: Path, song: str) -> Path | None:
    folder = songs_root / song
    mids = sorted(folder.glob("*.mid"))
    return mids[0] if mids else None


def _find_song_original(songs_root: Path, song: str) -> Path | None:
    folder = songs_root / song
    cand = folder / "original.wav"
    if cand.exists():
        return cand
    wavs = sorted(folder.glob("*.wav"))
    return wavs[0] if wavs else None


def build_gallery(renders_root: Path, songs_root: Path, out_dir: Path) -> dict:
    """Scan renders, generate PNGs, and write the master browse ``index.html``."""
    renders_root = Path(renders_root)
    songs_root = Path(songs_root)
    out_dir = Path(out_dir)
    assets = out_dir / "_assets"
    assets.mkdir(parents=True, exist_ok=True)

    timing_map = load_timing(renders_root)

    # Latency chart (once, at the top).
    latency_png = assets / "latency_by_ddim.png"
    latency_summary: dict = {}
    if timing_map:
        latency_summary = build_latency_chart(timing_map, latency_png)

    # Collect renders grouped by song.
    by_song: dict[str, list[dict]] = {}
    for wav in sorted(renders_root.glob("*/audio/*.wav")):
        stem = wav.stem
        meta = parse_stem(stem)
        if meta is None:
            continue
        run_dir = wav.parent.parent
        mel_pt = run_dir / "mels" / f"{stem}.mel.pt"
        by_song.setdefault(meta["song"], []).append({
            "stem": stem, "wav": wav, "mel_pt": mel_pt,
            "run_id": run_dir.name, **meta,
            "timing": timing_map.get(stem, {}),
        })

    n_pngs = 0
    sections: list[str] = []
    for song in sorted(by_song):
        renders = sorted(by_song[song], key=lambda r: (r["style"], int(r["step"])))

        # Per-song conditioning piano roll + original audio (once).
        pr_html = "<div class='note'>no source MIDI found</div>"
        midi = _find_song_midi(songs_root, song)
        if midi is not None:
            pr_png = assets / f"{song}__piano_roll.png"
            if not pr_png.exists():
                render_piano_roll_png(midi, pr_png, title=f"{song} — conditioning piano roll")
                n_pngs += 1
            pr_html = f'<img src="{_rel(pr_png, out_dir)}" alt="{_h(song)} piano roll">'

        orig_html = "<div class='note'>no original found</div>"
        orig = _find_song_original(songs_root, song)
        if orig is not None:
            orig_html = f'<audio controls preload="none" src="{_rel(orig, out_dir)}"></audio>'

        cards: list[str] = []
        for r in renders:
            mel_html = "<div class='note'>no mel</div>"
            if r["mel_pt"].exists():
                mel_png = assets / f"{r['stem']}__mel.png"
                if not mel_png.exists():
                    render_mel_png(r["mel_pt"], mel_png,
                                   title=f"{r['style']} · step {r['step']}")
                    n_pngs += 1
                mel_html = f'<img src="{_rel(mel_png, out_dir)}" alt="mel {_h(r["stem"])}">'

            wave_png = assets / f"{r['stem']}__wave.png"
            if not wave_png.exists():
                render_waveform_png(r["wav"], wave_png,
                                    title=f"{r['style']} · step {r['step']}")
                n_pngs += 1
            wave_html = f'<img src="{_rel(wave_png, out_dir)}" alt="wave {_h(r["stem"])}">'

            ti = r["timing"]
            lat = ""
            if ti:
                lat = (f"<span class='note'>infer {ti.get('infer_s','?')}s · "
                       f"RTF {ti.get('rtf','?')} · ddim {ti.get('ddim_steps','?')}</span>")
            audio_html = f'<audio controls preload="none" src="{_rel(r["wav"], out_dir)}"></audio>'

            cards.append(
                "<div class='card'>"
                f"<h4>{_h(r['style'])} &middot; step {_h(r['step'])} "
                f"<span class='note'>[{_h(r['run_id'])}]</span></h4>"
                f"{lat}"
                f"<div class='grid2'><div>{mel_html}</div><div>{wave_html}</div></div>"
                f"{audio_html}"
                "</div>"
            )

        sections.append(
            f"<section><h2 id='{_h(song)}'>{_h(song)} "
            f"<span class='note'>({len(renders)} renders)</span></h2>"
            "<div class='grid2'>"
            f"<div><h4>original</h4>{orig_html}</div>"
            f"<div><h4>conditioning score</h4>{pr_html}</div>"
            "</div>"
            f"<div class='cards'>{''.join(cards)}</div>"
            "</section><hr>"
        )

    # Table of contents + latency block.
    toc = " &middot; ".join(
        f"<a href='#{_h(s)}'>{_h(s)}</a>" for s in sorted(by_song)
    )
    latency_block = ""
    if latency_summary:
        parts = " &nbsp; ".join(
            f"<strong>{k}</strong>: RTF {v['mean_rtf']} "
            f"(infer {v['mean_infer_s']}s, n={v['n']})"
            for k, v in latency_summary.items()
        )
        latency_block = (
            "<h2>Latency (whole-song render pass = Mission 3c)</h2>"
            f"<p class='note'>{parts}</p>"
            f'<img src="{_rel(latency_png, out_dir)}" alt="latency by ddim">'
        )

    n_renders = sum(len(v) for v in by_song.values())
    extra_css = (
        "<style>"
        ".grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}"
        ".cards{display:flex;flex-direction:column;gap:20px;margin-top:16px}"
        ".card{border:1px solid #d0d7de;border-radius:8px;padding:12px 16px;background:#fbfcfd}"
        ".card h4{margin:0 0 6px 0}"
        "</style>"
    )
    html = (
        _HTML_HEAD.format(title="Demo render gallery — Israeli_3style") +
        extra_css +
        f"<p class='note'>{n_renders} renders across {len(by_song)} songs. "
        "Audio and mel tensors are referenced in place (not copied); this is the "
        "browse/triage hub. PNGs are slide-ready (dpi=200).</p>"
        f"<p><strong>Songs:</strong> {toc}</p>"
        + latency_block +
        "<hr>" + "".join(sections) +
        "</body></html>"
    )
    out_html = out_dir / "index.html"
    out_html.write_text(html, encoding="utf-8")

    summary = {
        "index_html": str(out_html),
        "n_songs": len(by_song),
        "n_renders": n_renders,
        "n_pngs_generated": n_pngs,
        "latency_summary": latency_summary,
    }
    (out_dir / "_gallery_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the master demo render gallery HTML.")
    ap.add_argument("--renders-root", required=True,
                    help="Dir containing <run_id>/audio/*.wav render folders.")
    ap.add_argument("--songs-root", required=True,
                    help="demo_external dir with <song>/{original.wav,*.mid}.")
    ap.add_argument("--out-dir", required=True,
                    help="Output dir for index.html + _assets/.")
    args = ap.parse_args(argv)

    summary = build_gallery(Path(args.renders_root), Path(args.songs_root),
                            Path(args.out_dir))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
