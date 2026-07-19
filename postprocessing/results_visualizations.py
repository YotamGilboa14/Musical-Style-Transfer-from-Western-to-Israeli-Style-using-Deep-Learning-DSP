"""Output-side result visualizations for TA deliverables.

Thin wrappers over the existing eval helpers (:mod:`postprocessing.fad_eval`,
:mod:`postprocessing.f1_eval`, :mod:`postprocessing.latency_eval`) that
produce **PowerPoint-friendly** asset bundles for every pipeline stage.

Every builder:
  * writes a self-contained ``<stage>.html`` next to standalone PNGs and WAVs;
  * places download-targets in a sibling ``_assets/`` subfolder;
  * generates a ``download_all.zip`` so the TA can right-click → download once
    and obtain everything needed for slides;
  * uses stable filenames of the form ``<stage>_<id>_<style>_<metric>.{png,wav}``.

The same module is used for every style version (Slakh v0, Israeli v1, …).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# These imports are intentionally lazy at call time only when needed, to keep
# `import postprocessing.results_visualizations` cheap on environments that
# do not have e.g. torch/Basic-Pitch installed.

_DEFAULT_DPI = 200
_FIGSIZE_WIDE = (16, 9)


# ---------------------------------------------------------------------------
# Shared HTML helpers
# ---------------------------------------------------------------------------
_HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       padding:24px; color:#1f2328; max-width:1200px; margin:0 auto }}
h1 {{ margin-top:0 }}
.downloads {{ background:#f6f8fa; border:1px solid #d0d7de; border-radius:6px;
             padding:12px 16px; margin:16px 0 24px 0 }}
.downloads a {{ display:inline-block; margin-right:14px; margin-bottom:4px;
                color:#0969da; text-decoration:none }}
.downloads a:hover {{ text-decoration:underline }}
table {{ border-collapse:collapse; width:100%; margin:16px 0 }}
th, td {{ padding:8px 12px; border-bottom:1px solid #d0d7de; text-align:left }}
th {{ background:#f6f8fa }}
img {{ max-width:100%; height:auto; border:1px solid #d0d7de; border-radius:6px;
       display:block; margin:12px 0 }}
audio {{ width:320px; vertical-align:middle }}
.note {{ color:#57606a; font-size:13px }}
</style></head><body>
<h1>{title}</h1>
"""


def _h(x) -> str:
    s = "" if x is None else str(x)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _download_bar(asset_paths: Sequence[Path], zip_path: Path, html_dir: Path) -> str:
    """Render a 'Download' bar linking each asset + the zip."""
    links = [
        f'<a href="{_rel(zip_path, html_dir)}" download>download_all.zip</a>'
    ]
    for p in asset_paths:
        links.append(f'<a href="{_rel(p, html_dir)}" download>{_h(p.name)}</a>')
    return (
        '<div class="downloads"><strong>Download:</strong> ' + " ".join(links) +
        '<div class="note">All PNGs and WAVs are standalone and slide-ready '
        '(dpi=200, ~1600&times;900).</div></div>'
    )


def _rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base)).replace("\\", "/")
    except ValueError:
        return p.as_uri()


def _make_zip(zip_path: Path, files: Iterable[Path], arc_root: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists():
                try:
                    arc = f.relative_to(arc_root)
                except ValueError:
                    arc = Path(f.name)
                zf.write(f, arc_name=str(arc))
    return zip_path


def _prepare_stage_dir(out_dir: Path) -> tuple[Path, Path]:
    """Create ``out_dir`` and ``out_dir/_assets``. Returns (out_dir, assets)."""
    out_dir = Path(out_dir)
    assets = out_dir / "_assets"
    assets.mkdir(parents=True, exist_ok=True)
    return out_dir, assets


# ---------------------------------------------------------------------------
# FAD
# ---------------------------------------------------------------------------
@dataclass
class _StyleFADInput:
    style_name: str        # human-readable, e.g. "Slakh rock"
    version_id: int        # 0/1/...
    generated_dir: str     # WAVs to score
    real_dir: str          # reference WAVs


def build_fad_visualizations(
    real_dir: str,
    style_inputs: Sequence[dict],
    out_dir: str | Path,
    *,
    use_pretrained: bool = True,
    sr: int = 22050,
) -> dict:
    """Produce All-FAD + per-style Group-FAD visualizations.

    Args:
        real_dir: directory with real reference WAVs (used for the All-FAD
            baseline). Group-FAD per style uses each style's own ``real_dir``.
        style_inputs: list of dicts with keys
            ``style_name``, ``version_id``, ``generated_dir``,
            ``real_dir`` (per-style real subset),
            optional ``version_manifest_csv``.
        out_dir: where to write ``fad_per_style.png``, ``fad_table.html``,
            ``fad_summary.json``, ``_assets/``, ``download_all.zip``.

    Returns:
        Dict with paths to the produced files and the raw scores.
    """
    from postprocessing.fad_eval import evaluate_all_fad, compute_group_fad

    out_dir, assets = _prepare_stage_dir(Path(out_dir))
    summary: Dict[str, object] = {"styles": []}

    # 1. All-FAD (real-vs-all-generated). We average the union of generated dirs
    # by passing the first as a representative — the user can extend this later
    # if a unified pool is desired. For now All-FAD is reported per style and
    # an "ALL" entry uses the supplied ``real_dir`` against the first style's
    # generated_dir as a sanity anchor.
    if style_inputs:
        anchor = style_inputs[0]
        try:
            all_details = evaluate_all_fad(
                real_dir, anchor["generated_dir"],
                sr=sr, use_pretrained=use_pretrained,
            )
            summary["all_fad"] = {
                "fad_score": all_details["fad_score"],
                "embedding_model": all_details.get("embedding_model"),
                "n_real_files": all_details.get("n_real_files"),
                "n_gen_files": all_details.get("n_gen_files"),
                "reference_style": anchor["style_name"],
            }
        except Exception as e:  # noqa: BLE001
            summary["all_fad"] = {"error": repr(e)}

    # 2. Group-FAD per style
    style_scores: List[dict] = []
    for s in style_inputs:
        try:
            r = compute_group_fad(
                real_dir=s["real_dir"],
                generated_dir=s["generated_dir"],
                version_id=int(s["version_id"]),
                version_manifest_csv=s.get("version_manifest_csv"),
                sr=sr,
                use_pretrained=use_pretrained,
            )
            style_scores.append({
                "style_name": s["style_name"],
                "version_id": int(s["version_id"]),
                "group_fad": r["group_fad"],
                "n_real_files": r["n_real_files"],
                "n_gen_files": r["n_gen_files"],
                "embedding_model": r["fad_details"].get("embedding_model"),
            })
        except Exception as e:  # noqa: BLE001
            style_scores.append({
                "style_name": s["style_name"],
                "version_id": int(s["version_id"]),
                "error": repr(e),
            })
    summary["styles"] = style_scores

    # 3. Bar chart
    png_path = assets / "fad_per_style.png"
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
    valid = [s for s in style_scores if "group_fad" in s]
    if valid:
        labels = [f"{s['style_name']}\nv{s['version_id']}" for s in valid]
        vals = [s["group_fad"] for s in valid]
        bars = ax.bar(labels, vals, color="#1f77b4")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Group-FAD (lower = closer to real)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No FAD scores available", ha="center", va="center", fontsize=14)
    ax.set_title("Group-FAD per style", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", dpi=_DEFAULT_DPI)
    plt.close(fig)

    # 4. JSON summary
    json_path = assets / "fad_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # 5. Zip
    zip_path = out_dir / "download_all.zip"
    _make_zip(zip_path, [png_path, json_path], arc_root=out_dir.parent)

    # 6. HTML
    html_path = out_dir / "fad_table.html"
    rows = []
    for s in style_scores:
        score = s.get("group_fad")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else _h(s.get("error", ""))
        rows.append(
            f"<tr><td>{_h(s['style_name'])}</td>"
            f"<td>{_h(s['version_id'])}</td>"
            f"<td>{score_str}</td>"
            f"<td>{_h(s.get('n_real_files', ''))}</td>"
            f"<td>{_h(s.get('n_gen_files', ''))}</td>"
            f"<td>{_h(s.get('embedding_model', ''))}</td></tr>"
        )
    all_fad = summary.get("all_fad", {})
    all_fad_block = ""
    if "fad_score" in all_fad:
        all_fad_block = (
            f"<p><strong>All-FAD anchor (vs {_h(all_fad.get('reference_style',''))})"
            f":</strong> {all_fad['fad_score']:.4f}  "
            f"<span class='note'>embedder: {_h(all_fad.get('embedding_model',''))}</span></p>"
        )
    html = (
        _HTML_HEAD.format(title="FAD evaluation") +
        _download_bar([png_path, json_path], zip_path, out_dir) +
        all_fad_block +
        f'<img src="{_rel(png_path, out_dir)}" alt="Group-FAD per style">'
        "<table><thead><tr><th>Style</th><th>version_id</th><th>Group-FAD</th>"
        "<th>#real</th><th>#gen</th><th>embedder</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    return {
        "html": str(html_path), "png": str(png_path),
        "json": str(json_path), "zip": str(zip_path),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------
def build_f1_visualizations(
    benchmark_pairs: Sequence[dict],
    out_dir: str | Path,
    *,
    basic_pitch_python: Optional[str] = None,
    onset_tolerance_s: float = 0.05,
) -> dict:
    """Produce F1 transcription-fidelity visualizations.

    Args:
        benchmark_pairs: list of dicts with keys
            ``name`` (str), ``style_name`` (str),
            ``generated_wav`` (path), ``reference_midi`` (path).
        out_dir: where to write artifacts.
        basic_pitch_python: optional override for the Basic-Pitch interpreter.
            Defaults to current Python.
    """
    from postprocessing.f1_eval import compute_f1

    out_dir, assets = _prepare_stage_dir(Path(out_dir))

    kwargs = {"onset_tolerance_s": onset_tolerance_s}
    if basic_pitch_python:
        kwargs["basic_pitch_python"] = basic_pitch_python

    results: List[dict] = []
    for pair in benchmark_pairs:
        try:
            r = compute_f1(
                generated_wav=Path(pair["generated_wav"]),
                reference_midi=Path(pair["reference_midi"]),
                **kwargs,
            )
            r["name"] = pair["name"]
            r["style_name"] = pair.get("style_name", "")
            results.append(r)
        except Exception as e:  # noqa: BLE001
            results.append({
                "name": pair["name"],
                "style_name": pair.get("style_name", ""),
                "error": repr(e),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
            })

    # PNG bar chart of F1, precision, recall per pair
    valid = [r for r in results if "error" not in r]
    png_path = assets / "f1_sensitivity.png"
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DEFAULT_DPI)
    if valid:
        names = [r["name"] for r in valid]
        x = np.arange(len(names))
        width = 0.25
        ax.bar(x - width, [r["precision"] for r in valid], width, label="precision", color="#1f77b4")
        ax.bar(x,         [r["recall"]    for r in valid], width, label="recall",    color="#ff7f0e")
        ax.bar(x + width, [r["f1"]        for r in valid], width, label="F1",        color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score (0 - 1)")
        ax.axhline(0.25, color="#888", linestyle="--", linewidth=1, label="Israeli F1 target = 0.25")
        ax.legend(loc="upper right")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No F1 results", ha="center", va="center", fontsize=14)
    ax.set_title("Note-level transcription F1 per benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", dpi=_DEFAULT_DPI)
    plt.close(fig)

    # Copy benchmark WAVs into assets (rename to stable scheme)
    asset_wavs: List[Path] = []
    for r in results:
        gw = Path(r.get("generated_wav", ""))
        if gw.exists():
            style = (r.get("style_name") or "style").replace(" ", "_")
            dest = assets / f"f1_{r['name']}_{style}.wav"
            try:
                shutil.copy2(gw, dest)
                asset_wavs.append(dest)
                r["asset_wav"] = str(dest)
            except Exception:  # noqa: BLE001
                pass

    json_path = assets / "f1_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    zip_path = out_dir / "download_all.zip"
    _make_zip(zip_path, [png_path, json_path, *asset_wavs], arc_root=out_dir.parent)

    rows = []
    for r in results:
        if "error" in r:
            rows.append(
                f"<tr><td>{_h(r['name'])}</td><td>{_h(r['style_name'])}</td>"
                f"<td colspan='5'>ERROR: {_h(r['error'])}</td></tr>"
            )
            continue
        audio_html = ""
        if r.get("asset_wav"):
            audio_html = (
                f'<audio controls src="{_rel(Path(r["asset_wav"]), out_dir)}"></audio>'
            )
        rows.append(
            f"<tr><td>{_h(r['name'])}</td><td>{_h(r['style_name'])}</td>"
            f"<td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>"
            f"<td><strong>{r['f1']:.3f}</strong></td>"
            f"<td>{r.get('n_predicted','')}/{r.get('n_reference','')}</td>"
            f"<td>{audio_html}</td></tr>"
        )

    html_path = out_dir / "f1_table.html"
    html = (
        _HTML_HEAD.format(title="F1 transcription fidelity") +
        _download_bar([png_path, json_path, *asset_wavs], zip_path, out_dir) +
        f'<img src="{_rel(png_path, out_dir)}" alt="F1 per benchmark">'
        "<table><thead><tr><th>Name</th><th>Style</th><th>P</th><th>R</th>"
        "<th>F1</th><th>pred/ref notes</th><th>generated audio</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    return {
        "html": str(html_path), "png": str(png_path),
        "json": str(json_path), "zip": str(zip_path),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def build_latency_visualizations(
    timing_infos: Dict[str, dict],
    out_dir: str | Path,
    *,
    sample_rate: int = 22050,
    hop_length: int = 256,
    segment_duration: float = 5.0,
) -> dict:
    """Produce latency visualizations from one or more timing dicts.

    Args:
        timing_infos: mapping of label -> timing_info dict produced by
            ``infer_mel(return_timing=True)``.
        out_dir: artifact output directory.
    """
    from postprocessing.latency_eval import evaluate_latency, plot_latency

    out_dir, assets = _prepare_stage_dir(Path(out_dir))
    summaries: Dict[str, dict] = {}
    pngs: List[Path] = []

    for label, timing in timing_infos.items():
        safe_label = label.replace(" ", "_").replace("/", "_")
        try:
            report = evaluate_latency(
                timing, sample_rate=sample_rate,
                hop_length=hop_length, segment_duration=segment_duration,
            )
            png = assets / f"latency_{safe_label}.png"
            plot_latency(report, output_path=str(png))
            pngs.append(png)
            summaries[label] = {
                "mean_ms": report.get("mean_ms"),
                "median_ms": report.get("median_ms"),
                "p95_ms": report.get("p95_ms"),
                "max_ms": report.get("max_ms"),
                "rtf": report.get("rtf"),
                "meets_realtime": report.get("meets_realtime"),
                "meets_target": report.get("meets_target"),
            }
        except Exception as e:  # noqa: BLE001
            summaries[label] = {"error": repr(e)}

    json_path = assets / "latency_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")

    zip_path = out_dir / "download_all.zip"
    _make_zip(zip_path, [*pngs, json_path], arc_root=out_dir.parent)

    rows = []
    imgs_html = []
    for label, s in summaries.items():
        if "error" in s:
            rows.append(
                f"<tr><td>{_h(label)}</td><td colspan='6'>ERROR: {_h(s['error'])}</td></tr>"
            )
            continue
        rows.append(
            f"<tr><td>{_h(label)}</td>"
            f"<td>{s['mean_ms']:.1f}</td><td>{s['median_ms']:.1f}</td>"
            f"<td>{s['p95_ms']:.1f}</td><td>{s['max_ms']:.1f}</td>"
            f"<td>{s['rtf']:.3f}</td>"
            f"<td>{'yes' if s.get('meets_target') else 'no'}</td></tr>"
        )
        safe_label = label.replace(' ', '_').replace('/', '_')
        png = assets / f"latency_{safe_label}.png"
        imgs_html.append(f'<h3>{_h(label)}</h3><img src="{_rel(png, out_dir)}" alt="latency {label}">')

    html_path = out_dir / "latency_table.html"
    html = (
        _HTML_HEAD.format(title="Inference latency") +
        _download_bar([*pngs, json_path], zip_path, out_dir) +
        "<table><thead><tr><th>Config</th><th>mean (ms)</th><th>median (ms)</th>"
        "<th>p95 (ms)</th><th>max (ms)</th><th>RTF</th><th>5s &le; 5s?</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>" +
        "".join(imgs_html) +
        "</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    return {
        "html": str(html_path), "json": str(json_path),
        "zip": str(zip_path), "pngs": [str(p) for p in pngs],
        "summary": summaries,
    }


# ---------------------------------------------------------------------------
# Transfer grid (qualitative deliverable)
# ---------------------------------------------------------------------------
def build_transfer_grid(
    items: Sequence[dict],
    out_dir: str | Path,
    *,
    title: str = "Style-transfer examples",
) -> dict:
    """Build a qualitative HTML grid linking input/generated mels + audio.

    Each ``items`` entry is a dict::

        {
            "name": "AuSep_1_tpt_33_Elise",
            "style_name": "Israeli",
            "input_mel_png":   "path/to/input_mel.png",   # standalone PNG
            "generated_mel_png": "path/to/generated_mel.png",
            "input_wav":  "path/to/input.wav",
            "generated_wav": "path/to/generated.wav",
            "scores": {"f1": 0.31, "fad": 12.4}   # optional
        }

    The function COPIES every asset into ``out_dir/_assets/`` with a stable
    ``transfer_<name>_<style>_<role>.{png,wav}`` filename so the bundle is
    self-contained.
    """
    out_dir, assets = _prepare_stage_dir(Path(out_dir))
    all_assets: List[Path] = []
    cards: List[str] = []

    for item in items:
        name = item.get("name", "item")
        style = (item.get("style_name") or "style").replace(" ", "_")
        base = f"transfer_{name}_{style}"
        copied: Dict[str, Optional[Path]] = {}
        for role, src_key in [
            ("input_mel", "input_mel_png"),
            ("generated_mel", "generated_mel_png"),
            ("input_audio", "input_wav"),
            ("generated_audio", "generated_wav"),
        ]:
            src = item.get(src_key)
            if not src:
                copied[role] = None
                continue
            src = Path(src)
            if not src.exists():
                copied[role] = None
                continue
            ext = src.suffix.lower()
            dest = assets / f"{base}_{role}{ext}"
            try:
                shutil.copy2(src, dest)
                copied[role] = dest
                all_assets.append(dest)
            except Exception:  # noqa: BLE001
                copied[role] = None

        def _img(p: Optional[Path], alt: str) -> str:
            if p is None:
                return f'<div class="note">missing: {_h(alt)}</div>'
            return f'<img src="{_rel(p, out_dir)}" alt="{_h(alt)}">'

        def _audio(p: Optional[Path], label: str) -> str:
            if p is None:
                return f'<div class="note">missing: {_h(label)}</div>'
            return (
                f'<div><strong>{_h(label)}:</strong> '
                f'<audio controls src="{_rel(p, out_dir)}"></audio></div>'
            )

        scores = item.get("scores") or {}
        score_html = ""
        if scores:
            score_html = (
                "<ul>"
                + "".join(f"<li><strong>{_h(k)}</strong>: {_h(v)}</li>" for k, v in scores.items())
                + "</ul>"
            )
        cards.append(
            f"<section><h2>{_h(name)} &mdash; {_h(item.get('style_name',''))}</h2>"
            "<div style='display:grid; grid-template-columns:1fr 1fr; gap:16px'>"
            f"<div><h4>input mel</h4>{_img(copied['input_mel'], 'input mel')}</div>"
            f"<div><h4>generated mel</h4>{_img(copied['generated_mel'], 'generated mel')}</div>"
            "</div>"
            + _audio(copied["input_audio"], "input audio")
            + _audio(copied["generated_audio"], "generated audio")
            + score_html
            + "</section><hr>"
        )

    zip_path = out_dir / "download_all.zip"
    _make_zip(zip_path, all_assets, arc_root=out_dir.parent)

    html_path = out_dir / "transfer_grid.html"
    html = (
        _HTML_HEAD.format(title=title) +
        _download_bar(all_assets, zip_path, out_dir) +
        "".join(cards) +
        "</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    return {
        "html": str(html_path), "zip": str(zip_path),
        "n_items": len(items), "n_assets": len(all_assets),
    }
