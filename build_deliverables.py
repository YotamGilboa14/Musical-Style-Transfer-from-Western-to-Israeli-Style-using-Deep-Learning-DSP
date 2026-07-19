"""Single-command TA-grading deliverables orchestrator.

Reads a small config (YAML or JSON) describing the run, then populates a
``deliverables/`` tree with PowerPoint-friendly artifacts produced by:

  * :mod:`preprocessing.data_quality` (gate report)
  * :mod:`preprocessing.dataset_visualizations` (input-side plots)
  * :mod:`postprocessing.results_visualizations` (FAD/F1/latency/transfer-grid)

Usage::

    python build_deliverables.py --config deliverables/config.yaml

The script is idempotent: it overwrites stage outputs but never deletes
unrelated files. Missing optional inputs (e.g. no benchmark pairs) skip the
corresponding stage and add a note to ``deliverables/00_overview/index.html``.

Config schema (all paths may be relative to the config file)::

    run_name: Israeli_Shalom_Arik_2026-03-15
    style_name: Israeli
    version_id: 1

    # Phase A — data quality / input-side
    splits_dir: /path/to/splits/             # contains train.csv val.csv test.csv
    min_hours: 3.0

    # Phase B — results
    fad:                                     # optional
      real_dir: ...
      style_inputs:
        - {style_name: "Slakh rock", version_id: 0, real_dir: ..., generated_dir: ...}
        - {style_name: "Israeli",    version_id: 1, real_dir: ..., generated_dir: ...}
    f1:                                      # optional
      benchmark_pairs:
        - {name: "song1", style_name: "Israeli", generated_wav: ..., reference_midi: ...}
    latency:                                 # optional — pre-loaded timing dicts
      timing_json: path/to/timing_infos.json # mapping label -> timing_info dict
    transfer:                                # optional qualitative grid
      items:
        - {name: ..., style_name: ..., input_mel_png: ..., generated_mel_png: ...,
           input_wav: ..., generated_wav: ..., scores: {f1: 0.31, fad: 12.4}}
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sibling packages importable when run as a script
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_config(path: Path) -> dict:
    """Load a deliverables config from either YAML or JSON."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("PyYAML is required for YAML configs. pip install pyyaml") from e
        return yaml.safe_load(text)
    return json.loads(text)


def _resolve(cfg_path: Path, p: Optional[str]) -> Optional[Path]:
    """Turn a config-relative path into an absolute one (None stays None).

    Paths in the config are written relative to the config file, so we join them
    onto the config's folder unless they are already absolute.
    """
    if p is None:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (cfg_path.parent / pp).resolve()


# ---------------------------------------------------------------------------
# Overview HTML
# ---------------------------------------------------------------------------
_OVERVIEW_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       padding:24px; max-width:1100px; margin:0 auto; color:#1f2328 }}
section {{ margin:20px 0; padding:16px; border:1px solid #d0d7de; border-radius:6px }}
h1 {{ margin-top:0 }} h2 {{ margin-top:0 }}
a {{ color:#0969da }} .skipped {{ color:#9a6700 }} .ok {{ color:#1a7f37 }}
.kv {{ font-family: monospace; background:#f6f8fa; padding:8px; border-radius:4px }}
</style></head><body>
"""


def _write_overview(out_dir: Path, run_name: str, sections: List[dict]) -> Path:
    """Write the top-level index page that links every deliverable stage."""
    parts = [_OVERVIEW_HEAD.format(title=f"Deliverables — {run_name}"),
             f"<h1>Deliverables: {run_name}</h1>",
             "<p>This page indexes every TA-grading artifact for this run. "
             "Each stage links to a self-contained HTML page with download "
             "links for slide-ready PNG/WAV assets.</p>"]
    for s in sections:
        status_html = (
            f'<span class="ok">[ok]</span>' if s["status"] == "ok"
            else f'<span class="skipped">[{s["status"]}]</span>'
        )
        link_html = ""
        if s.get("html"):
            rel = Path(s["html"]).relative_to(out_dir) if Path(s["html"]).is_absolute() else s["html"]
            link_html = f'<p><a href="{rel}">open {Path(s["html"]).name}</a></p>'
        notes = s.get("notes", "")
        parts.append(
            f"<section><h2>{s['title']} {status_html}</h2>"
            f"<p>{notes}</p>{link_html}</section>"
        )
    parts.append("</body></html>")
    overview_dir = out_dir / "00_overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    p = overview_dir / "index.html"
    p.write_text("\n".join(parts), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Reuse helpers (point a stage at already-built artifacts instead of recomputing)
# ---------------------------------------------------------------------------
_GALLERY_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       padding:24px; max-width:1200px; margin:0 auto; color:#1f2328 }}
h1 {{ margin-top:0 }}
a {{ color:#0969da }}
.kv {{ font-family: monospace; background:#f6f8fa; padding:4px 6px;
      border-radius:4px; font-size:12px; color:#57606a }}
figure {{ margin:18px 0; padding:12px; border:1px solid #d0d7de; border-radius:6px }}
figure img {{ max-width:100%; border:1px solid #e1e4e8; border-radius:4px }}
figcaption {{ margin-top:8px }}
table {{ border-collapse:collapse; margin:16px 0 }}
th,td {{ border:1px solid #d0d7de; padding:6px 10px; text-align:right }}
th:first-child, td:first-child {{ text-align:left }}
.bar {{ background:#f6f8fa; padding:10px 14px; border-radius:6px; margin:12px 0 }}
</style></head><body>
"""


def _copy_files(srcs: List[Path], dest_assets: Path) -> List[Path]:
    """Copy each existing source file into ``dest_assets`` (flat). Returns copies."""
    copied: List[Path] = []
    for s in srcs:
        if s is not None and s.exists() and s.is_file():
            dest = dest_assets / s.name
            shutil.copy2(s, dest)
            copied.append(dest)
    return copied


def _copy_glob(source_dir: Path, patterns: List[str], dest_assets: Path) -> List[Path]:
    """Copy every file under ``source_dir`` matching any glob pattern."""
    copied: List[Path] = []
    if source_dir is None or not source_dir.exists():
        return copied
    seen: set[str] = set()
    for pat in patterns:
        for s in sorted(source_dir.glob(pat)):
            if s.is_file() and s.name not in seen:
                shutil.copy2(s, dest_assets / s.name)
                copied.append(dest_assets / s.name)
                seen.add(s.name)
    return copied


def _reuse_gallery_stage(
    *,
    out_dir: Path,
    slug: str,
    title: str,
    blurb: str,
    source_dir: Path,
    files: List[str],
    dirs: List[str],
    globs: List[str],
    source_index: Optional[Path],
) -> tuple[Path, int]:
    """Copy pre-built PNG/JSON assets into a stage dir and write an index page.

    Returns ``(html_path, n_pngs)``.
    """
    stage_dir = out_dir / slug
    assets = stage_dir / "_assets"
    assets.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    copied += _copy_files([source_dir / f for f in files], assets)
    for d in dirs:
        sub = source_dir / d
        if sub.exists():
            # Prefix with the subdir name: sibling dirs (e.g. fad_pca / fad_bells)
            # often share identical file names, so a flat copy would collide.
            for s in sorted(sub.glob("*.png")):
                dest = assets / f"{d}__{s.name}"
                shutil.copy2(s, dest)
                copied.append(dest)
    copied += _copy_glob(source_dir, globs, assets)

    pngs = [c for c in copied if c.suffix.lower() == ".png"]

    zip_path = assets / "download_all.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in copied:
            zf.write(c, arcname=c.name)

    figs = "\n".join(
        f'<figure><img src="_assets/{p.name}"/>'
        f'<figcaption class="kv">{p.name}</figcaption></figure>'
        for p in pngs
    )
    src_link = ""
    if source_index is not None and source_index.exists():
        src_link = (
            f'<p class="bar">Full interactive view (all steps, both samplers): '
            f'<a href="{source_index.as_uri()}">{source_index.name}</a></p>'
        )
    body = (
        _GALLERY_HEAD.format(title=title)
        + f"<h1>{title}</h1>\n<p>{blurb}</p>\n"
        + f'<p class="bar"><a href="_assets/download_all.zip">'
        + f"&#11015; download all ({len(pngs)} PNG)</a></p>\n"
        + src_link
        + figs
        + "\n</body></html>"
    )
    html_path = stage_dir / "index.html"
    html_path.write_text(body, encoding="utf-8")
    return html_path, len(pngs)


def _stage_latency_reuse(reuse: dict, cfg_path: Path, out_dir: Path) -> dict:
    """Latency stage that reuses the demo render timings + pre-built chart.

    Reads each ``timing_infos.json`` (mapping label -> {infer_s, audio_s, rtf,
    ddim_steps, ...}), aggregates by sampler (ddim_steps), and embeds the
    pre-built ``latency_by_ddim.png`` chart.
    """
    stage_dir = out_dir / "04_latency"
    assets = stage_dir / "_assets"
    assets.mkdir(parents=True, exist_ok=True)

    # Aggregate every timing entry, grouped by ddim_steps.
    per_sampler: Dict[int, List[dict]] = {}
    n_entries = 0
    for tj in reuse.get("timing_jsons", []):
        p = _resolve(cfg_path, tj)
        if p is None or not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for _label, info in data.items():
            steps = int(info.get("ddim_steps", 0))
            per_sampler.setdefault(steps, []).append(info)
            n_entries += 1

    seg = float(reuse.get("segment_duration", 5.0))

    def _mean(vals: List[float]) -> float:
        # Average of a list, or NaN when there is nothing to average.
        return sum(vals) / len(vals) if vals else float("nan")

    rows = []
    summary: Dict[str, dict] = {}
    for steps in sorted(per_sampler):
        infos = per_sampler[steps]
        rtfs = [float(i["rtf"]) for i in infos if "rtf" in i]
        infers = [float(i["infer_s"]) for i in infos if "infer_s" in i]
        audios = [float(i["audio_s"]) for i in infos if "audio_s" in i]
        mean_rtf = _mean(rtfs)
        label = f"ddim{steps}"
        summary[label] = {
            "n": len(infos), "mean_rtf": mean_rtf,
            "mean_infer_s": _mean(infers), "mean_audio_s": _mean(audios),
            "sec_per_5s_segment": mean_rtf * seg,
        }
        rows.append(
            f"<tr><td>{label}</td><td>{len(infos)}</td>"
            f"<td>{mean_rtf:.3f}</td><td>{_mean(infers):.1f}</td>"
            f"<td>{_mean(audios):.1f}</td><td>{mean_rtf * seg:.2f}</td></tr>"
        )

    (assets / "latency_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Copy the pre-built chart.
    chart = _resolve(cfg_path, reuse.get("chart")) if reuse.get("chart") else None
    chart_html = ""
    copied_pngs: List[Path] = []
    if chart is not None and chart.exists():
        shutil.copy2(chart, assets / chart.name)
        copied_pngs.append(assets / chart.name)
        chart_html = f'<figure><img src="_assets/{chart.name}"/><figcaption class="kv">{chart.name}</figcaption></figure>'

    zip_path = assets / "download_all.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in [*copied_pngs, assets / "latency_summary.json"]:
            zf.write(c, arcname=c.name)

    blurb = reuse.get("blurb", "")
    body = (
        _GALLERY_HEAD.format(title="04. Latency / RTF")
        + "<h1>04. Latency / RTF</h1>\n"
        + f"<p>{blurb}</p>\n"
        + f'<p class="bar"><a href="_assets/download_all.zip">&#11015; download all</a></p>\n'
        + "<table><thead><tr><th>Sampler</th><th>n</th><th>mean RTF</th>"
        + "<th>mean infer (s)</th><th>mean audio (s)</th>"
        + f"<th>s / {seg:g}s segment</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>\n"
        + chart_html
        + "\n</body></html>"
    )
    html_path = stage_dir / "index.html"
    html_path.write_text(body, encoding="utf-8")
    return {
        "title": "04. Latency / RTF", "status": "ok",
        "notes": f"Aggregated {n_entries} timed render(s) across "
                 f"{len(per_sampler)} sampler(s).",
        "html": str(html_path),
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _stage_data_quality(cfg: dict, cfg_path: Path, out_dir: Path) -> dict:
    """Run the pre-training gate + dataset visualizations."""
    splits_dir = _resolve(cfg_path, cfg.get("splits_dir"))
    if splits_dir is None or not splits_dir.exists():
        return {
            "title": "01. Data quality gate",
            "status": "skipped",
            "notes": f"No splits_dir at {splits_dir}.",
        }
    from preprocessing.data_quality import (
        load_combined_manifest, run_full_gate,
    )
    from preprocessing.dataset_visualizations import (
        plot_mel_grid, plot_piano_roll_grid,
        plot_segment_length_histogram, plot_mfcc_similarity_heatmap,
        plot_dataset_stats_panel,
    )

    df, manifest_root = load_combined_manifest(splits_dir)
    expected_version = cfg.get("version_id")
    min_hours = float(cfg.get("min_hours", 3.0))

    report = run_full_gate(
        df, manifest_root,
        expected_version_id=expected_version,
        min_hours=min_hours,
    )
    stage_dir = out_dir / "01_data_quality"
    assets = stage_dir / "_assets"
    assets.mkdir(parents=True, exist_ok=True)
    report.print_summary()
    report.write_html(stage_dir / "gate_report.html",
                      title=f"Data Quality Gate — {cfg.get('run_name','')}")
    report.write_png_table(assets / "gate_report.png",
                           title=f"Data Quality Gate — {cfg.get('run_name','')}")
    report.write_json(assets / "gate_report.json")
    plot_dataset_stats_panel(df, save_path=assets / "dataset_stats.png")
    plot_mel_grid(df, manifest_root, save_path=assets / "mel_grid.png")
    plot_piano_roll_grid(df, manifest_root, save_path=assets / "piano_roll_grid.png")
    plot_segment_length_histogram(df, manifest_root, save_path=assets / "segment_length_hist.png")
    plot_mfcc_similarity_heatmap(df, manifest_root, save_path=assets / "mfcc_similarity.png")

    return {
        "title": "01. Data quality gate",
        "status": "ok",
        "notes": f"Overall: <strong>{report.overall}</strong>. "
                 f"{len(df)} segments; {df['song_id'].nunique() if 'song_id' in df.columns else '?'} songs.",
        "html": str(stage_dir / "gate_report.html"),
    }


def _stage_preprocessing(cfg: dict, cfg_path: Path, out_dir: Path) -> dict:
    """Render one preprocessing-demo PNG per featured song.

    Reads ``cfg['preprocessing']['featured_songs']`` (list of ``{name, wav,
    midi?}``), produces ``preprocessing_demo.png`` for each via
    :func:`preprocessing.dataset_visualizations.plot_preprocessing_demo`, and
    writes a small HTML index with a one-paragraph explainer of the DSP block.
    """
    pp_cfg = cfg.get("preprocessing")
    if not pp_cfg or not pp_cfg.get("featured_songs"):
        return {
            "title": "01b. Preprocessing demo", "status": "skipped",
            "notes": "No `preprocessing.featured_songs` section in config.",
        }

    import zipfile

    import matplotlib.pyplot as _plt
    from preprocessing.dataset_visualizations import plot_preprocessing_demo

    stage_dir = out_dir / "01b_preprocessing"
    assets = stage_dir / "_assets"
    assets.mkdir(parents=True, exist_ok=True)

    items: List[dict] = []
    for entry in pp_cfg["featured_songs"]:
        name = entry["name"]
        wav = _resolve(cfg_path, entry["wav"])
        midi = _resolve(cfg_path, entry.get("midi")) if entry.get("midi") else None
        if wav is None or not wav.exists():
            items.append({"name": name, "status": "missing-wav", "png": None})
            continue
        png_path = assets / f"{name}__preprocessing_demo.png"
        try:
            fig = plot_preprocessing_demo(
                wav_path=wav, midi_path=midi, save_path=png_path,
            )
            _plt.close(fig)
            items.append({"name": name, "status": "ok",
                          "png": png_path.name, "wav": wav.name,
                          "midi": midi.name if midi else None})
        except Exception as exc:  # noqa: BLE001
            items.append({"name": name, "status": f"error: {exc!r}", "png": None})

    # Bundle every produced PNG into a single zip for slide-deck download
    zip_path = assets / "download_all.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for it in items:
            if it.get("png"):
                zf.write(assets / it["png"], arcname=it["png"])

    # index.html
    rows = []
    for it in items:
        if it.get("png"):
            rows.append(
                f"<section><h3>{it['name']}</h3>"
                f'<p class="kv">WAV: {it.get("wav","?")} '
                f'&middot; MIDI: {it.get("midi") or "—"}</p>'
                f'<img src="_assets/{it["png"]}" style="max-width:100%;border:1px solid #ccc"/>'
                f"</section>"
            )
        else:
            rows.append(f"<section><h3>{it['name']}</h3>"
                        f"<p class='skipped'>{it['status']}</p></section>")

    body = """<!doctype html>
<html><head><meta charset="utf-8"><title>Preprocessing demo</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       padding:24px; max-width:1100px; margin:0 auto; color:#1f2328 }}
section {{ margin:20px 0; padding:16px; border:1px solid #d0d7de; border-radius:6px }}
.kv {{ font-family: monospace; background:#f6f8fa; padding:6px; border-radius:4px }}
.skipped {{ color:#9a6700 }}
</style></head><body>
<h1>Preprocessing demo</h1>
<p>Each panel shows one stage of the DSP block applied to a raw user WAV
before training/inference: (1) raw waveform at native SR, (2) full-bandwidth
linear STFT (pre-LPF), (3) resampled mono waveform at 22050 Hz, (4) mel
filter bank overlay (the LPF emerges from <code>fmax=8000 Hz</code>),
(5) log-mel in dB at shape <code>(80, T)</code>, (6) normalized mel
in <code>[-1, 1]</code> with cyan lines marking the 5-second segment
boundaries used at training time. A 7th panel adds the aligned
piano-roll companion (onset+sustain) when MIDI is supplied.</p>
<p><a href="_assets/download_all.zip">⬇ download all PNGs (zip)</a></p>
""" + "\n".join(rows) + "\n</body></html>"
    html_path = stage_dir / "index.html"
    html_path.write_text(body, encoding="utf-8")

    n_ok = sum(1 for it in items if it.get("png"))
    return {
        "title": "01b. Preprocessing demo", "status": "ok" if n_ok else "skipped",
        "notes": f"{n_ok}/{len(items)} featured songs rendered.",
        "html": str(html_path),
    }


def _stage_fad(cfg: dict, cfg_path: Path, out_dir: Path) -> dict:
    """Build (or reuse) the FAD figures stage.

    If the config points at an already-computed finalist-metrics run we just
    copy those figures; otherwise we compute FAD from scratch. Returns a small
    dict describing the stage (title / status / notes / html) for the overview.
    """
    fad_cfg = cfg.get("fad")
    if not fad_cfg:
        return {"title": "02. FAD", "status": "skipped", "notes": "No `fad` section in config."}
    reuse = fad_cfg.get("reuse")
    if reuse:
        source_dir = _resolve(cfg_path, reuse["source_dir"])
        html, n = _reuse_gallery_stage(
            out_dir=out_dir, slug="02_fad", title="02. FAD (finalist metrics)",
            blurb=reuse.get("blurb", ""), source_dir=source_dir,
            files=reuse.get("files", []), dirs=reuse.get("dirs", []),
            globs=reuse.get("globs", []),
            source_index=_resolve(cfg_path, reuse["source_index"]) if reuse.get("source_index") else None,
        )
        return {"title": "02. FAD", "status": "ok",
                "notes": f"Reused {n} FAD figure(s) from the finalist-metrics run.",
                "html": str(html)}
    from postprocessing.results_visualizations import build_fad_visualizations

    real_dir = _resolve(cfg_path, fad_cfg["real_dir"])
    style_inputs = []
    for s in fad_cfg["style_inputs"]:
        style_inputs.append({
            "style_name": s["style_name"],
            "version_id": int(s["version_id"]),
            "real_dir": str(_resolve(cfg_path, s["real_dir"])),
            "generated_dir": str(_resolve(cfg_path, s["generated_dir"])),
            "version_manifest_csv": str(_resolve(cfg_path, s["version_manifest_csv"]))
                if s.get("version_manifest_csv") else None,
        })
    res = build_fad_visualizations(
        real_dir=str(real_dir), style_inputs=style_inputs,
        out_dir=out_dir / "02_fad",
        use_pretrained=bool(fad_cfg.get("use_pretrained", True)),
        sr=int(fad_cfg.get("sr", 22050)),
    )
    return {"title": "02. FAD", "status": "ok",
            "notes": f"Scored {len(style_inputs)} style(s).",
            "html": res["html"]}


def _stage_f1(cfg: dict, cfg_path: Path, out_dir: Path) -> dict:
    """Build (or reuse) the note-level F1 stage (same reuse-or-compute pattern)."""
    f1_cfg = cfg.get("f1")
    if not f1_cfg:
        return {"title": "03. F1 transcription", "status": "skipped",
                "notes": "No `f1` section in config."}
    reuse = f1_cfg.get("reuse")
    if reuse:
        source_dir = _resolve(cfg_path, reuse["source_dir"])
        html, n = _reuse_gallery_stage(
            out_dir=out_dir, slug="03_f1", title="03. Note-level F1 (finalist metrics)",
            blurb=reuse.get("blurb", ""), source_dir=source_dir,
            files=reuse.get("files", []), dirs=reuse.get("dirs", []),
            globs=reuse.get("globs", []),
            source_index=_resolve(cfg_path, reuse["source_index"]) if reuse.get("source_index") else None,
        )
        return {"title": "03. F1 transcription", "status": "ok",
                "notes": f"Reused {n} F1 figure(s) from the finalist-metrics run.",
                "html": str(html)}
    from postprocessing.results_visualizations import build_f1_visualizations

    pairs = []
    for p in f1_cfg["benchmark_pairs"]:
        pairs.append({
            "name": p["name"],
            "style_name": p.get("style_name", cfg.get("style_name", "")),
            "generated_wav": str(_resolve(cfg_path, p["generated_wav"])),
            "reference_midi": str(_resolve(cfg_path, p["reference_midi"])),
        })
    res = build_f1_visualizations(
        benchmark_pairs=pairs,
        out_dir=out_dir / "03_f1",
        basic_pitch_python=f1_cfg.get("basic_pitch_python"),
        onset_tolerance_s=float(f1_cfg.get("onset_tolerance_s", 0.05)),
    )
    return {"title": "03. F1 transcription", "status": "ok",
            "notes": f"Scored {len(pairs)} benchmark(s).",
            "html": res["html"]}


def _stage_latency(cfg: dict, cfg_path: Path, out_dir: Path) -> dict:
    """Build (or reuse) the latency stage from the recorded timing JSON."""
    lat_cfg = cfg.get("latency")
    if not lat_cfg:
        return {"title": "04. Latency", "status": "skipped",
                "notes": "No `latency` section in config."}
    reuse = lat_cfg.get("reuse")
    if reuse:
        return _stage_latency_reuse(reuse, cfg_path, out_dir)
    from postprocessing.results_visualizations import build_latency_visualizations

    timing_json = _resolve(cfg_path, lat_cfg["timing_json"])
    if not timing_json or not timing_json.exists():
        return {"title": "04. Latency", "status": "skipped",
                "notes": f"timing_json not found: {timing_json}"}
    with open(timing_json, "r", encoding="utf-8") as fh:
        timing_infos = json.load(fh)
    res = build_latency_visualizations(
        timing_infos=timing_infos, out_dir=out_dir / "04_latency",
        sample_rate=int(lat_cfg.get("sample_rate", 22050)),
        hop_length=int(lat_cfg.get("hop_length", 256)),
        segment_duration=float(lat_cfg.get("segment_duration", 5.0)),
    )
    return {"title": "04. Latency", "status": "ok",
            "notes": f"Profiled {len(timing_infos)} config(s).",
            "html": res["html"]}


def _stage_transfer(cfg: dict, cfg_path: Path, out_dir: Path) -> dict:
    """Build (or reuse) the style-transfer grid stage."""
    tr_cfg = cfg.get("transfer")
    if not tr_cfg:
        return {"title": "05. Transfer grid", "status": "skipped",
                "notes": "No `transfer` section in config."}
    reuse = tr_cfg.get("reuse")
    if reuse:
        source_dir = _resolve(cfg_path, reuse["source_dir"])
        html, n = _reuse_gallery_stage(
            out_dir=out_dir, slug="05_transfer_grid",
            title=tr_cfg.get("title", "05. Style-transfer grid"),
            blurb=reuse.get("blurb", ""), source_dir=source_dir,
            files=reuse.get("files", []), dirs=reuse.get("dirs", []),
            globs=reuse.get("globs", []),
            source_index=_resolve(cfg_path, reuse["source_index"]) if reuse.get("source_index") else None,
        )
        return {"title": "05. Transfer grid", "status": "ok",
                "notes": f"Reused {n} transfer figure(s) from the demo gallery.",
                "html": str(html)}
    from postprocessing.results_visualizations import build_transfer_grid
    items = []
    for it in tr_cfg["items"]:
        items.append({
            "name": it["name"],
            "style_name": it.get("style_name", cfg.get("style_name", "")),
            "input_mel_png":     str(_resolve(cfg_path, it.get("input_mel_png"))) if it.get("input_mel_png") else None,
            "generated_mel_png": str(_resolve(cfg_path, it.get("generated_mel_png"))) if it.get("generated_mel_png") else None,
            "input_wav":         str(_resolve(cfg_path, it.get("input_wav"))) if it.get("input_wav") else None,
            "generated_wav":     str(_resolve(cfg_path, it.get("generated_wav"))) if it.get("generated_wav") else None,
            "scores": it.get("scores", {}),
        })
    res = build_transfer_grid(items=items, out_dir=out_dir / "05_transfer_grid",
                              title=tr_cfg.get("title", "Style-transfer examples"))
    return {"title": "05. Transfer grid", "status": "ok",
            "notes": f"Bundled {res['n_items']} item(s) / {res['n_assets']} asset(s).",
            "html": res["html"]}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_all(config_path: Path, out_dir: Path) -> Path:
    """Run every deliverable stage listed in the config and write the index.

    Each stage is wrapped in try/except so that one stage failing (for example a
    missing input file) is recorded as an error but does not abort the others.
    Returns the path to the overview HTML.
    """
    cfg = _load_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.get("run_name", "unnamed_run")

    sections: List[dict] = []
    for fn in (_stage_data_quality, _stage_preprocessing, _stage_fad,
               _stage_f1, _stage_latency, _stage_transfer):
        try:
            sections.append(fn(cfg, config_path, out_dir))
        except Exception as e:  # noqa: BLE001
            sections.append({
                "title": fn.__name__.replace("_stage_", "stage: "),
                "status": "error",
                "notes": f"ERROR: {e!r}",
            })
    overview = _write_overview(out_dir, run_name, sections)
    print(f"\nDeliverables written to: {out_dir}")
    print(f"Open this first: {overview}")
    return overview


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: build all deliverables from a config file."""
    ap = argparse.ArgumentParser(description="Build TA-grading deliverables.")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    ap.add_argument("--out_dir", default="deliverables",
                    help="Output root (default: ./deliverables)")
    args = ap.parse_args(argv)
    config_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir).resolve()
    build_all(config_path, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
