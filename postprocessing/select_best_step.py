"""
select_best_step.py — Composite z-score step selection with human tiebreak.

Used at the end of training to pick the best checkpoint step. Strategy:

  1. Run inference on a held-out song set across N candidate steps.
  2. Compute FAD (lower is better) and F1 (higher is better) per step.
  3. Z-score both metrics across the candidate set and combine::

        composite_z = z(FAD) + z(1 - F1)        # lower = better

  4. Take the top 3 lowest composite_z as finalists.
  5. Produce a composite ``index.html`` with audio players + mel images +
     metrics tables so a human listener can break the tie. The convention
     when audio quality is tied is to prefer the **earliest** step
     (smaller models / less overfit risk).

Outputs
-------
    <out_dir>/
        step_selection/
            step_<N>/
                audio/   *.wav        (copied from inference run)
                mels/    *.mel.pt
                metrics.json
            index.html
            ranking.csv

This script does NOT run inference itself — it consumes a directory of
``inference_runs/<run_id>/`` directories produced by
``run_inference_batch.py``. Pass each candidate run via ``--run`` (repeatable)
or point ``--runs-root`` at a parent directory containing several runs.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Iterable


def _read_run(run_dir: Path) -> dict | None:
    """Read one inference_runs/<run_id>/ directory. Returns None if invalid."""
    summary = run_dir / "_summary.csv"
    metrics = run_dir / "metrics.json"
    if not summary.exists() or not metrics.exists():
        return None
    with open(summary, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(metrics, "r", encoding="utf-8") as fh:
        per_stem = json.load(fh)
    return {"run_dir": run_dir, "rows": rows, "metrics": per_stem}


def _aggregate_step_metrics(rows: list[dict], per_stem: dict) -> dict[int, dict]:
    """Average FAD/F1 per step across all songs in a run."""
    by_step: dict[int, list[dict]] = {}
    for row in rows:
        step = int(row["step"])
        m = per_stem.get(row["stem"], {})
        by_step.setdefault(step, []).append(m)
    out: dict[int, dict] = {}
    for step, items in by_step.items():
        fads = [m["fad"] for m in items if m.get("fad") is not None]
        f1s = [m["f1"] for m in items if m.get("f1") is not None]
        out[step] = {
            "fad_mean": float(sum(fads) / len(fads)) if fads else None,
            "f1_mean": float(sum(f1s) / len(f1s)) if f1s else None,
            "n_songs": len({r["song"] for r in rows if int(r["step"]) == step}),
        }
    return out


def _zscore(values: list[float]) -> list[float]:
    """Standard z-score; returns zeros if values is degenerate (n<2 or std=0)."""
    if len(values) < 2:
        return [0.0] * len(values)
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0] * len(values)
    return [(v - mu) / sd for v in values]


def select_best(runs: Iterable[Path], out_dir: Path, top_k: int = 3) -> dict:
    """Aggregate metrics across runs, compute composite z-score, copy top-k."""
    out_dir = out_dir.resolve()
    sel_dir = out_dir / "step_selection"
    sel_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate every (run_id, step) into one row
    rows: list[dict] = []
    for run_dir in runs:
        info = _read_run(run_dir)
        if info is None:
            print(f"  ⚠ skipping invalid run dir: {run_dir}")
            continue
        agg = _aggregate_step_metrics(info["rows"], info["metrics"])
        for step, m in agg.items():
            rows.append({
                "run_id": run_dir.name,
                "step": step,
                "fad": m["fad_mean"],
                "f1": m["f1_mean"],
                "n_songs": m["n_songs"],
                "run_dir": str(run_dir),
            })

    if not rows:
        raise RuntimeError("no valid runs found")

    # Only rows that have BOTH FAD and F1 can be ranked
    rankable = [r for r in rows if r["fad"] is not None and r["f1"] is not None]
    if not rankable:
        print("  ⚠ no rows have both FAD and F1 — falling back to FAD-only ranking")
        rankable = [r for r in rows if r["fad"] is not None]
        if not rankable:
            raise RuntimeError("no row has FAD; cannot rank")
        fad_z = _zscore([r["fad"] for r in rankable])
        for r, z in zip(rankable, fad_z):
            r["composite_z"] = z
    else:
        fad_z = _zscore([r["fad"] for r in rankable])
        inv_f1_z = _zscore([(1.0 - r["f1"]) for r in rankable])
        for r, zf, zi in zip(rankable, fad_z, inv_f1_z):
            r["composite_z"] = zf + zi

    rankable.sort(key=lambda r: (r["composite_z"], r["step"]))  # lower is better, tie → earlier
    finalists = rankable[:top_k]

    # Write ranking.csv
    ranking_csv = sel_dir / "ranking.csv"
    with open(ranking_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "rank", "run_id", "step", "fad", "f1", "composite_z", "n_songs",
        ])
        w.writeheader()
        for i, r in enumerate(rankable, start=1):
            w.writerow({
                "rank": i, "run_id": r["run_id"], "step": r["step"],
                "fad": r["fad"], "f1": r["f1"],
                "composite_z": r["composite_z"], "n_songs": r["n_songs"],
            })

    # Copy finalists' audio + mels into step_<N>/
    for r in finalists:
        step_dir = sel_dir / f"step_{r['step']}"
        audio_dst = step_dir / "audio"
        mels_dst = step_dir / "mels"
        audio_dst.mkdir(parents=True, exist_ok=True)
        mels_dst.mkdir(parents=True, exist_ok=True)
        src_audio = Path(r["run_dir"]) / "audio"
        src_mels = Path(r["run_dir"]) / "mels"
        for f in src_audio.glob(f"*step_{r['step']}*.wav"):
            shutil.copy2(f, audio_dst / f.name)
        for f in src_mels.glob(f"*step_{r['step']}*.mel.pt"):
            shutil.copy2(f, mels_dst / f.name)
        with open(step_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump({
                "step": r["step"], "fad": r["fad"], "f1": r["f1"],
                "composite_z": r["composite_z"], "run_id": r["run_id"],
            }, fh, indent=2)

    # Composite HTML
    _write_index_html(sel_dir, rankable, finalists)

    return {
        "selection_dir": str(sel_dir),
        "n_candidates": len(rankable),
        "finalists": [{"step": r["step"], "composite_z": r["composite_z"],
                       "fad": r["fad"], "f1": r["f1"]} for r in finalists],
        "ranking_csv": str(ranking_csv),
    }


def _write_index_html(sel_dir: Path, rankable: list[dict], finalists: list[dict]) -> None:
    """Render a self-contained HTML page for human tiebreak."""
    rows_html = []
    for i, r in enumerate(rankable, start=1):
        is_finalist = r in finalists
        rows_html.append(
            "<tr{cls}><td>{rank}</td><td>{run}</td><td>{step}</td>"
            "<td>{fad}</td><td>{f1}</td><td>{cz:.3f}</td><td>{n}</td></tr>".format(
                cls=' style="background:#dff;font-weight:600;"' if is_finalist else "",
                rank=i, run=html.escape(r["run_id"]), step=r["step"],
                fad=f"{r['fad']:.3f}" if r.get("fad") is not None else "—",
                f1=f"{r['f1']:.3f}" if r.get("f1") is not None else "—",
                cz=r["composite_z"], n=r["n_songs"],
            )
        )

    finalist_sections = []
    for r in finalists:
        step_dir = sel_dir / f"step_{r['step']}"
        audio_files = sorted((step_dir / "audio").glob("*.wav"))
        players = "\n".join(
            f'<div><div>{html.escape(p.name)}</div>'
            f'<audio controls src="step_{r["step"]}/audio/{p.name}"></audio></div>'
            for p in audio_files
        )
        finalist_sections.append(
            f'<section><h3>step {r["step"]} '
            f'(composite_z={r["composite_z"]:.3f}, FAD={r["fad"]}, F1={r["f1"]})</h3>'
            f'<div style="display:flex;gap:1em;flex-wrap:wrap;">{players}</div></section>'
        )

    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Step selection</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 0.95em; }}
  th {{ background: #eef; }}
  audio {{ width: 320px; }}
</style>
</head><body>
<h1>Step selection — composite z-score</h1>
<p>composite_z = z(FAD) + z(1 - F1). Lower is better. Tie-break: prefer earlier step.</p>
<table>
<tr><th>rank</th><th>run_id</th><th>step</th><th>FAD</th><th>F1</th><th>composite_z</th><th>n_songs</th></tr>
{"".join(rows_html)}
</table>
<h2>Top {len(finalists)} finalists — listen for tiebreak</h2>
{"".join(finalist_sections)}
</body></html>"""
    (sel_dir / "index.html").write_text(body, encoding="utf-8")


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--run", action="append", type=Path, default=[],
                    help="Path to one inference_runs/<run_id>/ dir (repeatable)")
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="Auto-discover every subdirectory under this path as a run")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where to write step_selection/")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    runs: list[Path] = list(args.run)
    if args.runs_root and args.runs_root.exists():
        runs.extend(p for p in args.runs_root.iterdir()
                    if p.is_dir() and p.name != "step_selection" and not p.name.startswith("_"))
    if not runs:
        ap.error("no runs provided (use --run or --runs-root)")

    summary = select_best(runs, args.out_dir, top_k=args.top_k)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
