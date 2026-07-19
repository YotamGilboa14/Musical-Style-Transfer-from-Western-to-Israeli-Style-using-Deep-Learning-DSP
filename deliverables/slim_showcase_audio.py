"""Slim a built showcase bundle for the web: convert every WAV to MP3.

GitHub Pages serves the showcase, so we trade the large uncompressed WAVs for
small MP3s (about 10x smaller, no audible difference for a demo). The original
high-quality WAVs are untouched in deliverables/curated_examples and on Drive.

Steps: convert each *.wav under the bundle to *.mp3 (mono, 22.05 kHz, 96 kbps),
delete the WAV, then rewrite the .wav references in index.html and MANIFEST.csv.

Usage (from MusicProject/):
    .\\ml_env\\Scripts\\python.exe deliverables\\slim_showcase_audio.py project_showcase_lite
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def convert(bundle: Path) -> None:
    wavs = list(bundle.rglob("*.wav"))
    print(f"Converting {len(wavs)} WAV -> MP3 in {bundle} ...")
    ok = 0
    for w in wavs:
        mp3 = w.with_suffix(".mp3")
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(w),
             "-ac", "1", "-ar", "22050", "-b:a", "96k", str(mp3)],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and mp3.exists():
            w.unlink()
            ok += 1
        else:
            print(f"  !! ffmpeg failed on {w.name}: {r.stderr.strip()[:120]}")
    print(f"  converted {ok}/{len(wavs)}")

    # Rewrite .wav -> .mp3 in the page and the manifest.
    for name in ("index.html", "MANIFEST.csv"):
        f = bundle / name
        if f.exists():
            f.write_text(f.read_text(encoding="utf-8").replace(".wav", ".mp3"),
                         encoding="utf-8")
            print(f"  rewrote references in {name}")

    total = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    print(f"Bundle now: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    b = Path(sys.argv[1] if len(sys.argv) > 1 else "project_showcase_lite").resolve()
    convert(b)
