"""
Pipeline integration test — 4 configs.

Configs (all on the same downloaded WAV to keep YouTube I/O at one):
    A: source-sep OFF + local storage
    B: source-sep ON  + local storage
    C: source-sep OFF + Drive upload + local clean
    D: source-sep ON  + Drive upload + local clean

After all 4 ingest configs, sanity-check postprocessing on the segments produced
by config A:
    - data augmentation round-trip (JointAugment → vocoder → audio sanity)
    - BigVGAN vocoder round-trip + RTF latency
    - F1 (Basic-Pitch on vocoded WAV vs reference MIDI)
    - All-FAD + Group-FAD on shipped fixtures
    - FAD PCA visualization (with song-specific title)

Run from MusicProject/ with:
    $env:PYTHONIOENCODING='utf-8'; .\\ml_env\\Scripts\\python.exe tests\\pipeline_full_test.py
    # optional: --url <youtube_url> --artist X --album Y --song Z
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

# Ensure repo root is importable when invoked from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from preprocessing.youtube_downloader import (  # noqa: E402
    download_youtube_audio,
    extract_youtube_metadata,
)
from process_song_offline import process_song  # noqa: E402

# Defaults — overridable via CLI
DEFAULT_URL = "https://www.youtube.com/watch?v=3IRJ9oTakkE"
DEFAULT_ARTIST = "Arik Einstein"
DEFAULT_ALBUM = "Sa Leat"
DEFAULT_SONG = "Yeladim Shel Hachaim"
VERSION_ID = 1

ROOT = Path(__file__).resolve().parent / "_pipeline_full_test_out"
DOWNLOAD_DIR = ROOT / "_download"
RESULTS: list[tuple[str, str, str]] = []  # (test, status, detail)


def banner(text: str) -> None:
    """Print a titled separator line so each stage stands out in the test log."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def shape_check(song_dir: Path, label: str) -> int:
    """Assert the written tensors have the right shape and normalization.

    Checks that mel segments are [80, 430], piano rolls are [2, 128, 430], the
    counts match and the mel sits in [-1, 1]. Returns how many segments were
    produced.
    """
    import glob
    mel_files = sorted(glob.glob(str(song_dir / "**" / "mels" / "segment_*.pt"), recursive=True))
    pr_files = sorted(glob.glob(str(song_dir / "**" / "piano_rolls" / "segment_*.pt"), recursive=True))
    assert mel_files, f"{label}: no mel tensors written"
    assert len(mel_files) == len(pr_files), f"{label}: mel/pr count mismatch"
    mel = torch.load(mel_files[0], weights_only=True)
    pr = torch.load(pr_files[0], weights_only=True)
    assert mel.shape == (80, 430), f"{label}: mel shape {mel.shape}"
    assert pr.shape == (2, 128, 430), f"{label}: pr shape {pr.shape}"
    assert -1.0 <= mel.min() and mel.max() <= 1.0, f"{label}: mel not normalized"
    print(f"    {label}: {len(mel_files)} segments, mel{tuple(mel.shape)} pr{tuple(pr.shape)} "
          f"mel∈[{mel.min():.3f},{mel.max():.3f}]")
    return len(mel_files)


def run_config(name: str, sep: bool, drive: bool, source_wav: Path, out_root: Path,
               artist: str, album: str, song: str) -> Path | None:
    """Run one end-to-end pipeline configuration and record PASS/FAIL.

    A "configuration" is a combination of stem-separation on/off and Drive
    upload on/off. Returns the local song directory when the data stays on disk
    (so postprocessing can be tested on it), or None when it was uploaded and
    the local copy cleaned.
    """
    banner(f"CONFIG {name} — separate_stems={sep}, drive_upload={drive}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    row = {
        "url": "",
        "version_id": VERSION_ID,
        "artist": artist, "album": album, "song_name": song,
        "source_wav": str(source_wav),
    }
    t0 = time.time()
    res = process_song(row, out_root=out_root, skip_if_exists=False, separate_stems=sep)
    dt = time.time() - t0
    if res["status"] != "ok":
        RESULTS.append((name, "FAIL", f"process_song: {res.get('error')}"))
        return None
    song_dir = Path(res["song_dir"])
    n = shape_check(song_dir, f"config {name} tensors")
    print(f"    process_song: {dt:.1f}s, {n} segments")

    if drive:
        from preprocessing.drive_sync import (
            upload_song_to_drive, clean_local_song, get_or_create_music_data_folder,
        )
        print(f"    Resolving existing Drive folder MusicProject/MusicProjectData …")
        drive_id = get_or_create_music_data_folder("MusicProject/MusicProjectData")
        print(f"    Drive folder ID: {drive_id}")
        t0 = time.time()
        upload_song_to_drive(song_dir, drive_id)
        print(f"    Upload: {time.time()-t0:.1f}s")
        clean_local_song(song_dir)
        assert not song_dir.exists(), f"{name}: local song_dir still exists after clean"
        print(f"    Local copy cleaned. Data on Drive only.")
        RESULTS.append((name, "PASS", f"{n} segs, sep={sep}, uploaded+cleaned"))
        return None  # nothing local left to test postprocessing on
    else:
        RESULTS.append((name, "PASS", f"{n} segs, sep={sep}, local at {song_dir}"))
        return song_dir


def run_postprocessing_test(song_dir: Path, source_wav: Path,
                             song_label: str) -> None:
    """Run the postprocessing chain on the produced segments.

    Exercises augmentation, the vocoder, latency, F1, FAD and the
    visualizations end-to-end so a single test covers everything after the
    model output.
    """
    banner("POSTPROCESSING — augmentation + vocoder + latency + F1 + FAD + visualize")
    import glob
    import numpy as np

    mel_files = sorted(glob.glob(str(song_dir / "**" / "mels" / "segment_*.pt"), recursive=True))
    pr_files = sorted(glob.glob(str(song_dir / "**" / "piano_rolls" / "segment_*.pt"), recursive=True))
    assert mel_files and pr_files, "no segment tensors found for postprocessing"
    mel = torch.load(mel_files[0], weights_only=True)
    pr = torch.load(pr_files[0], weights_only=True)
    print(f"  Loaded mel {tuple(mel.shape)} + pr {tuple(pr.shape)} from segment_0000.pt")

    # ── 1. Vocoder factory ─────────────────────────────────────────────
    from postprocessing.vocoder_factory import create_vocoder
    voc = create_vocoder("bigvgan_22k")
    print(f"  Vocoder: {type(voc).__name__}")

    # ── 2. Augmentation block (T_aug) ──────────────────────────────────
    print("\n  [T_aug] Augmentation round-trip …")
    from preprocessing.augmentation import JointAugment
    aug_cfg = {
        "enabled": True,
        "pitch_shift":  {"p": 1.0, "max_semitones": 2},
        "time_stretch": {"p": 1.0, "max_pct": 0.05},
        "spec_augment": {"p": 1.0, "time_mask_max": 20, "freq_mask_max": 8,
                         "n_time": 2, "n_freq": 2},
    }
    aug = JointAugment(aug_cfg)
    mel_aug, pr_aug = aug(mel, pr)
    assert mel_aug.shape == mel.shape, f"aug changed mel shape: {mel_aug.shape}"
    assert pr_aug.shape == pr.shape, f"aug changed pr shape: {pr_aug.shape}"
    assert not torch.isnan(mel_aug).any(), "augmented mel has NaN"
    assert not torch.equal(mel_aug, mel), "augmentation produced identical mel (no-op)"
    delta = (mel_aug - mel).abs().mean().item()
    print(f"    shapes preserved, mean |Δmel|={delta:.4f}, no NaN")
    RESULTS.append(("aug_tensor", "PASS",
                    f"shape OK, |Δmel|={delta:.4f}, pr changed={not torch.equal(pr_aug, pr)}"))

    # ── 3. Vocoder round-trip + RTF latency ────────────────────────────
    print("\n  [T_vocode] BigVGAN round-trip + RTF …")
    out_wav = song_dir / "_pipeline_test_vocoded.wav"
    t0 = time.time()
    audio = voc.wav_to_wav(str(source_wav), str(out_wav))
    voc_dt = time.time() - t0
    assert audio.size > 0, "vocoder produced empty audio"
    audio_dur_s = audio.shape[0] / 22050.0
    rtf = voc_dt / audio_dur_s if audio_dur_s > 0 else float("inf")
    print(f"    vocoded {audio_dur_s:.1f}s in {voc_dt:.2f}s  →  RTF = {rtf:.3f} "
          f"({'real-time' if rtf <= 1.0 else 'slower than real-time'})")
    RESULTS.append(("postproc_vocode", "PASS", f"vocoded WAV @ {out_wav.name}"))
    RESULTS.append(("latency_rtf", "PASS",
                    f"vocoder RTF={rtf:.3f} ({voc_dt:.2f}s for {audio_dur_s:.1f}s audio)"))

    # ── 4. F1 metric on round-tripped WAV vs reference MIDI ────────────
    print("\n  [T_f1] F1 on round-tripped WAV vs reference MIDI …")
    from postprocessing.f1_eval import compute_f1
    project_root = Path(__file__).resolve().parent.parent
    bp_python = project_root / "basic_pitch_env" / "Scripts" / "python.exe"
    ref_midi = song_dir / f"{song_dir.name}.mid"
    if not ref_midi.exists():
        # Search song_dir top level for any .mid
        mids = list(song_dir.glob("*.mid"))
        ref_midi = mids[0] if mids else None
    if ref_midi is None or not bp_python.exists():
        print(f"    SKIP — ref_midi or basic_pitch_env missing "
              f"(midi={ref_midi}, bp_python_exists={bp_python.exists()})")
        RESULTS.append(("postproc_f1", "SKIP",
                        "missing reference MIDI or basic_pitch_env"))
    else:
        f1_res = compute_f1(out_wav, ref_midi, basic_pitch_python=str(bp_python))
        print(f"    F1={f1_res['f1']:.3f}  P={f1_res['precision']:.3f}  "
              f"R={f1_res['recall']:.3f}  matched={f1_res['matched']}/"
              f"{f1_res['n_predicted']} pred, {f1_res['n_reference']} ref")
        RESULTS.append(("postproc_f1", "PASS",
                        f"F1={f1_res['f1']:.3f} (P={f1_res['precision']:.3f}, "
                        f"R={f1_res['recall']:.3f})"))

    # ── 5. All-FAD + Group-FAD on shipped fixtures ─────────────────────
    print("\n  [T_fad] All-FAD + Group-FAD on shipped fixtures …")
    from postprocessing.fad_eval import compute_fad, compute_group_fad
    fad_real = project_root / "benchmark_output" / "fad_real"
    fad_gen = project_root / "benchmark_output" / "fad_generated"
    all_fad = compute_fad(str(fad_real), str(fad_gen))
    grp_fad = compute_group_fad(str(fad_real), str(fad_gen))
    print(f"    All-FAD = {all_fad:.4f}    Group-FAD = {grp_fad['group_fad']:.4f}")
    assert all_fad >= 0 and grp_fad["group_fad"] >= 0
    RESULTS.append(("postproc_fad", "PASS",
                    f"All-FAD={all_fad:.4f}, Group-FAD={grp_fad['group_fad']:.4f}"))

    # ── 6. FAD visualization with song-specific title ──────────────────
    print("\n  [T_viz] FAD visualization (PCA + ellipses) …")
    from postprocessing.fad_visualize import visualize_fad
    viz_out = song_dir / "_pipeline_test_fad_viz.png"
    viz_title = (
        f"Fréchet Audio Distance — {song_label}\n"
        "Real vs Generated   (VGGish embeddings, PCA projection)"
    )
    visualize_fad(str(fad_real), str(fad_gen), str(viz_out), title=viz_title)
    assert viz_out.exists() and viz_out.stat().st_size > 0, "FAD viz not produced"
    print(f"    FAD viz: {viz_out} ({viz_out.stat().st_size} bytes)")
    RESULTS.append(("postproc_visualize", "PASS", f"FAD PCA plot @ {viz_out.name}"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full-pipeline test."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",    default=DEFAULT_URL,    help="YouTube URL")
    p.add_argument("--artist", default=None,
                   help="Override artist (default: extracted from YouTube metadata)")
    p.add_argument("--album",  default=None,
                   help="Override album  (default: extracted from YouTube metadata)")
    p.add_argument("--song",   default=None,
                   help="Override song   (default: extracted from YouTube metadata)")
    p.add_argument("--fresh-download", action="store_true",
                   help="Force re-download even if a cached WAV exists")
    p.add_argument("--only", default="ABCD",
                   help="Subset of configs to run, e.g. 'D' or 'CD'. "
                        "Postprocessing runs only if 'A' is included.")
    return p.parse_args()


def main() -> int:
    """Run the selected pipeline configs and print a PASS/FAIL summary."""
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resolve metadata from YouTube unless overridden ──────────────
    banner(f"METADATA — {args.url}")
    if any(v is None for v in (args.artist, args.album, args.song)):
        meta = extract_youtube_metadata(args.url)
        artist = args.artist or meta["artist"]
        album  = args.album  or meta["album"]
        song   = args.song   or meta["song"]
        print(f"  YT title : {meta['title']}")
        print(f"  YT duration: {meta.get('duration', 0)} s")
    else:
        artist, album, song = args.artist, args.album, args.song
    print(f"  Artist : {artist}")
    print(f"  Album  : {album}")
    print(f"  Song   : {song}")
    song_label = f"{artist} — {song}"

    # Step 1: download once (cache shared by all 4 configs)
    banner(f"DOWNLOAD — {args.url}")
    existing = list(DOWNLOAD_DIR.glob("*.wav"))
    if existing and not args.fresh_download:
        wav = existing[0]
        print(f"  Reusing cached WAV: {wav}")
    else:
        if args.fresh_download:
            for f in existing:
                f.unlink()
            print("  --fresh-download: cleared cached WAVs")
        out = download_youtube_audio(args.url, DOWNLOAD_DIR, audio_format="wav")
        wav = out[0] if isinstance(out, tuple) else out
        print(f"  Downloaded: {wav}")
    wav = Path(wav)
    assert wav.exists() and wav.stat().st_size > 0

    # Step 2: configs (subset selectable via --only)
    only = args.only.upper()
    print(f"  Running configs: {' '.join(only)}")
    song_dir_A = None
    try:
        if "A" in only:
            song_dir_A = run_config("A", sep=False, drive=False,
                                    source_wav=wav, out_root=ROOT / "A_local_nosep",
                                    artist=artist, album=album, song=song)
        if "B" in only:
            run_config("B", sep=True, drive=False,
                       source_wav=wav, out_root=ROOT / "B_local_sep",
                       artist=artist, album=album, song=song)
        if "C" in only:
            run_config("C", sep=False, drive=True,
                       source_wav=wav, out_root=ROOT / "C_drive_nosep",
                       artist=artist, album=album, song=song)
        if "D" in only:
            run_config("D", sep=True, drive=True,
                       source_wav=wav, out_root=ROOT / "D_drive_sep",
                       artist=artist, album=album, song=song)

        # Step 3: postprocessing on config A's surviving local copy
        if "A" in only:
            if song_dir_A is not None and song_dir_A.exists():
                run_postprocessing_test(song_dir_A, wav, song_label=song_label)
            else:
                RESULTS.append(("postproc", "SKIP", "no local song_dir to test against"))
        else:
            RESULTS.append(("postproc", "SKIP", f"config A not in --only={only}"))

    except Exception as e:
        traceback.print_exc()
        RESULTS.append(("EXCEPTION", "FAIL", repr(e)))

    # Summary
    banner("SUMMARY")
    width = max(len(r[0]) for r in RESULTS)
    fail = 0
    for name, status, detail in RESULTS:
        flag = "OK " if status in ("PASS", "PASSED") else "!! "
        if status not in ("PASS", "PASSED", "SKIP"):
            fail += 1
        print(f"  {flag}{name.ljust(width)}  {status:<7}  {detail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
