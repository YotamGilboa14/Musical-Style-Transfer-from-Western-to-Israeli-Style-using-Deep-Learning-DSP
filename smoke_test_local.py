"""Local smoke tests — T1 through T7, plus metric verification tests.

T1       — Full Path B: YouTube download → local preprocess → upload to Drive → clean local
T6       — Model forward pass (shape check)
T7       — batch_ingest + split_dataset CLI round-trip
T_import — all new modules import cleanly
T_f1     — F1 metric end-to-end on Surprise Symphony trumpet/violin benchmark pair (V13)
T_fad    — Group-FAD identity + sensitivity check on benchmark WAVs (V15)
T_aug    — JointAugment audio round-trip: vocoded f0 ratio matches +2 semitones

Run:
  python smoke_test_local.py                    # all tests
  python smoke_test_local.py --skip-t1          # skip T1 (no internet / no Drive needed)
  python smoke_test_local.py --skip-t-aug       # skip T_aug (no BigVGAN download)

Basic-Pitch tests use basic_pitch_env/Scripts/python.exe (Python 3.10 env).
"""
import sys
import os
import argparse
import subprocess
import tempfile
from pathlib import Path

# Ensure CWD is always the MusicProject root, regardless of where the script is invoked
_PROJECT_ROOT = Path(__file__).parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS = {}

# ── T1: full Path B — download → preprocess → upload → clean ─────────────────
def test_t1():
    import torch
    from process_song_offline import process_song
    from preprocessing.drive_sync import (
        get_or_create_music_data_folder,
        upload_song_to_drive,
        clean_local_song,
    )

    LOCAL_OUT = Path("C:/tmp/smoke_t1")
    row = {
        "url":        "https://www.youtube.com/watch?v=3IRJ9oTakkE",
        "version_id": 1,
        "artist":     "Arik Einstein",
        "album":      "Sa Leat",
        "song_name":  "Yeladim Shel Hachaim",
    }

    # Step 1: process locally (writes to C:/tmp/smoke_t1)
    print("  Step 1/3 — downloading and preprocessing locally …")
    result = process_song(row, out_root=LOCAL_OUT, skip_if_exists=False)
    assert result["status"] == "ok", f"T1 FAIL (preprocess): {result['error']}"

    song_dir = Path(result["song_dir"])
    assert song_dir.exists(), "T1 FAIL: song_dir not created"

    # Verify tensor shapes
    import glob
    mel_files = sorted(glob.glob(str(song_dir / "**" / "mels" / "segment_*.pt"), recursive=True))
    pr_files  = sorted(glob.glob(str(song_dir / "**" / "piano_rolls" / "segment_*.pt"), recursive=True))
    assert len(mel_files) > 0, "T1 FAIL: no mel tensors written"
    assert len(mel_files) == len(pr_files), "T1 FAIL: mel/piano_roll count mismatch"

    mel = torch.load(mel_files[0], weights_only=True)
    pr  = torch.load(pr_files[0],  weights_only=True)
    assert mel.shape == (80, 430),      f"T1 FAIL: mel shape {mel.shape}"
    assert pr.shape  == (2, 128, 430),  f"T1 FAIL: pr shape {pr.shape}"
    assert mel.min() >= -1.0 and mel.max() <= 1.0, "T1 FAIL: mel not normalized"

    print(f"  Preprocessed: {len(mel_files)} segments, mel {mel.shape}, pr {pr.shape}")
    print(f"  mel range [{mel.min():.3f}, {mel.max():.3f}]")
    print(f"  piano_roll active frames: {(pr[1] > 0).sum().item()}")

    # Step 2: get/create Drive folder and upload
    print("  Step 2/3 — resolving Drive folder and uploading …")
    drive_folder_id = get_or_create_music_data_folder("MusicProject/MusicProjectData")
    upload_song_to_drive(song_dir, drive_folder_id)

    # Step 3: clean local copy
    print("  Step 3/3 — cleaning local copy …")
    clean_local_song(song_dir)
    assert not song_dir.exists(), "T1 FAIL: local song_dir still exists after clean"

    print(f"  Local copy removed. Data lives only on Drive.")
    print(f"  Drive folder ID: {drive_folder_id}")
    return "PASSED"

# ── T6: model forward pass ────────────────────────────────────────────────────
def test_t6():
    import torch
    from omegaconf import OmegaConf
    from model.unet import UNet1D

    cfg = OmegaConf.load("configs/default.yaml")
    n_versions = int(cfg.conditioning.n_versions)
    null_idx   = int(cfg.conditioning.null_version_idx)
    assert n_versions == 3, \
        f"T6 FAIL: n_versions={n_versions} (expected 3 for Israeli_3style: slakh 0 + artists 1 + military 2)"
    assert null_idx == n_versions, \
        f"T6 FAIL: null_version_idx={null_idx} must equal n_versions={n_versions}"

    model = UNet1D(
        mel_channels=cfg.model.mel_channels,
        score_channels=cfg.model.score_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=list(cfg.model.channel_mults),
        num_res_blocks_enc=cfg.model.num_res_blocks_enc,
        num_res_blocks_dec=cfg.model.num_res_blocks_dec,
        attention_levels=list(cfg.model.attention_levels),
        attn_heads=cfg.model.attention_heads,
        n_groups=cfg.model.n_groups,
        dropout=cfg.model.dropout,
        n_versions=n_versions,
        version_emb_dim=cfg.conditioning.version_emb_dim,
        time_emb_dim=cfg.conditioning.time_emb_dim,
    ).eval()
    B = 2
    mel   = torch.randn(B, 80, 430)
    score = torch.randn(B, 256, 430)
    t     = torch.randint(0, 1000, (B,))
    out_shapes = {}
    for vid_value in (0, 1, 2):
        vid = torch.full((B,), vid_value, dtype=torch.long)
        with torch.no_grad():
            out = model(mel, t, score, vid)
        assert out.shape == (B, 80, 430), \
            f"T6 FAIL: version_id={vid_value} output shape {out.shape}"
        out_shapes[vid_value] = tuple(out.shape)
    params = sum(p.numel() for p in model.parameters())
    print(f"  n_versions       = {n_versions}")
    print(f"  null_version_idx = {null_idx}")
    print(f"  out v0/v1/v2      = {out_shapes[0]} / {out_shapes[1]} / {out_shapes[2]}")
    print(f"  model params     = {params:,}")
    return "PASSED"


# ── T7: batch_ingest + split_dataset CLI round-trip ──────────────────────────
def test_t7():
    import csv
    import pandas as pd

    py = sys.executable
    tmp = Path(tempfile.mkdtemp(prefix="smoke_t7_"))

    # Write a minimal fake manifest instead of downloading YouTube
    # (batch_ingest will skip songs that are already processed;
    #  we pre-create the processed_data dir + manifest_song.csv to simulate)
    out_root = tmp / "data"

    for i, song in enumerate(["Song_A", "Song_B", "Song_C"]):
        song_dir = out_root / "TestArtist" / "TestAlbum" / song / "processed_data"
        song_dir.mkdir(parents=True, exist_ok=True)
        mels_dir = song_dir / "mels"
        prs_dir  = song_dir / "piano_rolls"
        mels_dir.mkdir(); prs_dir.mkdir()

        # Write fake manifest_song.csv (no actual tensor files needed for manifest test)
        manifest_path = song_dir / "manifest_song.csv"
        rows = []
        for seg in range(3):
            rows.append({
                "segment_path": str(Path("TestArtist/TestAlbum") / song / "processed_data/mels" / f"segment_{seg:04d}.pt"),
                "score_path":   str(Path("TestArtist/TestAlbum") / song / "processed_data/piano_rolls" / f"segment_{seg:04d}.pt"),
                "version_id":   1,
                "song_id":      f"TestArtist__TestAlbum__{song}",
                "artist":       "TestArtist",
                "album":        "TestAlbum",
                "song_name":    song,
                "segment_idx":  seg,
                "duration_s":   5.0,
            })
        fields = ["segment_path","score_path","version_id","song_id",
                  "artist","album","song_name","segment_idx","duration_s"]
        with open(manifest_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # Write a minimal batch_songs.csv pointing at these fake songs
    batch_csv = tmp / "batch_songs.csv"
    with open(batch_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index","url","version_id","artist",
                                            "album","song_name","enabled","notes"])
        w.writeheader()
        for i, song in enumerate(["Song_A","Song_B","Song_C"]):
            w.writerow({"index":i,"url":"","version_id":1,"artist":"TestArtist",
                        "album":"TestAlbum","song_name":song,"enabled":"true","notes":""})

    manifest_out = tmp / "manifest.csv"
    log_out      = tmp / "log.csv"

    # Run batch_ingest (all songs already have manifest_song.csv → will be "skipped")
    r = subprocess.run([
        py, "preprocessing/batch_ingest.py",
        "--csv",          str(batch_csv),
        "--out_root",     str(out_root),
        "--manifest_out", str(manifest_out),
        "--log",          str(log_out),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("  STDERR:", r.stderr[-500:])
    assert r.returncode == 0, f"T7 FAIL: batch_ingest exited {r.returncode}"
    assert manifest_out.exists(), "T7 FAIL: manifest.csv not created"

    df = pd.read_csv(manifest_out)
    assert len(df) == 9, f"T7 FAIL: expected 9 segments, got {len(df)}"
    assert set(df.columns) >= {"segment_path","score_path","version_id","song_id"}, \
        f"T7 FAIL: missing columns in manifest"

    # Run split_dataset
    split_dir = tmp / "splits"
    r2 = subprocess.run([
        py, "preprocessing/split_dataset.py",
        "--manifest", str(manifest_out),
        "--out_dir",  str(split_dir),
        "--train", "0.6", "--val", "0.2", "--test", "0.2",
        "--seed", "42",
    ], capture_output=True, text=True)
    print("  split output:", r2.stdout.strip())
    if r2.returncode != 0:
        print("  STDERR:", r2.stderr[-500:])
    assert r2.returncode == 0, f"T7 FAIL: split_dataset exited {r2.returncode}"

    # Leakage check
    train_ids = set(pd.read_csv(split_dir / "train.csv")["song_id"])
    val_ids   = set(pd.read_csv(split_dir / "val.csv")  ["song_id"])
    test_ids  = set(pd.read_csv(split_dir / "test.csv") ["song_id"])
    assert train_ids.isdisjoint(val_ids),  "T7 FAIL: LEAK train∩val"
    assert train_ids.isdisjoint(test_ids), "T7 FAIL: LEAK train∩test"
    assert val_ids.isdisjoint(test_ids),   "T7 FAIL: LEAK val∩test"

    total_segs = (len(pd.read_csv(split_dir / "train.csv")) +
                  len(pd.read_csv(split_dir / "val.csv")) +
                  len(pd.read_csv(split_dir / "test.csv")))
    assert total_segs == 9, f"T7 FAIL: split total {total_segs} != 9"

    print(f"  songs: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")
    print(f"  segments total: {total_segs}  (no leakage)")
    return "PASSED"


# ── T_import: verify all new modules import cleanly ──────────────────────────
def test_imports():
    import process_song_offline  # noqa
    from preprocessing import batch_ingest, split_dataset, slakh_adapter  # noqa
    from postprocessing import f1_eval, fad_eval  # noqa
    print("  process_song_offline         OK")
    print("  preprocessing.batch_ingest   OK")
    print("  preprocessing.split_dataset  OK")
    print("  preprocessing.slakh_adapter  OK")
    print("  postprocessing.f1_eval       OK")
    print("  postprocessing.fad_eval      OK")
    return "PASSED"


# ── T_f1: F1 metric on Surprise Symphony trumpet/violin pair (V13) ────────────
def test_f1():
    """
    V13 — Tests the full F1 pipeline on real audio from the trained model:
      1. Basic-Pitch transcribes the original trumpet WAV → reference MIDI
      2. compute_f1(original_trumpet, that_midi) should be HIGH  (>= 0.5)
         because the MIDI was derived from that same audio
      3. compute_f1(transferred_violin, that_midi) should be LOWER than step 2
         because the timbre change (trumpet→violin) affects transcription
    """
    from postprocessing.f1_eval import compute_f1

    # Locate basic_pitch_env Python (inside MusicProject/)
    project_root = Path(__file__).parent
    basic_pitch_py = project_root / "basic_pitch_env" / "Scripts" / "python.exe"
    if not basic_pitch_py.exists():
        raise RuntimeError(
            f"basic_pitch_env not found at {basic_pitch_py}\n"
            f"Create it with: python3.10 -m venv basic_pitch_env && "
            f"basic_pitch_env/Scripts/pip install basic-pitch"
        )

    bench = project_root / "benchmark_output" / "AuSep_2_tpt_15_Surprise"
    original_wav   = bench / "AuSep_2_tpt_15_Surprise_original.wav"
    transferred_wav = bench / "AuSep_2_tpt_15_Surprise_transferred_violin.wav"

    assert original_wav.exists(),    f"T_f1 FAIL: missing {original_wav}"
    assert transferred_wav.exists(), f"T_f1 FAIL: missing {transferred_wav}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Derive basic-pitch CLI executable from the venv Scripts dir
        scripts_dir = basic_pitch_py.parent
        basic_pitch_cli = scripts_dir / ("basic-pitch.exe" if sys.platform == "win32" else "basic-pitch")
        if not basic_pitch_cli.exists():
            basic_pitch_cli = scripts_dir / "basic-pitch"

        # Step 1: transcribe original → use as reference MIDI
        print("  Step 1/3 — transcribing original trumpet WAV with Basic-Pitch ...")
        bp_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        bp_result = subprocess.run(
            [str(basic_pitch_cli), str(tmp_path), str(original_wav)],
            capture_output=True, text=True, encoding="utf-8", env=bp_env
        )
        midi_candidates = list(tmp_path.glob("*.mid")) + list(tmp_path.glob("*.midi"))
        if not midi_candidates:
            raise RuntimeError(
                f"Basic-Pitch failed (exit {bp_result.returncode}):\n"
                f"  stderr: {bp_result.stderr[:500]}"
            )
        midi_candidates = list(tmp_path.glob("*.mid")) + list(tmp_path.glob("*.midi"))
        assert midi_candidates, f"T_f1 FAIL: Basic-Pitch produced no MIDI in {tmp_path}"
        reference_midi = midi_candidates[0]
        print(f"  Reference MIDI: {reference_midi.name}")

        # Step 2: F1 of original against itself → should be high
        print("  Step 2/3 — F1(original trumpet, reference MIDI) ...")
        res_orig = compute_f1(
            generated_wav=original_wav,
            reference_midi=reference_midi,
            basic_pitch_python=str(basic_pitch_py),
            tmp_dir=tmp_path / "orig_pred",
        )
        print(f"    F1={res_orig['f1']:.3f}  P={res_orig['precision']:.3f}  "
              f"R={res_orig['recall']:.3f}  "
              f"({res_orig['matched']}/{res_orig['n_reference']} ref notes matched)")
        assert res_orig["f1"] >= 0.5, (
            f"T_f1 FAIL: original-vs-self F1={res_orig['f1']:.3f} < 0.5\n"
            f"  predicted={res_orig['n_predicted']}, reference={res_orig['n_reference']}"
        )

        # Step 3: F1 of transferred violin against same MIDI → should be lower
        print("  Step 3/3 — F1(transferred violin, reference MIDI) ...")
        res_xfer = compute_f1(
            generated_wav=transferred_wav,
            reference_midi=reference_midi,
            basic_pitch_python=str(basic_pitch_py),
            tmp_dir=tmp_path / "xfer_pred",
        )
        print(f"    F1={res_xfer['f1']:.3f}  P={res_xfer['precision']:.3f}  "
              f"R={res_xfer['recall']:.3f}  "
              f"({res_xfer['matched']}/{res_xfer['n_reference']} ref notes matched)")
        assert res_xfer["f1"] < res_orig["f1"], (
            f"T_f1 FAIL: transferred F1={res_xfer['f1']:.3f} not < "
            f"original F1={res_orig['f1']:.3f} — metric not sensitive to timbre change"
        )

    print(f"  F1 sensitivity confirmed: original={res_orig['f1']:.3f} > "
          f"transferred={res_xfer['f1']:.3f}")
    return "PASSED"


# ── T_fad: Group-FAD identity + sensitivity (V15) ────────────────────────────
def test_group_fad():
    """
    V15 — Tests compute_group_fad() (no manifest = delegates to compute_fad):
      1. group_fad(real_dir, real_dir)  should be ~0 (same distribution)
      2. group_fad(real, gen) == all_fad(real, gen) within numerical tolerance
      3. group_fad(real, real) < group_fad(real, gen)  (identity < cross-dir)
    Uses benchmark_output/fad_real/ and fad_generated/ — 4 real + 3 generated violin WAVs.
    """
    from postprocessing.fad_eval import compute_fad, compute_group_fad

    project_root = Path(__file__).parent
    fad_real = project_root / "benchmark_output" / "fad_real"
    fad_gen  = project_root / "benchmark_output" / "fad_generated"

    assert fad_real.exists(), f"T_fad FAIL: missing {fad_real}"
    assert fad_gen.exists(),  f"T_fad FAIL: missing {fad_gen}"
    assert len(list(fad_real.glob("*.wav"))) >= 1, "T_fad FAIL: no WAVs in fad_real"
    assert len(list(fad_gen.glob("*.wav")))  >= 1, "T_fad FAIL: no WAVs in fad_generated"

    # 1. Self-FAD: same directory vs itself — must be near zero
    print("  Step 1/3 — Group-FAD(real, real) should be ~0 ...")
    self_fad = compute_group_fad(
        real_dir=str(fad_real),
        generated_dir=str(fad_real),
        version_manifest_csv=None,
    )
    print(f"    Group-FAD(real, real) = {self_fad['group_fad']:.4f}")
    assert self_fad["group_fad"] < 0.5, (
        f"T_fad FAIL: self Group-FAD={self_fad['group_fad']:.4f} >= 0.5 "
        f"(same dir should score near 0)"
    )

    # 2. Group-FAD == All-FAD when no manifest provided
    print("  Step 2/3 — Group-FAD(real, gen) == All-FAD(real, gen) ...")
    all_fad_score  = compute_fad(str(fad_real), str(fad_gen))
    group_fad_score = compute_group_fad(
        real_dir=str(fad_real),
        generated_dir=str(fad_gen),
        version_manifest_csv=None,
    )
    print(f"    All-FAD   = {all_fad_score:.4f}")
    print(f"    Group-FAD = {group_fad_score['group_fad']:.4f}")
    assert abs(group_fad_score["group_fad"] - all_fad_score) < 0.01, (
        f"T_fad FAIL: Group-FAD={group_fad_score['group_fad']:.4f} != "
        f"All-FAD={all_fad_score:.4f} (no manifest — should be identical)"
    )

    # 3. Self-FAD < cross-dir FAD (identity must score better)
    print("  Step 3/3 — Group-FAD(real, real) < Group-FAD(real, gen) ...")
    assert self_fad["group_fad"] < group_fad_score["group_fad"], (
        f"T_fad FAIL: self={self_fad['group_fad']:.4f} not < "
        f"cross={group_fad_score['group_fad']:.4f} — metric not sensitive"
    )

    print(f"  FAD sensitivity confirmed: "
          f"self={self_fad['group_fad']:.4f} < cross={group_fad_score['group_fad']:.4f}")
    return "PASSED"


# ── T_aug: JointAugment audio round-trip (pitch ratio) ──────────────────────
def test_t_aug_wrapper():
    from tests.test_augmentation import test_t_aug
    return test_t_aug()


# ── Runner ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-t1", action="store_true",
                    help="Skip T1 (no internet/Drive needed)")
parser.add_argument("--skip-t-aug", action="store_true",
                    help="Skip T_aug (avoids ~450 MB BigVGAN download on first run)")
args, _ = parser.parse_known_args()

tests = [
    ("T_import  module imports",         test_imports,         False),
    ("T1        Path B end-to-end",      test_t1,               args.skip_t1),
    ("T6        model forward pass",     test_t6,               False),
    ("T7        batch_ingest + split",   test_t7,               False),
    ("T_f1      F1 metric (V13)",        test_f1,               False),
    ("T_fad     Group-FAD (V15)",        test_group_fad,        False),
    ("T_aug     augmentation round-trip", test_t_aug_wrapper,   args.skip_t_aug),
]

print("\n" + "="*60)
print("  LOCAL SMOKE TESTS")
print("="*60)

for name, fn, skip in tests:
    print(f"\n[{name}]")
    if skip:
        RESULTS[name] = "SKIPPED"
        print("  → SKIPPED")
        continue
    try:
        status = fn()
    except Exception as e:
        status = f"FAILED — {e}"
    RESULTS[name] = status
    print(f"  → {status}")

print("\n" + "="*60)
for name, status in RESULTS.items():
    icon = "OK" if status == "PASSED" else ("--" if status == "SKIPPED" else "!!")
    print(f"  {icon}  {name:<40}  {status}")
print("="*60)

failed = [n for n,s in RESULTS.items() if s != "PASSED"]
if failed:
    sys.exit(1)
else:
    print("\nAll local tests passed.")
