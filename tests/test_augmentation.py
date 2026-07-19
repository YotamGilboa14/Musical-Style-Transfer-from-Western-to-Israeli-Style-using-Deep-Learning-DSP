"""
T_aug — Augmentation audio round-trip sanity check.

Verifies that ``JointAugment``'s pitch-shift produces audio whose fundamental
frequency actually shifts by the requested number of semitones.

Why this exists
---------------
``preprocessing/augmentation.py::_maybe_pitch_shift`` approximates a pitch
shift with ``torch.roll`` along the mel-bin axis.  Mel bins are non-uniform
in Hz, so a constant bin-roll is **not** a constant semitone shift in audio.
This test vocodes the pre/post-augment mels through BigVGAN v2 22 kHz and
estimates median f0 with ``librosa.pyin`` to verify the actual audio pitch
ratio falls in the workplan's [1.10, 1.16] window for a +2-semitone shift
(target ratio = 2 ** (2/12) ≈ 1.122).

Run as part of smoke_test_local.py:
    python smoke_test_local.py             # T_aug runs as one of the suite
    python smoke_test_local.py --skip-t-aug  # skip if BigVGAN download not desired
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Fixture paths — produced by tests/pipeline_full_test.py config A.
_FIXTURE_SONG_DIR = (
    _PROJECT_ROOT / "tests" / "_pipeline_full_test_out" / "A_local_nosep"
    / "שלום_חנוך" / "Singles" / "היה_כדאי"
)
_FIXTURE_MEL = _FIXTURE_SONG_DIR / "processed_data" / "mels" / "segment_0000.pt"
_FIXTURE_PR  = _FIXTURE_SONG_DIR / "processed_data" / "piano_rolls" / "segment_0000.pt"
_ARTIFACTS_DIR = _PROJECT_ROOT / "tests" / "_aug_artifacts"


def _ensure_fixtures() -> None:
    """Fail early with a clear message if the test fixture segments are missing."""
    if not _FIXTURE_MEL.exists() or not _FIXTURE_PR.exists():
        raise RuntimeError(
            f"T_aug FAIL: missing fixture segments. Run\n"
            f"  python tests/pipeline_full_test.py --only A\n"
            f"first to produce {_FIXTURE_MEL.parent}/segment_0000.pt"
        )


def _normalized_mel_to_log_magnitude(mel_norm, mel_min: float = -80.0, mel_max: float = 0.0):
    """Approximate inverse of process_song_offline's [-1, 1] normalization, then
    convert from dB to log-magnitude format expected by BigVGAN's generator.

    Same conversion as ``BigVGANVocoder.segments_to_wav``.
    """
    import numpy as np
    mel_db = (mel_norm + 1.0) / 2.0 * (mel_max - mel_min) + mel_min
    return mel_db * (np.log(10) / 20.0)


def _force_pitch_plus_two(joint_aug, mel, piano_roll):
    """Run ``JointAugment`` in a context where the pitch RNG path is forced to
    pick semitones=+2, while time_stretch and spec_augment are disabled.

    Returns (mel_aug, pr_aug).
    """
    # Disable time_stretch and spec_augment (probabilistic gates set to 0).
    joint_aug._ts = {"p": 0.0, "max_pct": 0.0}
    joint_aug._sa = {"p": 0.0}
    # Force pitch_shift to always trigger and always pick +2.
    joint_aug._ps = {"p": 1.0, "max_semitones": 2}

    orig_choice = random.choice
    orig_random = random.random
    expected_semis = [d for d in range(-2, 3) if d != 0]

    def fake_random():
        # Force the augmentation's probability gate to always fire.
        return 0.0  # Always passes the `>= p` gate

    def fake_choice(seq):
        # Force the pitch-shift to always choose +2 semitones for this test.
        if list(seq) == expected_semis:
            return 2
        return orig_choice(seq)

    random.choice = fake_choice  # type: ignore[assignment]
    random.random = fake_random  # type: ignore[assignment]
    try:
        return joint_aug(mel, piano_roll)
    finally:
        random.choice = orig_choice  # type: ignore[assignment]
        random.random = orig_random  # type: ignore[assignment]


def test_t_aug() -> str:
    """Pitch-shift round-trip: mel → JointAugment(+2 st) → BigVGAN → median f0 ratio."""
    import numpy as np
    import torch
    import librosa
    import soundfile as sf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from preprocessing.augmentation import JointAugment
    from preprocessing.dsp_preprocessor import DSPConfig
    from postprocessing.bigvgan_vocoder import BigVGANVocoder

    _ensure_fixtures()
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = DSPConfig()

    # ── 1. Load fixture mel + piano roll ─────────────────────────────────
    mel = torch.load(_FIXTURE_MEL, weights_only=True)        # [80, 430]
    pr  = torch.load(_FIXTURE_PR,  weights_only=True)        # [2, 128, 430]
    assert mel.shape == (80, 430), f"  bad mel shape: {mel.shape}"
    assert pr.shape  == (2, 128, 430), f"  bad pr shape: {pr.shape}"
    assert not torch.isnan(mel).any(), "  fixture mel contains NaN"
    assert -1.01 <= mel.min().item() and mel.max().item() <= 1.01, (
        f"  fixture mel out of [-1, 1]: [{mel.min():.3f}, {mel.max():.3f}]"
    )

    # ── 2. Run JointAugment with semitones forced to +2 ──────────────────
    aug = JointAugment({"enabled": True})
    random.seed(0)
    mel_aug, pr_aug = _force_pitch_plus_two(aug, mel, pr)

    assert mel_aug.shape == mel.shape, f"  shape changed: {mel_aug.shape}"
    assert pr_aug.shape  == pr.shape,  f"  pr shape changed: {pr_aug.shape}"
    assert not torch.isnan(mel_aug).any(), "  mel_aug contains NaN"
    assert mel_aug.abs().sum().item() > 0, "  mel_aug is all-zero"
    assert (mel_aug != mel).any(), "  augmentation was a no-op"

    # ── 3. Vocode original + augmented through BigVGAN v2 22kHz ──────────
    print("  Loading BigVGAN v2 22kHz vocoder (~450 MB on first run) ...")
    voc = BigVGANVocoder(model_name="bigvgan_22k", device="cpu")

    mel_orig_log = _normalized_mel_to_log_magnitude(mel.numpy()).astype(np.float32)
    mel_aug_log  = _normalized_mel_to_log_magnitude(mel_aug.numpy()).astype(np.float32)

    audio_orig = voc.mel_to_audio(torch.from_numpy(mel_orig_log).unsqueeze(0))
    audio_aug  = voc.mel_to_audio(torch.from_numpy(mel_aug_log).unsqueeze(0))
    sr = cfg.sample_rate

    assert not np.isnan(audio_orig).any(), "  audio_orig has NaN"
    assert not np.isnan(audio_aug).any(),  "  audio_aug has NaN"
    assert float(audio_orig.std()) > 1e-3, "  audio_orig is silent"
    assert float(audio_aug.std())  > 1e-3, "  audio_aug is silent"

    # ── 4. Save artifacts for human inspection ───────────────────────────
    sf.write(_ARTIFACTS_DIR / "orig.wav", audio_orig, sr)
    sf.write(_ARTIFACTS_DIR / "aug_pitch+2.wav", audio_aug, sr)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].imshow(mel.numpy(), aspect="auto", origin="lower")
    axes[0].set_title("orig mel (segment_0000)")
    axes[1].imshow(mel_aug.numpy(), aspect="auto", origin="lower")
    axes[1].set_title("aug mel (+2 semitones via torch.roll)")
    plt.tight_layout()
    plt.savefig(_ARTIFACTS_DIR / "mel_compare.png", dpi=80)
    plt.close(fig)

    # ── 5. Estimate median f0 on both audios via pyin ─────────────────────
    f0_orig, vp_orig, _ = librosa.pyin(
        audio_orig.astype(np.float32),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    f0_aug, vp_aug, _ = librosa.pyin(
        audio_aug.astype(np.float32),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    f0_orig_v = f0_orig[~np.isnan(f0_orig)]
    f0_aug_v  = f0_aug[~np.isnan(f0_aug)]

    assert len(f0_orig_v) >= 10, f"  too few voiced frames in orig: {len(f0_orig_v)}"
    assert len(f0_aug_v)  >= 10, f"  too few voiced frames in aug:  {len(f0_aug_v)}"

    f0_orig_med = float(np.median(f0_orig_v))
    f0_aug_med  = float(np.median(f0_aug_v))
    ratio = f0_aug_med / f0_orig_med
    target = 2.0 ** (2.0 / 12.0)

    print(f"  median f0 orig = {f0_orig_med:6.1f} Hz")
    print(f"  median f0 aug  = {f0_aug_med:6.1f} Hz")
    print(f"  ratio          = {ratio:.4f}  (target {target:.4f}; window [1.10, 1.16])")

    # ── 6. Persist numbers and assert the pitch-ratio gate ───────────────
    (_ARTIFACTS_DIR / "result.json").write_text(json.dumps({
        "f0_orig_hz": f0_orig_med,
        "f0_aug_hz":  f0_aug_med,
        "ratio":      ratio,
        "target_ratio": target,
        "window":     [1.10, 1.16],
        "n_voiced_orig": int(len(f0_orig_v)),
        "n_voiced_aug":  int(len(f0_aug_v)),
    }, indent=2))

    if not (1.10 <= ratio <= 1.16):
        raise AssertionError(
            f"T_aug FAIL: pitch ratio {ratio:.4f} not in [1.10, 1.16] "
            f"(target {target:.4f} for +2 semitones).\n"
            f"  → JointAugment._maybe_pitch_shift uses torch.roll on the\n"
            f"    mel-bin axis, which is not a constant-semitone shift in Hz.\n"
            f"  → See {_ARTIFACTS_DIR}/result.json for numbers and\n"
            f"    {_ARTIFACTS_DIR}/aug_pitch+2.wav for a listen-check.\n"
            f"  → Fix: resample audio in time-domain by 1/2**(semis/12), then\n"
            f"    recompute mel; or shift mel along Hz axis (not bin axis)."
        )

    return "PASSED"


if __name__ == "__main__":
    print("[T_aug   augmentation audio round-trip]")
    try:
        result = test_t_aug()
    except Exception as exc:
        print(f"  → FAILED — {exc}")
        sys.exit(1)
    print(f"  → {result}")
