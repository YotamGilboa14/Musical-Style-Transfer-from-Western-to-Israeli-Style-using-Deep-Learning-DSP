"""
augmentation.py — Joint data augmentation for mel + piano roll pairs.

Augmentations applied in order:
  1. pitch_shift  — shift both mel (Hz-aware bin interpolation) and piano roll
                    (1-semitone = 1-bin roll, MIDI-native) jointly
  2. time_stretch — stretch/squeeze both tensors along time, then crop/pad to fixed length
  3. spec_augment — time/freq masking on mel only (score untouched)

With enabled=False the module is a no-op: outputs are bit-exact copies of inputs.

Note on pitch shift accuracy
----------------------------
Mel bins are non-uniform in Hz. A constant bin-roll therefore produces a
different pitch ratio at different frequencies. To get a constant-semitone
shift in audio, we map each output mel bin's Hz center back to its source Hz
(target_hz / 2^(s/12)) and linearly interpolate along the mel-bin axis at the
corresponding fractional index. Verified by tests/test_augmentation.py (T_aug):
vocoded f0 ratio matches 2^(semis/12) within ±3%.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class JointAugment:
    """
    Args:
        config: dict with keys:
          enabled: bool
          pitch_shift:  {p, max_semitones}
          time_stretch: {p, max_pct}
          spec_augment: {p, time_mask_max, freq_mask_max, n_time, n_freq}
    """

    # Mel-bin Hz centers used for pitch shifting. Computed lazily once with
    # the project's standard 80-mel/22050Hz/fmax=8000Hz layout (DSPConfig).
    _MEL_CENTERS_HZ: np.ndarray | None = None

    def __init__(self, config: dict):
        """Read augmentation probabilities/ranges and initialize mel-bin centers."""
        self.enabled = config.get("enabled", True)
        self._ps = config.get("pitch_shift", {})
        self._ts = config.get("time_stretch", {})
        self._sa = config.get("spec_augment", {})
        if JointAugment._MEL_CENTERS_HZ is None:
            import librosa as _lr
            JointAugment._MEL_CENTERS_HZ = _lr.mel_frequencies(
                n_mels=80, fmin=0.0, fmax=8000.0
            ).astype(np.float64)

    def __call__(self, mel: Tensor, piano_roll: Tensor) -> tuple[Tensor, Tensor]:
        """
        mel:        [80, T]
        piano_roll: [2, 128, T]
        Returns tensors of the same shapes.
        """
        if not self.enabled:
            return mel.clone(), piano_roll.clone()

        mel = mel.clone()
        piano_roll = piano_roll.clone()

        mel, piano_roll = self._maybe_pitch_shift(mel, piano_roll)
        mel, piano_roll = self._maybe_time_stretch(mel, piano_roll)
        mel = self._maybe_spec_augment(mel)

        return mel, piano_roll

    # ──────────────────────────────────────────────────────────────────────
    # Pitch shift
    # ──────────────────────────────────────────────────────────────────────

    def _maybe_pitch_shift(self, mel: Tensor, piano_roll: Tensor):
        """Randomly pitch-shift mel and piano roll together, or return unchanged."""
        p = self._ps.get("p", 0.5)
        if random.random() >= p:
            return mel, piano_roll

        max_semi = int(self._ps.get("max_semitones", 2))
        # Exclude 0 so we always actually shift when triggered
        semitones = random.choice(
            [d for d in range(-max_semi, max_semi + 1) if d != 0]
        )

        # ── Mel: Hz-aware bin interpolation ─────────────────────────────
        mel = self._pitch_shift_mel_hz(mel, semitones)

        # ── Piano roll: shift along pitch axis (dim 1) ──────────────────
        # Piano-roll axis is MIDI-native (1 bin = 1 semitone), so torch.roll
        # by `semitones` is exact.
        pr_pitches = piano_roll.shape[1]  # 128
        if semitones != 0:
            piano_roll = torch.roll(piano_roll, shifts=semitones, dims=1)
            if semitones > 0:
                piano_roll[:, :semitones, :] = 0.0
            else:
                piano_roll[:, pr_pitches + semitones:, :] = 0.0

        return mel, piano_roll

    @classmethod
    def _pitch_shift_mel_hz(cls, mel: Tensor, semitones: int) -> Tensor:
        """Pitch-shift a [n_mels, T] mel by ``semitones`` using Hz-aware
        linear interpolation along the mel-bin axis.

        For each output mel bin ``j`` with Hz center ``f_j``, the source
        frequency is ``f_j / 2 ** (semitones / 12)`` and the source bin
        index is found by inverting the mel-center -> bin-index mapping.
        Bins whose source frequency falls outside the mel range are filled
        with the input's minimum value (= silence in [-1, 1] normalized mel).
        """
        if semitones == 0:
            return mel
        n_mels = mel.shape[0]
        centers = cls._MEL_CENTERS_HZ
        if centers is None or centers.shape[0] != n_mels:
            import librosa as _lr
            centers = _lr.mel_frequencies(
                n_mels=n_mels, fmin=0.0, fmax=8000.0
            ).astype(np.float64)
            cls._MEL_CENTERS_HZ = centers

        ratio = 2.0 ** (semitones / 12.0)
        src_freqs = centers / ratio                                       # [n_mels]
        valid_np = (src_freqs >= centers[0]) & (src_freqs <= centers[-1])
        # Fractional source index for each output bin (np.interp clamps OOR)
        src_idx_np = np.interp(src_freqs, centers, np.arange(n_mels, dtype=np.float64))
        src_idx = torch.from_numpy(src_idx_np).to(mel.dtype)              # [n_mels]
        floor_idx = src_idx.floor().long().clamp(0, n_mels - 1)
        ceil_idx  = (floor_idx + 1).clamp(0, n_mels - 1)
        frac = (src_idx - src_idx.floor()).unsqueeze(-1)                  # [n_mels, 1]
        out = mel[floor_idx] * (1.0 - frac) + mel[ceil_idx] * frac        # [n_mels, T]
        # Out-of-range bins -> silence baseline (= min of input mel)
        silence = mel.min().item()
        valid = torch.from_numpy(valid_np)
        if (~valid).any():
            out[~valid, :] = silence
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Time stretch
    # ──────────────────────────────────────────────────────────────────────

    def _maybe_time_stretch(self, mel: Tensor, piano_roll: Tensor):
        """Randomly stretch/compress time for both tensors, then restore length."""
        p = self._ts.get("p", 0.4)
        if random.random() >= p:
            return mel, piano_roll

        max_pct = float(self._ts.get("max_pct", 0.10))
        rate = random.uniform(1.0 - max_pct, 1.0 + max_pct)

        T_orig = mel.shape[-1]
        T_new = round(T_orig * rate)

        # Mel: [80, T] → interpolate as [1, 80, T] → [1, 80, T_new] → [80, T_new]
        mel_interp = F.interpolate(
            mel.unsqueeze(0),           # [1, 80, T]
            size=T_new,
            mode="linear",
            align_corners=False,
        ).squeeze(0)                    # [80, T_new]

        # Piano roll: [2, 128, T] → interpolate as [1, 2*128, T] → crop back to [2, 128, T_new]
        pr_interp = F.interpolate(
            piano_roll.view(1, -1, T_orig),  # [1, 256, T]
            size=T_new,
            mode="nearest",
        ).view(piano_roll.shape[0], piano_roll.shape[1], T_new)  # [2, 128, T_new]

        # Crop or zero-pad back to T_orig
        mel = _fit_to_length(mel_interp, T_orig)
        piano_roll = _fit_to_length(pr_interp, T_orig)

        return mel, piano_roll

    # ──────────────────────────────────────────────────────────────────────
    # SpecAugment (mel only)
    # ──────────────────────────────────────────────────────────────────────

    def _maybe_spec_augment(self, mel: Tensor) -> Tensor:
        """Randomly mask time/frequency regions of the mel only."""
        p = self._sa.get("p", 0.5)
        if random.random() >= p:
            return mel

        F_bins, T_frames = mel.shape
        n_freq = int(self._sa.get("n_freq", 2))
        n_time = int(self._sa.get("n_time", 2))
        freq_mask_max = int(self._sa.get("freq_mask_max", 8))
        time_mask_max = int(self._sa.get("time_mask_max", 30))

        mel = mel.clone()
        # In our [-1, 1] normalized mel, -1.0 is silence (minimum energy).
        # Masking with 0.0 would introduce mid-range energy instead of silence.
        silence_val = mel.min().item()

        # Frequency masks
        for _ in range(n_freq):
            f = random.randint(0, freq_mask_max)
            f0 = random.randint(0, max(0, F_bins - f))
            mel[f0: f0 + f, :] = silence_val

        # Time masks
        for _ in range(n_time):
            t = random.randint(0, time_mask_max)
            t0 = random.randint(0, max(0, T_frames - t))
            mel[:, t0: t0 + t] = silence_val

        return mel


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fit_to_length(x: Tensor, T: int) -> Tensor:
    """Center-crop or right-pad with zeros along the last dimension to length T."""
    T_x = x.shape[-1]
    if T_x == T:
        return x
    if T_x > T:
        # Center crop
        start = (T_x - T) // 2
        return x[..., start: start + T]
    # Zero-pad on the right
    pad_amount = T - T_x
    return F.pad(x, (0, pad_amount))


# ──────────────────────────────────────────────────────────────────────────────
# Offline source-pool augmentation (WAV + deterministic MIDI)
# ──────────────────────────────────────────────────────────────────────────────
#
# These functions operate on full-song WAV + MIDI files (NOT on already-extracted
# mel/piano-roll tensors like ``JointAugment`` above). They run once per song,
# locally, alongside ``process_song_offline.py --source-pool-mode``. The output
# WAV+MIDI pairs land in ``source_pool/<artist>/<album>/<song>/augmented/`` so
# every downstream version can pick them up at DSP time without re-running
# Basic-Pitch — the augmented MIDI is derived **deterministically** from the
# original MIDI by transposing pitches (pitch-shift) or scaling note times
# (time-stretch).
#
# Why deterministic MIDI derivation:
#   * Basic-Pitch is local-only (Python 3.10 + TF) and slow. Running it on every
#     augmented WAV would multiply ingest time.
#   * Pretty-MIDI lets us transpose / time-scale notes exactly, with no drift.
#   * The augmented WAV + augmented MIDI stay aligned by construction.
# ──────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from pathlib import Path as _Path

DEFAULT_AUGMENTATIONS: list[dict] = [
    {"kind": "pitch_shift", "semitones": +2, "suffix": "ps+2"},
    {"kind": "pitch_shift", "semitones": -2, "suffix": "ps-2"},
    {"kind": "time_stretch", "rate": 0.9, "suffix": "ts0.9"},
    {"kind": "time_stretch", "rate": 1.1, "suffix": "ts1.1"},
]
"""Default 4-augmentation policy applied to each source-pool song.

Each entry produces one (WAV, MIDI) pair in ``augmented/``. Override by passing
a custom ``augmentations`` list to :func:`augment_song`.
"""


def pitch_shift_wav(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Pitch-shift a mono waveform by ``semitones`` (librosa, high quality)."""
    import librosa as _lr
    return _lr.effects.pitch_shift(y=y.astype(np.float32), sr=sr, n_steps=float(semitones))


def time_stretch_wav(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch a mono waveform by ``rate`` (librosa, phase vocoder)."""
    import librosa as _lr
    if rate <= 0:
        raise ValueError(f"time_stretch rate must be > 0, got {rate}")
    return _lr.effects.time_stretch(y=y.astype(np.float32), rate=float(rate))


def pitch_shift_midi(in_midi: _Path, out_midi: _Path, semitones: int) -> None:
    """Transpose every note in ``in_midi`` by ``semitones`` and write to ``out_midi``.

    Notes whose final pitch falls outside MIDI range [0, 127] are dropped.
    """
    import pretty_midi as _pm
    pm = _pm.PrettyMIDI(str(in_midi))
    for inst in pm.instruments:
        kept = []
        for note in inst.notes:
            new_pitch = int(note.pitch) + int(semitones)
            if 0 <= new_pitch <= 127:
                note.pitch = new_pitch
                kept.append(note)
        inst.notes = kept
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi))


def time_scale_midi(in_midi: _Path, out_midi: _Path, rate: float) -> None:
    """Time-scale every note in ``in_midi`` by ``1 / rate`` and write to ``out_midi``.

    Matches :func:`time_stretch_wav` convention: rate > 1 → faster (shorter),
    rate < 1 → slower (longer). MIDI note times are divided by ``rate``.
    """
    import pretty_midi as _pm
    if rate <= 0:
        raise ValueError(f"time_scale rate must be > 0, got {rate}")
    pm = _pm.PrettyMIDI(str(in_midi))
    inv = 1.0 / float(rate)
    for inst in pm.instruments:
        for note in inst.notes:
            note.start = float(note.start) * inv
            note.end = float(note.end) * inv
        for bend in inst.pitch_bends:
            bend.time = float(bend.time) * inv
        for cc in inst.control_changes:
            cc.time = float(cc.time) * inv
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi))


def augment_song(
    song_dir: _Path,
    *,
    wav_name: str | None = None,
    midi_name: str | None = None,
    augmentations: list[dict] | None = None,
    sample_rate: int | None = None,
    skip_if_exists: bool = True,
) -> dict:
    """Generate augmented WAV+MIDI pairs for a single source-pool song.

    Reads ``song_dir/<song>.wav`` + ``song_dir/<song>.mid`` (auto-detected by
    default), applies each entry in ``augmentations`` (default
    :data:`DEFAULT_AUGMENTATIONS`), writes the result into
    ``song_dir/augmented/<song>_<suffix>.{wav,mid}``, and dumps an
    ``aug_spec.json`` describing the exact parameters used.

    Parameters
    ----------
    song_dir : Path
        ``source_pool/<artist>/<album>/<song>/`` directory.
    wav_name, midi_name : str, optional
        Filenames inside ``song_dir`` — defaults to the only ``.wav`` /
        ``.mid`` directly in ``song_dir``.
    augmentations : list[dict], optional
        Override the default 4-aug policy.
    sample_rate : int, optional
        Target SR for read+write. Default: native SR of the original WAV.
    skip_if_exists : bool
        Skip an entry if both its output WAV and MIDI already exist.

    Returns
    -------
    dict with keys ``aug_dir`` (Path), ``produced`` (list of suffixes written),
    ``skipped`` (list of suffixes skipped), ``errors`` (list of {suffix, error}).
    """
    import json as _json

    import librosa as _lr
    import soundfile as _sf

    song_dir = _Path(song_dir)
    if not song_dir.exists():
        raise FileNotFoundError(f"song_dir not found: {song_dir}")

    # Auto-detect WAV + MIDI if not explicitly named
    wavs = [p for p in song_dir.iterdir() if p.suffix.lower() == ".wav"]
    mids = [p for p in song_dir.iterdir() if p.suffix.lower() == ".mid"]
    if wav_name:
        wav_path = song_dir / wav_name
    elif len(wavs) == 1:
        wav_path = wavs[0]
    else:
        raise FileNotFoundError(
            f"could not auto-detect WAV in {song_dir} (found {len(wavs)})"
        )
    if midi_name:
        midi_path = song_dir / midi_name
    elif len(mids) == 1:
        midi_path = mids[0]
    else:
        raise FileNotFoundError(
            f"could not auto-detect MIDI in {song_dir} (found {len(mids)})"
        )

    augmentations = augmentations or DEFAULT_AUGMENTATIONS
    aug_dir = song_dir / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)

    # Load original audio once (re-used per aug)
    y_orig, sr_native = _lr.load(str(wav_path), sr=sample_rate, mono=True)
    sr = sample_rate or sr_native

    produced: list[str] = []
    skipped: list[str] = []
    errors: list[dict] = []
    spec_entries: list[dict] = []

    stem = wav_path.stem
    for entry in augmentations:
        suffix = entry["suffix"]
        out_wav = aug_dir / f"{stem}_{suffix}.wav"
        out_mid = aug_dir / f"{stem}_{suffix}.mid"

        if skip_if_exists and out_wav.exists() and out_mid.exists():
            skipped.append(suffix)
            spec_entries.append({**entry, "out_wav": out_wav.name,
                                 "out_mid": out_mid.name, "skipped": True})
            continue

        try:
            kind = entry["kind"]
            if kind == "pitch_shift":
                semis = int(entry["semitones"])
                y_aug = pitch_shift_wav(y_orig, sr, semis)
                pitch_shift_midi(midi_path, out_mid, semis)
            elif kind == "time_stretch":
                rate = float(entry["rate"])
                y_aug = time_stretch_wav(y_orig, rate)
                time_scale_midi(midi_path, out_mid, rate)
            else:
                raise ValueError(f"unknown augmentation kind: {kind!r}")

            _sf.write(str(out_wav), y_aug, sr)
            produced.append(suffix)
            spec_entries.append({**entry, "out_wav": out_wav.name,
                                 "out_mid": out_mid.name})
        except Exception as exc:  # noqa: BLE001
            errors.append({"suffix": suffix, "error": str(exc)})

    spec_path = aug_dir / "aug_spec.json"
    spec_path.write_text(_json.dumps({
        "source_wav": wav_path.name,
        "source_midi": midi_path.name,
        "sample_rate": int(sr),
        "augmentations": spec_entries,
        "method": {
            "wav_pitch_shift": "librosa.effects.pitch_shift",
            "wav_time_stretch": "librosa.effects.time_stretch",
            "midi_pitch_shift": "pretty_midi transpose (drop out-of-range)",
            "midi_time_stretch": "pretty_midi time-scale by 1/rate",
        },
    }, indent=2), encoding="utf-8")

    return {
        "aug_dir": aug_dir,
        "produced": produced,
        "skipped": skipped,
        "errors": errors,
    }
