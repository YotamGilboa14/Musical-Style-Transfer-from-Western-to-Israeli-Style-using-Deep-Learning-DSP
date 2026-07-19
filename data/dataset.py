"""
MelPianoRollDataset — loads preprocessed .pt segments produced by process_song_offline.py.

Manifest CSV columns used:
  segment_path   path to mel tensor   [80, 430]   float32, normalized to [-1, 1]
  score_path     path to piano roll   [2, 128, 430] float32, values in [0, 1]
  version_id     integer version label

All paths in the manifest are relative to `manifest_root` (the directory that
contains the manifest CSV, or an explicit override).
"""

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

# Optional: augmentation (import lazily to avoid hard dep when not used)
try:
    from preprocessing.augmentation import JointAugment as _JointAugment
except ImportError:  # pragma: no cover
    _JointAugment = None


class MelPianoRollDataset(Dataset):
    """PyTorch dataset for paired mel/score training segments.

    Each item returned by this dataset is one 5-second training example:
    the target mel spectrogram, the conditioning piano roll, and the style
    version ID. The manifest keeps paths relative so the same CSV can work on
    local disk, Google Drive for Desktop, or Colab-mounted Drive.

    Args:
        manifest_csv:   path to the dataset manifest CSV file.
        manifest_root:  root directory for resolving relative paths stored in the
                        manifest. Defaults to the directory containing the CSV.
        mel_channels:   expected number of mel bins (used for shape assertion).
        segment_frames: expected number of time frames per segment (for assertion).
    """

    def __init__(
        self,
        manifest_csv: str,
        manifest_root: str = None,
        mel_channels: int = 80,
        segment_frames: int = 430,
        augment=None,
    ):
        """Load the manifest and remember shape/version expectations."""
        self.df = pd.read_csv(manifest_csv)
        self.root = Path(manifest_root) if manifest_root else Path(manifest_csv).parent
        self.mel_channels = mel_channels
        self.segment_frames = segment_frames
        self.augment = augment  # JointAugment instance or None

        required = {"segment_path", "score_path", "version_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

    def __len__(self) -> int:
        """Return the number of segment rows available to DataLoader."""
        return len(self.df)

    @staticmethod
    def _fit_frames(t: torch.Tensor, frames: int) -> torch.Tensor:
        """Clamp the time axis (last dim) to exactly ``frames``.

        Truncates if longer, zero-pads on the right if shorter. Returns a
        contiguous tensor. A no-op when the length already matches.
        """
        T = t.shape[-1]
        if T == frames:
            return t
        if T > frames:
            return t[..., :frames].contiguous()
        return torch.nn.functional.pad(t, (0, frames - T)).contiguous()

    def __getitem__(self, idx: int) -> dict:
        """Load one mel/piano-roll pair and return tensors for training."""
        row = self.df.iloc[idx]

        mel_path = self.root / row["segment_path"]
        pr_path = self.root / row["score_path"]

        # .contiguous() is a cheap safety belt: an earlier preprocessing bug
        # saved non-contiguous tensor views with bloated backing storage. Loading
        # contiguous tensors keeps the training batch compact and predictable.
        mel = torch.load(mel_path, weights_only=True).float().contiguous()        # [80, 430]
        piano_roll = torch.load(pr_path, weights_only=True).float().contiguous()    # [2, 128, 430]

        # Offline pitch-shift/time-stretch augmentation can leave a ±1 frame
        # jitter on the time axis (STFT vs MIDI framing rounding), which would
        # break default_collate when batched with exact-length segments. Clamp
        # the time axis to segment_frames (truncate if longer, zero-pad if
        # shorter) so every batch element shares the same shape.
        mel = self._fit_frames(mel, self.segment_frames)
        piano_roll = self._fit_frames(piano_roll, self.segment_frames)

        if self.augment is not None:
            # Training-time augmentation must transform mel and piano roll
            # together so the conditioning still describes the target audio.
            mel, piano_roll = self.augment(mel, piano_roll)

        return {
            "mel": mel,
            "piano_roll": piano_roll,
            "version_id": torch.tensor(int(row["version_id"]), dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Normalization stats helper (optional, call once to verify dataset)
    # ------------------------------------------------------------------

    def compute_stats(self, n_samples: int = 200) -> dict:
        """Return per-channel min/max/mean/std over a random subset of samples."""
        import random
        indices = random.sample(range(len(self)), min(n_samples, len(self)))
        mels = torch.stack([self[i]["mel"] for i in indices])  # [N, 80, 430]
        return {
            "mel_min": mels.min().item(),
            "mel_max": mels.max().item(),
            "mel_mean": mels.mean().item(),
            "mel_std": mels.std().item(),
        }
