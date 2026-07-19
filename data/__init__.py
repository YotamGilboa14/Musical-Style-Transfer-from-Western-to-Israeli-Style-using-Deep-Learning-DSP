"""Dataset package: the PyTorch Dataset that feeds training.

Exposes MelPianoRollDataset so training code can do
``from data import MelPianoRollDataset``.
"""

from .dataset import MelPianoRollDataset

__all__ = ["MelPianoRollDataset"]
