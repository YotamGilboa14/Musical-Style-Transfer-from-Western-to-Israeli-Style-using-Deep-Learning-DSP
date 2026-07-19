"""Building blocks for the 1-D diffusion U-Net.

These are the small reusable pieces the U-Net is assembled from. We keep them in
their own file so unet.py reads as a clean wiring diagram instead of a wall of
layer definitions.

  ResBlock1D      - residual block with GroupNorm, SiLU, Conv1d and FiLM conditioning.
  SelfAttention1D - multi-head self-attention over the time axis.
  Downsample      - strided Conv1d (factor 2) that shrinks the time axis.
  Upsample        - nearest-neighbour x2 followed by Conv1d.

Everything runs in 1-D: the mel is treated as a sequence along time, and the 80
mel bins are just feature channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM


class ResBlock1D(nn.Module):
    """Residual block for 1-D feature maps.

    A residual block is a short stack of convolutions whose output is added back
    to its own input (the "+ skip" at the end). That shortcut is what lets us
    stack many blocks deep: each block only has to learn a small correction on
    top of what it received, and gradients have a direct path back through the
    addition, so training does not stall. FiLM sits in the middle so the
    timestep and style can reshape the features here too.

    Data flow:
        GN -> SiLU -> Conv1d(k=3) -> FiLM -> GN -> SiLU -> Dropout -> Conv1d(k=3)  +  skip

    When the block changes the channel count, the skip path uses a 1x1 conv so
    the shapes match before the addition; otherwise the input is passed through
    unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 256,
        n_groups: int = 32,
        dropout: float = 0.1,
    ):
        """Create the two-convolution residual block plus optional skip projection."""
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film = FiLM(out_channels, cond_dim)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, T]
            c: [B, cond_dim] conditioning vector
        Returns:
            [B, out_channels, T]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.film(h, c)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the time axis of a 1-D feature map.

    Convolutions only look at a small local window at a time, so they are good
    at local texture but weak at relating distant moments. Self-attention fixes
    that: every time position can look at every other position and decide how
    much each one matters, which helps the network keep phrasing and rhythm
    consistent across a whole segment. "Multi-head" means several of these
    comparisons run in parallel and can each focus on something different. We
    treat time as the sequence dimension and use full (non-causal) attention,
    since we generate a whole segment at once rather than left-to-right.

    Data flow: permute -> LayerNorm -> MultiheadAttention -> permute + residual.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        """Create non-causal multi-head attention over the time dimension."""
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T]
        """
        B, C, T = x.shape
        h = x.permute(0, 2, 1)          # [B, T, C]
        h = self.norm(h)
        h, _ = self.attn(h, h, h, need_weights=False)  # [B, T, C]
        h = h.permute(0, 2, 1)          # [B, C, T]
        return x + h


class Downsample(nn.Module):
    """
    Halve temporal resolution via strided convolution (stride=2).
    Also changes channel count if in_channels != out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Create the stride-2 convolution used to shrink the time axis."""
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the downsampled feature map."""
        return self.conv(x)


class Upsample(nn.Module):
    """
    Double temporal resolution via nearest-neighbour upsampling + Conv1d.
    Also changes channel count if in_channels != out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Create the convolution used after nearest-neighbour upsampling."""
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the upsampled feature map."""
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
