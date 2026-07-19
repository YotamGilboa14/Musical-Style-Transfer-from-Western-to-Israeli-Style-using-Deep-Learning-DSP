"""Timestep and version conditioning embeddings for the diffusion U-Net.

The denoiser needs to know two things at every layer: which diffusion step it is
on, and which style we are asking for. We turn each into a 128-d vector:

  Timestep: a fixed sinusoidal encoding (like the positional encoding in
            transformers) followed by a small MLP(128 -> 128) with SiLU.
  Version:  a learnable lookup table nn.Embedding(n_versions + 1, 128) followed
            by an MLP. The extra last slot (index n_versions) is the "null"
            token we use for classifier-free guidance, i.e. "no particular
            style".

Both come out at dim=128 and are concatenated outside this file into the
conditioning vector C = [time_emb | ver_emb] of length 256, which is what every
FiLM layer receives.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Turns a diffusion timestep t into a 128-d vector.

    We use the same sinusoidal trick as transformer positional encodings: a mix
    of sines and cosines at many different frequencies. Nearby timesteps get
    similar vectors and far-apart timesteps get very different ones, which gives
    the network a smooth, continuous sense of "how noisy is this input".
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        # dim must be even because we split it half into sines, half into cosines.
        assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
        self.dim = dim
        # A tiny MLP lets the network reshape the raw sinusoids into something
        # more useful before they drive the FiLM layers.
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] tensor of timestep indices (one per item in the batch).
        Returns:
            emb: [B, dim]
        """
        half = self.dim // 2
        # Build a geometric range of frequencies from high to low. Each channel
        # of the embedding oscillates at its own rate.
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )  # [half]
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        # Concatenate sin and cos of every frequency to get the full dim-d vector.
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        return self.mlp(emb)


class VersionEmbedding(nn.Module):
    """Learnable style embedding.

    nn.Embedding is just a trainable lookup table: give it an integer style ID
    and it returns that style's vector, which the model tunes during training.
    We allocate n_versions + 1 rows so the last one can act as the null token
    (unconditional "no style") that classifier-free guidance needs.
    """

    def __init__(self, n_versions: int, dim: int = 128):
        super().__init__()
        # +1 row for the null / unconditional token used by CFG.
        self.embedding = nn.Embedding(n_versions + 1, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, version_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            version_id: [B] tensor of style indices (0 .. n_versions, where the
                        last value is the null token).
        Returns:
            emb: [B, dim]
        """
        return self.mlp(self.embedding(version_id))
