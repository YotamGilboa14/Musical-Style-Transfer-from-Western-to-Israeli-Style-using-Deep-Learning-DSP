"""FiLM (Feature-wise Linear Modulation) layer.

This is how the timestep and the style/version get to steer the network. Rather
than just gluing the conditioning onto the features, FiLM turns the conditioning
vector C into a per-channel scale (gamma) and shift (beta) and applies them to a
feature map h:

    gamma = 1 + scale(C)
    beta  = shift(C)
    h'    = gamma * h + beta        # gamma and beta broadcast over time

The "1 +" on gamma is deliberate. At the start of training the two linear layers
output values close to zero, so gamma is about 1 and beta about 0, which makes
h' = h. The block therefore begins as an identity (it passes the features
through untouched) and only gradually learns how the timestep and the style
should reshape them. Starting from identity instead of a random scaling makes
early training much more stable.
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Applies a conditioning-driven scale and shift to a 1-D feature map.

    Args:
        in_channels: number of feature channels we modulate (C_feat).
        cond_dim:    length of the conditioning vector C. Default 256, which is
                     our 128-d timestep embedding concatenated with the 128-d
                     version embedding.
    """

    def __init__(self, in_channels: int, cond_dim: int = 256):
        super().__init__()
        # Two separate linear maps: one produces the scale, one the shift. Each
        # turns the cond_dim-long vector into one number per feature channel.
        self.scale = nn.Linear(cond_dim, in_channels)
        self.shift = nn.Linear(cond_dim, in_channels)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, C_feat, T] feature map (batch, channels, time).
            c: [B, cond_dim] conditioning vector, one per item in the batch.
        Returns:
            h': [B, C_feat, T] modulated feature map, same shape as h.
        """
        # unsqueeze(-1) adds a length-1 time axis so the per-channel scale and
        # shift broadcast across every time step of h.
        gamma = 1.0 + self.scale(c).unsqueeze(-1)  # [B, C_feat, 1]
        beta = self.shift(c).unsqueeze(-1)          # [B, C_feat, 1]
        return gamma * h + beta
