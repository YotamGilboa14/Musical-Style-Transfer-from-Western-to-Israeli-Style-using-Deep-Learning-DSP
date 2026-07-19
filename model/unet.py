"""UNet1D - the 1-D diffusion U-Net that actually generates the mel-spectrogram.

This is the heart of the project. A U-Net has an encoder that step by step
shrinks the time axis while widening the channels, a bottleneck in the middle,
and a decoder that grows the time axis back to the original length. Skip
connections copy each encoder level straight across to the matching decoder
level, so the network keeps the fine detail from the input while still reasoning
about the whole segment at the coarser levels.

What makes it a style-transfer model rather than a plain denoiser: the pitch
score is concatenated onto the noisy mel as extra input channels (so at every
time step the network sees which notes should sound), and the timestep + style
are injected into every ResBlock through FiLM.

Input:  concat([noisy_mel x_t [B,80,T], piano_roll_flat [B,256,T]]) -> [B,336,T]
Output: predicted noise eps_hat, shape [B,80,T]

Architecture (channels x time):
  Input conv            336  →  160    T=430
  Encoder level 0       160       430   (2 ResBlocks, no attn)
  Downsample            160  →  320    T=215
  Encoder level 1       320       215   (2 ResBlocks + attn)
  Downsample            320  →  480    T=108
  Encoder level 2       480       108   (2 ResBlocks + attn)
  Downsample            480  →  640    T=54
  Bottleneck            640        54   (ResBlock + attn + ResBlock)
  Upsample              640  →  480    T=108
  Decoder level 2       480+480   108   (3 ResBlocks + attn)   ← skip from enc2
  Upsample              480  →  320    T=215
  Decoder level 1       320+320   215   (3 ResBlocks + attn)   ← skip from enc1
  Upsample              320  →  160    T=430
  Decoder level 0       160+160   430   (3 ResBlocks, no attn) ← skip from enc0
  Output conv           160  →   80    T=430

Skip connections: encoder output is concatenated with decoder input at matching level.
FiLM conditioning: every ResBlock receives the combined C = [time_emb | ver_emb] ∈ R^{B,256}.
"""

import torch
import torch.nn as nn

from .embeddings import SinusoidalTimestepEmbedding, VersionEmbedding
from .blocks import ResBlock1D, SelfAttention1D, Downsample, Upsample


# ---------------------------------------------------------------------------
# Helper: a stack of ResBlock1D (+ optional attention after each)
# ---------------------------------------------------------------------------

class ResStack(nn.Module):
    """A repeated stack of ResBlock1D layers with optional attention after each."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        cond_dim: int,
        n_groups: int,
        dropout: float,
        use_attention: bool,
        attn_heads: int,
    ):
        """Build one encoder/decoder level from repeated residual blocks."""
        super().__init__()
        blocks = []
        attns = []
        for i in range(n_blocks):
            c_in = in_channels if i == 0 else out_channels
            blocks.append(ResBlock1D(c_in, out_channels, cond_dim, n_groups, dropout))
            attns.append(SelfAttention1D(out_channels, attn_heads) if use_attention else None)
        self.blocks = nn.ModuleList(blocks)
        # Store attention modules; use ModuleList with dummy Identity for None slots
        self.attns = nn.ModuleList(
            [a if a is not None else nn.Identity() for a in attns]
        )
        self.use_attention = use_attention

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply every residual block, and attention when this level uses it."""
        for block, attn in zip(self.blocks, self.attns):
            x = block(x, c)
            if self.use_attention:
                x = attn(x)
        return x


# ---------------------------------------------------------------------------
# UNet1D
# ---------------------------------------------------------------------------

class UNet1D(nn.Module):
    """
    1-D score/version-conditioned U-Net for diffusion noise prediction.

    Args:
        mel_channels:     F = 80
        score_channels:   2*128 = 256 (piano roll flattened)
        base_channels:    level-0 channels (production config uses 160 → levels 160/320/480/640)
        channel_mults:    multipliers per level, e.g. [1, 2, 3, 4]
        num_res_blocks_enc: ResBlocks per encoder level
        num_res_blocks_dec: ResBlocks per decoder level
        attention_levels: set of level indices where SelfAttention1D is applied (0 = top)
        attn_heads:       attention heads (must divide all channel counts)
        n_groups:         GroupNorm groups
        dropout:          dropout probability in ResBlock
        n_versions:       number of distinct version IDs (null token is added internally)
        version_emb_dim:  version embedding dimension
        time_emb_dim:     timestep embedding dimension
    """

    def __init__(
        self,
        mel_channels: int = 80,
        score_channels: int = 256,
        base_channels: int = 128,
        channel_mults: list = None,
        num_res_blocks_enc: int = 2,
        num_res_blocks_dec: int = 3,
        attention_levels: list = None,
        attn_heads: int = 8,
        n_groups: int = 32,
        dropout: float = 0.1,
        n_versions: int = 25,
        version_emb_dim: int = 128,
        time_emb_dim: int = 128,
    ):
        """Create the full encoder, bottleneck, decoder, and conditioning layers."""
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 3, 4]
        if attention_levels is None:
            attention_levels = [1, 2, 3]

        cond_dim = time_emb_dim + version_emb_dim  # 256

        # ── Conditioning embeddings ──────────────────────────────────────
        self.time_emb = SinusoidalTimestepEmbedding(time_emb_dim)
        self.ver_emb = VersionEmbedding(n_versions, version_emb_dim)

        # ── Channel counts per level ─────────────────────────────────────
        channels = [base_channels * m for m in channel_mults]  # e.g. base=160 → [160, 320, 480, 640]
        n_levels = len(channels)

        # ── Input projection: [B, mel+score, T] → [B, channels[0], T] ───
        self.input_conv = nn.Conv1d(
            mel_channels + score_channels, channels[0], kernel_size=3, padding=1
        )

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc_stacks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_ch = channels[0]
        for lvl in range(n_levels - 1):
            ch = channels[lvl]
            use_attn = lvl in attention_levels
            self.enc_stacks.append(
                ResStack(prev_ch, ch, num_res_blocks_enc, cond_dim, n_groups, dropout, use_attn, attn_heads)
            )
            self.downsamples.append(Downsample(ch, channels[lvl + 1]))
            prev_ch = channels[lvl + 1]

        # ── Bottleneck ────────────────────────────────────────────────────
        bn_ch = channels[-1]
        use_attn_bn = (n_levels - 1) in attention_levels
        self.bottleneck_res1 = ResBlock1D(prev_ch, bn_ch, cond_dim, n_groups, dropout)
        self.bottleneck_attn = SelfAttention1D(bn_ch, attn_heads) if use_attn_bn else nn.Identity()
        self.bottleneck_res2 = ResBlock1D(bn_ch, bn_ch, cond_dim, n_groups, dropout)

        # ── Decoder ──────────────────────────────────────────────────────
        self.upsamples = nn.ModuleList()
        self.dec_stacks = nn.ModuleList()

        for lvl in range(n_levels - 2, -1, -1):  # n_levels-2 … 0
            ch_up = channels[lvl + 1]
            ch_skip = channels[lvl]
            ch_out = channels[lvl]
            use_attn = lvl in attention_levels
            self.upsamples.append(Upsample(ch_up, ch_skip))
            # first ResBlock of decoder level merges skip: in_channels = ch_skip + ch_skip
            self.dec_stacks.append(
                ResStack(
                    ch_skip + ch_skip,  # concat with skip
                    ch_out,
                    num_res_blocks_dec,
                    cond_dim,
                    n_groups,
                    dropout,
                    use_attn,
                    attn_heads,
                )
            )

        # ── Output projection: [B, channels[0], T] → [B, mel_channels, T] ─
        self.output_norm = nn.GroupNorm(n_groups, channels[0])
        self.output_conv = nn.Conv1d(channels[0], mel_channels, kernel_size=3, padding=1)

    # -----------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        version_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:        [B, 80, T]  noisy mel at diffusion step t
            t:          [B]         diffusion timestep indices
            score:      [B, 256, T] flattened piano roll
            version_id: [B]         version indices (long)
        Returns:
            ε̂:          [B, 80, T]  predicted noise
        """
        # ── Conditioning vector ───────────────────────────────────────────
        c = torch.cat([self.time_emb(t), self.ver_emb(version_id)], dim=-1)  # [B, 256]

        # ── Input ─────────────────────────────────────────────────────────
        h = self.input_conv(torch.cat([x_t, score], dim=1))  # [B, 128, T]

        # ── Encoder ───────────────────────────────────────────────────────
        skips = []
        for enc, down in zip(self.enc_stacks, self.downsamples):
            h = enc(h, c)
            skips.append(h)
            h = down(h)

        # ── Bottleneck ────────────────────────────────────────────────────
        h = self.bottleneck_res1(h, c)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, c)

        # ── Decoder ───────────────────────────────────────────────────────
        for up, dec, skip in zip(self.upsamples, self.dec_stacks, reversed(skips)):
            h = up(h)
            # align time dimension (may differ by 1 due to odd input length)
            if h.shape[-1] != skip.shape[-1]:
                h = torch.nn.functional.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = dec(h, c)

        # ── Output ────────────────────────────────────────────────────────
        import torch.nn.functional as F
        h = F.silu(self.output_norm(h))
        return self.output_conv(h)
