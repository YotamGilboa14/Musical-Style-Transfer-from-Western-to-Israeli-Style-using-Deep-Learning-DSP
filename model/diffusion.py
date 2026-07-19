"""GaussianDiffusion - the training and sampling logic wrapped around the U-Net.

The U-Net only knows how to predict noise. This file adds everything around it
that turns "predict the noise" into "generate a styled mel":

  - cosine_beta_schedule  how much noise we add at each of the T steps
  - q_sample              the forward process: jump straight to a noisy version
  - p_losses              the training loss (L1) plus the CFG dropout that lets
                          one network act conditional and unconditional
  - ddim_sample           the reverse process: start from pure noise and walk
                          back to a clean mel, steered by score + style (CFG)
  - blend_overlap         stitches long songs together seamlessly at generation

All the schedule tensors are stored with register_buffer, which means PyTorch
treats them as part of the module (they move to the GPU with .to(device) and are
saved in the checkpoint) but never trains them.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from Nichol & Dhariwal (2021).

    The schedule decides how quickly we destroy the signal as t goes from 0 to
    T. A plain linear schedule adds noise too aggressively near the end, which
    wastes steps on almost-pure static; the cosine version spreads the noise
    more evenly, so more steps are spent where the model can still learn
    something useful. We clamp the per-step beta at 0.999 to avoid a fully
    singular final step.

    Returns beta_t in [0, 0.999] for t = 1 .. T.
    """
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    f_t = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999)


class GaussianDiffusion(nn.Module):
    """
    Wraps a UNet1D with the full diffusion process.

    Args:
        model:       UNet1D (or any module with the same forward signature)
        T:           number of diffusion timesteps (default 1000)
        n_versions:  number of version IDs (null token = n_versions index)
        cfg_score:   CFG guidance weight for score condition  (w_s)
        cfg_version: CFG guidance weight for version condition (w_v)
    """

    def __init__(
        self,
        model: nn.Module,
        T: int = 1000,
        n_versions: int = 25,
        cfg_score: float = 1.25,
        cfg_version: float = 1.25,
        cfg_drop_score: float = 0.10,
        cfg_drop_version: float = 0.10,
        cfg_drop_both: float = 0.05,
    ):
        """Store the denoiser and precompute the diffusion noise schedule."""
        super().__init__()
        self.model = model
        self.T = T
        self.null_version = n_versions  # index of the unconditional null token
        self.cfg_score = cfg_score
        self.cfg_version = cfg_version
        self.cfg_drop_score = cfg_drop_score
        self.cfg_drop_version = cfg_drop_version
        self.cfg_drop_both = cfg_drop_both

        # ── Precompute and register schedule buffers ──────────────────────
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    # ------------------------------------------------------------------
    # Forward noising
    # ------------------------------------------------------------------

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward noising: jump straight to step t in one shot.

        Diffusion adds noise gradually over T steps, but there is a closed-form
        shortcut so we never have to loop: x_t = sqrt(ab_t) * x_0 + sqrt(1-ab_t)
        * eps, where ab_t (alpha-bar) is how much of the original signal still
        survives at step t. During training we use this to noise a clean mel to
        a random level instantly and ask the network to predict the eps we added.

        Args:
            x_0:   [B, F, T_seg]  clean mel
            t:     [B]            timestep indices
            noise: optional pre-sampled noise (same shape as x_0)
        Returns:
            x_t:   [B, F, T_seg]
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        # view(-1,1,1) reshapes the per-sample scalars so they broadcast over
        # the channel and time axes of x_0.
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x_0 + sqrt_1mab * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        x_0: torch.Tensor,
        score: torch.Tensor,
        version_id: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L1 diffusion loss with classifier-free guidance dropout.

        CFG dropout (applied independently per sample in the batch):
          - Drop score with prob cfg_drop_score   → zero score tensor
          - Drop version with prob cfg_drop_version → null version token
          - Drop both simultaneously with prob cfg_drop_both (on top of above)

        Args:
            x_0:        [B, 80, T_seg]  clean mel
            score:      [B, 256, T_seg] flattened piano roll
            version_id: [B]             version indices (long)
            t:          [B]             sampled diffusion timesteps
        Returns:
            loss: scalar tensor
        """
        B = x_0.shape[0]
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # ── CFG dropout masks ─────────────────────────────────────────────
        # During training we deliberately hide conditions from the model on a
        # random subset of examples. That teaches one network how to denoise
        # unconditionally, with score only, with version only, and with both.
        # Inference can then subtract/add those predictions for guidance.
        # Score dropout
        score_keep = (torch.rand(B, device=x_0.device) >= self.cfg_drop_score).float()
        score_masked = score * score_keep.view(B, 1, 1)

        # Version dropout replaces the real style ID with the learned null
        # token. The null token is n_versions, never an actual style label.
        ver_keep = torch.rand(B, device=x_0.device) >= self.cfg_drop_version
        v = torch.where(
            ver_keep,
            version_id,
            torch.full_like(version_id, self.null_version),
        )

        # Joint drop (applied on top): zero both conditioning signals. This is
        # the true unconditional case used as the CFG anchor during sampling.
        joint_keep = torch.rand(B, device=x_0.device) >= self.cfg_drop_both
        score_masked = score_masked * joint_keep.float().view(B, 1, 1)
        v = torch.where(joint_keep, v, torch.full_like(v, self.null_version))

        pred_noise = self.model(x_t, t, score_masked, v)
        return F.l1_loss(pred_noise, noise)

    # ------------------------------------------------------------------
    # Overlap blending (used inside DDIM loop)
    # ------------------------------------------------------------------

    @staticmethod
    def blend_overlap(
        x0_segments: torch.Tensor, T_ol: int = 32
    ) -> torch.Tensor:
        """
        Linearly blend adjacent segment boundaries in the predicted x̂_0 tensor.
        Applied *inside* the DDIM loop before re-noising.

        Args:
            x0_segments: [N_seg, F, T_seg]  predicted clean mels for all segments
            T_ol:        number of overlap frames to blend
        Returns:
            blended tensor of same shape
        """
        if x0_segments.shape[0] <= 1 or T_ol == 0:
            return x0_segments
        mask = torch.linspace(1.0, 0.0, T_ol, device=x0_segments.device).view(1, 1, T_ol)
        result = x0_segments.clone()
        for i in range(x0_segments.shape[0] - 1):
            left_tail = result[i, :, -T_ol:]
            right_head = result[i + 1, :, :T_ol]
            blended = mask * left_tail + (1.0 - mask) * right_head
            result[i, :, -T_ol:] = blended
            result[i + 1, :, :T_ol] = blended
        return result

    # ------------------------------------------------------------------
    # DDIM sampling with CFG
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        score: torch.Tensor,
        version_id: torch.Tensor,
        N: int = 100,
        cfg_score: Optional[float] = None,
        cfg_version: Optional[float] = None,
        overlap_frames: int = 32,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling (η=0, deterministic) with compound CFG.

        Supports single-segment OR multi-segment (overlapped) generation:
          - Single segment: score shape [B, 256, T_seg]
          - Multi-segment:  score shape [N_seg, 256, T_seg]

        For multi-segment, blend_overlap is applied at every DDIM step to the
        predicted x̂_0 before re-noising, ensuring seamless boundaries.

        3 forward passes per step:
          ε_uncond  = model(x_t, t, score_zero,  null_version)
          ε_score   = model(x_t, t, score,        null_version)
          ε_version = model(x_t, t, score_zero,  version)

          ε̂ = ε_uncond + w_s*(ε_score - ε_uncond) + w_v*(ε_version - ε_uncond)

        Args:
            score:      [B_or_Nseg, 256, T_seg] piano roll (already flattened)
            version_id: [B_or_Nseg]              version indices (long)
            N:          number of DDIM steps
            cfg_score:  override default w_s
            cfg_version: override default w_v
            overlap_frames: blend window size for multi-segment
            return_intermediates: if True, also return list of x̂_0 at each step
        Returns:
            x_0: [B_or_Nseg, 80, T_seg] sampled (denoised) mel spectrograms
        """
        w_s = cfg_score if cfg_score is not None else self.cfg_score
        w_v = cfg_version if cfg_version is not None else self.cfg_version

        device = score.device
        B = score.shape[0]
        F_mel = self.model.output_conv.out_channels if hasattr(self.model, 'output_conv') else 80
        T_seg = score.shape[-1]

        # Strided timestep sequence τ_N > τ_{N-1} > … > τ_0 ≥ 0
        tau = torch.linspace(self.T - 1, 0, N, dtype=torch.long, device=device)

        # Start from pure noise. DDIM then walks this tensor backward from high
        # noise to low noise until it becomes a generated mel spectrogram.
        x = torch.randn(B, F_mel, T_seg, device=device)

        null_v = torch.full((B,), self.null_version, dtype=torch.long, device=device)
        score_zero = torch.zeros_like(score)
        intermediates = []

        for i, t_cur in enumerate(tau):
            t_batch = t_cur.expand(B)

            # 3-pass compound CFG. Notice there is no "full score+version" pass
            # here: the code estimates the score direction and the version
            # direction separately, both relative to the unconditional anchor.
            eps_uncond = self.model(x, t_batch, score_zero, null_v)
            eps_score = self.model(x, t_batch, score, null_v)
            eps_version = self.model(x, t_batch, score_zero, version_id)

            eps_hat = (
                eps_uncond
                + w_s * (eps_score - eps_uncond)
                + w_v * (eps_version - eps_uncond)
            )

            # DDIM update. First recover the clean-mel estimate x0 from the
            # guided noise prediction, then move to the next lower-noise level.
            ab_cur = self.alpha_bars[t_cur]
            sqrt_ab_cur = self.sqrt_alpha_bars[t_cur]
            sqrt_1mab_cur = self.sqrt_one_minus_alpha_bars[t_cur]

            # Predicted x̂_0
            x0_pred = (x - sqrt_1mab_cur * eps_hat) / sqrt_ab_cur
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Overlap blending (multi-segment). Blending x0 inside the loop
            # encourages adjacent 5-second windows to agree before re-noising,
            # reducing clicks or abrupt mel changes at segment boundaries.
            if B > 1 and overlap_frames > 0:
                x0_pred = self.blend_overlap(x0_pred, overlap_frames)

            if return_intermediates:
                intermediates.append(x0_pred.clone())

            if i < N - 1:
                t_next = tau[i + 1]
                ab_next = self.alpha_bars[t_next]
                sqrt_ab_next = ab_next.sqrt()
                sqrt_1mab_next = (1.0 - ab_next).sqrt()
                # Re-noise toward t_next (η=0 → no stochastic term)
                x = sqrt_ab_next * x0_pred + sqrt_1mab_next * eps_hat
            else:
                x = x0_pred

        if return_intermediates:
            return x, intermediates
        return x

    # ------------------------------------------------------------------
    # Convenience: concat overlapped segments after sampling
    # ------------------------------------------------------------------

    @staticmethod
    def concat_with_overlap(
        segments: torch.Tensor, overlap_frames: int = 32
    ) -> torch.Tensor:
        """
        Concatenate N sampled segments, discarding the overlap from each join.

        Args:
            segments:      [N_seg, F, T_seg]
            overlap_frames: number of frames blended at each boundary
        Returns:
            full mel:      [F, T_total]  where T_total = N_seg*T_seg - (N_seg-1)*overlap_frames
        """
        if segments.shape[0] == 1:
            return segments[0]
        parts = [segments[0]]
        for i in range(1, segments.shape[0]):
            parts.append(segments[i, :, overlap_frames:])
        return torch.cat(parts, dim=-1)
