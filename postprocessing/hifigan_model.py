"""
HiFi-GAN Generator Model (V1)
==============================

Neural vocoder that converts mel-spectrograms to audio waveforms.
This is the Generator-only portion needed for inference.

Architecture: Transposed convolution upsampling + Multi-Receptive Field Fusion (MRF)
- Input: mel-spectrogram (80, T)
- Output: waveform (1, T * hop_length)
- Upsample factors: [8, 8, 2, 2] = 256x total (matches hop_length=256)

Source: Adapted from jik876/hifi-gan (MIT License)
Paper: "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"
       Jungil Kong, Jaehyeon Kim, Jaekyoung Bae (2020)

Author: Yotam & Gal - StyleTransfer Music Project  
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm


# ============================================================================
# CONSTANTS
# ============================================================================

# Leaky ReLU negative slope used throughout HiFi-GAN
LRELU_SLOPE = 0.1


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def init_weights(m, mean=0.0, std=0.01):
    """Initialize Conv layer weights with normal distribution.
    
    This is the standard initialization used in the original HiFi-GAN.
    Applied to all Conv1d and ConvTranspose1d layers.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """Calculate padding to maintain temporal resolution.
    
    For a dilated convolution, pad = (kernel_size * dilation - dilation) / 2
    This ensures the output length matches the input length ('same' padding).
    """
    return int((kernel_size * dilation - dilation) / 2)


class AttrDict(dict):
    """Dictionary subclass that allows attribute-style access.
    
    Used to load HiFi-GAN config.json as an object with dot notation.
    Example: config.upsample_rates instead of config['upsample_rates']
    """
    def __init__(self, *args, **kwargs):
        """Store dictionary items as object attributes too."""
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# ============================================================================
# RESIDUAL BLOCKS
# ============================================================================

class ResBlock1(nn.Module):
    """Multi-Receptive Field Fusion (MRF) Residual Block - Type 1.
    
    Uses 3 dilated convolution pairs with dilations (1,3,5).
    Each pair: dilated_conv -> activation -> 1x_conv -> activation -> residual add.
    This creates 3 different receptive field sizes, capturing patterns at
    multiple time scales (critical for modeling periodic audio signals).
    
    Used in V1 config (highest quality).
    """
    
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        """Create the three dilated residual convolution pairs."""
        super(ResBlock1, self).__init__()
        self.h = h
        
        # First convolutions in each pair: dilated convolutions with increasing dilation
        # Dilation (1,3,5) gives receptive fields of 3, 7, 11 samples respectively
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        
        # Second convolutions: always dilation=1, refine the dilated features
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
    
    def forward(self, x):
        """Apply 3 residual blocks with different dilations.
        
        Each block: x -> LeakyReLU -> dilated_conv -> LeakyReLU -> 1x_conv -> + x
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x  # Residual connection
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference (slightly faster)."""
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    """Multi-Receptive Field Fusion (MRF) Residual Block - Type 2.
    
    Lighter version with only 2 dilated convolutions (dilation 1, 3).
    Used in V2/V3 configs for faster inference.
    """
    
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        """Create the lighter two-layer dilated residual block."""
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)
    
    def forward(self, x):
        """Apply the lightweight residual block and return updated features."""
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization from this residual block for inference."""
        for l in self.convs:
            remove_weight_norm(l)


# ============================================================================
# GENERATOR
# ============================================================================

class Generator(nn.Module):
    """HiFi-GAN Generator - Converts mel-spectrograms to audio waveforms.
    
    Architecture Overview:
    1. conv_pre: Project 80 mel channels → upsample_initial_channel (e.g., 512)
    2. Upsampling blocks: Transpose convolutions that increase temporal resolution
       - V1 rates: [8, 8, 2, 2] → total 256x upsampling (matches hop_length)
       - Channel count halves at each stage: 512 → 256 → 128 → 64 → 32
    3. MRF blocks: At each resolution, apply multi-receptive-field residual blocks
       - Multiple kernel sizes (3, 7, 11) capture different temporal patterns
    4. conv_post: Project to 1 channel (mono audio)
    5. tanh: Squash output to [-1, 1] range
    
    Input shape:  (batch, 80, mel_frames)
    Output shape: (batch, 1, mel_frames * 256)
    """
    
    def __init__(self, h):
        """
        Args:
            h: AttrDict config with keys:
               - upsample_rates: list of int (e.g., [8, 8, 2, 2])
               - upsample_kernel_sizes: list of int (e.g., [16, 16, 4, 4])
               - upsample_initial_channel: int (e.g., 512)
               - resblock: str ('1' or '2')
               - resblock_kernel_sizes: list of int (e.g., [3, 7, 11])
               - resblock_dilation_sizes: list of list of int
        """
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        
        # Initial convolution: mel channels (80) → initial_channel
        # kernel_size=7 with padding=3 preserves temporal dimension
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        
        # Choose residual block type based on config
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        
        # Transposed convolution upsampling layers
        # Each layer doubles/quadruples/etc. the temporal resolution
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            # Channel count halves at each stage
            in_ch = h.upsample_initial_channel // (2 ** i)
            out_ch = h.upsample_initial_channel // (2 ** (i + 1))
            # padding=(k-u)//2 ensures exact upsampling by factor u
            self.ups.append(weight_norm(
                ConvTranspose1d(in_ch, out_ch, k, u, padding=(k - u) // 2)))
        
        # Multi-Receptive Field Fusion blocks after each upsampling
        # For each upsampling stage, we have num_kernels residual blocks
        # with different kernel sizes to capture patterns at different scales
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        
        # Final convolution: project to single-channel audio
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    
    def forward(self, x):
        """Generate audio waveform from mel-spectrogram.
        
        Args:
            x: (batch, 80, mel_frames) mel-spectrogram tensor
            
        Returns:
            (batch, 1, audio_samples) waveform tensor in [-1, 1]
        """
        # Project mel features to high-dimensional representation
        x = self.conv_pre(x)
        
        for i in range(self.num_upsamples):
            # Upsample: increase temporal resolution
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            
            # Multi-Receptive Field Fusion: average outputs from multiple ResBlocks
            # Each ResBlock has a different kernel size, capturing different patterns
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels  # Average across receptive fields
        
        # Final activation + projection to mono audio
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)  # Squash to [-1, 1]
        
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization from all layers.
        
        Called before inference to slightly speed up the forward pass.
        Weight norm is useful during training but unnecessary for inference.
        """
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
