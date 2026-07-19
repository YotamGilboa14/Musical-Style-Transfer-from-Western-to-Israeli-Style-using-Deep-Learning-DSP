"""
tt-vae-gan: Timbre Transfer with VAE-GAN
=========================================
Forked from ebadawy/voice_conversion (https://github.com/ebadawy/voice_conversion)
Adapted for the Music Style Transfer pipeline.

Architecture:
    - 1 shared Encoder → latent space
    - N Generators (one per timbre/instrument)
    - N Discriminators (PatchGAN)
    - CycleGAN-style unpaired training with KL + reconstruction + cycle losses

Original paper context: "One-to-Many Voice Conversion" using VAE-GAN
Our adaptation: Instrument timbre transfer (e.g., guitar → oud)
"""

from .models import Encoder, Generator, Discriminator, ResidualBlock
from .models import weights_init_normal, LambdaLR

__all__ = [
    "Encoder",
    "Generator", 
    "Discriminator",
    "ResidualBlock",
    "weights_init_normal",
    "LambdaLR",
]
