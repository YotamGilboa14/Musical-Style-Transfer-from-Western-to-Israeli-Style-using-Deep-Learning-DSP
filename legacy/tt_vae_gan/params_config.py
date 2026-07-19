"""
Parameter Configuration Selector
=================================
Routes all param imports through a single switchable module.

Usage:
    # Set BEFORE importing utils, preprocess, data_proc, etc.:
    import models.tt_vae_gan.params_config as params_config
    params_config.use_pipeline_params()    # Switch to 80-mel, 22kHz

    # Or via environment variable (e.g., from CLI or Colab):
    os.environ['TT_VAE_GAN_USE_PIPELINE'] = '1'

    # Or via CLI flags in train.py / preprocess.py:
    python -m models.tt_vae_gan.train --use_pipeline_params ...
"""

import os
import importlib

# Cached reference to the active params module
_active_params = None


def use_original_params():
    """Switch to original URMP params (128 mels, 16kHz)."""
    global _active_params
    from . import params as _p
    _active_params = _p
    return _active_params


def use_pipeline_params():
    """Switch to our pipeline params (80 mels, 22050Hz)."""
    global _active_params
    from . import params_pipeline as _p
    _active_params = _p
    return _active_params


def get_params():
    """Get the currently active params module.
    
    Auto-selects based on TT_VAE_GAN_USE_PIPELINE environment variable.
    Default: original params (for backward compatibility).
    """
    global _active_params
    if _active_params is not None:
        return _active_params
    
    if os.environ.get('TT_VAE_GAN_USE_PIPELINE', '0') == '1':
        return use_pipeline_params()
    else:
        return use_original_params()
