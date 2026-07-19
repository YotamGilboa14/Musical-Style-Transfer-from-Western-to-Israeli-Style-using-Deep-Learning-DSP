"""Legacy VAE-GAN proof-of-concept - dataset loader (superseded).

Part of the early trumpet-to-violin timbre-transfer experiment we built before
switching to the diffusion model. Kept for reference and learning only; the
final pipeline does not use it. See legacy/README for the full context.
"""

# Forked from ebadawy/voice_conversion/src/data_proc.py
# PyTorch Dataset for loading preprocessed mel spectrograms

import torch
import numpy as np
import pickle
import random

try:
    from .params_config import get_params as _get_params
    _p = _get_params()
except ImportError:
    import params as _p
num_samples = _p.num_samples


class DataProc(torch.utils.data.Dataset):
    """Dataset that loads preprocessed mel spectrograms from pickle files
    and returns random crops of num_samples frames from each speaker."""

    def __init__(self, args, split):
        self.args = args
        self.data_dict = pickle.load(
            open('%s/data_%s.pickle' % (args.dataset, split), 'rb'))

    def __len__(self):
        total_len = 0
        for i in range(len(self.data_dict.keys())):
            tmp = np.sum([j.shape[1] for j in self.data_dict[i]])
            total_len = max(total_len, tmp / 128)
        return int(total_len)

    def __getitem__(self, item):
        rslt = []
        n_spkrs = len(self.data_dict.keys())

        for i in range(0, n_spkrs):
            # Choose random item based on proportional distribution
            # (length of each sample)
            tmp_lens = [j.shape[1] for j in self.data_dict[i]]
            item = np.random.choice(
                len(tmp_lens), p=tmp_lens / np.sum(tmp_lens))
            rslt.append(self.random_sample(i, item))

        # Prepares a random sample per speaker
        samples = {}
        for i in range(0, n_spkrs):
            samples[i] = np.array(rslt)[i, :]
        return samples

    def augment(self, data, sample_rate=16000, pitch_shift=0.5):
        if pitch_shift == 0:
            return data
        return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=pitch_shift)

    def random_sample(self, i, item):
        n_samples = num_samples
        data = self.data_dict[i][item]
        assert data.shape[1] >= n_samples
        rand_i = random.randint(0, data.shape[1] - n_samples)
        data = data[:, rand_i:rand_i + n_samples]
        return np.array([data])
