"""Legacy VAE-GAN proof-of-concept - preprocessing (superseded).

Part of the early trumpet-to-violin timbre-transfer experiment we built before
switching to the diffusion model. Kept for reference and learning only; the
final pipeline does not use it. See legacy/README for the full context.
"""

# Forked from ebadawy/voice_conversion/src/preprocess.py
# Preprocesses WAV files into mel spectrograms and saves as pickle files

import argparse
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Use relative imports when run as module, absolute when run as script
try:
    from .params_config import get_params as _get_params
    _p = _get_params()
    from .utils import ls, preprocess_wav, melspectrogram
except ImportError:
    from params_config import get_params as _get_params
    _p = _get_params()
    from utils import ls, preprocess_wav, melspectrogram

num_samples = _p.num_samples


def preprocess_dataset(dataset_path, n_spkrs=2, test_size=0.1, eval_size=0.1):
    """
    Preprocess WAV files from a structured dataset into mel spectrograms.
    
    Expected directory structure:
        dataset_path/
        ├── spkr_1/
        │   ├── sample1.wav
        │   └── sample2.wav
        └── spkr_2/
            ├── sample1.wav
            └── sample2.wav
    
    Args:
        dataset_path: Path to root dataset directory
        n_spkrs: Number of speaker/instrument directories
        test_size: Fraction for test split
        eval_size: Fraction for eval split
    """
    not_train_size = test_size + eval_size

    # Stores preprocessed spectrograms
    train_feats = defaultdict(list)
    eval_feats = defaultdict(list)
    test_feats = defaultdict(list)

    # Stores corresponding wav filenames
    train_refs = {}
    eval_refs = {}
    test_refs = {}

    def get_spect(wav, spkr):
        wav_path = os.path.join(dataset_path, 'spkr_%s' % (spkr + 1), wav)
        sample = preprocess_wav(wav_path)
        return melspectrogram(sample)

    for spkr in range(n_spkrs):
        # Cross-platform WAV listing
        spkr_dir = os.path.join(dataset_path, 'spkr_%s' % (spkr + 1))
        wavs = [f for f in os.listdir(spkr_dir) if f.endswith('.wav')]

        if len(wavs) < 3:
            print(f"Warning: spkr_{spkr + 1} has only {len(wavs)} WAV files. "
                  f"Need at least 3 for train/eval/test split.")
            continue

        # Train/eval/test split
        train_refs[spkr], temp_refs = train_test_split(
            wavs, test_size=not_train_size, random_state=42)
        if len(temp_refs) < 2:
            print("Not enough samples to split into eval and test sets.")
            eval_refs[spkr] = temp_refs
            test_refs[spkr] = []
        else:
            eval_refs[spkr], test_refs[spkr] = train_test_split(
                temp_refs, test_size=(test_size / not_train_size),
                random_state=42)

        # Process each split
        for wav in tqdm(train_refs[spkr], total=len(train_refs[spkr]),
                        desc="spkr_%d_train" % (spkr + 1)):
            spect = get_spect(wav, spkr)
            if spect.shape[1] >= num_samples:
                train_feats[spkr].append(spect)

        for wav in tqdm(eval_refs[spkr], total=len(eval_refs[spkr]),
                        desc="spkr_%d_eval" % (spkr + 1)):
            spect = get_spect(wav, spkr)
            if spect.shape[1] >= num_samples:
                eval_feats[spkr].append(spect)

        for wav in tqdm(test_refs[spkr], total=len(test_refs[spkr]),
                        desc="spkr_%d_test" % (spkr + 1)):
            spect = get_spect(wav, spkr)
            if spect.shape[1] >= num_samples:
                test_feats[spkr].append(spect)

    # Save preprocessed spectrograms
    pickle.dump(train_feats, open(os.path.join(dataset_path, 'data_train.pickle'), 'wb'))
    pickle.dump(eval_feats, open(os.path.join(dataset_path, 'data_eval.pickle'), 'wb'))
    pickle.dump(test_feats, open(os.path.join(dataset_path, 'data_test.pickle'), 'wb'))

    # Save corresponding filenames
    pickle.dump(train_refs, open(os.path.join(dataset_path, 'refs_train.pickle'), 'wb'))
    pickle.dump(eval_refs, open(os.path.join(dataset_path, 'refs_eval.pickle'), 'wb'))
    pickle.dump(test_refs, open(os.path.join(dataset_path, 'refs_test.pickle'), 'wb'))

    print(f"\nPreprocessing complete!")
    for spkr in range(n_spkrs):
        print(f"  spkr_{spkr + 1}: train={len(train_feats[spkr])}, "
              f"eval={len(eval_feats[spkr])}, test={len(test_feats[spkr])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="path to dataset")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="fraction for test split")
    parser.add_argument("--eval_size", type=float, default=0.1,
                        help="fraction for eval split")
    parser.add_argument("--n_spkrs", type=int, default=2,
                        help="number of speakers/instruments for conversion")
    parser.add_argument("--use_pipeline_params", action="store_true",
                        help="use pipeline params (80 mels, 22050 Hz)")
    opt = parser.parse_args()

    # Activate pipeline params if requested
    if opt.use_pipeline_params:
        os.environ['TT_VAE_GAN_USE_PIPELINE'] = '1'
        try:
            from .params_config import use_pipeline_params
            p = use_pipeline_params()
        except ImportError:
            from params_config import use_pipeline_params
            p = use_pipeline_params()
        print(f"Using PIPELINE params: {p.num_mels} mels, {p.sample_rate} Hz")

    print(opt)

    preprocess_dataset(opt.dataset, opt.n_spkrs, opt.test_size, opt.eval_size)
