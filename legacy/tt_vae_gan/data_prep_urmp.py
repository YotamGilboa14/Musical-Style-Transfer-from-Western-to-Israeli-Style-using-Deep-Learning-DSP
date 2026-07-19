"""
URMP Dataset Preparation Script
================================
Adapted from RussellSB/tt-vae-gan/data_prep/urmp.py

Prepares instrument-separated WAV files from the URMP dataset
into the directory structure expected by tt-vae-gan preprocessing.

Expected URMP structure:
    urmp_root/
    ├── 01_Jupiter_tpt_tpt/
    │   ├── AuSep_1_tpt_01_Jupiter.wav
    │   └── AuSep_2_tpt_01_Jupiter.wav
    ├── 02_Sonata_vn_vc/
    │   ├── AuSep_1_vn_02_Sonata.wav
    │   └── AuSep_2_vc_02_Sonata.wav
    ...

Output structure (for voice_conversion):
    outdir/
    ├── spkr_1/   (instrument 1, e.g., trumpet)
    │   ├── AuSep_1_tpt_01_Jupiter.wav
    │   └── ...
    └── spkr_2/   (instrument 2, e.g., violin)
        ├── AuSep_1_vn_02_Sonata.wav
        └── ...

Usage:
    python -m models.tt_vae_gan.data_prep_urmp --dataroot /path/to/urmp/
    python -m models.tt_vae_gan.data_prep_urmp --dataroot /path/to/urmp/ --instruments tpt vn fl

Available URMP instrument codes:
    tpt = trumpet, vn = violin, fl = flute, vc = cello, cl = clarinet,
    sax = saxophone, tbn = trombone, hn = horn, ob = oboe, bn = bassoon,
    tba = tuba, va = viola, db = double bass
"""

import argparse
import glob
import os
import shutil
from tqdm import tqdm


# ─── Default instrument pairs ──────────────────────────────────────────────────
# These match the pretrained URMP weights: G1=trumpet, G2=violin
DEFAULT_INSTRUMENTS = ['tpt', 'vn']


def get_audiosep_wavs(dataroot, instrument_code):
    """Find all audio-separated WAV files for a given instrument.
    
    URMP naming convention: AuSep_<track>_<instrument>_<piece>.wav
    """
    pattern = os.path.join(dataroot, '**', f'AuSep*{instrument_code}*.wav')
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"  Warning: No files found for instrument '{instrument_code}'")
        print(f"  Searched pattern: {pattern}")
    return sorted(files)


def prepare_instrument(outdir, instrument_files, instrument_code):
    """Copy instrument files to the speaker directory."""
    os.makedirs(outdir, exist_ok=True)
    for f in tqdm(instrument_files, desc=f"Extracting {instrument_code}"):
        shutil.copy(f, outdir)
    return len(instrument_files)


def prepare_urmp(dataroot, outdir, instruments):
    """
    Prepare URMP dataset for VAE-GAN training.
    
    Args:
        dataroot: Path to URMP dataset root
        outdir: Output directory (will contain spkr_1/, spkr_2/, etc.)
        instruments: List of instrument codes to extract
    """
    print(f"URMP Data Preparation")
    print(f"  Source: {dataroot}")
    print(f"  Output: {outdir}")
    print(f"  Instruments: {instruments}")
    print()

    total_files = 0
    for i, ins_code in enumerate(instruments):
        spkr_dir = os.path.join(outdir, f'spkr_{i + 1}')
        files = get_audiosep_wavs(dataroot, ins_code)
        count = prepare_instrument(spkr_dir, files, ins_code)
        total_files += count
        print(f"  spkr_{i + 1} ({ins_code}): {count} files")

    print(f"\nTotal: {total_files} files across {len(instruments)} instruments")
    print(f"\nNext step: preprocess the data:")
    print(f"  python -m models.tt_vae_gan.preprocess --dataset {outdir} "
          f"--n_spkrs {len(instruments)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare URMP dataset for VAE-GAN training")
    parser.add_argument("--dataroot", type=str,
                        default='../../datasets/urmp/',
                        help="root directory of the URMP dataset")
    parser.add_argument("--outdir", type=str,
                        default='models/tt_vae_gan/data/data_urmp/',
                        help="output directory for prepared data")
    parser.add_argument("--instruments", nargs='+',
                        default=DEFAULT_INSTRUMENTS,
                        help="instrument codes to extract "
                             "(default: tpt vn, matching pretrained weights)")
    args = parser.parse_args()
    print(args)

    prepare_urmp(args.dataroot, args.outdir, args.instruments)
