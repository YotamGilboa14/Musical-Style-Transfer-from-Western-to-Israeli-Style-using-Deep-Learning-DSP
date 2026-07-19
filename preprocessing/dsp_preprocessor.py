"""
DSP Preprocessing Pipeline for DDPM Audio Style Transfer
=========================================================

Converts raw audio/MIDI into aligned training tensors
with automatic Google Drive synchronization for team collaboration.

Features:
- Mel-Spectrogram extraction (22050 Hz, 80 mels, fmax=8000 Hz)
- [-1, 1] normalization for diffusion model compatibility
- 2-channel Piano Roll (onset + sustain) with frame alignment
- 5-second segmentation for DDPM training
- Dual-Save: Local storage + immediate Drive sync
- Visualization generation for presentations
- Dataset manifest tracking
- DSP params aligned with HiFi-GAN UNIVERSAL_V1 vocoder

Author: Yotam & Gal - StyleTransfer Music Project
Date: January 7, 2026
"""

import os
import sys
import pickle
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np
import librosa
import pretty_midi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DSPConfig:
    """Digital Signal Processing Configuration
    
    Parameters aligned with HiFi-GAN UNIVERSAL_V1 vocoder for
    seamless mel-to-waveform reconstruction.
    """
    sample_rate: int = 22050          # Hz (matches HiFi-GAN)
    hop_length: int = 256             # Samples per frame (matches HiFi-GAN)
    n_fft: int = 1024                 # FFT window size (matches HiFi-GAN)
    n_mels: int = 80                  # Mel-frequency bins (matches HiFi-GAN)
    fmin: float = 0.0                 # Min mel frequency in Hz (matches HiFi-GAN)
    fmax: float = 8000.0              # Max mel frequency in Hz (matches HiFi-GAN)
    segment_duration: float = 5.0     # Seconds
    
    @property
    def segment_samples(self) -> int:
        """Total samples in one segment"""
        return int(self.sample_rate * self.segment_duration)
    
    @property
    def segment_frames(self) -> int:
        """Total frames in one segment (for alignment)"""
        return int(self.segment_samples / self.hop_length)
    
    @property
    def midi_fs(self) -> float:
        """MIDI sampling rate for perfect alignment"""
        return self.sample_rate / self.hop_length


@dataclass
class PathConfig:
    """Project Path Configuration"""
    project_root: Path
    
    @property
    def input_audio_dir(self) -> Path:
        """Directory where downloaded source audio is expected."""
        return self.project_root / "youtube_downloads"
    
    @property
    def input_midi_dir(self) -> Path:
        """Directory where Basic-Pitch MIDI output is expected."""
        return self.project_root / "midi_output"
    
    @property
    def output_root(self) -> Path:
        """Root directory for generated tensors, manifests, and visualizations."""
        return self.project_root / "processed_data"
    
    @property
    def mels_dir(self) -> Path:
        """Directory for per-segment mel tensor files."""
        return self.output_root / "mels"
    
    @property
    def piano_rolls_dir(self) -> Path:
        """Directory for per-segment piano-roll tensor files."""
        return self.output_root / "piano_rolls"
    
    @property
    def visualizations_dir(self) -> Path:
        """Directory for diagnostic mel/piano-roll figures."""
        return self.output_root / "visualizations"
    
    @property
    def manifest_path(self) -> Path:
        """Path to the local dataset manifest CSV."""
        return self.output_root / "dataset_manifest.csv"
    
    def ensure_local_structure(self):
        """Create all required local directories"""
        for dir_path in [self.mels_dir, self.piano_rolls_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Local folder structure created at: {self.output_root}")


@dataclass
class SegmentMetadata:
    """Metadata for a processed segment"""
    segment_id: str
    local_mel_path: str
    local_score_path: str
    drive_mel_id: str
    drive_score_id: str
    style_label: str


# ============================================================================
# GOOGLE DRIVE SYNC MANAGER
# ============================================================================

class DriveSyncManager:
    """
    Manages Google Drive folder structure and file uploads.
    Reuses authentication from existing token.pickle.
    """
    
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    def __init__(self, root_folder_id: str, credentials_path: str = 'credentials.json'):
        """Prepare Drive API state and authenticate before uploads."""
        self.root_folder_id = root_folder_id
        self.credentials_path = credentials_path
        self.service = None
        self.folder_cache: Dict[str, str] = {}  # path -> folder_id
        
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate using existing token or create new one"""
        creds = None
        token_path = 'token.pickle'
        
        # Load existing token
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # Refresh or create new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("Refreshing Google Drive authentication...")
                creds.refresh(Request())
            else:
                print("Authenticating with Google Drive (browser will open)...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save token for future use
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('drive', 'v3', credentials=creds)
        print("✓ Google Drive authentication successful")
    
    def _find_folder_by_name(self, parent_id: str, folder_name: str) -> Optional[str]:
        """Find a folder by name within a parent folder"""
        query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        
        try:
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)',
                pageSize=1
            ).execute()
            
            files = results.get('files', [])
            return files[0]['id'] if files else None
        except HttpError as e:
            print(f"Error searching for folder '{folder_name}': {e}")
            return None
    
    def _create_folder(self, parent_id: str, folder_name: str) -> str:
        """Create a new folder in Drive"""
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        
        try:
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            print(f"  Created Drive folder: {folder_name}")
            return folder['id']
        except HttpError as e:
            print(f"Error creating folder '{folder_name}': {e}")
            raise
    
    def ensure_folder_structure(self) -> Dict[str, str]:
        """
        Ensure the required folder structure exists in Drive.
        Returns dict mapping folder names to their IDs.
        """
        print(f"Setting up Google Drive folder structure...")
        folders = {}
        
        # Ensure processed_data/ folder
        processed_data_id = self._find_folder_by_name(self.root_folder_id, 'processed_data')
        if not processed_data_id:
            processed_data_id = self._create_folder(self.root_folder_id, 'processed_data')
        folders['processed_data'] = processed_data_id
        
        # Ensure subfolders
        for subfolder_name in ['mels', 'piano_rolls', 'visualizations']:
            folder_id = self._find_folder_by_name(processed_data_id, subfolder_name)
            if not folder_id:
                folder_id = self._create_folder(processed_data_id, subfolder_name)
            folders[subfolder_name] = folder_id
            self.folder_cache[subfolder_name] = folder_id
        
        print("✓ Drive folder structure verified")
        return folders
    
    def upload_file(self, local_path: Path, drive_folder_name: str, 
                   mime_type: str = None) -> str:
        """
        Upload a file to a specific Drive folder.
        Returns the file ID in Drive.
        """
        if drive_folder_name not in self.folder_cache:
            raise ValueError(f"Unknown Drive folder: {drive_folder_name}")
        
        parent_id = self.folder_cache[drive_folder_name]
        file_name = local_path.name
        
        # Detect mime type if not provided
        if mime_type is None:
            if local_path.suffix == '.pt':
                mime_type = 'application/octet-stream'
            elif local_path.suffix == '.png':
                mime_type = 'image/png'
            else:
                mime_type = 'application/octet-stream'
        
        file_metadata = {
            'name': file_name,
            'parents': [parent_id]
        }
        
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
        
        try:
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            return file['id']
        except HttpError as e:
            print(f"Error uploading {file_name}: {e}")
            raise


# ============================================================================
# DSP PROCESSING FUNCTIONS
# ============================================================================

def load_and_resample_audio(audio_path: Path, target_sr: int = 22050) -> np.ndarray:
    """Load audio file and resample to target sample rate"""
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        print(f"  Loaded audio: {audio_path.name} ({len(y)/sr:.2f}s)")
        return y
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        raise


def extract_mel_spectrogram(y: np.ndarray, config: DSPConfig) -> np.ndarray:
    """
    Extract mel-spectrogram and convert to log-scale (dB).
    Uses fmin/fmax aligned with HiFi-GAN vocoder expectations.
    Returns: (n_mels, n_frames) array in dB scale (NOT normalized)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def normalize_mel(mel_spec_db: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize mel-spectrogram from dB scale to [-1, 1] range.
    Required for diffusion model training (data must match Gaussian noise scale).
    
    Args:
        mel_spec_db: (n_mels, n_frames) mel-spectrogram in dB scale
    
    Returns:
        normalized: (n_mels, n_frames) mel-spectrogram in [-1, 1] range
        mel_min: minimum dB value (for denormalization)
        mel_max: maximum dB value (for denormalization)
    """
    mel_min = float(mel_spec_db.min())
    mel_max = float(mel_spec_db.max())
    
    # Avoid division by zero for silent segments
    if mel_max - mel_min < 1e-8:
        return np.zeros_like(mel_spec_db), mel_min, mel_max
    
    normalized = 2.0 * (mel_spec_db - mel_min) / (mel_max - mel_min) - 1.0
    return normalized, mel_min, mel_max


def denormalize_mel(normalized_mel: np.ndarray, mel_min: float, mel_max: float) -> np.ndarray:
    """
    Denormalize mel-spectrogram from [-1, 1] back to dB scale.
    Used by vocoder pipeline for audio reconstruction.
    
    Args:
        normalized_mel: mel-spectrogram normalized to [-1, 1]
        mel_min: original minimum dB value
        mel_max: original maximum dB value
    
    Returns:
        mel_spec_db: mel-spectrogram in dB scale
    """
    return (normalized_mel + 1.0) / 2.0 * (mel_max - mel_min) + mel_min


def load_midi_to_piano_roll(midi_path: Path, config: DSPConfig, 
                            total_duration: float) -> np.ndarray:
    """
    Load MIDI and convert to 2-channel piano roll with perfect frame alignment.
    
    Channel 0 (Onset):   1 at the frame where a note begins
    Channel 1 (Sustain): 1 at all frames where a note is active
    
    Returns: (2, 128, n_frames) float32 array
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Calculate piano roll with aligned sampling rate
        piano_roll = midi_data.get_piano_roll(fs=config.midi_fs)
        
        # Sustain channel: any frame with velocity > 0
        sustain = (piano_roll > 0).astype(np.float32)
        
        # Onset channel: detect note beginnings (0 -> active transitions)
        onset = np.zeros_like(sustain)
        onset[:, 0] = sustain[:, 0]  # First frame: active notes are onsets
        if sustain.shape[1] > 1:
            onset[:, 1:] = np.maximum(0, np.diff(sustain, axis=1))  # Positive transitions
        
        # Stack into (2, 128, T)
        piano_roll_2ch = np.stack([onset, sustain], axis=0)
        
        # Ensure correct length (match audio frames)
        target_frames = int(total_duration * config.midi_fs)
        current_frames = piano_roll_2ch.shape[2]
        
        if current_frames < target_frames:
            # Pad with zeros
            padding = target_frames - current_frames
            piano_roll_2ch = np.pad(piano_roll_2ch, ((0, 0), (0, 0), (0, padding)), 
                                    mode='constant', constant_values=0)
        elif current_frames > target_frames:
            # Truncate
            piano_roll_2ch = piano_roll_2ch[:, :, :target_frames]
        
        print(f"  Loaded MIDI: {Path(midi_path).name} -> Piano roll shape: {piano_roll_2ch.shape} (onset+sustain)")
        return piano_roll_2ch
    
    except Exception as e:
        print(f"Error loading MIDI {midi_path}: {e}")
        raise


def segment_data(mel_spec: np.ndarray, piano_roll: np.ndarray, 
                config: DSPConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Segment mel-spectrogram and piano roll into fixed-duration chunks.
    
    Args:
        mel_spec: (n_mels, T) mel-spectrogram (normalized or raw)
        piano_roll: (2, 128, T) onset/sustain piano roll
        config: DSP configuration
    
    Returns: list of (mel_segment, piano_roll_segment) tuples
    """
    n_frames = mel_spec.shape[1]
    segment_frames = config.segment_frames
    
    segments = []
    
    for start_frame in range(0, n_frames, segment_frames):
        end_frame = min(start_frame + segment_frames, n_frames)
        
        # Skip if segment is too short (less than 80% of target)
        if (end_frame - start_frame) < int(segment_frames * 0.8):
            break
        
        mel_segment = mel_spec[:, start_frame:end_frame]
        # Piano roll is (2, 128, T) - slice on the last axis
        piano_segment = piano_roll[:, :, start_frame:end_frame]
        
        # Pad if necessary to exact segment length
        if mel_segment.shape[1] < segment_frames:
            pad_width = segment_frames - mel_segment.shape[1]
            mel_segment = np.pad(mel_segment, ((0, 0), (0, pad_width)), 
                                mode='constant', constant_values=mel_segment.min())
            piano_segment = np.pad(piano_segment, ((0, 0), (0, 0), (0, pad_width)), 
                                  mode='constant', constant_values=0)
        
        segments.append((mel_segment, piano_segment))
    
    return segments


def create_visualization(mel_segment: np.ndarray, piano_roll_segment: np.ndarray,
                        config: DSPConfig, song_name: str, output_path: Path,
                        mel_min: float = None, mel_max: float = None):
    """
    Create high-quality visualization showing aligned mel-spectrogram and piano roll.
    Saves as PNG for presentations.
    
    Args:
        mel_segment: (n_mels, T) - can be normalized [-1,1] or raw dB
        piano_roll_segment: (2, 128, T) onset/sustain or (128, T) legacy binary
        config: DSP configuration
        song_name: Name for the title
        output_path: Where to save the PNG
        mel_min: If provided, mel is normalized and will be denormalized for display
        mel_max: If provided, mel is normalized and will be denormalized for display
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Time axis (in seconds)
    time_axis = np.arange(mel_segment.shape[1]) * config.hop_length / config.sample_rate
    
    # Denormalize mel for display if normalization stats provided
    display_mel = mel_segment
    if mel_min is not None and mel_max is not None:
        display_mel = denormalize_mel(mel_segment, mel_min, mel_max)
    
    # Plot Mel-Spectrogram (dB scale)
    im1 = axes[0].imshow(
        display_mel,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, time_axis[-1], 0, config.n_mels]
    )
    axes[0].set_ylabel('Mel Frequency Bins', fontsize=12)
    axes[0].set_title(f'Mel-Spectrogram: {song_name} (Segment 1, fmax={config.fmax:.0f}Hz)', 
                     fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')
    cbar1.set_label('Magnitude (dB)', fontsize=10)
    
    # Handle both legacy (128, T) and new (2, 128, T) piano roll
    if piano_roll_segment.ndim == 3:
        # New format: (2, 128, T) - show sustain with onset overlay
        sustain = piano_roll_segment[1]  # (128, T)
        onset = piano_roll_segment[0]    # (128, T)
    else:
        # Legacy format: (128, T)
        sustain = piano_roll_segment
        onset = None
    
    # Plot Piano Roll (sustain channel)
    im2 = axes[1].imshow(
        sustain,
        aspect='auto',
        origin='lower',
        cmap='Greys',
        extent=[0, time_axis[-1], 0, 128],
        interpolation='nearest'
    )
    
    # Overlay onsets as red dots if available
    if onset is not None:
        onset_y, onset_x = np.where(onset > 0)
        if len(onset_x) > 0:
            onset_times = onset_x * config.hop_length / config.sample_rate
            axes[1].scatter(onset_times, onset_y, c='red', s=1, alpha=0.6, label='Onsets')
            axes[1].legend(loc='upper right', fontsize=9)
    
    axes[1].set_ylabel('MIDI Note Number', fontsize=12)
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    title_suffix = 'Onset + Sustain' if onset is not None else 'Binary'
    axes[1].set_title(f'Aligned Piano Roll ({title_suffix})', fontsize=14, fontweight='bold')
    axes[1].set_yticks([21, 40, 60, 80, 100, 108])  # Common piano range markers
    axes[1].set_yticklabels(['A0', 'E2', 'C4', 'G#5', 'E7', 'C8'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  \u2713 Visualization saved: {output_path.name}")


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

class DSPPreprocessor:
    """Main pipeline orchestrator"""
    
    def __init__(self, project_root: str, drive_root_id: str):
        """Create DSP, path, and Drive managers for the legacy pipeline class."""
        self.config = DSPConfig()
        self.paths = PathConfig(Path(project_root))
        self.drive_manager = DriveSyncManager(drive_root_id)
        
        # Setup
        self.paths.ensure_local_structure()
        self.drive_manager.ensure_folder_structure()
        
        # Manifest tracking
        self.manifest_entries: List[SegmentMetadata] = []
    
    def find_matching_midi(self, audio_path: Path) -> Optional[Path]:
        """
        Find corresponding MIDI file for an audio file.
        Convention: song.wav -> song_basic_pitch.mid
        """
        audio_stem = audio_path.stem
        
        # Try exact match with _basic_pitch suffix
        midi_path = self.paths.input_midi_dir / f"{audio_stem}_basic_pitch.mid"
        if midi_path.exists():
            return midi_path
        
        # Try without _basic_pitch (in case of manual naming)
        midi_path = self.paths.input_midi_dir / f"{audio_stem}.mid"
        if midi_path.exists():
            return midi_path
        
        return None
    
    def process_song(self, audio_path: Path) -> int:
        """
        Process a single song: load, align, segment, save, and upload.
        Returns number of segments created.
        """
        print(f"\n{'='*70}")
        print(f"Processing: {audio_path.name}")
        print(f"{'='*70}")
        
        # Find matching MIDI
        midi_path = self.find_matching_midi(audio_path)
        if not midi_path:
            print(f"  ⚠ No matching MIDI found for {audio_path.name}, skipping...")
            return 0
        
        # Load audio
        y = load_and_resample_audio(audio_path, self.config.sample_rate)
        duration = len(y) / self.config.sample_rate
        
        # Extract mel-spectrogram
        mel_spec = extract_mel_spectrogram(y, self.config)
        
        # Load MIDI as piano roll
        piano_roll = load_midi_to_piano_roll(midi_path, self.config, duration)
        
        # Segment into 5-second chunks
        segments = segment_data(mel_spec, piano_roll, self.config)
        print(f"  Created {len(segments)} segments ({self.config.segment_duration}s each)")
        
        if len(segments) == 0:
            print(f"  ⚠ No valid segments created, skipping...")
            return 0
        
        # Process each segment
        song_stem = audio_path.stem
        for idx, (mel_seg, piano_seg) in enumerate(segments):
            segment_id = f"{song_stem}_seg{idx:03d}"
            
            # Convert to PyTorch tensors
            mel_tensor = torch.from_numpy(mel_seg).float()
            piano_tensor = torch.from_numpy(piano_seg).float()
            
            # Save locally
            mel_local_path = self.paths.mels_dir / f"{segment_id}_mel.pt"
            piano_local_path = self.paths.piano_rolls_dir / f"{segment_id}_score.pt"
            
            torch.save(mel_tensor, mel_local_path)
            torch.save(piano_tensor, piano_local_path)
            
            # Upload to Drive
            print(f"  Uploading segment {idx+1}/{len(segments)} to Drive...", end=' ')
            mel_drive_id = self.drive_manager.upload_file(mel_local_path, 'mels')
            piano_drive_id = self.drive_manager.upload_file(piano_local_path, 'piano_rolls')
            print("✓")
            
            # Record metadata
            metadata = SegmentMetadata(
                segment_id=segment_id,
                local_mel_path=str(mel_local_path.relative_to(self.paths.project_root)),
                local_score_path=str(piano_local_path.relative_to(self.paths.project_root)),
                drive_mel_id=mel_drive_id,
                drive_score_id=piano_drive_id,
                style_label=song_stem  # Can be customized later
            )
            self.manifest_entries.append(metadata)
        
        # Create visualization for first segment only
        print(f"  Creating visualization for first segment...")
        vis_path = self.paths.visualizations_dir / f"{song_stem}_vis.png"
        create_visualization(
            segments[0][0], segments[0][1],
            self.config, song_stem, vis_path
        )
        
        # Upload visualization
        print(f"  Uploading visualization to Drive...", end=' ')
        self.drive_manager.upload_file(vis_path, 'visualizations', 'image/png')
        print("✓")
        
        return len(segments)
    
    def save_manifest(self):
        """Save dataset manifest CSV"""
        if not self.manifest_entries:
            print("No manifest entries to save.")
            return
        
        with open(self.paths.manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'segment_id', 'local_mel_path', 'local_score_path',
                'drive_mel_id', 'drive_score_id', 'style_label'
            ])
            writer.writeheader()
            for entry in self.manifest_entries:
                writer.writerow(asdict(entry))
        
        print(f"\n✓ Manifest saved: {self.paths.manifest_path}")
        print(f"  Total segments: {len(self.manifest_entries)}")
    
    def run(self):
        """Execute full preprocessing pipeline"""
        print("\n" + "="*70)
        print("DSP PREPROCESSING PIPELINE - DDPM Audio Style Transfer")
        print("="*70)
        print(f"Project Root: {self.paths.project_root}")
        print(f"Input Audio: {self.paths.input_audio_dir}")
        print(f"Input MIDI: {self.paths.input_midi_dir}")
        print(f"Output: {self.paths.output_root}")
        print(f"Drive Root ID: {self.drive_manager.root_folder_id}")
        print("="*70)
        
        # Find all audio files
        audio_files = list(self.paths.input_audio_dir.glob("*.wav"))
        audio_files.extend(list(self.paths.input_audio_dir.glob("*.mp3")))
        
        if not audio_files:
            print(f"\n⚠ No audio files found in {self.paths.input_audio_dir}")
            return
        
        print(f"\nFound {len(audio_files)} audio files to process\n")
        
        # Process each song
        total_segments = 0
        for audio_path in audio_files:
            try:
                n_segments = self.process_song(audio_path)
                total_segments += n_segments
            except Exception as e:
                print(f"  ✗ Error processing {audio_path.name}: {e}")
                continue
        
        # Save manifest
        self.save_manifest()
        
        # Summary
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Total audio files processed: {len(audio_files)}")
        print(f"Total segments created: {total_segments}")
        print(f"Local output: {self.paths.output_root}")
        print(f"Google Drive: synced to folder ID {self.drive_manager.root_folder_id}")
        print("="*70 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main execution"""
    # Configuration
    PROJECT_ROOT = r"C:\Users\yotam\StyleTransfer_Gal_Yotam\MusicProject"
    DRIVE_ROOT_FOLDER_ID = "1HOJSH_nFinzd0BCEZwH-zTYsGJmAP8UF"
    
    try:
        preprocessor = DSPPreprocessor(PROJECT_ROOT, DRIVE_ROOT_FOLDER_ID)
        preprocessor.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
