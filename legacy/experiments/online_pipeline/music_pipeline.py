"""
Music Pipeline - Master Orchestration Script
=============================================

Interactive pipeline that coordinates YouTube download, MIDI transcription,
and DSP preprocessing into a single song-centric workflow with Drive sync.

Features hierarchical metadata: Artist/Album/Song structure with version tracking.

Author: Yotam & Gal - StyleTransfer Music Project
Date: January 7, 2026
"""

import os
import sys
import subprocess
import csv
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
import torch

# Import from preprocessing modules (same directory)
from youtube_downloader import download_youtube_audio
# Note: audio_tp_midi_poc will be called as subprocess (requires Python 3.10)
from gdrive_uploader import authenticate_google_drive, find_or_create_nested_folder, upload_file_to_drive
from dsp_preprocessor import (
    DSPConfig,
    load_and_resample_audio,
    extract_mel_spectrogram,
    normalize_mel,
    denormalize_mel,
    load_midi_to_piano_roll,
    segment_data,
    create_visualization
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DRIVE_ROOT_FOLDER_ID = "1HOJSH_nFinzd0BCEZwH-zTYsGJmAP8UF"
MASTER_DATA_FOLDER = "MusicProjectData"
MANIFEST_CSV = "dataset_manifest.csv"


# ============================================================================
# METADATA STRUCTURES
# ============================================================================

@dataclass
class SegmentMetadata:
    """Metadata for a single segment"""
    artist: str
    album: str
    song_name: str
    version_id: int
    segment_idx: int
    segment_path: str
    score_path: str
    mel_min: float
    mel_max: float


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanitize_filename(name: str) -> str:
    """Sanitize song name for use as filename/folder name"""
    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.strip()


def print_step(step_num: int, total_steps: int, message: str):
    """Print formatted progress message"""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}/{total_steps}: {message}")
    print(f"{'='*70}")


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class MusicPipeline:
    """
    Master orchestration class for the complete music processing pipeline.
    Coordinates download, transcription, and DSP processing for a single song.
    """
    
    def __init__(self, youtube_url: str, artist: str, album: str, song_name: str, version_id: int):
        """
        Initialize pipeline. version_id is REQUIRED (no default).
        """
        self.youtube_url = youtube_url
        self.artist = sanitize_filename(artist)
        self.album = sanitize_filename(album)
        self.song_name = sanitize_filename(song_name)
        self.version_id = version_id
        
        # Local hierarchical paths: MusicProjectData/Artist/Album/Song/
        # Path relative to project root (one level up from preprocessing/)
        project_root = Path(__file__).parent.parent
        self.master_root = project_root / MASTER_DATA_FOLDER
        self.artist_dir = self.master_root / self.artist
        self.album_dir = self.artist_dir / self.album
        self.local_root = self.album_dir / self.song_name
        self.local_root.mkdir(parents=True, exist_ok=True)
        
        self.wav_path = self.local_root / f"{self.song_name}.wav"
        self.midi_path = self.local_root / f"{self.song_name}.mid"
        self.processed_dir = self.local_root / "processed_data"
        self.mels_dir = self.processed_dir / "mels"
        self.piano_rolls_dir = self.processed_dir / "piano_rolls"
        self.vis_path = self.processed_dir / "visualization.png"
        
        # Manifest path (master CSV at root)
        self.manifest_path = self.master_root / MANIFEST_CSV
        
        # Metadata tracking
        self.segment_metadata: List[SegmentMetadata] = []
        
        # DSP config
        self.dsp_config = DSPConfig()
        
        # Drive service
        self.drive_service = None
        self.song_folder_id = None
        self.processed_folder_id = None
        self.mels_folder_id = None
        self.piano_rolls_folder_id = None
    
    def setup_drive_structure(self):
        """Setup Google Drive folder structure: MusicProjectData/Artist/Album/Song/"""
        print("\n🔐 Authenticating with Google Drive...")
        self.drive_service = authenticate_google_drive()
        print("✓ Drive authentication successful")
        
        print(f"\n📁 Setting up Drive hierarchy: {self.artist}/{self.album}/{self.song_name}")
        
        # Create hierarchical folder structure
        song_folder_path = f"{MASTER_DATA_FOLDER}/{self.artist}/{self.album}/{self.song_name}"
        self.song_folder_id = find_or_create_nested_folder(
            self.drive_service, song_folder_path
        )
        
        # Create processed_data subfolder
        processed_path = f"{MASTER_DATA_FOLDER}/{self.artist}/{self.album}/{self.song_name}/processed_data"
        self.processed_folder_id = find_or_create_nested_folder(
            self.drive_service, processed_path
        )
        
        # Create mels subfolder
        mels_path = f"{MASTER_DATA_FOLDER}/{self.artist}/{self.album}/{self.song_name}/processed_data/mels"
        self.mels_folder_id = find_or_create_nested_folder(
            self.drive_service, mels_path
        )
        
        # Create piano_rolls subfolder
        piano_rolls_path = f"{MASTER_DATA_FOLDER}/{self.artist}/{self.album}/{self.song_name}/processed_data/piano_rolls"
        self.piano_rolls_folder_id = find_or_create_nested_folder(
            self.drive_service, piano_rolls_path
        )
        
        print("✓ Drive folder structure ready")
    
    def setup_local_structure(self):
        """Create local folder structure"""
        print(f"\n📂 Creating local folder structure...")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.mels_dir.mkdir(parents=True, exist_ok=True)
        self.piano_rolls_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Local structure created at: {self.local_root}")
    
    def phase1_download(self):
        """Phase 1: Download audio from YouTube"""
        print_step(1, 4, "YouTube Audio Download")
        
        print(f"🎵 Downloading from: {self.youtube_url}")
        
        # Download to temporary location
        temp_output = download_youtube_audio(
            self.youtube_url,
            str(self.local_root),
            audio_format='wav'
        )
        
        # The downloader returns the path to the downloaded file
        # We need to rename it to our standard format
        if temp_output and Path(temp_output).exists():
            temp_path = Path(temp_output)
            if temp_path != self.wav_path:
                # Remove target if it already exists
                if self.wav_path.exists():
                    self.wav_path.unlink()
                # Rename to standard format
                temp_path.rename(self.wav_path)
                print(f"✓ Renamed to: {self.wav_path.name}")
        else:
            # If download_youtube_audio doesn't return path, find the newest wav
            wav_files = list(self.local_root.glob("*.wav"))
            if wav_files:
                newest_wav = max(wav_files, key=lambda p: p.stat().st_mtime)
                if newest_wav != self.wav_path:
                    # Remove target if it already exists
                    if self.wav_path.exists():
                        self.wav_path.unlink()
                    newest_wav.rename(self.wav_path)
            else:
                raise FileNotFoundError("Downloaded WAV file not found")
        
        print(f"✓ Downloaded: {self.wav_path.name}")
        
        # Upload to Drive
        print("☁️  Uploading WAV to Drive...", end=' ')
        upload_file_to_drive(
            self.drive_service,
            str(self.wav_path),
            self.song_folder_id
        )
        print("✓")
    
    def phase2_transcribe(self):
        """Phase 2: Convert audio to MIDI using Basic-Pitch"""
        print_step(2, 4, "MIDI Transcription (Basic-Pitch)")
        
        print(f"🎹 Converting {self.wav_path.name} to MIDI...")
        print("⚠️  Using Python 3.10 environment (basic_pitch_env)")
        
        # Check if MIDI output already exists from a previous run
        expected_midi_in_output = Path("midi_output") / f"{self.wav_path.stem}_basic_pitch.mid"
        if expected_midi_in_output.exists():
            print(f"  Found existing MIDI file, removing to allow fresh conversion...")
            expected_midi_in_output.unlink()
        
        # Call the Basic-Pitch script using subprocess with Python 3.10
        basic_pitch_python = Path("..") / "basic_pitch_env" / "Scripts" / "python.exe"
        basic_pitch_script = Path(__file__).parent / "audio_tp_midi_poc.py"
        
        # Build command
        cmd = [
            str(basic_pitch_python),
            str(basic_pitch_script),
            str(self.wav_path)
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        
        # Execute conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        # Note: Basic-Pitch writes warnings to stderr, so check for actual MIDI file instead of return code
        # The conversion creates filename_basic_pitch.mid in midi_output/
        # We need to move it to our song folder
        expected_midi = Path("midi_output") / f"{self.wav_path.stem}_basic_pitch.mid"
        
        if expected_midi.exists():
            # Move to song folder and rename
            expected_midi.rename(self.midi_path)
            print(f"✓ Transcribed and moved: {self.midi_path.name}")
        else:
            print(f"❌ MIDI conversion failed!")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            raise FileNotFoundError(f"MIDI file not found at: {expected_midi}")
        
        # Upload to Drive
        print("☁️  Uploading MIDI to Drive...", end=' ')
        upload_file_to_drive(
            self.drive_service,
            str(self.midi_path),
            self.song_folder_id
        )
        print("✓")
    
    def phase3_dsp_processing(self):
        """Phase 3: DSP Processing - Extract and segment features"""
        print_step(3, 4, "DSP Processing & Feature Extraction")
        
        print("🔊 Loading and processing audio...")
        
        # Load audio
        y = load_and_resample_audio(self.wav_path, self.dsp_config.sample_rate)
        duration = len(y) / self.dsp_config.sample_rate
        print(f"  Audio duration: {duration:.2f}s")
        
        # Extract mel-spectrogram
        print("  Extracting mel-spectrogram...")
        mel_spec = extract_mel_spectrogram(y, self.dsp_config)
        print(f"  Mel-spectrogram shape: {mel_spec.shape}")
        
        # Load MIDI as piano roll
        print("  Converting MIDI to piano roll...")
        piano_roll = load_midi_to_piano_roll(self.midi_path, self.dsp_config, duration)
        print(f"  Piano roll shape: {piano_roll.shape}")
        
        # Segment data
        print(f"  Segmenting into {self.dsp_config.segment_duration}s chunks...")
        segments = segment_data(mel_spec, piano_roll, self.dsp_config)
        n_segments = len(segments)
        print(f"  Created {n_segments} segments")
        
        if n_segments == 0:
            print("⚠️  No valid segments created (audio too short?)")
            return
        
        # Save and upload each segment
        print(f"\n💾 Saving and uploading {n_segments} segments...")
        for idx, (mel_seg, piano_seg) in enumerate(segments):
            # Convert to tensors
            mel_tensor = torch.from_numpy(mel_seg).float()
            piano_tensor = torch.from_numpy(piano_seg).float()
            
            # Generate filenames
            segment_id = f"{self.song_name}_seg{idx:03d}"
            mel_filename = f"{segment_id}_mel.pt"
            piano_filename = f"{segment_id}_score.pt"
            
            # Local paths
            mel_local = self.mels_dir / mel_filename
            piano_local = self.piano_rolls_dir / piano_filename
            
            # Save locally
            torch.save(mel_tensor, mel_local)
            torch.save(piano_tensor, piano_local)
            
            # Upload to Drive
            print(f"  Segment {idx+1}/{n_segments}: ", end='')
            upload_file_to_drive(self.drive_service, str(mel_local), self.mels_folder_id)
            upload_file_to_drive(self.drive_service, str(piano_local), self.piano_rolls_folder_id)
            print("✓")
            
            # Track metadata (relative paths from MusicProjectData root)
            segment_rel_path = str(mel_local.relative_to(self.master_root))
            score_rel_path = str(piano_local.relative_to(self.master_root))
            
            metadata = SegmentMetadata(
                artist=self.artist,
                album=self.album,
                song_name=self.song_name,
                version_id=self.version_id,
                segment_path=segment_rel_path,
                score_path=score_rel_path
            )
            self.segment_metadata.append(metadata)
        
        # Create visualization for first segment
        print(f"\n🎨 Creating visualization for first segment...")
        create_visualization(
            segments[0][0], segments[0][1],
            self.dsp_config, self.song_name, self.vis_path
        )
        print(f"✓ Visualization saved: {self.vis_path.name}")
        
        # Upload visualization
        print("☁️  Uploading visualization to Drive...", end=' ')
        upload_file_to_drive(
            self.drive_service,
            str(self.vis_path),
            self.processed_folder_id
        )
        print("✓")
        
        print(f"\n✓ DSP Processing complete: {n_segments} segments created")
    
    def deduplicate_and_save_manifest(self):
        """
        Remove old entries for this artist/album/song from manifest,
        then append new entries.
        """
        print("\n💾 Updating master manifest...")
        
        # Load existing manifest
        existing_entries = []
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Keep entries that don't match this song
                    if not (row['artist'] == self.artist and 
                           row['album'] == self.album and 
                           row['song_name'] == self.song_name):
                        existing_entries.append(row)
        
        # Write back deduplicated entries + new entries
        with open(self.manifest_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['artist', 'album', 'song_name', 'version_id', 'segment_idx',
                          'segment_path', 'score_path', 'mel_min', 'mel_max']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write existing (non-duplicate) entries
            for entry in existing_entries:
                # Migrate old entries: add new fields with defaults if missing
                for field in ['segment_idx', 'mel_min', 'mel_max']:
                    if field not in entry:
                        entry[field] = ''
                writer.writerow(entry)
            
            # Write new entries
            for metadata in self.segment_metadata:
                writer.writerow(asdict(metadata))
        
        print(f"✓ Manifest updated: {self.manifest_path}")
        print(f"  New segments added: {len(self.segment_metadata)}")
        print(f"  Total entries: {len(existing_entries) + len(self.segment_metadata)}")
    
    def run(self):
        """Execute the complete pipeline"""
        print("\n" + "="*70)
        print("🎵 MUSIC PROCESSING PIPELINE")
        print("="*70)
        print(f"Artist: {self.artist}")
        print(f"Album: {self.album}")
        print(f"Song: {self.song_name}")
        print(f"Version: {self.version_id}")
        print(f"YouTube URL: {self.youtube_url}")
        print(f"Local Output: {self.local_root}")
        print("="*70)
        
        try:
            # Setup
            self.setup_local_structure()
            self.setup_drive_structure()
            
            # Phase 1: Download
            self.phase1_download()
            
            # Phase 2: Transcription
            self.phase2_transcribe()
            
            # Phase 3: DSP Processing
            self.phase3_dsp_processing()
            
            # Phase 4: Update manifest
            self.deduplicate_and_save_manifest()
            
            # Success summary
            print_step(4, 4, "PIPELINE COMPLETE ✓")
            print(f"\n🎉 Successfully processed: {self.artist} - {self.album} - {self.song_name}")
            print(f"\n📊 Output Summary:")
            print(f"  Local hierarchy: {self.artist}/{self.album}/{self.song_name}/")
            print(f"  - {self.wav_path.name}")
            print(f"  - {self.midi_path.name}")
            print(f"  - processed_data/")
            print(f"    - visualization.png")
            print(f"    - mels/ ({len(self.segment_metadata)} tensor files)")
            print(f"    - piano_rolls/ ({len(self.segment_metadata)} tensor files)")
            print(f"\n☁️  All files synced to Google Drive")
            print(f"  Drive path: {MASTER_DATA_FOLDER}/{self.artist}/{self.album}/{self.song_name}/")
            print(f"\n📋 Manifest: {self.manifest_path}")
            print("\n" + "="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Run pipeline in interactive mode with user input"""
    print("\n" + "="*70)
    print("🎵 INTERACTIVE MUSIC PIPELINE")
    print("="*70)
    print("\nThis pipeline will:")
    print("  1. Download audio from YouTube")
    print("  2. Convert audio to MIDI (Basic-Pitch)")
    print("  3. Extract features and create training tensors")
    print("  4. Upload everything to Google Drive")
    print("  5. Update master manifest CSV")
    print("\n" + "="*70)
    
    # Get user input
    print("\n📝 Please provide the following information:\n")
    
    youtube_url = input("YouTube URL: ").strip()
    if not youtube_url:
        print("❌ YouTube URL is required!")
        sys.exit(1)
    
    artist = input("Artist Name: ").strip()
    if not artist:
        print("❌ Artist name is required!")
        sys.exit(1)
    
    album = input("Album Name: ").strip()
    if not album:
        print("❌ Album name is required!")
        sys.exit(1)
    
    song_name = input("Song Name: ").strip()
    if not song_name:
        print("❌ Song name is required!")
        sys.exit(1)
    
    version_input = input("Version ID (integer, required): ").strip()
    if not version_input:
        print("❌ Version ID is required!")
        sys.exit(1)
    
    try:
        version_id = int(version_input)
    except ValueError:
        print("❌ Version ID must be a valid integer!")
        sys.exit(1)
    
    # Confirm
    print(f"\n✓ Artist: {artist}")
    print(f"✓ Album: {album}")
    print(f"✓ Song: {song_name}")
    print(f"✓ Version: {version_id}")
    print(f"✓ YouTube URL: {youtube_url}")
    print(f"✓ Folder structure: MusicProjectData/{artist}/{album}/{song_name}/")
    
    confirm = input("\nProceed with pipeline? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Pipeline cancelled.")
        sys.exit(0)
    
    # Run pipeline
    pipeline = MusicPipeline(youtube_url, artist, album, song_name, version_id)
    pipeline.run()


def batch_mode(batch_data: List[Dict[str, str]]):
    """
    Run pipeline in batch mode for multiple songs.
    
    Args:
        batch_data: List of dicts with keys: url, artist, album, song_name, version_id
    """
    total_songs = len(batch_data)
    print(f"\n🎵 BATCH MODE: Processing {total_songs} songs\n")
    
    for idx, data in enumerate(batch_data, 1):
        print(f"\n{'#'*70}")
        print(f"SONG {idx}/{total_songs}: {data['artist']} - {data['album']} - {data['song_name']}")
        print(f"{'#'*70}\n")
        
        version_id = int(data['version_id'])
        pipeline = MusicPipeline(
            data['url'], 
            data['artist'], 
            data['album'], 
            data['song_name'],
            version_id
        )
        pipeline.run()
        
        print(f"\n✓ Completed {idx}/{total_songs} songs\n")
    
    print(f"\n🎉 BATCH COMPLETE: All {total_songs} songs processed!\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Music Processing Pipeline - YouTube to DDPM Training Data"
    )
    parser.add_argument(
        '--url', type=str,
        help='YouTube URL (skip interactive mode)'
    )
    parser.add_argument(
        '--artist', type=str,
        help='Artist name (skip interactive mode)'
    )
    parser.add_argument(
        '--album', type=str,
        help='Album name (skip interactive mode)'
    )
    parser.add_argument(
        '--name', type=str,
        help='Song name (skip interactive mode)'
    )
    parser.add_argument(
        '--version', type=int, required=False,
        help='Version ID (required for direct mode)'
    )
    parser.add_argument(
        '--batch', type=str,
        help='Path to CSV file with columns: url,artist,album,song_name,version_id (all required)'
    )
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch:
        batch_data = []
        with open(args.batch, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, 1):
                # Validate all required fields exist
                if 'version_id' not in row or not row['version_id']:
                    print(f"❌ Error: version_id is required for row {row_num} in batch CSV")
                    sys.exit(1)
                try:
                    int(row['version_id'])  # Validate it's an integer
                except ValueError:
                    print(f"❌ Error: version_id must be an integer in row {row_num}")
                    sys.exit(1)
                
                batch_data.append({
                    'url': row['url'],
                    'artist': row['artist'],
                    'album': row['album'],
                    'song_name': row['song_name'],
                    'version_id': row['version_id']
                })
        batch_mode(batch_data)
    
    # Direct mode
    elif args.url and args.artist and args.album and args.name:
        if args.version is None:
            print("❌ Error: --version is required for direct mode")
            sys.exit(1)
        pipeline = MusicPipeline(args.url, args.artist, args.album, args.name, args.version)
        pipeline.run()
    
    # Interactive mode (default)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
