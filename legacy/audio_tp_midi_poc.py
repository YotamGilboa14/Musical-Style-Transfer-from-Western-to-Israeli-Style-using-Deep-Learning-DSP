"""
Audio to MIDI Conversion Script
================================
This script converts audio files (WAV or MP3) to MIDI format using Spotify's Basic-Pitch.
Basic-Pitch is a polyphonic audio-to-MIDI transcription model that handles multiple instruments.
"""

import os
import argparse
from pathlib import Path
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH


def validate_audio_file(audio_path):
    """
    Validates that the input audio file exists and has a supported format.
    
    Parameters:
    -----------
    audio_path : str or Path
        Path to the audio file to validate
    
    Returns:
    --------
    Path
        Validated Path object pointing to the audio file
    
    Raises:
    -------
    FileNotFoundError
        If the audio file does not exist
    ValueError
        If the audio file format is not supported (not .wav or .mp3)
    """
    # Convert string to Path object for easier file handling
    audio_path = Path(audio_path)
    
    # Check if the file exists
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check if the file has a supported extension
    supported_formats = ['.wav', '.mp3']
    if audio_path.suffix.lower() not in supported_formats:
        raise ValueError(f"Unsupported audio format. Supported formats: {', '.join(supported_formats)}")
    
    return audio_path


def create_output_directory(output_dir):
    """
    Creates the output directory if it doesn't exist.
    
    Parameters:
    -----------
    output_dir : str or Path
        Path to the output directory
    
    Returns:
    --------
    Path
        Path object pointing to the output directory
    """
    # Convert to Path object
    output_dir = Path(output_dir)
    
    # Create directory and any necessary parent directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def convert_audio_to_midi(audio_path, output_dir, onset_threshold=0.5, frame_threshold=0.3,
                         minimum_note_length=127.70, minimum_frequency=None, maximum_frequency=None):
    """
    Converts an audio file to MIDI format using Spotify's Basic-Pitch model.
    
    Basic-Pitch is a polyphonic transcription model that can handle multiple
    simultaneous notes, making it suitable for complex music with instruments,
    vocals, and percussion.
    
    Parameters:
    -----------
    audio_path : str or Path
        Path to the input audio file (WAV or MP3 format)
    
    output_dir : str or Path
        Directory where the output MIDI file will be saved
    
    onset_threshold : float, optional (default=0.5)
        Threshold for note onset detection. Higher values = fewer, more confident notes.
        Range: 0.0 to 1.0. Typical values: 0.3-0.7
    
    frame_threshold : float, optional (default=0.3)
        Threshold for note frame detection. Controls note duration accuracy.
        Range: 0.0 to 1.0. Typical values: 0.1-0.5
    
    minimum_note_length : float, optional (default=127.70)
        Minimum note length in milliseconds. Shorter notes will be filtered out.
        Helps reduce noise and very short artifacts.
    
    minimum_frequency : float, optional (default=None)
        Minimum frequency in Hz to consider. Notes below this will be ignored.
        Useful for filtering out low-frequency noise.
    
    maximum_frequency : float, optional (default=None)
        Maximum frequency in Hz to consider. Notes above this will be ignored.
        Useful for focusing on specific instrument ranges.
    
    Returns:
    --------
    Path
        Path to the generated MIDI file
    
    Raises:
    -------
    Exception
        If the conversion process fails
    """
    # Validate inputs
    audio_path = validate_audio_file(audio_path)
    output_dir = create_output_directory(output_dir)
    
    print(f"Converting audio file: {audio_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters:")
    print(f"  - Onset threshold: {onset_threshold}")
    print(f"  - Frame threshold: {frame_threshold}")
    print(f"  - Minimum note length: {minimum_note_length} ms")
    print("Note: This may take several minutes for long audio files...")
    print("-" * 60)
    
    try:
        # Call the Basic-Pitch prediction function
        # This function:
        # 1. Loads the pre-trained neural network model
        # 2. Processes the audio file in chunks
        # 3. Predicts note onsets, pitches, and durations for polyphonic audio
        # 4. Converts predictions to MIDI format
        # 5. Saves the MIDI file to the output directory
        
        print("Loading Basic-Pitch model...")
        print("Transcribing audio (this will take a few minutes)...")
        
        predict_and_save(
            audio_path_list=[audio_path],           # List of audio files to process
            output_directory=output_dir,             # Where to save output files
            save_midi=True,                          # Save as MIDI file
            sonify_midi=False,                       # Don't create audio from MIDI
            save_model_outputs=False,                # Don't save raw model outputs
            save_notes=False,                        # Don't save notes as CSV
            model_or_model_path=ICASSP_2022_MODEL_PATH,  # Use the default trained model
            onset_threshold=onset_threshold,         # Note onset sensitivity
            frame_threshold=frame_threshold,         # Note frame sensitivity
            minimum_note_length=minimum_note_length, # Filter short notes
            minimum_frequency=minimum_frequency,     # Low frequency cutoff
            maximum_frequency=maximum_frequency      # High frequency cutoff
        )
        
        # Construct the expected output MIDI file path
        # Basic-Pitch saves files with the same name as input but with .mid extension
        midi_filename = audio_path.stem + "_basic_pitch.mid"
        output_midi_path = output_dir / midi_filename
        
        if output_midi_path.exists():
            print(f"\n✓ Successfully converted to MIDI: {output_midi_path}")
            return output_midi_path
        else:
            raise Exception("MIDI file was not created")
            
    except Exception as e:
        print(f"\n✗ Error during conversion: {str(e)}")
        raise


def main():
    """
    Main function to handle command-line interface for audio-to-MIDI conversion.
    
    Parses command-line arguments and executes the conversion process.
    """
    # Create argument parser for command-line interface
    parser = argparse.ArgumentParser(
        description="Convert audio files (WAV/MP3) to MIDI using Spotify's Basic-Pitch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_tp_midi_poc.py input.wav
  python audio_tp_midi_poc.py input.mp3 -o output_folder
  python audio_tp_midi_poc.py song.wav --onset-threshold 0.6 --frame-threshold 0.4
  python audio_tp_midi_poc.py song.mp3 --min-note-length 100 --min-freq 80 --max-freq 1000
        """
    )
    
    # Required argument: input audio file
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to the input audio file (WAV or MP3 format)'
    )
    
    # Optional argument: output directory
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='midi_output',
        help='Output directory for MIDI files (default: midi_output)'
    )
    
    # Optional argument: onset threshold
    parser.add_argument(
        '--onset-threshold',
        type=float,
        default=0.5,
        help='Onset threshold (0.0-1.0, default: 0.5). Higher = fewer notes'
    )
    
    # Optional argument: frame threshold
    parser.add_argument(
        '--frame-threshold',
        type=float,
        default=0.3,
        help='Frame threshold (0.0-1.0, default: 0.3). Controls note duration'
    )
    
    # Optional argument: minimum note length
    parser.add_argument(
        '--min-note-length',
        type=float,
        default=127.70,
        help='Minimum note length in milliseconds (default: 127.70)'
    )
    
    # Optional argument: minimum frequency
    parser.add_argument(
        '--min-freq',
        type=float,
        default=None,
        help='Minimum frequency in Hz (default: None)'
    )
    
    # Optional argument: maximum frequency
    parser.add_argument(
        '--max-freq',
        type=float,
        default=None,
        help='Maximum frequency in Hz (default: None)'
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Display script header
    print("=" * 60)
    print("Audio to MIDI Conversion")
    print("=" * 60)
    
    try:
        # Execute the conversion
        output_file = convert_audio_to_midi(
            audio_path=args.audio_file,
            output_dir=args.output,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
            minimum_note_length=args.min_note_length,
            minimum_frequency=args.min_freq,
            maximum_frequency=args.max_freq
        )
        
        print("=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"Conversion failed: {str(e)}")
        print("=" * 60)
        exit(1)


# Entry point of the script
if __name__ == "__main__":
    main()
