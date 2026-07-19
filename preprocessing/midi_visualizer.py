"""
MIDI Visualizer Script
======================
This script visualizes MIDI files as piano rolls, showing notes over time.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pretty_midi
import numpy as np


def validate_midi_file(midi_path):
    """
    Validates that the input MIDI file exists and has correct format.
    
    Parameters:
    -----------
    midi_path : str or Path
        Path to the MIDI file to validate
    
    Returns:
    --------
    Path
        Validated Path object pointing to the MIDI file
    
    Raises:
    -------
    FileNotFoundError
        If the MIDI file does not exist
    ValueError
        If the file format is not supported (not .mid or .midi)
    """
    # Convert string to Path object for easier file handling
    midi_path = Path(midi_path)
    
    # Check if the file exists
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    
    # Check if the file has a supported extension
    supported_formats = ['.mid', '.midi']
    if midi_path.suffix.lower() not in supported_formats:
        raise ValueError(f"Unsupported file format. Supported formats: {', '.join(supported_formats)}")
    
    return midi_path


def visualize_midi_piano_roll(midi_path, output_path=None, show_plot=True):
    """
    Visualizes a MIDI file as a piano roll (notes over time).
    
    This function loads a MIDI file and creates a visual representation
    where the x-axis represents time and y-axis represents pitch (piano keys).
    Each note is shown as a horizontal bar.
    
    Parameters:
    -----------
    midi_path : str or Path
        Path to the MIDI file to visualize
    
    output_path : str or Path, optional
        If provided, saves the visualization to this path (PNG format)
    
    show_plot : bool, optional (default=True)
        If True, displays the plot in a window
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Validate and load MIDI file
    midi_path = validate_midi_file(midi_path)
    
    print(f"Loading MIDI file: {midi_path.name}")
    
    try:
        # Load MIDI file using pretty_midi
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Set title
        ax.set_title(f'Piano Roll Visualization: {midi_path.name}', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Color palette for different instruments
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Track all notes across all instruments
        all_pitches = []
        
        # Process each instrument
        for inst_idx, instrument in enumerate(midi_data.instruments):
            # Use different color for each instrument
            color = colors[inst_idx % len(colors)]
            
            print(f"Processing instrument {inst_idx + 1}/{len(midi_data.instruments)}: "
                  f"{instrument.name if instrument.name else 'Unnamed'} "
                  f"({len(instrument.notes)} notes)")
            
            # Draw each note as a rectangle
            for note in instrument.notes:
                # Note dimensions: start time, duration, pitch
                start_time = note.start
                duration = note.end - note.start
                pitch = note.pitch
                
                # Track pitch for y-axis limits
                all_pitches.append(pitch)
                
                # Create rectangle for note
                # Height of 0.8 to leave small gaps between notes
                rect = patches.Rectangle(
                    (start_time, pitch - 0.4),  # (x, y) position
                    duration,                    # width (duration)
                    0.8,                        # height
                    linewidth=0.5,
                    edgecolor='black',
                    facecolor=color,
                    alpha=0.7
                )
                ax.add_patch(rect)
        
        # Configure plot appearance
        if all_pitches:
            # Set y-axis limits based on actual pitch range
            min_pitch = min(all_pitches) - 2
            max_pitch = max(all_pitches) + 2
            ax.set_ylim(min_pitch, max_pitch)
            
            # Add piano key labels (note names)
            # Show labels for every octave
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            y_ticks = []
            y_labels = []
            
            for pitch in range(int(min_pitch), int(max_pitch) + 1):
                if pitch % 12 == 0:  # C notes (start of octave)
                    y_ticks.append(pitch)
                    octave = (pitch // 12) - 1
                    y_labels.append(f'C{octave}')
            
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
        else:
            print("Warning: No notes found in MIDI file")
        
        # Set x-axis to show time in seconds
        ax.set_xlim(0, midi_data.get_end_time())
        
        # Labels and grid
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pitch (MIDI Note Number)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add statistics
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        duration = midi_data.get_end_time()
        
        stats_text = (
            f'Total Notes: {total_notes}\n'
            f'Duration: {duration:.2f}s\n'
            f'Instruments: {len(midi_data.instruments)}'
        )
        
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {output_path}")
        
        # Show plot if requested
        if show_plot:
            print("\n✓ Displaying visualization...")
            plt.show()
        
        return fig
        
    except Exception as e:
        print(f"\n✗ Error visualizing MIDI file: {str(e)}")
        raise


def main():
    """
    Main function to handle command-line interface for MIDI visualization.
    """
    parser = argparse.ArgumentParser(
        description="Visualize MIDI files as piano rolls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize and display
  python midi_visualizer.py input.mid
  
  # Visualize and save to file
  python midi_visualizer.py input.mid -o visualization.png
  
  # Save without displaying
  python midi_visualizer.py input.mid -o output.png --no-show
        """
    )
    
    # Required argument: MIDI file
    parser.add_argument(
        'midi_file',
        type=str,
        help='Path to the MIDI file to visualize'
    )
    
    # Optional argument: output file
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Save visualization to this file (PNG format)'
    )
    
    # Optional argument: disable display
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (only save to file)'
    )
    
    args = parser.parse_args()
    
    # Display header
    print("=" * 60)
    print("MIDI Piano Roll Visualizer")
    print("=" * 60)
    
    try:
        # Create visualization
        visualize_midi_piano_roll(
            midi_path=args.midi_file,
            output_path=args.output,
            show_plot=not args.no_show
        )
        
        print("=" * 60)
        print("Visualization completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"Visualization failed: {str(e)}")
        print("=" * 60)
        exit(1)


if __name__ == "__main__":
    main()
