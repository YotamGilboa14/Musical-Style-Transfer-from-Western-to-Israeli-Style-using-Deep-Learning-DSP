"""
MIDI Quality Diagnostic Script
==============================
This script compares the original audio with the generated MIDI to identify conversion issues.
"""

import argparse
from pathlib import Path
import librosa
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt


def load_audio(audio_path):
    """Load audio file and return waveform and sample rate."""
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(str(audio_path))
    return y, sr


def load_midi(midi_path):
    """Load MIDI file."""
    print(f"Loading MIDI: {midi_path}")
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    return midi_data


def analyze_audio(y, sr):
    """Analyze audio characteristics."""
    print("\n" + "=" * 60)
    print("AUDIO ANALYSIS")
    print("=" * 60)
    
    # Duration
    duration = len(y) / sr
    print(f"Duration: {duration:.2f} seconds")
    
    # Tempo estimation
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Estimated Tempo: {float(tempo):.1f} BPM")
    
    # Detect if music is monophonic or polyphonic
    # Use harmonic-percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Chromagram to see pitch classes over time
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    
    # Count active pitches per time frame
    active_pitches = np.sum(chroma > 0.5 * np.max(chroma), axis=0)
    avg_active_pitches = np.mean(active_pitches)
    
    print(f"Average simultaneous notes: {avg_active_pitches:.1f}")
    
    if avg_active_pitches > 2:
        print("⚠️  POLYPHONIC music detected (multiple notes at once)")
        print("   Simple pitch tracking struggles with polyphonic audio!")
    else:
        print("✓ MONOPHONIC music (single melody line)")
    
    # Check for vocals vs instruments
    # Spectral centroid indicates brightness
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = np.mean(spectral_centroid)
    
    print(f"Spectral Centroid: {avg_centroid:.1f} Hz")
    
    if avg_centroid > 3000:
        print("⚠️  High frequencies detected - possibly vocals or high instruments")
        print("   These can be challenging for pitch detection")
    
    # Check for percussive elements
    percussive_ratio = np.sum(np.abs(y_percussive)) / np.sum(np.abs(y))
    print(f"Percussive content: {percussive_ratio*100:.1f}%")
    
    if percussive_ratio > 0.3:
        print("⚠️  Significant percussion detected")
        print("   Percussion can create false pitch detections")
    
    return {
        'duration': duration,
        'tempo': tempo,
        'polyphony': avg_active_pitches,
        'spectral_centroid': avg_centroid,
        'percussive_ratio': percussive_ratio
    }


def analyze_midi(midi_data):
    """Analyze MIDI characteristics."""
    print("\n" + "=" * 60)
    print("MIDI ANALYSIS")
    print("=" * 60)
    
    total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    duration = midi_data.get_end_time()
    
    print(f"Total notes: {total_notes}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Instruments: {len(midi_data.instruments)}")
    
    # Analyze note distribution
    all_pitches = []
    all_velocities = []
    note_durations = []
    
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            all_pitches.append(note.pitch)
            all_velocities.append(note.velocity)
            note_durations.append(note.end - note.start)
    
    if all_pitches:
        print(f"\nPitch range: {min(all_pitches)} - {max(all_pitches)}")
        print(f"   ({pretty_midi.note_number_to_name(min(all_pitches))} - "
              f"{pretty_midi.note_number_to_name(max(all_pitches))})")
        
        avg_duration = np.mean(note_durations)
        print(f"Average note duration: {avg_duration:.3f} seconds")
        
        if avg_duration < 0.1:
            print("⚠️  Very short notes detected - may sound choppy")
        
        # Check for stuck notes (very long durations)
        max_duration = max(note_durations)
        if max_duration > 2.0:
            print(f"⚠️  Very long notes detected (up to {max_duration:.2f}s)")
            print("   These may be detection errors")
    
    return {
        'total_notes': total_notes,
        'duration': duration,
        'avg_note_duration': avg_duration if all_pitches else 0
    }


def visualize_comparison(audio_path, midi_path, y, sr, midi_data):
    """Create side-by-side comparison visualization."""
    print("\n" + "=" * 60)
    print("Creating comparison visualization...")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # Plot 1: Original audio waveform
    times = np.arange(len(y)) / sr
    axes[0].plot(times, y, alpha=0.6)
    axes[0].set_title('Original Audio Waveform', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Chromagram (pitch content over time)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', 
                                    ax=axes[1], cmap='coolwarm')
    axes[1].set_title('Audio Pitch Content (Chromagram)', fontsize=14, fontweight='bold')
    fig.colorbar(img, ax=axes[1])
    
    # Plot 3: MIDI piano roll
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            axes[2].plot([note.start, note.end], [note.pitch, note.pitch], 
                        linewidth=3, alpha=0.7)
    
    axes[2].set_title('Generated MIDI Notes', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('MIDI Note Number')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, midi_data.get_end_time())
    
    plt.tight_layout()
    
    # Save comparison
    output_path = Path(midi_path).parent / f"{Path(midi_path).stem}_diagnostic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison saved to: {output_path}")
    
    plt.show()


def suggest_improvements(audio_stats, midi_stats):
    """Suggest improvements based on analysis."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    issues = []
    
    # Check polyphony
    if audio_stats['polyphony'] > 2:
        issues.append({
            'issue': 'Polyphonic music',
            'impact': 'Simple pitch tracking only captures one note at a time',
            'solution': 'Use specialized polyphonic transcription tools like Basic-Pitch or MT3'
        })
    
    # Check percussion
    if audio_stats['percussive_ratio'] > 0.3:
        issues.append({
            'issue': 'High percussion content',
            'impact': 'Drums create false pitch detections',
            'solution': 'Pre-process audio to separate harmonic content, or adjust threshold'
        })
    
    # Check note duration
    if midi_stats['avg_note_duration'] < 0.1:
        issues.append({
            'issue': 'Very short notes',
            'impact': 'Results in choppy, unrealistic MIDI',
            'solution': 'Increase magnitude threshold or smooth pitch detection'
        })
    
    # Check spectral content
    if audio_stats['spectral_centroid'] > 3000:
        issues.append({
            'issue': 'Complex high-frequency content',
            'impact': 'Difficult for pitch detection algorithms',
            'solution': 'Apply low-pass filtering or use better transcription model'
        })
    
    if issues:
        for idx, issue in enumerate(issues, 1):
            print(f"\n{idx}. {issue['issue']}")
            print(f"   Impact: {issue['impact']}")
            print(f"   Solution: {issue['solution']}")
    else:
        print("\n✓ No major issues detected")
        print("  The conversion parameters may just need tuning")
    
    print("\n" + "-" * 60)
    print("GENERAL TIPS:")
    print("1. Try increasing --threshold parameter (0.15-0.2) for cleaner output")
    print("2. For polyphonic music, consider using specialized tools")
    print("3. Pre-process audio: remove vocals, isolate melody, reduce percussion")
    print("4. Test with simpler audio (single instrument, clear melody)")


def main():
    """Parse CLI arguments and run the audio-vs-MIDI diagnostic report."""
    parser = argparse.ArgumentParser(
        description="Diagnose MIDI conversion quality issues",
        epilog="Example: python midi_diagnostic.py audio.wav midi.mid"
    )
    
    parser.add_argument('audio_file', type=str, help='Original audio file (WAV/MP3)')
    parser.add_argument('midi_file', type=str, help='Generated MIDI file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MIDI QUALITY DIAGNOSTIC TOOL")
    print("=" * 60)
    
    try:
        # Load files
        y, sr = load_audio(args.audio_file)
        midi_data = load_midi(args.midi_file)
        
        # Analyze both
        audio_stats = analyze_audio(y, sr)
        midi_stats = analyze_midi(midi_data)
        
        # Visualize
        visualize_comparison(args.audio_file, args.midi_file, y, sr, midi_data)
        
        # Suggest improvements
        suggest_improvements(audio_stats, midi_stats)
        
        print("\n" + "=" * 60)
        print("Diagnostic completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
