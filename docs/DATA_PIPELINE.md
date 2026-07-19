# Data pipeline

The first block of the project builds the dataset, because no Israeli-style
dataset exists. The code lives in `preprocessing/`, driven by
`process_song_offline.py` for a single song and `preprocessing/batch_ingest.py`
for a list.

## Why this is the hard part

Most music-generation research trains on very large, clean, Western datasets
that someone else already collected and labelled. For Israeli music none of that
exists, so we built our own set by hand: we chose artists and military-band
repertoire we considered representative of each style, found the recordings,
downloaded the audio, transcribed it, and turned it into training tensors.

Every step costs data. Many songs are simply not available in good quality, some
artists have only a handful of tracks online, and older recordings were mastered
decades ago so loudness and audio quality jump around a lot. We also could not
blindly grab everything — the songs in a style have to actually sound like that
style, which meant listening and curating rather than scraping. On top of the
small size there is little variety inside each style, and because we have no
clean reference transcriptions, our pitch labels come from an automatic
transcriber, which adds its own noise.

Each style ends up resting on only about **five to six hours** of music
(roughly 11 hours of Israeli music, or 17 including the Slakh reference).

## Steps

For each song:

1. **Download** the audio (`preprocessing/youtube_downloader.py`, using
   `yt-dlp`), driven by CSV lists (`batch_songs.csv`, `batch_songs_military.csv`).
2. **Transcribe** to MIDI with **Basic-Pitch** — this becomes the piano-roll
   content condition. (Basic-Pitch runs in its own Python 3.10 environment; see
   `docs/TRAINING.md`.)
3. **DSP** (`preprocessing/dsp_preprocessor.py`, using `librosa`): resample to
   22.05 kHz, apply the 80-bin mel filterbank (which also acts as an 8 kHz
   low-pass), log-compress, normalize to `[-1, 1]`, and cut into 5-second
   segments. In parallel, `pretty_midi` builds the aligned 256-channel
   piano-roll.
4. **Tensorize** the segments and upload them to Google Drive.

Optional stem separation (`preprocessing/source_separator.py`, Demucs) exists
but is off by default for the Israeli pipeline.

## Why mel-spectrograms and not raw audio or a plain STFT

Raw audio is far too dense to generate directly. A short-time Fourier transform
(STFT) helps by turning the signal into a time-frequency picture, but it keeps a
very fine, linear frequency axis with a lot of redundant detail. The
mel-spectrogram goes one step further: it warps the frequency axis to match how
human hearing spaces pitches, keeps only 80 bands, and log-compresses the
magnitudes so loud and quiet parts sit on a comparable scale. That gives a
compact, image-like input the U-Net can handle, and it matches exactly what the
vocoder expects on the way back to audio.

## Augmentation (`preprocessing/augmentation.py`)

Because the data is small, we augment it during training only (never at test
time), and each trick has a clear musical meaning:

- **Pitch-shift** — move the whole clip up or down by up to two semitones (a
  semitone is one piano key), so the model hears the same phrase in a slightly
  different key.
- **Time-stretch** — speed the clip up or slow it down by up to 10% without
  changing its pitch.
- **SpecAugment** — hide a few random stripes of the spectrogram so the model
  cannot lean on any single frequency band or moment.

The pitch and tempo changes are applied to the mel and the piano-roll together,
so the notes always stay lined up with the sound. Offline, each song is also
pre-expanded into pitch ±2 and time-stretch 0.9/1.1 copies so every version can
reuse them without re-running Basic-Pitch.

## Splitting (`preprocessing/split_dataset.py`)

We split into train / validation / test **grouped by song**, so a song and all
of its augmentations always land in the same split — otherwise the model could
"cheat" by seeing an augmented copy of a validation song during training. A few
songs per style are held out entirely as demo/inference songs.

## Dataset composition

See `configs/version_*.yaml` for the exact song lists. In short:

- **v0** — Slakh2100 Western rock, clean ground-truth MIDI, used as a reference
  and down-weighted during training (~5.5 h, 42 tracks).
- **v1** — Israeli artists, a blend of about 90 songs across a dozen artists
  (~5.7 h).
- **v2** — Israeli military-band songs, about 120 songs (~5.6 h).
