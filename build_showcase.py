"""Portable PROJECT SHOWCASE builder.

Produces a single self-contained ``project_showcase/`` bundle that tells the
project story following the block diagram (Preprocessing -> Training ->
Postprocessing) using **playable audio, images and numbers**, and doubles as a
drag-and-drop asset library for building a poster / project book / slides.

Design goals
------------
* **Portable**: every asset is *physically copied* into ``assets/`` and every
  link in ``index.html`` is *relative*. Zip the folder, open anywhere, drag a
  ``.wav`` straight into PowerPoint.
* **Reuse first**: already-built figures/metrics are copied, not recomputed.
  Only the genuinely-missing visuals are generated (original-audio mels,
  MIDI-as-audio renders, a combined loss-curve comparison, and — if available —
  the diffusion denoising progression).
* **Traceable**: ``MANIFEST.csv`` records every asset with its stage, type,
  caption and which specification it supports.

Run (from ``MusicProject/``; ``ml_env`` has every needed library)::

    .\\ml_env\\Scripts\\python.exe .\\build_showcase.py \\
        --config .\\deliverables\\showcase_config.yaml \\
        --out_dir .\\project_showcase

Then open ``project_showcase/index.html``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

# Mel display parameters (for freshly-computed original-audio mels).
MEL_SR = 22050
MEL_N_MELS = 80
MEL_HOP = 256
MEL_N_FFT = 1024


# ===========================================================================
# Block diagrams (Mermaid) - copied verbatim from README.md (source of truth).
# ===========================================================================
DIAGRAM_MACRO = """flowchart LR
    URL(("Song URL"))
    PRE["Preprocessing"]
    TENS[("Mel-spectrogram tensors /<br/>piano-roll tensors")]
    VER(["Style ID (version)"])
    MODEL["Model block<br/>training + inference"]
    GMEL[("Generated<br/>mel-spectrogram")]
    POST["Postprocessing"]
    WAV(("Generated audio (WAV)"))
    MET[("Metric test results")]
    VIS[("Visualizations")]
    PVIS[("Preprocessing<br/>visualizations")]
    URL --> PRE --> TENS --> MODEL --> GMEL --> POST
    VER --> MODEL
    PRE -.-> PVIS
    POST --> WAV
    POST --> MET
    POST --> VIS
    classDef preprocess fill:#fff,stroke:#22c55e,stroke-width:2px,color:#222
    classDef model fill:#fff,stroke:#a855f7,stroke-width:3px,color:#222
    classDef postprocess fill:#fff,stroke:#22c55e,stroke-width:2px,color:#222
    classDef io fill:#fff,stroke:#64748b,color:#666
    class PRE preprocess
    class MODEL model
    class POST postprocess
    class URL,TENS,VER,GMEL,WAV,MET,VIS,PVIS io"""

DIAGRAM_PREPROCESS = """flowchart LR
    URL(("Song URL"))
    DL["Download audio (WAV)"]
    AUD[("Working audio")]
    BP["Basic-Pitch<br/>note transcription (MIDI)"]
    MEL["Mel-spectrogram<br/>extraction"]
    PR["Piano roll<br/>(note grid)"]
    SEG["5-second segments"]
    AUG["Optional augmentation"]
    TENS[("Mel-spectrogram tensors /<br/>piano-roll tensors")]
    VIZ[("Diagnostic<br/>visualizations")]
    URL --> DL --> AUD
    AUD --> BP --> PR --> SEG
    AUD --> MEL --> SEG
    SEG --> TENS
    SEG -. "augment=true" .-> AUG --> TENS
    SEG -. "diagnostic" .-> VIZ
    classDef custom fill:#fff,stroke:#22c55e,stroke-width:2px,color:#222
    classDef oss fill:#fff,stroke:#4f8cff,stroke-width:2px,color:#222
    classDef modified fill:#fff,stroke:#f59e0b,stroke-width:2px,color:#222
    classDef data fill:#fff,stroke:#64748b,color:#666
    class DL,BP modified
    class MEL,PR,SEG,AUG custom
    class URL,AUD,TENS,VIZ data"""

DIAGRAM_MODEL = """flowchart LR
    subgraph TRAINPHASE["1 - TRAINING (loop over the whole dataset, ~250,000 weight updates)"]
        direction TB
        MEL[("Real mel-spectrogram<br/>[80 x 430]")]
        EPS[("Random noise<br/>[80 x 430]")]
        ADD["Add the noise to the real mel<br/>at a random strength t"]
        XT[("Noisy mel-spectrogram<br/>[80 x 430]")]
        PRT[("Piano roll = the notes<br/>[256 x 430]")]
        UNETT["U-Net denoiser<br/>(the trainable network)"]
        FILMT["FiLM layers<br/>(scale + shift inside<br/>every network layer)"]
        VERT(["Style ID = 0 / 1 / 2"])
        TT(["Noise strength t"])
        PRED[("Predicted noise<br/>[80 x 430]")]
        LOSS["L1 loss: predicted noise vs<br/>the noise actually added"]
        MEL --> ADD
        EPS --> ADD
        ADD --> XT
        XT -- "input" --> UNETT
        PRT -- "which notes are played" --> UNETT
        VERT -- "which style it is" --> FILMT
        TT --> FILMT
        FILMT -. "conditions the network<br/>from the side" .-> UNETT
        UNETT -- "output" --> PRED
        PRED --> LOSS
        EPS -- "the correct answer" --> LOSS
        LOSS -- "update weights, repeat" --> UNETT
    end
    subgraph INFERPHASE["2 - INFERENCE (style transfer of a new song, iterative denoising)"]
        direction TB
        NOISE[("Start: pure random noise<br/>[80 x 430]")]
        PRI[("Piano roll of the<br/>input song [256 x 430]")]
        UNETI["U-Net denoiser<br/>(weights frozen)"]
        FILMI["FiLM layers<br/>(scale + shift inside<br/>every network layer)"]
        VERI(["Chosen target style ID"])
        TI(["Current step number"])
        STEP["Remove a small amount<br/>of the predicted noise<br/>(one denoising step)"]
        GMEL[("Generated mel-spectrogram<br/>[80 x 430] - same notes,<br/>new style of sound")]
        NOISE --> UNETI
        PRI -- "keep these notes" --> UNETI
        VERI -- "paint this style" --> FILMI
        TI -- "how noisy is it now" --> FILMI
        FILMI -. "conditions the network<br/>from the side" .-> UNETI
        UNETI --> STEP
        STEP -- "feed back - repeat 100 times,<br/>slightly cleaner each time" --> UNETI
        STEP -- "after the last step" --> GMEL
    end
    TRAINPHASE ~~~ INFERPHASE
    classDef model fill:#fff,stroke:#a855f7,stroke-width:3px,color:#222
    classDef cond fill:#fff,stroke:#f59e0b,stroke-width:2px,color:#222
    classDef custom fill:#fff,stroke:#22c55e,stroke-width:2px,color:#222
    classDef data fill:#fff,stroke:#64748b,color:#666
    class UNETT,UNETI model
    class FILMT,FILMI cond
    class ADD,LOSS,STEP custom
    class MEL,EPS,XT,PRT,VERT,TT,PRED,NOISE,PRI,VERI,TI,GMEL data"""

DIAGRAM_POSTPROCESS = """flowchart LR
    GMEL[("Generated<br/>mel-spectrogram")]
    PREP["Mel preparation<br/>(undo normalization)"]
    BIGVGAN["BigVGAN v2 (vocoder)<br/>mel-spectrogram to waveform"]
    WAV(("Generated audio (WAV)"))
    MET["Metric tests<br/>realism / note accuracy / speed"]
    RES[("Metric results")]
    VIS[("Visualizations")]
    GMEL --> PREP --> BIGVGAN --> WAV
    WAV --> MET --> RES
    MET --> VIS
    classDef custom fill:#fff,stroke:#22c55e,stroke-width:2px,color:#222
    classDef oss fill:#fff,stroke:#4f8cff,stroke-width:2px,color:#222
    classDef data fill:#fff,stroke:#64748b,color:#666
    class PREP,MET,VIS custom
    class BIGVGAN oss
    class GMEL,WAV,RES data"""

DIAGRAM_CLEAN_NOISY = """flowchart LR
    A["Synthesized music dataset<br/>with aligned MIDI transcription<br/>(exact, per-instrument notes)"] -- "clean piano roll" --> M["Diffusion model"]
    B["Israeli audio<br/>automatic note transcription<br/>(Basic-Pitch, imperfect notes)"] -- "noisy piano roll" --> M
    M --> V1["Architecture check:<br/>validation loss stable,<br/>hearing test passed"]
    M --> V2["Israeli output:<br/>judged by listening"]
    classDef clean fill:#fff,stroke:#22c55e,stroke-width:2px,color:#222
    classDef noisy fill:#fff,stroke:#f59e0b,stroke-width:2px,color:#222
    classDef model fill:#fff,stroke:#a855f7,stroke-width:2px,color:#222
    classDef gate fill:#fff,stroke:#64748b,color:#666
    class A clean
    class B noisy
    class M model
    class V1,V2 gate"""

# Detailed model architecture (real config numbers: base 160, mults [1,2,3,4]).
DIAGRAM_UNET = """flowchart TB
    IN["Input stack<br/>noisy mel-spectrogram [80] + piano roll [256]<br/>336 channels, 430 time frames"]
    IC["Input convolution 3x1<br/>336 -> 160 channels"]
    E0["Encoder level 0<br/>2x residual block, 160 channels, 430 frames<br/>(no attention)"]
    D0["Downsample 160 -> 320 channels<br/>215 frames"]
    E1["Encoder level 1<br/>2x residual block + self-attention<br/>320 channels, 215 frames"]
    D1["Downsample 320 -> 480 channels<br/>108 frames"]
    E2["Encoder level 2<br/>2x residual block + self-attention<br/>480 channels, 108 frames"]
    D2["Downsample 480 -> 640 channels<br/>54 frames"]
    BN["Bottleneck<br/>residual block + attention + residual block<br/>640 channels, 54 frames"]
    U2["Upsample 640 -> 480 channels<br/>108 frames"]
    C2["Decoder level 2<br/>3x residual block + self-attention<br/>480+480 channels, 108 frames"]
    U1["Upsample 480 -> 320 channels<br/>215 frames"]
    C1["Decoder level 1<br/>3x residual block + self-attention<br/>320+320 channels, 215 frames"]
    U0["Upsample 320 -> 160 channels<br/>430 frames"]
    C0["Decoder level 0<br/>3x residual block, 160+160 channels, 430 frames<br/>(no attention)"]
    OC["Output convolution 3x1<br/>160 -> 80 channels"]
    OUT["Predicted noise<br/>[80 x 430]"]
    COND(["FiLM conditioning C<br/>= time embedding (128) + version embedding (128)"])
    IN --> IC --> E0 --> D0 --> E1 --> D1 --> E2 --> D2 --> BN
    BN --> U2 --> C2 --> U1 --> C1 --> U0 --> C0 --> OC --> OUT
    E0 -. "skip connection" .-> C0
    E1 -. "skip connection" .-> C1
    E2 -. "skip connection" .-> C2
    COND -. "scale + shift at every residual block" .-> E1
    COND -. " " .-> BN
    COND -. " " .-> C1
    classDef enc fill:#fff,stroke:#22c55e,stroke-width:2px,color:#222
    classDef dec fill:#fff,stroke:#4f8cff,stroke-width:2px,color:#222
    classDef mid fill:#fff,stroke:#a855f7,stroke-width:3px,color:#222
    classDef io fill:#fff,stroke:#64748b,color:#666
    classDef cond fill:#fff,stroke:#f59e0b,stroke-width:2px,color:#222
    class E0,E1,E2,D0,D1,D2 enc
    class C0,C1,C2,U0,U1,U2 dec
    class BN mid
    class IN,IC,OC,OUT io
    class COND cond"""

DIAGRAM_CONDITIONING = """flowchart LR
    T(["Diffusion timestep t"]) --> TE["Sinusoidal embedding<br/>-> small MLP, 128 values"]
    V(["Version ID 0/1/2<br/>(+ null token for guidance)"]) --> VE["Learned embedding table<br/>128 values"]
    TE --> CAT["Conditioning C<br/>concatenate -> 256 values"]
    VE --> CAT
    CAT --> FILM["FiLM at every residual block<br/>gamma = 1 + Linear(C)<br/>beta = Linear(C)<br/>h' = gamma * h + beta"]
    PR[("Piano-roll score<br/>2 x 128 pitches -> 256 channels")] --> UNET["U-Net<br/>(score stacked with noisy<br/>mel-spectrogram)"]
    FILM --> UNET
    UNET --> EPS["Predicted noise"]
    CFG["Classifier-free guidance<br/>3 forward passes: full / drop-score / drop-version<br/>w_s = w_v = 1.25"] -.-> EPS
    classDef cond fill:#fff,stroke:#f59e0b,stroke-width:2px,color:#222
    classDef model fill:#fff,stroke:#a855f7,stroke-width:3px,color:#222
    classDef io fill:#fff,stroke:#64748b,color:#666
    class TE,VE,CAT,FILM,CFG cond
    class UNET,EPS model
    class T,V,PR io"""

# Populated by ``export_all_diagrams`` at build time: diagram source ->
# {"svg": rel_path, "png": rel_path}. Used by ``_mermaid`` to add download links.
_DIAGRAM_EXPORTS: Dict[str, Dict[str, str]] = {}
MERMAID_INK = "https://mermaid.ink"


# ===========================================================================
# Manifest
# ===========================================================================
class Manifest:
    """Accumulates one row per copied/generated asset for MANIFEST.csv."""

    def __init__(self) -> None:
        """Start with an empty list of asset rows."""
        self.rows: List[Dict[str, str]] = []

    def add(self, *, stage: str, rel_path: str, kind: str,
            title: str, caption: str = "", spec: str = "") -> None:
        """Record one asset (stage, relative path, type, caption, which spec it supports)."""
        self.rows.append({
            "stage": stage,
            "asset": rel_path,
            "type": kind,
            "title": title,
            "caption": caption,
            "spec": spec,
        })

    def write(self, path: Path) -> None:
        """Write all recorded rows out to MANIFEST.csv."""
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["stage", "asset", "type", "title", "caption", "spec"])
            w.writeheader()
            w.writerows(self.rows)


# ===========================================================================
# Small utilities
# ===========================================================================
def _copy(src: Path, dest: Path) -> Optional[Path]:
    """Copy ``src`` -> ``dest`` (creating parents). Returns dest or None."""
    if not src or not src.exists() or not src.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def _copy_audio(src: Path, dest: Path, lite: bool = False,
                seconds: float = 45.0) -> Optional[Path]:
    """Copy an audio file, trimming to ``seconds`` when ``lite`` is set.

    Full mode is a byte copy. Lite mode loads the first ``seconds`` and
    re-writes a small preview WAV; on any failure it falls back to a full copy.
    """
    if not lite or not src or not src.exists() or not src.is_file():
        return _copy(src, dest)
    try:
        import numpy as np
        import soundfile as sf
        info = sf.info(str(src))
        frames = int(seconds * info.samplerate)
        data, sr = sf.read(str(src), frames=frames, dtype="float32")
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), data, sr)
        return dest
    except Exception as e:
        print(f"    [lite] trim failed on {src.name} ({e}); copying full")
        return _copy(src, dest)


def _rel(path: Path, root: Path) -> str:
    """POSIX relative path of ``path`` under ``root`` (for portable href/src)."""
    return path.relative_to(root).as_posix()


def _esc(text: str) -> str:
    """HTML-escape a value so it is safe to drop into the page."""
    return html.escape(str(text), quote=True)


def _parse_render_base(base: str) -> Optional[Dict[str, str]]:
    """Parse ``Song__step_NNNN__style_Style_sampler__role_transferred``.

    Returns dict(song, step, style, sampler) or None if it does not match.
    """
    parts = base.split("__")
    if len(parts) < 3 or not parts[1].startswith("step_") \
            or not parts[2].startswith("style_"):
        return None
    stylefull = parts[2][len("style_"):]
    style, _, sampler = stylefull.rpartition("_")
    return {
        "song": parts[0],
        "step": parts[1][len("step_"):],
        "style": style,
        "sampler": sampler,
    }


# ===========================================================================
# Generators (only for the genuinely-missing visuals)
# ===========================================================================
def _synthesize_midi(mid_path: Path, out_wav: Path, seconds: float = 30.0) -> bool:
    """Render a MIDI file to a playable WAV (harmonic sine synth, no soundfont)."""
    try:
        import numpy as np
        import pretty_midi
        import soundfile as sf
    except Exception as e:  # pragma: no cover
        print(f"    [midi] skipped ({e})")
        return False
    try:
        pm = pretty_midi.PrettyMIDI(str(mid_path))
        audio = pm.synthesize(fs=MEL_SR)  # sum-of-harmonics sine
        if seconds and len(audio) > int(seconds * MEL_SR):
            audio = audio[: int(seconds * MEL_SR)]
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0:
            audio = 0.9 * audio / peak
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), audio.astype("float32"), MEL_SR)
        return True
    except Exception as e:
        print(f"    [midi] failed on {mid_path.name}: {e}")
        return False


def _dsp_demo_png(wav_path: Path, midi_path: Optional[Path], out_png: Path,
                  title: str, seconds: float = 30.0) -> bool:
    """Render the multi-panel DSP walkthrough for one song (reuses the
    project's ``plot_preprocessing_demo``)."""
    if not wav_path or not wav_path.exists():
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from preprocessing.dataset_visualizations import plot_preprocessing_demo
    except Exception as e:  # pragma: no cover
        print(f"    [dsp] skipped ({e})")
        return False
    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plot_preprocessing_demo(
            wav_path=wav_path,
            midi_path=midi_path if (midi_path and midi_path.exists()) else None,
            save_path=out_png, max_seconds=seconds, title=title)
        plt.close("all")
        return out_png.exists()
    except Exception as e:
        print(f"    [dsp] failed on {wav_path.name}: {e}")
        return False


def _augmentation_png(wav_path: Path, midi_path: Optional[Path], out_png: Path,
                      title: str) -> bool:
    """Render the training-time data augmentation (``JointAugment``) applied to
    one real 5 s segment: pitch-shift and time-stretch on mel + piano-roll
    jointly, plus SpecAugment on the mel."""
    if not wav_path or not wav_path.exists():
        return False
    try:
        import random as _random
        import numpy as np
        import torch
        import torch.nn.functional as F
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from preprocessing.dsp_preprocessor import (
            DSPConfig, load_and_resample_audio, extract_mel_spectrogram,
            normalize_mel, load_midi_to_piano_roll)
        from preprocessing.augmentation import JointAugment, _fit_to_length
    except Exception as e:  # pragma: no cover
        print(f"    [aug] skipped ({e})")
        return False
    try:
        cfg = DSPConfig()
        y = load_and_resample_audio(Path(wav_path), cfg.sample_rate)
        mel_db = extract_mel_spectrogram(y, cfg)
        mel_norm, _mmin, _mmax = normalize_mel(mel_db)
        seg = cfg.segment_frames
        T = mel_norm.shape[1]
        start = min(seg * 6, max(0, T - seg))  # a musically active segment
        mel = torch.from_numpy(mel_norm[:, start:start + seg]).float()

        has_midi = midi_path is not None and Path(midi_path).exists()
        if has_midi:
            dur = len(y) / cfg.sample_rate
            pr_full = np.asarray(load_midi_to_piano_roll(Path(midi_path), cfg, dur))
            pr = torch.from_numpy(pr_full[:, :, start:start + seg]).float()
        else:
            pr = torch.zeros(2, 128, seg)

        # --- deterministic variants ---
        mel_p2 = JointAugment._pitch_shift_mel_hz(mel, +2)
        pr_p2 = torch.roll(pr, shifts=2, dims=1); pr_p2[:, :2, :] = 0.0
        mel_m2 = JointAugment._pitch_shift_mel_hz(mel, -2)

        def _stretch(m: torch.Tensor, rate: float) -> torch.Tensor:
            # Time-stretch a mel by resampling along time, then refit to seg frames.
            t_new = round(seg * rate)
            mi = F.interpolate(m.unsqueeze(0), size=t_new, mode="linear",
                               align_corners=False).squeeze(0)
            return _fit_to_length(mi, seg)
        mel_ts09 = _stretch(mel, 0.9)
        mel_ts11 = _stretch(mel, 1.1)

        _random.seed(7)
        sa = JointAugment({"enabled": True, "pitch_shift": {"p": 0.0},
                           "time_stretch": {"p": 0.0},
                           "spec_augment": {"p": 1.0, "time_mask_max": 30,
                                            "freq_mask_max": 8, "n_time": 2,
                                            "n_freq": 2}})
        mel_sa = sa._maybe_spec_augment(mel)

        def _mel_ax(ax, m, ttl):
            # Draw one mel panel (magma, fixed [-1,1] scale) with a title.
            ax.imshow(m.numpy(), aspect="auto", origin="lower", cmap="magma",
                      vmin=-1, vmax=1)
            ax.set_title(ttl, fontsize=10, loc="left")
            ax.set_xlabel("frames"); ax.set_ylabel("mel bin")

        def _pr_ax(ax, p, ttl):
            # Draw one piano-roll panel (collapse the 2 onset/sustain channels).
            comp = p.sum(axis=0).numpy()
            ax.imshow(comp, aspect="auto", origin="lower", cmap="Greens",
                      vmin=0, vmax=2)
            ax.set_title(ttl, fontsize=10, loc="left")
            ax.set_xlabel("frames"); ax.set_ylabel("MIDI pitch")

        fig = plt.figure(figsize=(16, 12), dpi=170)
        gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.22)
        _mel_ax(fig.add_subplot(gs[0, 0]), mel, "mel - original segment")
        _mel_ax(fig.add_subplot(gs[0, 1]), mel_p2, "mel - pitch shift +2 semitones")
        _mel_ax(fig.add_subplot(gs[0, 2]), mel_m2, "mel - pitch shift -2 semitones")
        _mel_ax(fig.add_subplot(gs[1, 0]), mel_ts09, "mel - time stretch x0.9 (faster)")
        _mel_ax(fig.add_subplot(gs[1, 1]), mel_ts11, "mel - time stretch x1.1 (slower)")
        _mel_ax(fig.add_subplot(gs[1, 2]), mel_sa, "mel - SpecAugment (time+freq masks)")
        _pr_ax(fig.add_subplot(gs[2, 0]), pr, "score - original piano roll")
        _pr_ax(fig.add_subplot(gs[2, 1]), pr_p2, "score - pitch shift +2 (moves with mel)")
        ax_note = fig.add_subplot(gs[2, 2]); ax_note.axis("off")
        ax_note.text(0.0, 0.9,
                     "JointAugment (training-time, on the fly):\n\n"
                     "- pitch shift +/-2 semitones (p=0.5)\n"
                     "  mel: Hz-aware bin interpolation\n"
                     "  score: MIDI-native 1-bin roll\n"
                     "  -> mel + score stay aligned\n\n"
                     "- time stretch +/-10% (p=0.4)\n"
                     "  both tensors, refit to 430 frames\n\n"
                     "- SpecAugment (p=0.5), mel only\n"
                     "  2 freq masks <=8 bins,\n"
                     "  2 time masks <=30 frames",
                     fontsize=9, va="top", family="monospace")
        fig.suptitle(title, fontsize=13, fontweight="bold")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return out_png.exists()
    except Exception as e:
        print(f"    [aug] failed on {wav_path.name}: {e}")
        return False


def _augmentation_audio(wav_path: Path, out_dir: Path,
                        clip_seconds: float = 25.0) -> List[Tuple[str, Path]]:
    """Render audible augmentation samples from a real clip using the project's
    WAV-level augmentations (offline source-pool policy: pitch +/-2, stretch
    0.9/1.1). Returns list of (label, wav_path)."""
    if not wav_path or not wav_path.exists():
        return []
    try:
        import numpy as np
        import librosa
        import soundfile as sf
        from preprocessing.augmentation import pitch_shift_wav, time_stretch_wav
    except Exception as e:  # pragma: no cover
        print(f"    [aug-audio] skipped ({e})")
        return []

    def _norm(a):
        # Peak-normalise a waveform to 0.9 so the clips play at a similar volume.
        peak = float(np.max(np.abs(a))) if len(a) else 0.0
        return (0.9 * a / peak).astype("float32") if peak > 0 else a.astype("float32")

    try:
        sr = 22050
        total = librosa.get_duration(path=str(wav_path))
        offset = 30.0 if total > 30.0 + clip_seconds else 0.0
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True,
                            offset=offset, duration=clip_seconds)
        out_dir.mkdir(parents=True, exist_ok=True)
        specs = [
            ("original", y),
            ("pitch shift +2 semitones", pitch_shift_wav(y, sr, +2)),
            ("pitch shift -2 semitones", pitch_shift_wav(y, sr, -2)),
            ("time stretch x0.9 (slower)", time_stretch_wav(y, 0.9)),
            ("time stretch x1.1 (faster)", time_stretch_wav(y, 1.1)),
        ]
        results: List[Tuple[str, Path]] = []
        for label, audio in specs:
            fname = "aug_" + label.split(" (")[0].replace(" ", "_") \
                .replace("+", "p").replace("-", "m") + ".wav"
            dest = out_dir / fname
            sf.write(str(dest), _norm(np.asarray(audio)), sr)
            results.append((label, dest))
        return results
    except Exception as e:
        print(f"    [aug-audio] failed on {wav_path.name}: {e}")
        return []


def _clip_wav(src: Path, dest: Path, clip_seconds: float,
              offset_if_long: float = 30.0) -> bool:
    """Write a short clip of ``src`` to ``dest`` (offset into the song if it is
    long enough, else from the start)."""
    try:
        import librosa
        import soundfile as sf
        total = librosa.get_duration(path=str(src))
        offset = offset_if_long if total > offset_if_long + clip_seconds else 0.0
        y, sr = librosa.load(str(src), sr=None, mono=True,
                             offset=offset, duration=clip_seconds)
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), y, sr)
        return dest.exists()
    except Exception as e:
        print(f"    [aug-audio] clip failed on {src.name}: {e}")
        return False


def _augmentation_real_audio(sample_dir: Path, out_dir: Path,
                             clip_seconds: float = 25.0
                             ) -> Tuple[List[Tuple[str, Path]], Optional[Path],
                                        Optional[Path]]:
    """Copy the REAL offline-augmented WAVs actually produced for training from a
    source-pool song directory (``<stem>.wav`` + ``augmented/<stem>_{ps,ts}*.wav``).

    Returns (samples, original_wav, original_midi) where samples is an ordered
    list of (label, clipped_wav_path)."""
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        return [], None, None
    orig_wavs = sorted(sample_dir.glob("*.wav"))
    if not orig_wavs:
        return [], None, None
    orig_wav = orig_wavs[0]
    stem = orig_wav.stem
    orig_midi = next(iter(sorted(sample_dir.glob("*.mid"))), None)
    aug_dir = sample_dir / "augmented"
    plan = [
        ("original (real source-pool song)", orig_wav),
        ("pitch shift +2 semitones", aug_dir / f"{stem}_ps+2.wav"),
        ("pitch shift -2 semitones", aug_dir / f"{stem}_ps-2.wav"),
        ("time stretch x0.9 (slower)", aug_dir / f"{stem}_ts0.9.wav"),
        ("time stretch x1.1 (faster)", aug_dir / f"{stem}_ts1.1.wav"),
    ]
    results: List[Tuple[str, Path]] = []
    for label, src in plan:
        if not src.exists():
            continue
        dest = out_dir / f"aug_{src.stem}.wav"
        if _clip_wav(src, dest, clip_seconds):
            results.append((label, dest))
    return results, orig_wav, orig_midi


def _mel_png(wav_path: Path, out_png: Path, title: str, seconds: float = 30.0) -> bool:
    """Compute + save a mel-spectrogram image of an audio file."""
    try:
        import librosa
        import librosa.display
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        print(f"    [mel] skipped ({e})")
        return False
    try:
        y, sr = librosa.load(str(wav_path), sr=MEL_SR, mono=True,
                             duration=seconds if seconds else None)
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=MEL_N_FFT, hop_length=MEL_HOP, n_mels=MEL_N_MELS)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=MEL_HOP, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"    [mel] failed on {wav_path.name}: {e}")
        return False


def _step_gallery_png(mel_viz_dir: Path, out_png: Path) -> bool:
    """Grid of the generated mel every 10k steps, cropped from the mel_viz frames.

    Each mel_viz frame is a 3-panel snapshot (score / generated / target); we keep
    only the middle "generated" panel and drop its subtitle and x-axis label, then
    lay the crops out on a 5-column grid so you can watch the sample improve as the
    model trains. Same view used in the project book.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except Exception as e:
        print(f"    [gallery] skipped ({e})")
        return False
    # 2k, then every 10k up to 240k, plus the final 248k checkpoint.
    steps = [2000] + list(range(10000, 250000, 10000))
    if 248000 not in steps:
        steps.append(248000)
    imgs = []
    for s in steps:
        p = mel_viz_dir / f"step_{s:06d}.png"
        if not p.exists():
            continue
        arr = mpimg.imread(str(p))
        h, w = arr.shape[:2]
        x0, x1 = int(0.375 * w), int(0.650 * w)
        y0, y1 = int(0.235 * h), int(0.90 * h)
        imgs.append((s, arr[y0:y1, x0:x1]))
    if not imgs:
        print(f"    [gallery] no mel_viz frames under {mel_viz_dir}")
        return False
    cols = 5
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, rows * 1.55), dpi=150)
    axes = axes.ravel()
    for ax, (s, im) in zip(axes, imgs):
        ax.imshow(im)
        ax.set_title(f"{s // 1000}k", fontsize=8)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle("Generated mel-spectrogram as training progresses (every 10k steps)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return True


def _final_loss_png(loss_csv: Path, out_png: Path, label: str) -> bool:
    """Two-panel figure for ONE training run: training loss | validation loss.

    The advisor asked to show the final run only, with the two losses explicit
    and separated, so a non-ML reader can see what each curve means. Train is
    logged much more often than val, so the train panel shows the raw noisy
    curve (faint) plus a moving average (bold).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        print(f"    [loss] skipped ({e})")
        return False

    def _num(v: Any) -> Optional[float]:
        # Parse a value to float, returning None for blanks/bad values/NaN.
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f else None  # drop NaN

    tr_s, tr_y, va_s, va_y = [], [], [], []
    try:
        with loss_csv.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                step = _num(row.get("step"))
                if step is None:
                    continue
                t = _num(row.get("train_loss"))
                v = _num(row.get("val_loss"))
                if t is not None:
                    tr_s.append(step); tr_y.append(t)
                if v is not None:
                    va_s.append(step); va_y.append(v)
    except Exception as e:
        print(f"    [loss] could not read {loss_csv}: {e}")
        return False
    if not tr_s or not va_s:
        return False

    try:
        tr_s, tr_y = np.array(tr_s), np.array(tr_y)
        va_s, va_y = np.array(va_s), np.array(va_y)
        win = max(1, len(tr_y) // 200)          # ~0.5% moving-average window
        if win > 1:
            kern = np.ones(win) / win
            tr_smooth = np.convolve(tr_y, kern, mode="valid")  # no edge dip
            off = (win - 1) // 2
            tr_s_smooth = tr_s[off: off + len(tr_smooth)]
        else:
            tr_smooth, tr_s_smooth = tr_y, tr_s

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        ax1.plot(tr_s, tr_y, color="#4f8cff", linewidth=0.5, alpha=0.25)
        ax1.plot(tr_s_smooth, tr_smooth, color="#1565C0", linewidth=1.8,
                 label="training loss (moving average)")
        ax1.set_title("Training loss - on songs the model learns from")
        ax1.set_xlabel("training step")
        ax1.set_ylabel("L1 noise-prediction error")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.25)

        ax2.plot(va_s, va_y, color="#C62828", linewidth=1.8, marker="o",
                 markersize=2.5, label="validation loss")
        ax2.set_title("Validation loss - on held-out songs it never trains on")
        ax2.set_xlabel("training step")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.25)

        fig.suptitle(f"{label}: training vs. validation loss", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"    [loss] plot failed: {e}")
        return False


# ===========================================================================
# HTML rendering
# ===========================================================================
def _page_head(title: str) -> str:
    """Return the <head> + opening tags and inline CSS for the showcase page."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{ --fg:#1f2328; --muted:#57606a; --line:#d0d7de; --bg:#f6f8fa; --accent:#0969da; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
         color:var(--fg); margin:0; line-height:1.55; }}
  header.hero {{ background:linear-gradient(135deg,#0b1e3f,#3b0b5f); color:#fff;
         padding:40px 24px; text-align:center; }}
  header.hero h1 {{ margin:0 0 6px; font-size:2rem; }}
  header.hero p {{ margin:2px 0; opacity:.9; }}
  nav.toc {{ position:sticky; top:0; background:#fff; border-bottom:1px solid var(--line);
         padding:10px 16px; z-index:5; display:flex; flex-wrap:wrap; gap:14px; }}
  nav.toc a {{ color:var(--accent); text-decoration:none; font-size:.92rem; }}
  main {{ max-width:1180px; margin:0 auto; padding:24px; }}
  section {{ margin:34px 0; padding-top:8px; border-top:1px solid var(--line); }}
  section:first-of-type {{ border-top:none; }}
  h2 {{ font-size:1.5rem; }}
  h3 {{ margin:22px 0 8px; }}
  p.lead {{ font-size:1.02rem; color:#333; }}
  .diagram {{ background:#fff; border:1px solid var(--line); border-radius:10px;
         padding:14px; margin:16px 0; overflow-x:auto; text-align:center; }}
  .card {{ border:1px solid var(--line); border-radius:10px; padding:16px; margin:16px 0;
         background:#fff; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
  figure {{ margin:0; }}
  figure img {{ width:100%; border:1px solid #e1e4e8; border-radius:6px; }}
  figcaption {{ font-size:.85rem; color:var(--muted); margin-top:6px; }}
  audio {{ width:100%; margin:6px 0; }}
  video {{ width:100%; border:1px solid #e1e4e8; border-radius:6px; margin:6px 0; }}
  figure.wide {{ margin:14px 0 4px; }}
  .kv {{ font-family:ui-monospace,Menlo,Consolas,monospace; background:var(--bg);
         padding:2px 6px; border-radius:4px; font-size:.8rem; color:var(--muted); }}
  table {{ border-collapse:collapse; width:100%; margin:14px 0; font-size:.94rem; }}
  th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
  th {{ background:var(--bg); }}
  .pass {{ color:#1a7f37; font-weight:600; }}
  .fail {{ color:#cf222e; font-weight:600; }}
  details > summary {{ cursor:pointer; font-weight:600; margin:8px 0; }}
  .pill {{ display:inline-block; background:var(--bg); border:1px solid var(--line);
         border-radius:999px; padding:2px 10px; font-size:.8rem; color:var(--muted); }}
  .note {{ background:#fff8c5; border:1px solid #d4a72c; border-radius:8px; padding:10px 12px; }}
  .dl {{ font-size:.85rem; color:var(--muted); margin:8px 0 0; text-align:center; }}
  .dl a {{ color:var(--accent); text-decoration:none; }}
  footer {{ color:var(--muted); font-size:.85rem; text-align:center; padding:30px; }}
</style></head><body>
"""


def _mermaid(diagram: str) -> str:
    """Return the live Mermaid diagram block, plus SVG/PNG download links if exported."""
    live = f'<div class="diagram"><pre class="mermaid">\n{diagram}\n</pre></div>'
    exp = _DIAGRAM_EXPORTS.get(diagram)
    if exp:
        links = []
        if exp.get("svg"):
            links.append(f'<a href="{_esc(exp["svg"])}" download>SVG (vector)</a>')
        if exp.get("png"):
            links.append(f'<a href="{_esc(exp["png"])}" download>PNG</a>')
        if links:
            live += ('<p class="dl">Download this diagram for the poster / book / '
                     'slides: ' + ' &middot; '.join(links) + '</p>')
    return live


def _audio(rel: str, label: str) -> str:
    """Return an HTML audio player (with a small label) for one clip."""
    return (f'<div><span class="kv">{_esc(label)}</span>'
            f'<audio controls preload="none" src="{_esc(rel)}"></audio></div>')


def _figure(rel: str, caption: str) -> str:
    """Return an HTML figure (lazy-loaded image + caption)."""
    return (f'<figure><img loading="lazy" src="{_esc(rel)}">'
            f'<figcaption>{_esc(caption)}</figcaption></figure>')


# ===========================================================================
# Stage builders
# ===========================================================================
def build_overview(cfg: dict, specs: List[dict]) -> str:
    """Build section 0: the intro, the macro pipeline diagram and the specs board."""
    ov = cfg.get("overview", {})
    intro = ov.get("intro", "")
    results = ov.get("results", "")
    rows = []
    n_pass = 0
    n_scored = 0
    for s in specs:
        ok = s.get("pass", None)
        if ok is not None:
            n_scored += 1
            if ok:
                n_pass += 1
        badge = ('<span class="pass">MET</span>' if ok
                 else '<span class="fail">NOT MET</span>' if ok is False
                 else "-")
        rows.append(
            f"<tr><td>{_esc(s.get('name',''))}</td>"
            f"<td>{_esc(s.get('target',''))}</td>"
            f"<td>{_esc(s.get('achieved',''))}</td>"
            f"<td>{badge}</td>"
            f"<td>{_esc(s.get('note',''))}</td></tr>")
    summary = (f'<p class="pill">Specifications met: '
               f"<strong>{n_pass} of {n_scored}</strong></p>") if n_scored else ""
    board = (
        "<h3>Specifications scoreboard</h3>"
        + summary
        + '<table><thead><tr><th>Specification</th><th>Target</th>'
        "<th>Achieved</th><th>Status</th><th>Explanation</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>")
    results_block = (
        f'<div class="card"><h3>Did we meet the specifications?</h3>'
        f'<p class="lead">{_esc(results)}</p></div>') if results else ""
    return (
        '<section id="overview"><h2>0. Overview</h2>'
        f'<p class="lead">{_esc(intro)}</p>'
        "<h3>Full pipeline block diagram</h3>"
        + _mermaid(DIAGRAM_MACRO)
        + '<p class="pill">Legend: green = built by us &middot; blue = open-source '
          "as-is &middot; orange = adjusted open-source &middot; purple = active "
          "model block &middot; gray = data/artifacts</p>"
        + board
        + results_block
        + "</section>")


def build_preprocessing(cfg: dict, drive: dict, out_root: Path,
                        assets_root: Path, man: Manifest,
                        lite: bool = False, seconds: float = 45.0) -> str:
    """Build section 1: for each demo song, the audio, score, mel and DSP figure."""
    version_dir = Path(drive["version_dir"])
    demo_dir = version_dir / "demo_external"
    gallery = demo_dir / "_gallery" / "_assets"
    stage_dir = assets_root / "01_preprocessing"
    songs = cfg.get("preprocessing", {}).get("featured_songs", [])

    cards = []
    for song in songs:
        name = song["name"]
        label = song.get("label", name)
        song_src = demo_dir / name
        dest = stage_dir / name
        media: List[str] = []

        # original audio
        orig = _copy_audio(song_src / "original.wav", dest / "original.wav",
                           lite=lite, seconds=seconds)
        if orig:
            rel = _rel(orig, out_root)
            media.append(_audio(rel, "original audio"))
            man.add(stage="01_preprocessing", rel_path=rel, kind="audio",
                    title=f"{label} - original", caption="input song",
                    spec="preprocessing input")

        # MIDI rendered as audio (score you can hear)
        mids = list(song_src.glob("*.mid"))
        if mids:
            midi_wav = dest / "score_as_audio.wav"
            if _synthesize_midi(mids[0], midi_wav,
                                seconds=(seconds if lite else 30.0)):
                rel = _rel(midi_wav, out_root)
                media.append(_audio(rel, "score as audio (MIDI synth)"))
                man.add(stage="01_preprocessing", rel_path=rel, kind="audio",
                        title=f"{label} - score as audio",
                        caption="Basic-Pitch MIDI rendered to sound",
                        spec="score conditioning")

        # original mel (generated)
        if orig:
            mel_png = dest / "original_mel.png"
            if _mel_png(orig, mel_png, f"{label} - original mel"):
                rel = _rel(mel_png, out_root)
                media.append(_figure(rel, "mel spectrogram of the original"))
                man.add(stage="01_preprocessing", rel_path=rel, kind="image",
                        title=f"{label} - mel", caption="model target space",
                        spec="mel representation")

        # piano-roll image (reuse from gallery)
        pr_src = gallery / f"{name}__piano_roll.png"
        pr = _copy(pr_src, dest / "piano_roll.png")
        if pr:
            rel = _rel(pr, out_root)
            media.append(_figure(rel, "piano-roll score (conditioning)"))
            man.add(stage="01_preprocessing", rel_path=rel, kind="image",
                    title=f"{label} - piano roll", caption="score conditioning",
                    spec="score conditioning")

        # DSP walkthrough (downsample -> LPF -> mel -> normalize -> segment)
        # Shown FULL-WIDTH under the card grid (advisor: enlarge, serial).
        dsp_html = ""
        dsp_png = dest / "dsp_pipeline.png"
        if _dsp_demo_png(song_src / "original.wav",
                         mids[0] if mids else None, dsp_png,
                         f"{label} - DSP preprocessing"):
            rel = _rel(dsp_png, out_root)
            dsp_html = (
                '<figure class="wide">'
                f'<img loading="lazy" src="{_esc(rel)}">'
                '<figcaption>DSP walkthrough (full width): raw waveform to '
                'normalized mel + 5 s segments (downsample, mel-filterbank '
                'LPF at 8 kHz, log-mel dB, [-1,1] normalization, segment '
                'boundaries)</figcaption></figure>')
            man.add(stage="01_preprocessing", rel_path=rel, kind="image",
                    title=f"{label} - DSP pipeline",
                    caption="full signal-processing chain applied to the song",
                    spec="DSP preprocessing")

        cards.append(
            f'<div class="card"><h3>{_esc(label)}</h3>'
            f'<div class="grid">{"".join(media)}</div>'
            f'{dsp_html}</div>')

    return (
        '<section id="preprocessing"><h2>1. Preprocessing</h2>'
        '<p class="lead">Before any learning can happen we have to build the '
        "dataset, because no Israeli-style dataset exists. Each song is "
        "downloaded, transcribed into notes (MIDI) by a tool called Basic-Pitch, "
        "and turned into two things: a piano-roll of the notes to keep and a "
        "mel-spectrogram of the sound, both sliced into 5-second pieces. The DSP "
        "panel for each song shows this signal chain step by step: resample the "
        "audio to 22.05 kHz, apply the mel filter-bank (which also acts as an "
        "8 kHz low-pass), take the log so loud and soft parts sit on a similar "
        "scale, and normalize to a fixed range. For the demo songs you can hear "
        "the original, hear the transcribed score played back as sound (what the "
        "model is told to keep), and see the piano-roll, the mel and the full "
        "DSP pipeline.</p>"
        + _mermaid(DIAGRAM_PREPROCESS)
        + _piano_roll_explainer(out_root, assets_root, man)
        + "".join(cards)
        + "</section>")


def _piano_roll_explainer(out_root: Path, assets_root: Path,
                          man: Manifest) -> str:
    """'What is a piano roll?' card: definition, annotated still and a demo
    video (piano roll of Ode to Joy with a moving cursor, synced audio).

    Piano roll is a music term, not an EE one, so the page explains it before
    using it. Assets are prebuilt by deliverables/build_piano_roll_video.py.
    """
    src_dir = _HERE / "deliverables" / "piano_roll_explainer"
    dest = assets_root / "01_preprocessing" / "_piano_roll_explainer"
    parts: List[str] = []

    still = _copy(src_dir / "piano_roll_annotated.png",
                  dest / "piano_roll_annotated.png")
    if still:
        rel = _rel(still, out_root)
        parts.append(_figure(
            rel, "the anatomy of a piano roll: time runs left to right, each "
            "row is one piano key, each bar is one note"))
        man.add(stage="01_preprocessing", rel_path=rel, kind="image",
                title="piano roll - annotated explainer",
                caption="what a piano roll is", spec="score conditioning")

    video = _copy(src_dir / "piano_roll_demo.mp4", dest / "piano_roll_demo.mp4")
    if video:
        rel = _rel(video, out_root)
        parts.append(
            '<div><span class="kv">watch + listen: the cursor traces the '
            'notes exactly as they sound</span>'
            f'<video controls preload="none" src="{_esc(rel)}"></video></div>')
        man.add(stage="01_preprocessing", rel_path=rel, kind="video",
                title="piano roll - demo video",
                caption="Ode to Joy piano roll with moving cursor + audio",
                spec="score conditioning")

    if not parts:
        return ""
    return (
        '<div class="card"><h3>What is a piano roll?</h3>'
        '<p>A piano roll is how musicians write notes on a computer - the '
        'digital version of sheet music, named after the punched paper rolls '
        'of old self-playing pianos. It is a simple chart: <b>time runs left '
        'to right, each row is one piano key</b>, and every bar means "this '
        'note starts here, lasts this long". Higher on the page = higher '
        'note. The demo below shows the piano roll of a melody everyone '
        'knows (Beethoven\'s Ode to Joy) while it plays - the red cursor '
        'and the lit-up bars show exactly what you are hearing. This chart '
        'is what our model receives as its "keep these notes" instruction '
        'for every song.</p>'
        f'<div class="grid">{"".join(parts)}</div>'
        '<p class="dl">More examples of piano rolls in the wild: '
        '<a href="https://www.youtube.com/results?search_query=piano+tutorial+falling+notes" '
        'target="_blank" rel="noopener">YouTube "falling notes" piano videos</a> '
        '(the same chart, rotated vertically).</p></div>')


def _architecture_block() -> str:
    """Detailed model architecture, conditioning, and engineering decisions."""
    spec_rows = [
        ("Backbone", "1-D U-Net, 4 levels, ~32 M params",
         "Mel frames as 1-D sequences; strided conv is efficient over time"),
        ("Channels", "160 -> 320 -> 480 -> 640 (mults 1,2,3,4)",
         "Widen as time is downsampled 430 -> 215 -> 108 -> 54"),
        ("Attention", "Self-attention at levels 1, 2, 3 + bottleneck",
         "Long-range temporal structure where feature maps are small"),
        ("Noise schedule", "Cosine (Nichol & Dhariwal 2021), T=1000",
         "Avoids the harsh endpoints of a linear schedule"),
        ("Sampler", "DDIM, N=100 steps, eta=0 (deterministic)",
         "~10x faster than full 1000-step ancestral sampling"),
        ("Loss", "L1 on predicted noise",
         "Smoother gradient landscape than L2 for mel targets"),
        ("Conditioning", "FiLM at every ResBlock (gamma, beta from C)",
         "Injects score + version at all temporal scales"),
        ("Guidance", "Classifier-free, w_s = w_v = 1.25 (3 passes)",
         "Independent control of score fidelity and style strength"),
        ("Optim", "AdamW, lr 1e-4 -> 1e-5, EMA 0.999, bf16, 250k steps",
         "Stable long run; late checkpoints settle into the LR minimum"),
        ("Output", "80-bin mel -> BigVGAN v2 vocoder @ 22 kHz",
         "Same vocoder pipeline as the earlier models"),
    ]
    rows = "".join(
        f"<tr><td>{_esc(a)}</td><td>{_esc(b)}</td><td>{_esc(c)}</td></tr>"
        for a, b, c in spec_rows)
    why = "".join(f"<li>{_esc(t)}</li>" for t in [
        "Handles POLYPHONIC music - several instruments sounding at once - which "
        "the simpler earlier approaches we tried could not.",
        "Score conditioning splits WHAT from HOW: the 2-channel piano roll "
        "(onset + sustain) says what notes to play, the version embedding says "
        "in which style (which artist / corpus) to play them.",
        "Mel-spectrogram in / mel-spectrogram out - it drops straight into the "
        "existing DSP + vocoder pipeline; only the middle block changed.",
    ])
    return (
        '<h3>Full model architecture</h3>'
        '<p class="lead">The production model is a score-conditioned denoising '
        'diffusion U-Net (DDPM). It predicts the noise added to a 5 s mel '
        'segment, conditioned on a piano-roll score and a style/version ID. The '
        'detailed U-Net layer stack and the FiLM + classifier-free-guidance '
        'conditioning are covered in the project book.</p>'
        + '<div class="card"><h3>Architecture &amp; training decisions</h3>'
          '<table><thead><tr><th>Component</th><th>Choice</th>'
          '<th>Why</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
        + '<div class="card"><h3>Why diffusion over DDSP / VAE-GAN?</h3>'
          '<ul>' + why + '</ul>'
          '<p class="pill">Input: noisy mel [80 x 430] + piano-roll [256 x 430] '
          '&middot; Conditioning C = time_emb(128) + version_emb(128) = 256-d '
          '&middot; Output: predicted noise [80 x 430]</p></div>')


def build_training(cfg: dict, drive: dict, out_root: Path,
                   assets_root: Path, man: Manifest,
                   lite: bool = False, seconds: float = 45.0) -> str:
    """Build section 2: campaigns, final-run train/val loss, gallery, denoising, augmentation."""
    drive_root = Path(drive["drive_root"])
    ckpt_root = drive_root / "checkpoints"
    stage_dir = assets_root / "02_training"
    tcfg = cfg.get("training", {})
    campaigns = tcfg.get("campaigns", [])
    denoising_dir = tcfg.get("denoising_dir")
    denoising_src = (_HERE / denoising_dir) if denoising_dir else None

    final_loss_csv: Optional[Tuple[str, str, Path]] = None  # (cid, label, csv)
    cards = []
    for camp in campaigns:
        cid = camp["id"]
        label = camp.get("label", cid)
        blurb = camp.get("blurb", "")
        src = ckpt_root / cid
        dest = stage_dir / cid
        media: List[str] = []

        # Loss-curve images are still copied as reusable assets (the book uses
        # them), but per advisor feedback the page shows loss ONLY for the
        # final run - as an explicit train vs. validation two-panel figure.
        _copy(src / "loss_curves.png", dest / "loss_curves.png")

        # Training-progress gallery: only the final Israeli_3style campaign shows
        # it, built from that run's mel_viz frames (same crop/layout as the book).
        # The two earlier campaigns don't carry a gallery.
        if camp.get("step_gallery"):
            gal_png = dest / "step_gallery.png"
            if _step_gallery_png(src / "mel_viz", gal_png):
                rel = _rel(gal_png, out_root)
                media.append(_figure(
                    rel, "generated mel every 10k steps (2k -> 248k) - sample quality "
                    "improving as the model trains"))
                man.add(stage="02_training", rel_path=rel, kind="image",
                        title=f"{cid} - step gallery", caption="model progress",
                        spec="sample quality")

        loss_csv = src / "loss_log.csv"
        if loss_csv.exists():
            if camp.get("step_gallery"):        # marks the final campaign
                final_loss_csv = (cid, label, loss_csv)
            copied_csv = _copy(loss_csv, dest / "loss_log.csv")
            if copied_csv:
                rel = _rel(copied_csv, out_root)
                man.add(stage="02_training", rel_path=rel, kind="data",
                        title=f"{cid} - loss log", caption="raw loss/lr/grad",
                        spec="training stability")

        # denoising progression (generated separately into assets/02_training/<cid>)
        if denoising_src is not None:
            _copy(denoising_src / f"{cid}_denoising.png", dest / "denoising.png")
        for dn in sorted(dest.glob("denoising*.png")):
            rel = _rel(dn, out_root)
            media.append(_figure(rel, "diffusion denoising progression (noise -> clean mel)"))
            man.add(stage="02_training", rel_path=rel, kind="image",
                    title=f"{cid} - denoising", caption="reverse diffusion",
                    spec="diffusion behaviour")

        cards.append(
            f'<div class="card"><h3>{_esc(label)}</h3>'
            f"<p>{_esc(blurb)}</p>"
            f'<div class="grid">{"".join(media) or "<em>artifacts pending</em>"}</div></div>')

    # Final-run train/val loss (advisor: final run only, both losses explicit)
    loss_block = ""
    if final_loss_csv is not None:
        cid, flabel, fcsv = final_loss_csv
        tv_png = stage_dir / "final_train_val_loss.png"
        if _final_loss_png(fcsv, tv_png, "Final 3-style model"):
            rel = _rel(tv_png, out_root)
            man.add(stage="02_training", rel_path=rel, kind="image",
                    title=f"{cid} - train vs validation loss",
                    caption="final run, two-panel", spec="training stability")
            explain = (
                "Before training we split the collected songs into two disjoint "
                "sets. About 90% become the training set: the model sees these "
                "segments over and over and adjusts its weights to reduce its "
                "error on them. The remaining songs form the validation set: "
                "the model never trains on them - at regular intervals we only "
                "measure the same error on them, without changing any weights. "
                "The split is done per song, not per 5-second segment; "
                "otherwise two slices of the same song could land on both "
                "sides, and the check would be meaningless.")
            reading = (
                "How to read the curves: the training loss (left) is the "
                "error on songs the model fits directly; the validation loss "
                "(right) is the honest test - the same error measured on "
                "songs the model has never heard. If the model were merely "
                "memorizing, the left curve would keep dropping while the "
                "right one flattened or rose. In our run the two settle at "
                "almost the same level (about 0.12), which tells us the model "
                "generalizes to unseen songs instead of memorizing - but also "
                "that after ~100k steps the loss alone can no longer "
                "distinguish the late checkpoints. That is why the final "
                "checkpoints (210k-250k) were chosen by listening tests and "
                "by the FAD / F1 metrics, not by the loss value.")
            loss_block = (
                '<div class="card"><h3>Training vs. validation loss '
                '(final 3-style run)</h3>'
                f"<p>{_esc(explain)}</p>"
                + _figure(rel, "left: loss on training songs (raw + moving "
                          "average); right: loss on held-out validation songs "
                          "- same L1 noise-prediction error, disjoint songs")
                + f"<p>{_esc(reading)}</p></div>")

    denoise_note = ""
    if not any(stage_dir.glob("*/denoising*.png")):
        denoise_note = (
            '<p class="note">Diffusion denoising-progression visuals '
            "(pure noise &rarr; clean mel) are generated by a dedicated pass "
            "(<span class=\"kv\">ddim_sample(return_intermediates=True)</span>) "
            "and will appear here once produced.</p>")

    # data augmentation subsection (training-time JointAugment on a real segment)
    aug_block = ""
    acfg = tcfg.get("augmentation", {})
    featured = cfg.get("preprocessing", {}).get("featured_songs", [])
    demo_song = acfg.get("demo_song") or (featured[0]["name"] if featured else None)
    aug_clip = (seconds if lite else 25.0)

    # Prefer the REAL offline-augmented source-pool song actually used in training.
    sample_dir = acfg.get("sample_dir")
    real_aud, real_wav, real_midi = ([], None, None)
    if sample_dir:
        real_aud, real_wav, real_midi = _augmentation_real_audio(
            Path(sample_dir), stage_dir / "augmentation_audio", aug_clip)

    # Visual: use the real source-pool song if available, else a featured demo.
    viz_wav = real_wav if real_wav else (
        Path(drive["version_dir"]) / "demo_external" / demo_song / "original.wav"
        if demo_song else None)
    viz_midi = real_midi if real_wav else (
        next(iter(sorted((Path(drive["version_dir"]) / "demo_external"
                          / demo_song).glob("*.mid"))), None)
        if demo_song else None)
    viz_name = (Path(sample_dir).name if real_wav else demo_song) or "song"

    if viz_wav is not None:
        aug_png = stage_dir / "augmentation.png"
        if _augmentation_png(viz_wav, viz_midi, aug_png,
                             f"Training-time data augmentation - {viz_name}"):
            rel = _rel(aug_png, out_root)
            man.add(stage="02_training", rel_path=rel, kind="image",
                    title="data augmentation", caption="JointAugment variants",
                    spec="data augmentation")
            aug_block = (
                '<div class="card"><h3>Data augmentation (training time)</h3>'
                '<p>' + _esc(acfg.get("narrative",
                    "Every training batch is augmented on the fly by JointAugment: "
                    "pitch-shift and time-stretch are applied jointly to the mel and "
                    "the piano-roll score (so the conditioning stays aligned), and "
                    "SpecAugment masks time/frequency bands on the mel only. Offline, "
                    "each source-pool song is also expanded with pitch +/-2 and "
                    "time-stretch 0.9/1.1 WAV+MIDI copies.")) + "</p>"
                + _figure(rel, "one real 5 s segment under each augmentation; the "
                          "bottom row shows the piano-roll shifting together with the "
                          "mel under pitch-shift")
                + "</div>")

    # Audible samples: real drive-augmented WAVs if present, else synth fallback.
    aud = real_aud
    audio_note = ("These are the actual augmented WAVs generated for the "
                  "source-pool song and fed into training (offline pitch/time "
                  "policy).")
    if not aud and viz_wav is not None:
        aud = _augmentation_audio(viz_wav, stage_dir / "augmentation_audio",
                                  clip_seconds=aug_clip)
        audio_note = ("The same pitch-shift and time-stretch, applied at the "
                      "waveform level (the offline source-pool policy).")
    if aud:
        players = []
        for label, path in aud:
            rel = _rel(path, out_root)
            players.append(_audio(rel, label))
            man.add(stage="02_training", rel_path=rel, kind="audio",
                    title=f"augmentation - {label}",
                    caption="audible waveform augmentation",
                    spec="data augmentation")
        aug_block += (
            '<div class="card"><h3>Hear the augmentation</h3>'
            f'<p>{audio_note} Play the original, then each augmented copy.</p>'
            f'<div class="grid">{"".join(players)}</div></div>')

    return (
        '<section id="training"><h2>2. Training</h2>'
        '<p class="lead">' + _esc(tcfg.get("narrative", "")) + "</p>"
        + _mermaid(DIAGRAM_MODEL)
        + _architecture_block()
        + "<h3>Why train on a synthesized dataset first? Clean vs. noisy note labels</h3>"
        + _mermaid(DIAGRAM_CLEAN_NOISY)
        + aug_block
        + loss_block
        + "".join(cards)
        + denoise_note
        + "</section>")


def _copy_reuse_group(source_dir: Path, files: List[str], dirs: List[str],
                      dest: Path) -> List[Tuple[Path, str]]:
    """Copy named files + PNGs from subdirs (prefixed to avoid name collisions).

    ``files`` entries may include a subdirectory (e.g. ``fad_pca/x.png``) to
    hand-pick single figures instead of pulling a whole directory; those get
    the same ``subdir__name`` prefix as full-directory copies so captions and
    filenames stay consistent.

    Returns list of (dest_path, display_name).
    """
    out: List[Tuple[Path, str]] = []
    for f in files:
        name = f.replace("/", "__").replace("\\", "__")
        c = _copy(source_dir / f, dest / name)
        if c:
            out.append((c, name))
    for d in dirs:
        sub = source_dir / d
        if sub.exists():
            for s in sorted(sub.glob("*.png")):
                name = f"{d}__{s.name}"
                c = _copy(s, dest / name)
                if c:
                    out.append((c, name))
    return out


def _eval_caption(key: str, name: str) -> str:
    """Human 'how to read this figure' caption for an evaluation asset.

    The grader is not an ML person, so every figure says what is plotted and
    what to look for, instead of just its filename.
    """
    stem = Path(name).stem
    label = stem
    for prefix in ("fad_bells_overlay__overlay__", "fad_bells_overlay__",
                   "fad_bells__", "fad_pca__", "f1_pianoroll__",
                   "f1_heatmap__"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    label = label.replace("__", " / ").replace("_", " ")
    n = stem.lower()
    if key == "purity":
        if "purity_" in n:
            return (f"{label} - every song of the dataset measured against the "
                    "rest of its own dataset (leave-one-song-out FAD, same "
                    "VGGish embedding distance as the main FAD metric). How to "
                    "read: short green bars = the song sits comfortably inside "
                    "its dataset's sound; the dashed line is the standard "
                    "statistical outlier fence (Q3 + 1.5 IQR); every song in "
                    "both datasets is below FAD 5, the 'near-indistinguishable' "
                    "band - so each dataset really is one coherent style.")
        return label
    if key == "fad":
        if "clusters_2d_tsne" in n:
            return ("t-SNE map of the embedding space - every dot is 0.96 s of "
                    "audio; circles = real songs, diamonds = our generated "
                    "audio, colored by style. t-SNE keeps neighbors together, "
                    "so mixed colors in a region mean those sounds are hard to "
                    "tell apart. How to read: the two real Israeli datasets "
                    "interleave (they are genuinely close styles), and the "
                    "generated audio occupies the same region as real music "
                    "while keeping some texture of its own - the visual twin "
                    "of a low-but-not-zero FAD.")
        if "clusters_3d" in n:
            return ("Every sample as a dot in 3-D. We take the 128-number "
                    "fingerprint of each 0.96 s of audio and keep its three "
                    "biggest directions of variation (3-D PCA). Circles = real "
                    "songs (blue = Israeli Artists, green = Israeli Military); "
                    "diamonds = our generated audio, colored by style. How to "
                    "read: the real Artists and Military circles sit in the "
                    "same region (the two Israeli styles genuinely sound "
                    "related), and the generated diamonds land inside/next to "
                    "the real clouds with a small offset - the generated audio "
                    "lives where real music lives, and FAD measures that small "
                    "remaining gap as a single number. (Real Western-rock audio "
                    "is not stored, so that style appears as generated-only.)")
        if "clusters_2d_pca" in n:
            return ("All samples as dots in the embedding space (PCA to 2-D) - "
                    "circles = real songs, diamonds = generated audio, colored "
                    "by style. How to read: the generated clouds sit inside / "
                    "next to the real clouds with a visible but small offset - "
                    "the generated audio lives in the space of real music, "
                    "with its own texture. FAD summarizes exactly this "
                    "picture as one number.")
        if "bells_overlay" in n:
            return (f"{label} - all finalist checkpoints vs. the real style, on ONE "
                    "shared axis (the 128-D embeddings projected onto the line from "
                    "the real mean to the generated mean). The black filled bell is "
                    "real audio; each colored bell is one checkpoint. How to read: "
                    "the closer a bell sits to the black one, the more realistic "
                    "that checkpoint - the starred bell is the lowest (best) FAD.")
        if "bells" in n:
            return (f"{label} - real (blue) vs. generated (red) audio as two "
                    "distributions on the single most-separating direction of the "
                    "128-D embedding space (the Fisher axis). How to read: strong "
                    "overlap = even along their most different direction the two "
                    "are hard to tell apart (good). Do not compare separate bell "
                    "figures to each other - each has its own axis; use the "
                    "overlay figure for that.")
        if "pca" in n:
            return (f"{label} - every dot is 0.96 s of audio mapped to a point "
                    "(VGGish embedding, squashed from 128-D to the 2 directions "
                    "with the most variance). Blue dots = real audio in this "
                    "style, red diamonds = our generated audio; ellipses = the "
                    "2-sigma/3-sigma spread of each cloud. How to read: the more the red "
                    "cloud sits inside the blue one, the more the generated audio "
                    "'lives in the same space' as the real style. The FAD number "
                    "printed on the plot is computed in the full 128-D space.")
        if "by_step" in n:
            return ("FAD per finalist checkpoint (grouped by style and sampler). "
                    "How to read: lower bar = generated audio closer to the real "
                    "style; all bars sit far below our target of 9.")
        return label
    if key == "f1":
        if "pianoroll" in n:
            return (f"{label} - the honest view of content preservation: the "
                    "reference notes (transcribed from the original song) and the "
                    "notes transcribed from our styled output, drawn on the same "
                    "piano roll (time left-to-right, pitch bottom-to-top). How to "
                    "read: where the two colors overlap or line up vertically, the "
                    "melody survived the transfer; isolated marks are transcription "
                    "noise. Judge the alignment of the note patterns, not the "
                    "exact pixel overlap.")
        if "heatmap" in n:
            return (f"{label} - note-level F1 for every (checkpoint x song) pair, "
                    "darker = higher. How to read: use it to compare checkpoints "
                    "and songs against each other; the absolute numbers are "
                    "pessimistic because both sides come from an imperfect "
                    "automatic transcriber.")
        if "precision_recall" in n:
            return ("Precision (of the notes we generated, how many match the "
                    "reference), recall (of the reference notes, how many we "
                    "reproduced) and their combination F1, per finalist. How to "
                    "read: comparative tool between checkpoints - the absolute "
                    "scale is deflated by transcription noise on both sides.")
        if "scatter" in n or "vs_fad" in n:
            return ("Trade-off map: each point is one finalist checkpoint, x = FAD "
                    "(realism, lower is better), y = F1 (content, higher is "
                    "better). How to read: the best compromises sit toward the "
                    "top-left; the dashed line marks the checkpoints no other "
                    "checkpoint beats on both axes at once (the Pareto frontier).")
        return label
    return label


_FAD_EXPLAINER_HTML = (
    '<div class="card"><h3>How the FAD test actually works (from audio to one '
    'number)</h3>'
    '<p><b>Step 1 - from sound to numbers.</b> A pre-trained network called '
    'VGGish listens to every 0.96-second window of audio and summarizes it as '
    'a list of 128 numbers (an "embedding") that captures how that moment '
    'sounds - texture, instruments, brightness. A folder of audio therefore '
    'becomes a cloud of points in a 128-dimensional space, where similar-'
    'sounding moments land near each other.</p>'
    '<p><b>Step 2 - compare the two clouds.</b> We take the cloud of real '
    'songs in a style and the cloud of our generated audio for that style, '
    'fit a Gaussian to each (its center &mu; and spread &Sigma;), and compute the '
    'Fr&eacute;chet distance between the two Gaussians. That distance is the FAD '
    'score: 0 means identical distributions, and lower means the generated '
    'audio is statistically closer to the real style. Our scores of about '
    '2.2-2.5 are far under the target of 9.</p>'
    '<p><b>Step 3 - how the pictures show it (128-D &rarr; 2-D &rarr; 1-D).</b> A '
    '128-dimensional cloud cannot be drawn, so we flatten it in two honest '
    'ways. The <i>PCA scatter</i> keeps the two directions along which the '
    'points vary the most and drops the rest - good for seeing the two clouds '
    'and their overlap. The <i>bell curves</i> go further: every point is '
    'projected onto a single line - the line connecting the real cloud\'s '
    'center to the generated cloud\'s center. This is the one direction along '
    'which the two clouds differ the <em>most</em>, so it is the most '
    'pessimistic possible 1-D view; each cloud then becomes an ordinary bell '
    'curve. If even these bells overlap heavily, the two distributions are '
    'genuinely close. In every figure the printed FAD value is computed in '
    'the full 128-D space - the pictures are only there to make it visible.</p>'
    '</div>')

_F1_EXPLAINER_HTML = (
    '<div class="card"><h3>How to read the note-preservation (F1) figures</h3>'
    '<p>F1 asks: did the notes of the original song survive the style '
    'transfer? We transcribe the original and the generated audio with the '
    'same automatic tool (Basic-Pitch), then match notes one-to-one - a match '
    'means same pitch and an onset within 50 ms. Precision = how many of our '
    'generated notes are real; recall = how many of the real notes we kept; '
    'F1 combines both. Because <em>both</em> sides pass through an imperfect '
    'transcriber (and our score tracks pitch only, not instruments), the '
    'absolute numbers come out low (3-4.5%) even when the melody is clearly '
    'preserved by ear - so we use F1 only to compare checkpoints, and the '
    'piano-roll overlay figures as the honest visual check.</p></div>')


def build_evaluation(cfg: dict, out_root: Path, assets_root: Path,
                     man: Manifest) -> str:
    """Build section 3: the FAD, F1 and latency figures (reused from the metrics run)."""
    ecfg = cfg.get("evaluation", {})
    stage_dir = assets_root / "03_evaluation"
    blocks = []
    for key, heading in (("purity", "Dataset purity (is each style one coherent sound?)"),
                         ("fad", "Frechet Audio Distance (realism)"),
                         ("f1", "Note-level F1 (score fidelity)")):
        sub = ecfg.get(key, {})
        if not sub:
            continue
        source_dir = Path(sub["source_dir"])
        dest = stage_dir / key
        copied = _copy_reuse_group(source_dir, sub.get("files", []),
                                   sub.get("dirs", []), dest)
        # Additional source folders (e.g. the embedding-cluster figures live
        # in a different metrics directory than the finalist FAD assets).
        for extra in sub.get("extra_sources", []) or []:
            copied += _copy_reuse_group(Path(extra["dir"]),
                                        extra.get("files", []),
                                        extra.get("dirs", []), dest)
        figs = []
        for c, name in copied:
            if c.suffix.lower() == ".png":
                rel = _rel(c, out_root)
                figs.append(_figure(rel, _eval_caption(key, name)))
                man.add(stage="03_evaluation", rel_path=rel, kind="image",
                        title=f"{key}: {name}", caption=heading,
                        spec=key.upper())
            else:
                rel = _rel(c, out_root)
                man.add(stage="03_evaluation", rel_path=rel, kind="data",
                        title=f"{key}: {name}", caption=heading, spec=key.upper())
        explainer = (_FAD_EXPLAINER_HTML if key == "fad"
                     else _F1_EXPLAINER_HTML if key == "f1" else "")
        blocks.append(
            f'<div class="card"><h3>{_esc(heading)}</h3>'
            f"<p>{_esc(sub.get('blurb',''))}</p></div>"
            + explainer
            + f'<div class="card"><div class="grid">{"".join(figs)}</div></div>')

    # latency
    lat = ecfg.get("latency", {})
    if lat:
        dest = stage_dir / "latency"
        chart = _copy(Path(lat["chart"]), dest / "latency_by_ddim.png")
        fig = ""
        if chart:
            rel = _rel(chart, out_root)
            fig = _figure(rel, "real-time factor by ddim steps")
            man.add(stage="03_evaluation", rel_path=rel, kind="image",
                    title="latency chart", caption="RTF by sampler",
                    spec="RTF < 1.0")
        blocks.append(
            f'<div class="card"><h3>Latency / real-time factor</h3>'
            f"<p>{_esc(lat.get('blurb',''))}</p>{fig}</div>")

    return (
        '<section id="evaluation"><h2>3. Postprocessing &amp; evaluation</h2>'
        '<p class="lead">The generated mel is vocoded to audio with BigVGAN, then '
        "scored with FAD (realism), note-level F1 (score fidelity) and latency "
        "(real-time factor).</p>"
        + _mermaid(DIAGRAM_POSTPROCESS)
        + "".join(blocks)
        + "</section>")


def _curated_manifest(cfg: dict):
    """Load the curated-clips manifest (built from the listening worksheet)."""
    icfg = cfg.get("inference", {})
    curated_dir = _HERE / icfg.get("curated_dir", "deliverables/curated_examples")
    manifest_csv = curated_dir / "manifest.csv"
    if not manifest_csv.exists():
        return curated_dir, None
    rows = list(csv.DictReader(manifest_csv.open(encoding="utf-8-sig")))
    return curated_dir, rows


def _style_display(style: str) -> str:
    """Presentation-friendly style name (e.g. Slakh_v0 -> Western rock)."""
    return {
        "Slakh_v0": "Western rock",
        "Slakh": "Western rock",
        "Israeli_Artists": "Israeli Artists",
        "Israeli_Military": "Israeli Military",
    }.get(style, style.replace("_", " "))


def _clip_tag(r: dict) -> str:
    """Human label for one curated clip, e.g. 'Israeli Artists · ddim100 · step 224k'."""
    return (f"{_style_display(r['style'])} \u00b7 {r['sampler']} "
            f"\u00b7 step {int(r['step']) // 1000}k")


def build_inference(cfg: dict, out_root: Path, assets_root: Path,
                    man: Manifest) -> str:
    """Build section 4: curated style-transfer results, grouped by song.

    Uses the clips built from the listening worksheet (already cut to the chosen
    window and split good/broken). For each song we play the original and then
    the same window re-rendered in each style, so the comparison is direct.
    """
    curated_dir, rows = _curated_manifest(cfg)
    stage_dir = assets_root / "04_inference_results"
    labels = {s["name"]: s.get("label", s["name"])
              for s in cfg.get("preprocessing", {}).get("featured_songs", [])}
    labels.update(cfg.get("inference", {}).get("song_labels", {}) or {})

    if not rows:
        return ('<section id="inference"><h2>4. Inference results (style transfer)</h2>'
                '<p class="note">Curated examples not found &mdash; run '
                '<span class="kv">deliverables/build_curated_examples.py</span> '
                'first.</p></section>')

    by_song: Dict[str, Dict[str, list]] = {}
    for r in rows:
        b = by_song.setdefault(r["song"], {"original": [], "good": [], "broken": []})
        b[r["bucket"]].append(r)

    order = [s["name"] for s in cfg.get("preprocessing", {}).get("featured_songs", [])]
    songs = [s for s in order if s in by_song] + [s for s in by_song if s not in order]

    # Optional filtering for the refined/final bundle: a song whitelist and a
    # cap on clips per song (clips are picked round-robin across styles so
    # every style stays represented).
    icfg = cfg.get("inference", {})
    whitelist = icfg.get("featured_songs")
    if whitelist:
        songs = [s for s in songs if s in whitelist]
    max_clips = int(icfg.get("max_clips_per_song", 0))

    cards = []
    for song in songs:
        buckets = by_song[song]
        if not buckets["good"]:
            continue  # all-broken songs live in the struggles section
        dest = stage_dir / song
        label = labels.get(song, song.replace("_", " "))

        orig_block = ""
        for r in buckets["original"]:
            copied = _copy(curated_dir / r["rel_path"], dest / "original.wav")
            if copied:
                rel = _rel(copied, out_root)
                orig_block = ('<div class="card"><h4>Original (source song)</h4>'
                              + _audio(rel, "original \u00b7 " + r["window"]) + "</div>")
                man.add(stage="04_inference_results", rel_path=rel, kind="audio",
                        title=f"{song} - original", caption="A/B reference",
                        spec="inference input")

        good = sorted(buckets["good"],
                      key=lambda m: (m["style"], m["sampler"], m["step"]))
        if max_clips and len(good) > max_clips:
            by_style: Dict[str, list] = {}
            for r in good:
                by_style.setdefault(r["style"], []).append(r)
            picked, i = [], 0
            while len(picked) < max_clips and any(by_style.values()):
                for st in sorted(by_style):
                    if by_style[st] and len(picked) < max_clips:
                        picked.append(by_style[st].pop(0))
                i += 1
            good = sorted(picked, key=lambda m: (m["style"], m["sampler"], m["step"]))

        players = []
        for r in good:
            copied = _copy(curated_dir / r["rel_path"], dest / Path(r["rel_path"]).name)
            if not copied:
                continue
            rel = _rel(copied, out_root)
            players.append(_audio(rel, _clip_tag(r)))
            man.add(stage="04_inference_results", rel_path=rel, kind="audio",
                    title=f"{song} - {_clip_tag(r)}", caption="style transfer output",
                    spec="style transfer")

        window = buckets["good"][0]["window"]
        cards.append(
            f'<div class="card"><h3>{_esc(label)}</h3>'
            + orig_block
            + f'<p>Transferred versions ({len(players)}) &mdash; the same '
              f'{_esc(window)} window as the original, across styles, samplers and '
              'checkpoints:</p>'
            + f'<div class="grid">{"".join(players)}</div></div>')

    return (
        '<section id="inference"><h2>4. Inference results (style transfer)</h2>'
        '<p class="lead">For five songs chosen from outside the training styles, '
        'listen to the original and then the same short window re-rendered in each '
        'learned style. The notes stay fixed while the model repaints the sound. '
        'Every clip is the exact segment we picked by ear for the clearest '
        'comparison.</p>'
        + "".join(cards)
        + "</section>")


def build_struggles(cfg: dict, out_root: Path, assets_root: Path,
                    man: Manifest) -> str:
    """Build section 5: honest failure cases (hard styles + seam artifacts)."""
    curated_dir, rows = _curated_manifest(cfg)
    if not rows:
        return ""
    broken = [r for r in rows if r["bucket"] == "broken"]
    if not broken:
        return ""

    stage_dir = assets_root / "05_struggles"
    labels = {s["name"]: s.get("label", s["name"])
              for s in cfg.get("preprocessing", {}).get("featured_songs", [])}
    labels.update(cfg.get("inference", {}).get("song_labels", {}) or {})

    by_song: Dict[str, list] = {}
    for r in broken:
        by_song.setdefault(r["song"], []).append(r)

    order = [s["name"] for s in cfg.get("preprocessing", {}).get("featured_songs", [])]
    songs = [s for s in order if s in by_song] + [s for s in by_song if s not in order]

    # Optional filtering for the refined/final bundle.
    icfg = cfg.get("inference", {})
    whitelist = icfg.get("struggle_songs")
    if whitelist:
        songs = [s for s in songs if s in whitelist]
    max_clips = int(icfg.get("max_struggle_clips_per_song", 0))

    cards = []
    for song in songs:
        dest = stage_dir / song
        label = labels.get(song, song.replace("_", " "))
        players = []
        broken_rows = sorted(by_song[song],
                             key=lambda m: (m["style"], m["sampler"], m["step"]))
        if max_clips:
            broken_rows = broken_rows[:max_clips]
        for r in broken_rows:
            copied = _copy(curated_dir / r["rel_path"], dest / Path(r["rel_path"]).name)
            if not copied:
                continue
            rel = _rel(copied, out_root)
            players.append(_audio(rel, _clip_tag(r)))
            man.add(stage="05_struggles", rel_path=rel, kind="audio",
                    title=f"{song} - {_clip_tag(r)} (struggle)",
                    caption="failure case", spec="known limitation")
        cards.append(
            f'<div class="card"><h3>{_esc(label)}</h3>'
            f'<div class="grid">{"".join(players)}</div></div>')

    return (
        '<section id="struggles"><h2>5. Where the model struggles</h2>'
        '<p class="lead">We show the weak cases on purpose. Two things go wrong. '
        'First, when the source is far from anything in our small training set '
        '&mdash; heavy metal like Metallica &mdash; the model has little to lean on, '
        'so the output is noisy and only loosely follows the song. Second, on some '
        'checkpoints you can hear the joins between the 5-second segments, or a '
        'machine-like buzz, when the generated mel is blurry and the vocoder has to '
        'fill in. These are the honest limits of training on only a few hours of '
        'hand-collected audio.</p>'
        + "".join(cards)
        + "</section>")


# ===========================================================================
# Assembly
# ===========================================================================
def _fetch(url: str, timeout: int = 60, retries: int = 3) -> Optional[bytes]:
    """Fetch a URL with a few retries; return the bytes or None if it keeps failing."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # offline / service down / timeout -> retry
            if attempt == retries - 1:
                print(f"  [diagram] fetch failed ({e})")
    return None


def _export_diagram(diagram: str, out_dir: Path, name: str) -> Dict[str, Path]:
    """Render a Mermaid diagram to SVG (vector) + PNG via mermaid.ink."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"code": diagram, "mermaid": {"theme": "neutral"}}
    b = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    out: Dict[str, Path] = {}
    svg_p = out_dir / f"{name}.svg"
    if svg_p.exists() and svg_p.stat().st_size > 0:
        out["svg"] = svg_p
    else:
        svg = _fetch(f"{MERMAID_INK}/svg/{b}?bgColor=white")
        if svg and svg.lstrip()[:5].lower() in (b"<?xml", b"<svg "):
            svg_p.write_bytes(svg)
            out["svg"] = svg_p
    png_p = out_dir / f"{name}.png"
    if png_p.exists() and png_p.stat().st_size > 0:
        out["png"] = png_p
    else:
        png = _fetch(f"{MERMAID_INK}/img/{b}?type=png&width=1800&bgColor=white")
        if png and png[:8] == b"\x89PNG\r\n\x1a\n":
            png_p.write_bytes(png)
            out["png"] = png_p
    return out


def export_all_diagrams(out_root: Path, assets_root: Path, man: "Manifest") -> None:
    """Save every block diagram as SVG + PNG files (for drop-in deliverables).

    Exports the full set - including the detailed U-Net and conditioning
    diagrams that the page itself does not embed - so the bundle's
    ``assets/diagrams/`` folder is a complete drop-in library for the poster,
    book and slides.
    """
    catalog = [
        ("00_pipeline_macro", "Full pipeline block diagram", DIAGRAM_MACRO),
        ("01_preprocessing", "Preprocessing block diagram", DIAGRAM_PREPROCESS),
        ("02_model", "Model (training + inference) block diagram", DIAGRAM_MODEL),
        ("02_clean_vs_noisy", "Clean vs noisy score conditioning", DIAGRAM_CLEAN_NOISY),
        ("02_unet_architecture", "Detailed U-Net architecture", DIAGRAM_UNET),
        ("02_conditioning", "FiLM + classifier-free-guidance conditioning", DIAGRAM_CONDITIONING),
        ("03_postprocessing", "Postprocessing block diagram", DIAGRAM_POSTPROCESS),
    ]
    diag_dir = assets_root / "diagrams"
    for name, title, diagram in catalog:
        res = _export_diagram(diagram, diag_dir, name)
        if not res:
            print(f"  [diagram] {name}: SKIPPED (offline?) - live render still shown")
            continue
        entry: Dict[str, str] = {}
        for kind, path in res.items():
            rel = _rel(path, out_root)
            entry[kind] = rel
            man.add(stage="diagrams", rel_path=rel, kind="image",
                    title=f"{title} ({kind.upper()})",
                    caption="block diagram export for deliverables",
                    spec="block diagram")
        _DIAGRAM_EXPORTS[diagram] = entry
        print(f"  [diagram] {name}: {' + '.join(k.upper() for k in res)}")


def _collect_presentation_assets(out_root: Path, assets_root: Path) -> None:
    """Gather every diagram + figure into a single flat ``presentation_assets/``.

    The showcase spreads its images across ``assets/<stage>/...`` for the page,
    but for building slides it is far easier to have one folder to drag from.
    We copy every diagram (SVG + PNG) and every visualization PNG (loss curves,
    piano-roll explainer, DSP walkthroughs, FAD/F1/latency/purity/cluster
    figures) here with stage-prefixed names so nothing collides. Audio and
    per-render clips are left in ``assets/`` (they are not slide images).
    """
    pres = out_root / "presentation_assets"
    if pres.exists():
        shutil.rmtree(pres)
    pres.mkdir(parents=True, exist_ok=True)

    # 1) all diagrams (SVG + PNG) - flat, keep their clean names.
    diag_src = assets_root / "diagrams"
    if diag_src.exists():
        for f in sorted(diag_src.iterdir()):
            if f.suffix.lower() in (".png", ".svg"):
                shutil.copy2(f, pres / f"diagram_{f.name}")

    # 2) every visualization PNG under assets/, stage-prefixed, de-duplicated.
    skip_stages = {"diagrams", "04_inference_results", "05_struggles"}
    seen: set = set()
    for png in sorted(assets_root.rglob("*.png")):
        rel_parts = png.relative_to(assets_root).parts
        stage = rel_parts[0]
        if stage in skip_stages:
            continue
        flat = "__".join(rel_parts).replace(" ", "_")
        if flat in seen:
            continue
        seen.add(flat)
        shutil.copy2(png, pres / flat)

    # 3) the piano-roll explainer video (a real slide asset).
    for mp4 in sorted(assets_root.rglob("piano_roll_demo.mp4")):
        shutil.copy2(mp4, pres / "piano_roll_demo.mp4")
        break

    n = len(list(pres.iterdir()))
    print(f"  [presentation] {n} slide-ready files -> presentation_assets/")


def _vendor_mermaid(out_root: Path) -> str:
    """Download mermaid.min.js locally for offline use; return a script tag."""
    vendor = out_root / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    local = vendor / "mermaid.min.js"
    try:
        with urllib.request.urlopen(MERMAID_CDN, timeout=30) as resp:
            local.write_bytes(resp.read())
        print(f"  [mermaid] vendored -> {local.name} "
              f"({local.stat().st_size // 1024} KB)")
        return '<script src="vendor/mermaid.min.js"></script>'
    except Exception as e:
        print(f"  [mermaid] CDN fallback ({e})")
        return f'<script src="{MERMAID_CDN}"></script>'


def _load_config(path: Path) -> dict:
    """Load the showcase config from YAML or JSON."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: read the config and build the whole showcase bundle."""
    ap = argparse.ArgumentParser(description="Build the portable project showcase.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--lite", action="store_true",
                    help="trim every audio clip to a short preview for a small bundle")
    args = ap.parse_args(argv)

    cfg = _load_config(args.config.resolve())
    out_root = args.out_dir.resolve()
    assets_root = out_root / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)

    lite = bool(args.lite)
    preview_seconds = float(cfg.get("lite", {}).get("preview_seconds", 45.0))

    drive = {
        "drive_root": cfg["drive_root"],
        "version_dir": cfg["version_dir"],
    }
    specs = cfg.get("specs", [])
    man = Manifest()

    print("Building project showcase ->", out_root)
    if lite:
        print(f" - LITE mode: audio trimmed to {preview_seconds:.0f}s previews")
    print(" - exporting block diagrams (SVG + PNG)")
    export_all_diagrams(out_root, assets_root, man)
    print(" - overview")
    sec_overview = build_overview(cfg, specs)
    print(" - preprocessing")
    sec_pre = build_preprocessing(cfg, drive, out_root, assets_root, man,
                                  lite=lite, seconds=preview_seconds)
    print(" - training")
    sec_train = build_training(cfg, drive, out_root, assets_root, man,
                               lite=lite, seconds=preview_seconds)
    print(" - evaluation")
    sec_eval = build_evaluation(cfg, out_root, assets_root, man)
    print(" - inference results")
    sec_infer = build_inference(cfg, out_root, assets_root, man)
    print(" - where it struggles")
    sec_struggles = build_struggles(cfg, out_root, assets_root, man)

    script_tag = _vendor_mermaid(out_root)

    toc = (
        '<nav class="toc"><strong>Jump to:</strong>'
        '<a href="#overview">Overview</a>'
        '<a href="#preprocessing">1. Preprocessing</a>'
        '<a href="#training">2. Training</a>'
        '<a href="#evaluation">3. Evaluation</a>'
        '<a href="#inference">4. Inference results</a>'
        '<a href="#struggles">5. Where it struggles</a>'
        '<a href="MANIFEST.csv">Manifest</a></nav>')

    hero = (
        '<header class="hero">'
        f'<h1>{_esc(cfg.get("title", "Project Showcase"))}</h1>'
        f'<p>{_esc(cfg.get("subtitle", ""))}</p>'
        f'<p>{_esc(cfg.get("authors", ""))}</p>'
        + (f'<p class="pill">LITE preview build &middot; audio trimmed to '
           f'{preview_seconds:.0f}s clips</p>' if lite else "")
        + '</header>')

    page = (
        _page_head(cfg.get("title", "Project Showcase"))
        + hero + toc + "<main>"
        + sec_overview + sec_pre + sec_train + sec_eval + sec_infer + sec_struggles
        + "</main>"
        + '<footer>Portable showcase &middot; every asset under '
          '<span class="kv">assets/</span> is a real file with a relative link '
          "&middot; see MANIFEST.csv</footer>"
        + script_tag
        + '<script>mermaid.initialize({startOnLoad:true, theme:"neutral", '
          'securityLevel:"loose"});</script>'
        + "</body></html>")

    (out_root / "index.html").write_text(page, encoding="utf-8")
    man.write(out_root / "MANIFEST.csv")

    _collect_presentation_assets(out_root, assets_root)

    print(f"Done. {len(man.rows)} assets catalogued.")
    print(f"Open: {out_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
