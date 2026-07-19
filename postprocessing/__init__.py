"""Postprocessing package: everything after the model produces a mel.

This is where a generated mel becomes audio and gets scored. It holds the
vocoder wrappers (BigVGAN, plus a HiFi-GAN reference we compared against) and
the evaluation code (latency, FAD, note-level F1). We re-export the most common
entry points here so callers can import them straight from ``postprocessing``.
"""

from postprocessing.latency_eval import evaluate_latency, plot_latency, print_latency_report
from postprocessing.fad_eval import compute_fad, evaluate_all_fad
