# Results

We set four measurable targets and met three of them. The evaluation code is in
`postprocessing/` (FAD, note-level F1, latency); the numbers below come from our
finalist checkpoints.

## Scoreboard

| Target | What it measures | Result | Met? |
|--------|------------------|--------|------|
| All-FAD ≤ 9 | overall audio realism | ~2.2–2.5 pooled | yes |
| Group-FAD ≤ 7 | realism per style | ~2.2–2.5 per style | yes |
| Note-F1 ≥ 30% | content preservation | ~3–4.5% | no (metric-limited) |
| Latency ≤ 5 s / 5 s segment | speed | RTF 0.13 (100 steps) / 0.20 (200 steps) | yes |

## Realism — FAD

Fréchet Audio Distance measures how close our generated audio is to real songs
in the same style — lower is better. We embed real and generated audio with a
pre-trained VGGish network, fit a Gaussian to each set, and compare them. Across
the finalist checkpoints FAD sits around **2.2–2.5**, far under the target of 9,
so the generated audio lands close to the distribution of real music in each
style. Pooled all-FAD is 2.22 (Artists, 100 steps) and 2.20 (Military, 100
steps); the best single checkpoints reach 2.20 (Artists, step 224k) and 2.43
(Military, step 238k).

## Content preservation — F1, and why it is low

Note-level F1 checks how many of the original notes survive the transfer. It
comes out at roughly **3–4.5%**, far below the 30% we originally aimed for. We
think this is mostly a property of the metric rather than of what we hear:

- both sides of the comparison are automatic Basic-Pitch transcriptions, which
  are noisy on their own;
- our score keeps pitch only and does not model instruments;
- transcribing stylized audio, with new timbres and effects, makes Basic-Pitch
  less reliable still;
- the match is strict (pitch + onset within ±50 ms, polyphonic).

When we instead overlay the reference and the transcribed output as piano-rolls,
the melody and harmony line up well, so we treat those overlays as the more
honest measure of content preservation, and use F1 only as a comparative
tie-breaker between checkpoints.

## Speed — latency

We report the real-time factor (RTF = seconds of compute per second of audio;
below 1 is faster than real-time). We measure **0.133** at 100 DDIM steps
(~0.67 s to generate each 5 s of audio, ~36 s for a full song) and **0.202** at
200 steps — both comfortably faster than real-time.

## Dataset checks (coherence and clustering)

Before judging the model we check the data itself. `postprocessing/dataset_purity_fad.py`
computes a leave-one-song-out FAD for every song against the rest of its own
dataset: every song in both Israeli styles sits below FAD 5 (the
"near-indistinguishable" band), with medians around 1.2, so each collection is
sonically coherent. `postprocessing/embedding_cluster_viz.py` renders the same
VGGish embeddings as 2-D/3-D scatter plots of real vs. generated audio per
style, which shows the generated clouds sitting inside the real-music region —
the visual counterpart of the low FAD.

## Honest scope

We deliberately kept the goals modest and measurable. The system does what we
set out to do — the output is recognizable at the same time as the original song
and as the target style — but the audio quality is limited by the small,
self-collected, automatically-transcribed dataset, not by the model design (the
loss curves are stable and the Slakh baseline sounds clean). The best late
checkpoints sit around 238k–248k steps. See the project book for the full
analysis and the comparison with Ben-Maman et al., who train on about 58 hours
of real annotated audio versus our five to six hours per style.
