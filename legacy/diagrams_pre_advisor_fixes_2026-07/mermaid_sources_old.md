# Diagram sources — snapshot BEFORE advisor-fix round (2026-07-19)

These are the Mermaid sources exactly as they were in `build_showcase.py`
before the advisor-requested changes (remove Demucs branch, split model
training/inference, rename Slakh, expand abbreviations, single vocoder).
The rendered PNG/SVG exports in this folder match these sources.

## 00_pipeline_macro (DIAGRAM_MACRO)

```mermaid
flowchart LR
    URL(("Song URL"))
    PRE["Preprocessing"]
    TENS[("Mel tensors / Piano-roll tensors")]
    VER(["Version ID"])
    MODEL["Model block\ntraining + inference"]
    GMEL[("Generated mel")]
    POST["Postprocessing"]
    WAV(("Generated WAV"))
    MET[("Metric test results")]
    VIS[("Visualizations")]
    PVIS[("Preprocess visualizations")]
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
    class URL,TENS,VER,GMEL,WAV,MET,VIS,PVIS io
```

## 01_preprocessing (DIAGRAM_PREPROCESS)

```mermaid
flowchart LR
    URL(("Song URL"))
    DL["Download WAV"]
    DEM["Optional stems"]
    AUD[("Working audio")]
    BP["Basic-Pitch MIDI"]
    MEL["Mel extraction"]
    PR["Piano roll"]
    SEG["5s segments"]
    AUG["Optional augmentation"]
    TENS[("Mel tensors / PR tensors")]
    VIZ[("Diagnostic visualizations")]
    URL --> DL --> AUD
    DL -. "separate-stems" .-> DEM --> AUD
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
    class DEM oss
    class MEL,PR,SEG,AUG custom
    class URL,AUD,TENS,VIZ data
```

## 02_model (DIAGRAM_MODEL)

```mermaid
flowchart LR
    MEL[("Mel tensors")]
    PR[("PR tensors")]
    VER(["Version ID"])
    NOISE[("Noise x_T")]
    TRAIN["Training\nDDPM"]
    CKPT[("Checkpoint")]
    INFER["Inference\nDDIM + CFG"]
    GMEL[("Generated mel")]
    MEL --> TRAIN
    PR --> TRAIN
    VER --> TRAIN
    TRAIN --> CKPT --> INFER
    PR --> INFER
    VER --> INFER
    NOISE --> INFER
    INFER --> GMEL
    classDef model fill:#fff,stroke:#a855f7,stroke-width:3px,color:#222
    classDef data fill:#fff,stroke:#64748b,color:#666
    class TRAIN,INFER model
    class MEL,PR,VER,NOISE,CKPT,GMEL data
```

## 02_clean_vs_noisy (DIAGRAM_CLEAN_NOISY)

```mermaid
flowchart LR
    A["Slakh2100\nGround-truth MIDI\n(per instrument)"] -->|"Clean piano roll"| M["Diffusion Model"]
    B["Israeli audio\nBasic-Pitch MIDI\n(automated, ~noisy)"] -->|"Noisy piano roll"| M
    M --> V1["Architecture check\nval loss stable\nhearing test passed"]
    M --> V2["Israeli output\nsubjective quality"]
    classDef clean fill:#fff,stroke:#22c55e,stroke-width:2px,color:#222
    classDef noisy fill:#fff,stroke:#f59e0b,stroke-width:2px,color:#222
    classDef model fill:#fff,stroke:#a855f7,stroke-width:2px,color:#222
    classDef gate fill:#fff,stroke:#64748b,color:#666
    class A clean
    class B noisy
    class M model
    class V1,V2 gate
```

## 03_postprocessing (DIAGRAM_POSTPROCESS)

```mermaid
flowchart LR
    GMEL[("Generated mel")]
    PREP["Mel prep"]
    FACTORY["Vocoder wrapper"]
    BIGVGAN["BigVGAN v2"]
    HIFIGAN["HiFi-GAN reference"]
    WAV(("Generated WAV"))
    MET["Metric tests"]
    RES[("Metric results")]
    VIS[("Visualizations")]
    GMEL --> PREP --> FACTORY --> BIGVGAN --> WAV
    FACTORY -. "optional" .-> HIFIGAN -.-> WAV
    WAV --> MET --> RES
    MET --> VIS
    classDef custom fill:#fff,stroke:#22c55e,stroke-width:2px,color:#222
    classDef oss fill:#fff,stroke:#4f8cff,stroke-width:2px,color:#222
    classDef data fill:#fff,stroke:#64748b,color:#666
    class PREP,FACTORY,MET,VIS custom
    class BIGVGAN,HIFIGAN oss
    class GMEL,WAV,RES data
```

## 02_unet_architecture (DIAGRAM_UNET) — unchanged in this round

```mermaid
flowchart TB
    IN["Input concat\nnoisy mel [80] + piano-roll [256]\n336 ch, T=430"]
    IC["Input conv 3x1\n336 -> 160"]
    E0["Encoder L0\n2x ResBlock, 160 ch, T=430\n(no attention)"]
    D0["Downsample 160 -> 320\nT=215"]
    E1["Encoder L1\n2x ResBlock + Self-Attn\n320 ch, T=215"]
    D1["Downsample 320 -> 480\nT=108"]
    E2["Encoder L2\n2x ResBlock + Self-Attn\n480 ch, T=108"]
    D2["Downsample 480 -> 640\nT=54"]
    BN["Bottleneck\nResBlock + Attn + ResBlock\n640 ch, T=54"]
    U2["Upsample 640 -> 480\nT=108"]
    C2["Decoder L2\n3x ResBlock + Self-Attn\n480+480 ch, T=108"]
    U1["Upsample 480 -> 320\nT=215"]
    C1["Decoder L1\n3x ResBlock + Self-Attn\n320+320 ch, T=215"]
    U0["Upsample 320 -> 160\nT=430"]
    C0["Decoder L0\n3x ResBlock, 160+160 ch, T=430\n(no attention)"]
    OC["Output conv 3x1\n160 -> 80"]
    OUT["Predicted noise\n[80, 430]"]
    COND(["FiLM conditioning C\n= time_emb 128 + version_emb 128"])
    IN --> IC --> E0 --> D0 --> E1 --> D1 --> E2 --> D2 --> BN
    BN --> U2 --> C2 --> U1 --> C1 --> U0 --> C0 --> OC --> OUT
    E0 -. "skip" .-> C0
    E1 -. "skip" .-> C1
    E2 -. "skip" .-> C2
    COND -. "gamma,beta @ every ResBlock" .-> E1
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
    class COND cond
```

## 02_conditioning (DIAGRAM_CONDITIONING) — abbreviation cleanup only

```mermaid
flowchart LR
    T(["Timestep t"]) --> TE["Sinusoidal embedding\n-> MLP, 128-d"]
    V(["Version ID 0/1/2\n(+ null token for CFG)"]) --> VE["nn.Embedding\n128-d"]
    TE --> CAT["Conditioning C\nconcat -> 256-d"]
    VE --> CAT
    CAT --> FILM["FiLM per ResBlock\ngamma = 1 + Linear(C)\nbeta = Linear(C)\nh' = gamma * h + beta"]
    PR[("Piano-roll score\n2 x 128 pitches -> 256 ch")] --> UNET["U-Net\n(score concat with noisy mel)"]
    FILM --> UNET
    UNET --> EPS["Predicted noise"]
    CFG["Classifier-free guidance\n3 forward passes: full / drop-score / drop-version\nw_s = w_v = 1.25"] -.-> EPS
    classDef cond fill:#fff,stroke:#f59e0b,stroke-width:2px,color:#222
    classDef model fill:#fff,stroke:#a855f7,stroke-width:3px,color:#222
    classDef io fill:#fff,stroke:#64748b,color:#666
    class TE,VE,CAT,FILM,CFG cond
    class UNET,EPS model
    class T,V,PR io
```
