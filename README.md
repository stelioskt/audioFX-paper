# AudioFX Emotion Pipeline — README

## Overview

This project investigates **how audio effects influence emotion recognition in music**.  
We approach the problem through two complementary tasks:

- **Regression** → predicting continuous **Valence** and **Arousal (VA)** values.  
- **Classification** → predicting discrete (*Excitement, Anger, Sadness, Calmness*) or multi-label (*tension, nostalgia, wonder, …*) emotions.  

The pipeline applies **audio effects** (e.g., reverb, distortion, delay) at increasing levels, extracts **embeddings** from large pretrained audio models (**MERT, CLAP, Qwen2-Audio**), and evaluates how both **predictions** and **representations** change.

We design three experiments:

1. **Emotion Shift vs FX** — quantify how predicted emotions drift with different effects and levels.  
2. **Accuracy vs FX Level** — measure robustness of models to transformations in audio.  
3. **Embedding Trajectories** — visualize how embeddings move in representation space under effects.  

**Flow of information**:  
**Dataset → Embeddings → Audio Effects → Prediction Models → Experiments → Outputs**  

---

## 1. Datasets

- **EMOPIA** → single-label classification.  
  - **Labels**: `Excitement`, `Anger`, `Sadness`, `Calmness`  

- **DEAM** → continuous valence/arousal (per-second & per-song) → regression.  

- **witheFlow (WTF)**:  
  - `wtf_va` → regression (continuous VA).  
  - `wtf_lb` → multi-label classification.  
    - **Labels**: `sadness`, `nostalgia`, `peacefulness`, `neutral`, `tenderness`, `joyfulActivation`, `wonder`, `transcendence`, `power`, `tension`

---

## 2. Embedding Models

### **MERT-v1-330M**  
- Parameters: **330M** (24 layers × 1024 hidden).  
- Training: **160,000 hours of music audio**, masked language modeling with RVQ-VAE & CQT teachers.  
- Input: 24 kHz audio, 75 frames/s.  
- Goal: capture **musical/acoustic emotion cues**.  

### **CLAP**  
- Parameters: ~**630M**.  
- Architecture: dual encoder → audio (Transformer/HTS-AT) + text (RoBERTa).  
- Training: **128K–630K audio-text pairs** with contrastive learning.  
- Input: log-Mel spectrograms, 48 kHz.  
- Goal: learn **shared space for audio-text**; strong zero-shot generalization.  

### **Qwen2-Audio-7B**  
- Parameters: **7B**.  
- Architecture: Whisper-style audio encoder + Qwen-style decoder.  
- Training: **>300K hours** of speech, music, and environmental audio.  
- Goal: broad **audio-language understanding** (ASR, classification, QA, chat).  

---

## 3. Audio Effects

We apply six audio effects at levels **1 → 10**. Each effect has **parameters that scale linearly with level**, simulating stronger audio processing.

- **Reverb** *(adds space/room effect)*  
  - Parameter: `room_size` (0.18 → 0.90)  
  - Low levels = small room; high levels = large, echoey hall.  

- **Delay** *(echo effect)*  
  - Parameters:  
    - `delay_seconds` (0.077 → 0.32s) → time between repeats  
    - `feedback` (0.15 → 0.60) → how much of the delayed signal feeds back  
  - Low levels = subtle echo; high levels = strong, repeating echoes.  

- **Distortion** *(adds grit & harmonic saturation)*  
  - Parameter: `drive_db` (6.8 → 23 dB)  
  - Low = mild warmth; high = aggressive guitar-amp style distortion.  

- **EQ (band-pass)** *(filters frequencies)*  
  - Parameters:  
    - `low_cutoff` (500 → 4100 Hz)  
    - `high_cutoff` (21.4k → 16.0k Hz at 44.1k sample rate)  
  - Low levels = broad spectrum; high levels = narrower, “telephone-like” sound.  

- **Chorus** *(thickens by modulating pitch)*  
  - Parameters:  
    - `rate` (0.7 → 2.5 Hz)  
    - `depth` (0.19 → 1.0)  
    - `feedback` (0.12 → 0.30)  
  - Low = subtle doubling; high = lush, swirling chorus effect.  

- **Phaser** *(shifts frequency phase for sweeping effect)*  
  - Parameters:  
    - `rate` (0.3 → 1.2 Hz)  
    - `depth` (0.34 → 1.6)  
    - `feedback` (0.13 → 0.40)  
  - Low = gentle shimmer; high = dramatic sweeping “whoosh”.  

**Real World Scenarios (inspired from combinations of effects used by known artists):**  
- *Pink Floyd*: {reverb: 3, eq: 6, delay: 9, chorus: 7, phaser: 2}  
- *Raging Against The Machine*: {distortion: 9, eq: 5, reverb: 1, delay: 2, chorus: 6, phaser: 1}  
- *U2*: {distortion: 4, eq: 7, reverb: 8, delay: 10, chorus: 2, phaser: 3}  

---

## 4. Prediction Pipelines

### Regression (DEAM, wtf_va)  
- Model: `XGBRegressor` (valence & arousal).  
- Metrics: MAE, MSE, R².  

### Single-label classification (EMOPIA)  
- Model: `XGBClassifier`.  
- Metrics: Accuracy, Precision, Recall, F1 (weighted).  

### Multi-label classification (wtf_lb)  
- Model: `OneVsRest(XGBClassifier)`.  
- Metrics: F1-micro, F1-macro (threshold=0.5).  

---

## 5. Experiments

### **Experiment 1 — Emotion Shift vs FX**  
**Purpose**: measure how predicted emotions drift with effects.  
- Regression → heatmaps & trends of valence/arousal.  
- Classification → label proportions, radar charts.  
- Multi-label → per-label trends.  

### **Experiment 2 — Accuracy vs FX Level**  
**Purpose**: evaluate robustness.  
- Regression → MSE, MAE, R² vs levels.  
- EMOPIA → Accuracy, Precision, Recall, F1.  
- WTF-LB → F1-micro, F1-macro.  
- Scenarios → Original vs bands.  

### **Experiment 3 — Embedding Trajectories**  
**Purpose**: visualize representational drift.  
- Clean & select features → ElasticNet or LogisticRegression.  
- Reduce with UMAP.  
- Plot per-effect trajectories & scenario comparisons.  

---

## 6. Setup & Environment

### Install dependencies
```bash
pip install -r requirements.txt
```

👉 Hugging Face models (MERT, CLAP, Qwen2-Audio) download automatically at first run. You can comment them out in later runs in order to speed up the process.

### .env file
```ini
NUM_LEVEL=10
SAMPLE_RATIO=0.2
RANDOM_STATE=42

DATA_DIR=data/
MODEL_DIR=models/
EMOPIA_DIR=emopia/
DEAM_DIR=deam/
WITHEFLOW_DIR=witheflow/
OUTPUT_DIR=outputs/
```

---

## 7. How to Run

### Full pipeline
```bash
python main.py
```

Runs the full flow: extract → predict → experiments → save outputs.

### Individual experiments
```bash
python emotion.py      # Experiment 1
python accuracy.py     # Experiment 2
python embedding.py    # Experiment 3
```

---

## 8. Outputs

- `outputs/emotion/` → emotion heatmaps, radar charts, CSVs.  
- `outputs/accuracy/` → metric curves & scenario plots.  
- `outputs/embedding/` → UMAP trajectories & scenario grids.  
- `data/` → intermediate pickles (`*_features_fx.pkl`, `*_results_fx.pkl`).  
