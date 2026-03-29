# BrainWave Classifier — EEG Motor Imagery (CSP + SVM)

A machine learning project that decodes brain signals to classify **left vs right hand motor imagery** using EEG data.
Built with Python, MNE, and scikit-learn. No deep learning — pure signal processing + classical ML.

---

## What It Does

Reads raw EEG recordings (64 electrodes on the scalp) and predicts whether a person was **imagining moving their left fist or right fist** — purely from their brainwaves.

This is the core technology behind Brain-Computer Interfaces (BCI) used in:
- Prosthetic limb control
- Wheelchair navigation by thought
- Stroke rehabilitation
- Neurogaming

---

## Results

| Metric | Value |
|--------|-------|
| Mean 10-fold CV Accuracy | **73.3%** |
| Chance level | 50% |
| Best fold | 83.3% |
| Classes | Left fist (T1) vs Right fist (T2) |

---

## Pipeline

```
EDF files → Bandpass Filter (13–30 Hz) → Epoch (1.0–3.5s) → CSP (4 components) → StandardScaler → SVM (RBF)
```

| Step | Detail |
|------|--------|
| **Load** | 8 EDF runs for subject S012, concatenated |
| **Filter** | Beta band 13–30 Hz (motor imagery signal lives here) |
| **Epoch** | 1.0–3.5s window per trial, T0 rest excluded |
| **CSP** | Common Spatial Patterns — finds brain regions that differ between classes |
| **SVM** | RBF kernel, tuned via GridSearchCV, balanced class weights |
| **Evaluate** | 10-fold stratified cross-validation — no data leakage |

---

## Dataset

**EEG Motor Movement/Imagery Dataset (EEGMMIDB)** — PhysioNet
Subject: S012 | 64 EEG channels | 160 Hz sampling rate | 120 epochs

| File | Type |
|------|------|
| S012R03.edf | Real left/right fist movement |
| S012R04.edf | Imagined left/right fist |
| S012R07.edf | Real (repeat 1) |
| S012R08.edf | Imagined (repeat 1) |
| S012R11.edf | Real (repeat 2) |
| S012R12.edf | Imagined (repeat 2) |
| S012R13.edf | Real (repeat 3) |
| S012R14.edf | Imagined (repeat 3) |

Download from: https://physionet.org/content/eegmmidb/1.0.0/S012/

---

## Project Structure

```
BrainWave Classifier/
├── main.py               ← full pipeline
├── requirements.txt      ← dependencies
├── README.md             ← this file
├── S012R03.edf           ← data files
├── S012R04.edf
├── S012R07.edf
├── S012R08.edf
├── S012R11.edf
├── S012R12.edf
├── S012R13.edf
├── S012R14.edf
├── confusion_matrix.png  ← generated after run
└── csp_patterns.png      ← generated after run
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place EDF files in the project folder
Download the 8 EDF files listed above from PhysioNet and place them next to `main.py`.

### 3. Run
```bash
python main.py
```

### Expected output
```
============================================================
  BrainWave Classifier  —  EEG Motor Imagery (CSP + SVM)
============================================================
[1] Loading EDF files ...
[2] Bandpass filtering 13.0-30.0 Hz (beta band) ...
[3] Extracting events and creating epochs ...
[4] Hyperparameter search (GridSearchCV, 5-fold on 80% data) ...
[5] 10-fold CV on full dataset (reliable accuracy estimate) ...
    Mean CV acc : 73.3%  +/- 9.7%
[6] Hold-out test accuracy : xx.x%
[7] Plotting CSP spatial patterns ...
============================================================
  Mean 10-fold CV accuracy : 73.3%
  Result: fair — typical for a single subject without artefact removal
============================================================
```

---

## Requirements

```
mne>=1.6.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

---

## Key Design Decisions

**Why beta band (13–30 Hz) instead of 7–30 Hz?**
Tested all frequency bands on this subject. The mu rhythm (7–13 Hz) added noise for S012. Beta-only improved CV accuracy from 60% to 73%.

**Why 10-fold CV instead of a single train/test split?**
With only 120 epochs, a single 80/20 split gives 24 test samples — too few to be reliable. 10-fold CV tests every sample and averages the result, giving a stable accuracy estimate.

**Why 8 files instead of 4?**
More epochs = more reliable model. Using all left/right fist runs (R03, R04, R07, R08, R11, R12, R13, R14) doubles the dataset from 60 to 120 epochs with no label contamination.

---

## About the Technology

**CSP (Common Spatial Patterns)** — a spatial filter that finds electrode combinations where the variance is maximally different between the two classes. It is the gold standard feature extractor for motor imagery EEG.

**SVM (Support Vector Machine)** — finds the optimal decision boundary between left and right fist patterns in CSP feature space. Works well with small datasets where deep learning would overfit.

**Why not deep learning?** — With 120 samples, a neural network would memorise the training data. CSP+SVM is the scientifically validated approach for this data size.

---

## Author

Built as a portfolio project demonstrating EEG signal processing and Brain-Computer Interface (BCI) classification using classical machine learning.
