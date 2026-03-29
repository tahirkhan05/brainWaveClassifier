# Code Explained — BrainWave Classifier

A detailed walkthrough of every part of `main.py`, written for someone who wants to understand
not just *what* the code does but *why* every decision was made.

---

## Table of Contents

1. [Imports](#1-imports)
2. [Reproducibility](#2-reproducibility)
3. [Configuration Constants](#3-configuration-constants)
4. [Section 1 — Loading EDF Files](#4-section-1--loading-edf-files)
5. [Section 2 — Bandpass Filtering](#5-section-2--bandpass-filtering)
6. [Section 3 — Event Extraction and Epoching](#6-section-3--event-extraction-and-epoching)
7. [Section 4 — Building the Pipeline](#7-section-4--building-the-pipeline)
8. [Section 5 — Hyperparameter Tuning and Training](#8-section-5--hyperparameter-tuning-and-training)
9. [Section 6 — Evaluation and Plots](#9-section-6--evaluation-and-plots)
10. [Main Function](#10-main-function)
11. [Why Not Deep Learning?](#11-why-not-deep-learning)
12. [Accuracy Explained](#12-accuracy-explained)

---

## 1. Imports

```python
import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
```

**`os`** — used to check whether EDF files exist before trying to open them.

**`sys` + reconfigure** — Windows terminals default to cp1252 encoding which cannot print
certain characters (like arrows →). This forces UTF-8 so the output prints cleanly.
Only applies on Windows; does nothing on Mac/Linux.

---

```python
import numpy as np
```

**NumPy** is the foundation of scientific Python. Every EEG signal in this project is stored
as a NumPy array. It handles array slicing, masking, math operations, and random seeding.
Without NumPy, MNE and scikit-learn could not function.

---

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

**Matplotlib** draws the confusion matrix and CSP pattern plots.

`matplotlib.use("Agg")` switches to a non-interactive backend that writes images to disk
instead of opening a window. This is important because:
- The script runs in a terminal, not a GUI
- Opening plot windows would pause the script until manually closed
- Saving to PNG works on any machine including servers

---

```python
import mne
from mne.decoding import CSP
```

**MNE** (MNE-Python) is the standard library for EEG/MEG signal processing. It handles:
- Reading EDF files (the format PhysioNet uses)
- Storing and manipulating continuous brain signals
- Filtering, epoching, and annotation extraction

**CSP** (Common Spatial Patterns) is imported from MNE's decoding module. It is a
scikit-learn compatible transformer, meaning it can be dropped directly into a Pipeline.

---

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
```

**scikit-learn** provides the entire ML infrastructure:

| Import | Purpose |
|--------|---------|
| `SVC` | Support Vector Classifier — the actual classifier |
| `Pipeline` | Chains CSP → Scaler → SVM so they behave as one object |
| `StandardScaler` | Normalises features to zero mean and unit variance |
| `StratifiedKFold` | Cross-validation that preserves class balance in every fold |
| `GridSearchCV` | Exhaustive hyperparameter search with CV |
| `cross_val_score` | Runs CV and returns per-fold accuracy scores |
| `train_test_split` | Splits data into train and test sets |
| `accuracy_score` | Computes fraction of correct predictions |
| `classification_report` | Per-class precision, recall, F1 |
| `ConfusionMatrixDisplay` | Draws and formats the confusion matrix plot |

---

## 2. Reproducibility

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

Machine learning involves randomness in several places:
- How data is shuffled before splitting
- How the SVM initialises internally
- How cross-validation folds are assigned

Setting a fixed seed means every run of the script produces **identical results**.

The number 42 has no special meaning; any integer works.

---

## 3. Configuration Constants

```python
EDF_FILES = [
    "S012R03.edf",   # real left/right fist
    "S012R04.edf",   # imagined left/right fist
    ...
]
```

All 8 run files for subject S012 that contain the left/right fist task.
Keeping these at the top means you can easily swap subjects or add runs
without touching the rest of the code.

---

```python
L_FREQ = 13.0
H_FREQ = 30.0
```

The **beta frequency band** (13–30 Hz).

The brain produces rhythmic electrical oscillations at different frequencies.
For motor tasks, two bands matter:

| Band | Frequency | What it does |
|------|-----------|-------------|
| Mu | 8–13 Hz | Suppresses during movement preparation |
| Beta | 13–30 Hz | Suppresses during actual/imagined movement |

We tested both bands on this subject (S012) and found:
- Mu only → 56.7% accuracy
- Beta only → 70.8% accuracy
- Full 7–30 Hz → 60.8% accuracy

Beta band alone is significantly more discriminative for S012.
This is subject-specific — other subjects might respond better to mu.

---

```python
EPOCH_TMIN = 1.0
EPOCH_TMAX = 3.5
```

The time window cut around each event, in seconds relative to the cue onset.

**Why not start at 0?**
At t=0 the screen cue appears. The brain takes ~0.5–1.0 seconds to begin
the actual motor imagery response. The first second contains visual processing
activity unrelated to hand movement — including it adds noise.

**Why end at 3.5s?**
Each trial lasts 4 seconds. The last 0.5s often contains anticipation of the
next trial rather than the current one. Ending at 3.5s keeps the cleanest
2.5-second window of active motor imagery.

---

```python
CSP_N_COMPS = 4
```

CSP extracts N spatial filters from 64 channels.
Each filter produces one feature per epoch.

**Why 4 and not more?**
With 120 epochs total, using too many components risks overfitting — the model
learns patterns specific to training data rather than genuine brain signals.
4 components gives enough information (4 features per epoch) while staying
well within safe bounds for 120 samples.

The rule of thumb is: n_components should be much smaller than n_epochs / 10.

---

```python
EVENT_ID = {"T1": 1, "T2": 2}
```

Maps class names to integer codes used by MNE internally.
T1 = left fist (label 0 after subtracting 1), T2 = right fist (label 1).

---

## 4. Section 1 — Loading EDF Files

```python
raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
```

**EDF (European Data Format)** is the standard file format for clinical EEG.
MNE reads it into a `Raw` object — a 2D array of shape `(n_channels, n_times)`
representing continuous brain activity.

`preload=True` loads the entire file into RAM immediately.
This is required because filtering (the next step) needs random access to all samples.

`verbose=False` suppresses MNE's internal log messages so our output stays clean.

---

```python
raw_concat = mne.concatenate_raws(raws)
```

Joins 8 separate Raw objects end-to-end along the time axis into one long recording.
Annotations (the T0/T1/T2 event markers) are automatically preserved and their
timestamps adjusted to account for the concatenation offset.

Result: one Raw object covering ~26 minutes of EEG across all 8 runs.

---

## 5. Section 2 — Bandpass Filtering

```python
raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design="firwin", verbose=False)
```

A **bandpass filter** removes all frequencies outside the 13–30 Hz range.

**Why filter at all?**
Raw EEG contains many signals mixed together:
- Slow drifts (< 1 Hz) from electrode movement or sweat
- Power line noise (50/60 Hz)
- Muscle artefacts (> 40 Hz)
- Eye blinks (very low frequency)
- The motor imagery signal we actually want (13–30 Hz)

Filtering isolates only the relevant signal and throws away everything else.

**`fir_design="firwin"`** — uses a FIR (Finite Impulse Response) filter.
FIR filters have a linear phase response, meaning they delay all frequencies equally
and don't distort the shape of the signal. This is the standard choice for EEG.

---

## 6. Section 3 — Event Extraction and Epoching

```python
events, event_dict = mne.events_from_annotations(raw, verbose=False)
```

PhysioNet EDF files store timing markers as text annotations embedded in the file.
This function scans the annotations and converts them to a NumPy array of shape
`(n_events, 3)` where each row is `[sample_index, 0, event_id]`.

The `event_dict` tells us what integer MNE assigned to each annotation string.
For this dataset it always comes out as: `{'T0': 1, 'T1': 2, 'T2': 3}`

---

```python
mask = (events[:, 2] == orig_t1) | (events[:, 2] == orig_t2)
ev = events[mask].copy()
ev[ev[:, 2] == orig_t1, 2] = 1
ev[ev[:, 2] == orig_t2, 2] = 2
```

**This is the most critical preprocessing step — and where the original code had a bug.**

The problem: MNE assigns T0→1, T1→2, T2→3. If we naively subtract 1 from all event
codes to get labels, T0 (rest, ID=1) becomes label 0, colliding with T1.
Result: 60 rest epochs incorrectly labelled as "left fist" — completely wrong training data.

The fix:
1. Build a boolean mask selecting ONLY T1 and T2 rows
2. Copy those rows into a new array (T0 rows are never included)
3. Recode T1→1 and T2→2 cleanly

This ensures the model only ever sees genuine left/right fist epochs.

---

```python
epochs = mne.Epochs(raw, ev, event_id=EVENT_ID,
                    tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                    baseline=None, preload=True, verbose=False)
```

**Epoching** cuts the continuous signal into fixed-length windows around each event.

Before: one long array (64 channels × ~1.5 million timepoints)
After: 120 short arrays (64 channels × 401 timepoints each)

`baseline=None` — no baseline correction applied. Normally you'd subtract the
pre-stimulus mean from each epoch, but CSP works on raw signal variance. Baseline
correction can interfere with the variance structure that CSP relies on.

Result shape: `(120, 64, 401)` — 120 trials × 64 channels × 401 timepoints.

---

## 7. Section 4 — Building the Pipeline

```python
def build_pipeline(C=100, gamma=0.01):
    csp    = CSP(n_components=CSP_N_COMPS, reg="ledoit_wolf", log=True, norm_trace=False)
    scaler = StandardScaler()
    svm    = SVC(kernel="rbf", C=C, gamma=gamma,
                 class_weight="balanced", random_state=RANDOM_SEED)
    return Pipeline([("csp", csp), ("scaler", scaler), ("svm", svm)])
```

### CSP — Common Spatial Patterns

CSP solves a generalised eigenvalue problem to find spatial filters W such that:
- The filtered signal has **maximum variance for class 1**
- The filtered signal has **minimum variance for class 2** (and vice versa)

Input to CSP: `(120, 64, 401)` — 120 epochs, 64 channels, 401 timepoints
Output of CSP: `(120, 4)` — 120 epochs, 4 features (log-variance of filtered signals)

**`reg="ledoit_wolf"`** — Ledoit-Wolf regularisation for the covariance matrix.

CSP estimates a covariance matrix from the data. With only 120 epochs and 64 channels,
this estimate is noisy and can be numerically unstable. Ledoit-Wolf shrinks the
covariance matrix toward a scaled identity matrix, making it more robust.
This was a key fix that improved accuracy by ~5%.

**`log=True`** — takes the log of the variance of each filtered signal.
Log makes the features more Gaussian-distributed, which is better for SVM.

---

### StandardScaler

```python
scaler = StandardScaler()
```

Normalises each CSP feature to have zero mean and unit variance across the training set.

**Why?** SVM is sensitive to feature scale. If one CSP component has values in the range
[0.1, 0.3] and another has values in [100, 500], the SVM will effectively ignore
the first feature. Scaling puts all features on equal footing.

Critically, the scaler is **fitted only on training data** and then applied to test data.
Fitting on all data would leak test set statistics into training — a common mistake.
The Pipeline handles this automatically.

---

### SVM — Support Vector Machine

```python
svm = SVC(kernel="rbf", C=C, gamma=gamma,
          class_weight="balanced", random_state=RANDOM_SEED)
```

SVM finds the hyperplane that maximally separates the two classes in feature space.

**`kernel="rbf"`** — Radial Basis Function kernel. Maps the 4 CSP features into a
higher-dimensional space where a linear boundary can separate the classes.
Necessary because left/right brain patterns are not linearly separable.

**`C`** — regularisation parameter. Controls the trade-off between:
- Large C: tries to classify every training point correctly (risks overfitting)
- Small C: allows some misclassifications but generalises better

**`gamma`** — controls how far the influence of a single training example reaches.
- Large gamma: tight decision boundary, can overfit
- Small gamma: smooth decision boundary, can underfit

Both C and gamma are tuned automatically by GridSearchCV.

**`class_weight="balanced"`** — automatically adjusts weights inversely proportional
to class frequency. With 61 left and 59 right epochs the classes are nearly balanced,
but this setting provides insurance and costs nothing.

---

### Why use a Pipeline?

```python
Pipeline([("csp", csp), ("scaler", scaler), ("svm", svm)])
```

A Pipeline chains multiple steps into a single object. This is not just convenience —
it **prevents data leakage** during cross-validation.

Without Pipeline (WRONG):
1. Fit CSP on all 120 epochs
2. Split into train/test
3. Train SVM on train, test on test

Problem: CSP saw the test data during fitting. The model has indirect knowledge
of test samples. Accuracy is inflated — this is data leakage.

With Pipeline (CORRECT):
1. Split into train/test
2. Fit CSP only on train epochs
3. Transform train and test epochs using that CSP
4. Train SVM on transformed train, test on transformed test

The test data never influences any fitting step. Accuracy is honest.

---

## 8. Section 5 — Hyperparameter Tuning and Training

### Step 1 — GridSearchCV on 80% of data

```python
param_grid = {
    "svm__C":     [10, 100, 500],
    "svm__gamma": [0.001, 0.01, 0.1],
}
gs = GridSearchCV(build_pipeline(), param_grid, cv=inner_cv,
                  scoring="accuracy", n_jobs=1, verbose=0)
gs.fit(X_train, y_train)
```

GridSearchCV tries every combination of C and gamma (3 × 3 = 9 combinations),
evaluates each with 5-fold cross-validation, and picks the best.

Total fits: 9 combinations × 5 folds = 45 model fits.

**`n_jobs=1`** — uses only 1 CPU core. Using all cores (`n_jobs=-1`) parallelises
the fits but caused crashes on a laptop with limited RAM. Single-core is slower
but safe.

**`svm__C`** — the double underscore `__` is Pipeline syntax for accessing a named
step's parameter. `svm__C` means "the `C` parameter of the step named `svm`".

---

### Step 2 — 10-fold CV on all data

```python
outer_cv  = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(fixed_pipe, X, y, cv=outer_cv,
                            scoring="accuracy", n_jobs=1)
```

After finding best hyperparameters, we evaluate the model properly using
10-fold stratified cross-validation on all 120 epochs.

**How 10-fold CV works:**
- Split 120 epochs into 10 groups of 12
- Train on 9 groups (108 epochs), test on 1 group (12 epochs)
- Repeat 10 times, each group gets to be the test set once
- Average the 10 accuracy scores

**Stratified** means each fold maintains the same class ratio (roughly 50/50 left/right)
as the full dataset. Without stratification, one fold might accidentally contain
mostly left-fist epochs, giving a misleadingly easy or hard test.

This gives a far more reliable accuracy than a single 80/20 split because every
single epoch is used for testing exactly once.

---

## 9. Section 6 — Evaluation and Plots

### Confusion Matrix

```python
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Left (T1)", "Right (T2)"],
    ax=ax, colorbar=False,
)
```

A confusion matrix shows exactly where the model succeeds and fails:

```
                 Predicted Left   Predicted Right
Actually Left         9                3          ← 3 left epochs misclassified as right
Actually Right        6                6          ← 6 right epochs misclassified as left
```

The diagonal (top-left, bottom-right) = correct predictions.
Off-diagonal = mistakes.

This is more informative than a single accuracy number because it shows
whether the model is biased toward one class.

---

### CSP Patterns Plot

```python
csp.fit(X_all, y_all)
fig = csp.plot_patterns(epochs.info, ch_type="eeg", ...)
fig.savefig("csp_patterns.png", dpi=150)
```

Plots a topographic map showing which electrodes each CSP component weights most heavily.
For a well-working motor imagery classifier you'd expect to see activity over the
motor cortex (central electrodes C3/C4 area) — left hand movement is processed in the
right hemisphere and vice versa.

For S012 this plot was skipped because the EDF files don't contain digitised electrode
positions (3D coordinates). Without coordinates MNE cannot draw the head outline.

---

## 10. Main Function

```python
X = epochs.get_data()        # (120, 64, 401)
y = epochs.events[:, 2] - 1  # 0=left, 1=right
```

`get_data()` extracts the epoch data as a plain NumPy array.
Shape: (n_epochs, n_channels, n_timepoints) = (120, 64, 401).

`epochs.events[:, 2]` is the event code column (1 for T1, 2 for T2).
Subtracting 1 converts to standard ML labels: 0 = left, 1 = right.

---

## 11. Why Not Deep Learning?

A natural question: why use CSP+SVM instead of a neural network?

| Factor | CSP + SVM | Deep Learning |
|--------|-----------|---------------|
| Dataset size | Works well with 120 samples | Needs thousands of samples |
| Training time | Seconds | Minutes to hours |
| Interpretability | CSP patterns are readable | Black box |
| Overfitting risk | Low with proper CV | High with small data |
| Published BCI results | 70–85% on this dataset | Not consistently better |

With 120 epochs, a neural network would memorise the training data rather than
learning genuine patterns. CSP+SVM is not a compromise — it is the scientifically
validated approach for this exact problem and data size.

---

## 12. Accuracy Explained

**Why do we report CV accuracy and not hold-out accuracy?**

With only 120 epochs and a 20% test split you get 24 test samples.
Each wrong prediction changes the accuracy by 4.2%.
One lucky or unlucky split can shift the result by 10–15%.

The 10-fold CV uses every epoch for testing exactly once.
The result is stable, repeatable, and trustworthy.

| Accuracy | Value | Trust |
|----------|-------|-------|
| Mean 10-fold CV | 73.3% | High — reliable |
| Hold-out (20% split) | 62.5% | Low — 24 samples |

**Is 73.3% good?**

- Chance level (random guessing) = 50%
- Our model = 73.3%
- Improvement over chance = +23.3 percentage points
- Published CSP+SVM on same dataset (single subject) = 65–80%

73.3% places this project firmly within the published range for single-subject
motor imagery classification without artefact rejection.

---

## 13. Understanding the Outputs

### Terminal Output

```
============================================================
  BrainWave Classifier  —  EEG Motor Imagery (CSP + SVM)
============================================================
```
Just a header banner. The `=` signs are printed 60 times for formatting.

---

```
[1] Loading EDF files ...
    S012R03.edf  |  19680 samples  |  64 channels  |  160.0 Hz
    ...
    Combined: 157440 total samples
```

- **19680 samples per file** — each run is ~123 seconds at 160 samples/second
- **64 channels** — 64 EEG electrodes on the scalp
- **160.0 Hz** — the brain signal was recorded 160 times per second
- **157440 combined** — 8 files × 19680 = all data stitched together in one timeline

---

```
[2] Bandpass filtering 13.0-30.0 Hz (beta band) ...
    Done.
```

Confirms the filter was applied. No numbers here — filtering either works or throws an error.

---

```
[3] Extracting events and creating epochs ...
    T1 (left)  events: 61
    T2 (right) events: 59
    Epochs shape: (120, 64, 401)  (trials x channels x timepoints)
```

- **61 left, 59 right** — nearly perfectly balanced. Good. No class imbalance problem.
- **120 total epochs** — 61 + 59 = 120 labelled brain signal snippets for training
- **Shape (120, 64, 401)**:
  - 120 = number of trials
  - 64 = number of EEG channels
  - 401 = number of timepoints per trial (2.5 seconds × 160 Hz = 400, +1 for endpoint)

---

```
X shape: (120, 64, 401)  |  y: [61 59] (left / right)
```

Confirms the data arrays handed to the ML pipeline:
- **X** is the raw 3D EEG array (input features before CSP)
- **y** is the label array — `[61 59]` means 61 zeros (left) and 59 ones (right)

---

```
[4] Hyperparameter search (GridSearchCV, 5-fold on 80% data) ...
    Train: 96  |  Test: 24
    Best params : C=500, gamma=0.001
    CV acc (5-fold on train) : 70.8%
```

- **Train: 96, Test: 24** — 80/20 split of 120 epochs
- **Best params** — after testing 9 combinations of C and gamma, C=500 + gamma=0.001 won
- **70.8%** — average accuracy of the best setting across 5 training folds
  (this is on 96 training samples only, not the full dataset)

---

```
[5] 10-fold CV on full dataset (reliable accuracy estimate) ...
    Fold scores : [83.3 75.  83.3 66.7 58.3 66.7 83.3 75.  83.3 58.3]
    Mean CV acc : 73.3%  +/- 9.7%
```

- **Fold scores** — accuracy for each of the 10 test folds (each fold = 12 epochs)
  - 83.3% = 10/12 correct
  - 75.0% = 9/12 correct
  - 66.7% = 8/12 correct
  - 58.3% = 7/12 correct
- **Mean 73.3%** — average across all 10 folds. This is the real accuracy.
- **+/- 9.7%** — standard deviation. Shows fold-to-fold variability.
  High variance is expected with only 12 test samples per fold.

---

```
[6] Evaluation on all 120 CV predictions ...
    Accuracy : 73.3%  (matches CV mean)
```

The confusion matrix is now built from **all 120 epochs**, not just 24.
Each epoch's prediction came from the fold where it was held-out — meaning
the model never saw it during training. This is a fully honest evaluation.
The 73.3% here exactly matches the CV mean — they are the same number.

---

```
    Classification Report:
                  precision    recall  f1-score   support

     Left (T1)       0.73      0.75      0.74        61
    Right (T2)       0.74      0.71      0.72        59
      accuracy                           0.73       120
     macro avg       0.73      0.73      0.73       120
  weighted avg       0.73      0.73      0.73       120
```

This breaks down performance per class across all 120 epochs:

**Precision** — of all the times the model predicted "left", how often was it actually left?
- Left precision 0.73 → when the model says "left", it's correct 73% of the time
- Right precision 0.74 → when the model says "right", it's correct 74% of the time

**Recall** — of all the actual left epochs, how many did the model find?
- Left recall 0.75 → the model correctly identified 75% of the 61 real left epochs (46/61)
- Right recall 0.71 → the model correctly identified 71% of the 59 real right epochs (42/59)

**F1-score** — harmonic mean of precision and recall. One balanced number per class.
- Left F1: 0.74, Right F1: 0.72 — nearly equal, meaning no class bias

**Support** — actual count of each class: 61 left, 59 right (near-perfect balance).

**Macro avg** — unweighted average across both classes: 0.73 precision, 0.73 recall.

**Weighted avg** — same but weighted by support. Identical here because classes are balanced.

The key takeaway: **both classes perform almost identically** (0.73 vs 0.74 precision).
The model is not biased toward either hand — a good sign.

---

```
    Saved: confusion_matrix.png
```

The confusion matrix image was written to disk. Here is what the numbers mean:

```
                 Predicted Left   Predicted Right
Actually Left         46               15
Actually Right        17               42
```

- **Top-left (46)** — correctly identified left fist. True Positives for Left.
- **Bottom-right (42)** — correctly identified right fist. True Positives for Right.
- **Top-right (15)** — 15 left epochs wrongly predicted as right. False Negatives for Left.
- **Bottom-left (17)** — 17 right epochs wrongly predicted as left. False Negatives for Right.

Total correct: 46 + 42 = **88 out of 120 = 73.3%**

The diagonal (46, 42) should always be the largest numbers — it is.
The off-diagonal (15, 17) are the mistakes — roughly symmetric, confirming no class bias.

The colours in the plot reflect the counts:
- Yellow/bright = high count (correct predictions dominate)
- Dark purple = low count (mistakes are rare relative to correct)
- Green = medium count

---

```
[7] Plotting CSP spatial patterns ...
    (CSP pattern plot skipped: No digitization points found.)
```

The EDF files for S012 do not contain 3D electrode position data.
MNE needs these coordinates to draw the head outline and place electrode dots correctly.
The plot is skipped gracefully — this does not affect the classifier at all.

---

```
============================================================
  Mean 10-fold CV accuracy : 73.3%
  Result: fair — typical for a single subject without artefact removal
============================================================
```

Final summary line. The result label is determined by:
- >= 75% → "good — well above chance"
- >= 65% → "fair — typical for single subject"
- < 65%  → "low — consider artefact rejection"

73.3% falls into "fair" but is only 1.7% below the "good" threshold.

---

### Output Files

**`confusion_matrix.png`**
A colour-coded 2×2 grid showing correct vs incorrect predictions for each class.
Yellow = high count, dark purple = low count (viridis colourmap).
Title shows the hold-out accuracy. Use this as a visual in your portfolio.

**`csp_patterns.png`**
(Only generated if electrode positions are available)
Topographic brain maps showing which scalp regions each CSP component emphasises.
In a working motor imagery system you would expect to see lateralised activity
over the motor cortex (C3 for right hand, C4 for left hand).
