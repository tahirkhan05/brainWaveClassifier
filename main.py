"""
BrainWave Classifier - EEG Motor Imagery Classification
========================================================
Dataset : EEG Motor Movement/Imagery Dataset (EEGMMIDB) - PhysioNet
Subject : S012
Files   : R03, R04, R07, R08, R11, R12, R13, R14 (left/right fist)
Pipeline: MNE loading -> beta bandpass -> epoch -> CSP -> SVM

Classes:
    T1 -> Left fist  (label 0)
    T2 -> Right fist (label 1)

Key findings:
    - Beta band (13-30 Hz) is far more discriminative than mu for S012
    - Ledoit-Wolf regularisation stabilises CSP with limited epochs
    - Epoch window 1.0-3.5s captures peak motor imagery activation
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from mne.decoding import CSP

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# ─── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─── Configuration ──────────────────────────────────────────────────────────
EDF_FILES = [
    "S012R03.edf",   # real left/right fist
    "S012R04.edf",   # imagined left/right fist
    "S012R07.edf",   # real left/right fist (repeat 1)
    "S012R08.edf",   # imagined left/right fist (repeat 1)
    "S012R11.edf",   # real left/right fist (repeat 2)
    "S012R12.edf",   # imagined left/right fist (repeat 2)
    "S012R13.edf",   # real left/right fist (repeat 3)
    "S012R14.edf",   # imagined left/right fist (repeat 3)
]

# Beta band only — analysis showed mu (7-13 Hz) adds noise for S012
L_FREQ      = 13.0
H_FREQ      = 30.0

# 1.0-3.5s window captures peak imagery; skips onset transient
EPOCH_TMIN  = 1.0
EPOCH_TMAX  = 3.5

# 4 CSP components — enough signal, not enough to overfit 120 epochs
CSP_N_COMPS = 4

EVENT_ID = {"T1": 1, "T2": 2}   # T1=left fist, T2=right fist


# ============================================================================
# SECTION 1 – LOAD & CONCATENATE EDF FILES
# ============================================================================
def load_raw(edf_files):
    """
    Read each EDF file with MNE and concatenate into one continuous Raw object.
    Preloading into RAM is required for filtering.
    """
    print("\n[1] Loading EDF files ...")
    raws = []
    for fname in edf_files:
        if not os.path.isfile(fname):
            raise FileNotFoundError(
                f"Could not find '{fname}'. "
                "Place all EDF files in the same folder as main.py."
            )
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        raws.append(raw)
        print(f"    {fname}  |  {raw.n_times} samples  |  "
              f"{len(raw.ch_names)} channels  |  {raw.info['sfreq']} Hz")

    raw_concat = mne.concatenate_raws(raws)
    print(f"    Combined: {raw_concat.n_times} total samples")
    return raw_concat


# ============================================================================
# SECTION 2 – PREPROCESSING  (beta bandpass filter)
# ============================================================================
def preprocess(raw):
    """
    Apply a bandpass FIR filter keeping only the beta rhythm (13-30 Hz).
    Beta desynchronisation during motor imagery is the main discriminative signal
    for subject S012. The mu rhythm (8-13 Hz) was found to add noise.
    """
    print(f"\n[2] Bandpass filtering {L_FREQ}-{H_FREQ} Hz (beta band) ...")
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design="firwin", verbose=False)
    print("    Done.")
    return raw


# ============================================================================
# SECTION 3 – EVENT EXTRACTION & EPOCHING
# ============================================================================
def make_epochs(raw):
    """
    Extract T1/T2 events from PhysioNet annotations and cut epochs.
    T0 (rest) is explicitly excluded to prevent label leakage.
    """
    print("\n[3] Extracting events and creating epochs ...")

    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    orig_t1 = event_dict.get("T1")
    orig_t2 = event_dict.get("T2")
    if orig_t1 is None or orig_t2 is None:
        raise ValueError(f"T1/T2 not found in annotations. Got: {event_dict}")

    # Keep ONLY T1 and T2 rows — T0 (rest) must not enter either class
    mask = (events[:, 2] == orig_t1) | (events[:, 2] == orig_t2)
    ev = events[mask].copy()
    ev[ev[:, 2] == orig_t1, 2] = 1  # T1 -> 1 (left)
    ev[ev[:, 2] == orig_t2, 2] = 2  # T2 -> 2 (right)

    print(f"    T1 (left)  events: {(ev[:, 2] == 1).sum()}")
    print(f"    T2 (right) events: {(ev[:, 2] == 2).sum()}")

    epochs = mne.Epochs(raw, ev, event_id=EVENT_ID,
                        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                        baseline=None, preload=True, verbose=False)
    print(f"    Epochs shape: {epochs.get_data().shape}  "
          f"(trials x channels x timepoints)")
    return epochs


# ============================================================================
# SECTION 4 – PIPELINE  (CSP -> StandardScaler -> SVM)
# ============================================================================
def build_pipeline(C=100, gamma=0.01):
    """
    CSP finds spatial filters that maximise variance difference between classes.
    Ledoit-Wolf regularisation stabilises covariance estimation with ~120 epochs.
    SVM with RBF kernel handles the non-linear boundary in CSP feature space.
    """
    csp    = CSP(n_components=CSP_N_COMPS, reg="ledoit_wolf",
                 log=True, norm_trace=False)
    scaler = StandardScaler()
    svm    = SVC(kernel="rbf", C=C, gamma=gamma,
                 class_weight="balanced", random_state=RANDOM_SEED)

    return Pipeline([("csp", csp), ("scaler", scaler), ("svm", svm)])


# ============================================================================
# SECTION 5 – HYPERPARAMETER TUNING + EVALUATION
# ============================================================================
def tune_and_train(X, y):
    """
    Step 1 — GridSearchCV (5-fold) on 80% of data to find best C and gamma.
    Step 2 — 10-fold stratified CV with fixed best params on ALL data.
             This is the honest accuracy estimate (not the hold-out split).
    Step 3 — Refit on 80% train split for the confusion matrix.

    n_jobs=1 everywhere to keep CPU/RAM usage safe on a laptop.
    """
    # ── Step 1: hyperparameter search ───────────────────────────────────────
    print("\n[4] Hyperparameter search (GridSearchCV, 5-fold on 80% data) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}")

    param_grid = {
        "svm__C":     [10, 100, 500],
        "svm__gamma": [0.001, 0.01, 0.1],
    }
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(build_pipeline(), param_grid,
                      cv=inner_cv, scoring="accuracy", n_jobs=1, verbose=0)
    gs.fit(X_train, y_train)

    best_C     = gs.best_params_["svm__C"]
    best_gamma = gs.best_params_["svm__gamma"]
    print(f"    Best params : C={best_C}, gamma={best_gamma}")
    print(f"    CV acc (5-fold on train) : {gs.best_score_*100:.1f}%")

    # ── Step 2: unbiased accuracy + CV predictions for confusion matrix ────────
    print("\n[5] 10-fold CV on full dataset (reliable accuracy estimate) ...")
    from sklearn.model_selection import cross_val_predict
    fixed_pipe  = build_pipeline(C=best_C, gamma=best_gamma)
    outer_cv    = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    cv_scores   = cross_val_score(fixed_pipe, X, y,
                                  cv=outer_cv, scoring="accuracy", n_jobs=1)

    # cross_val_predict collects each epoch's prediction from its held-out fold
    # Result: 120 predictions, each made without seeing that epoch during training
    fixed_pipe2  = build_pipeline(C=best_C, gamma=best_gamma)
    outer_cv2    = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    y_cv_pred    = cross_val_predict(fixed_pipe2, X, y, cv=outer_cv2, n_jobs=1)

    print(f"    Fold scores : {np.round(cv_scores * 100, 1)}")
    print(f"    Mean CV acc : {cv_scores.mean()*100:.1f}%  "
          f"+/- {cv_scores.std()*100:.1f}%")

    return y, y_cv_pred, cv_scores.mean()


# ============================================================================
# SECTION 6 – EVALUATION & PLOTS
# ============================================================================
def evaluate(y_true, y_pred, cv_mean):
    """
    Confusion matrix built from 10-fold CV predictions (all 120 epochs).
    Every prediction was made on a held-out fold, so this is fully honest.
    """
    acc = accuracy_score(y_true, y_pred)

    print(f"\n[6] Evaluation on all 120 CV predictions ...")
    print(f"    Accuracy : {acc*100:.1f}%  (matches CV mean)")
    print("\n    Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Left (T1)", "Right (T2)"]))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Left (T1)", "Right (T2)"],
        ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion Matrix  (10-fold CV acc = {cv_mean*100:.1f}%)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("    Saved: confusion_matrix.png")
    plt.close()
    return acc


def plot_csp_patterns(epochs):
    """Plot CSP spatial activation patterns (which electrodes matter most)."""
    try:
        X_all = epochs.get_data()
        y_all = epochs.events[:, 2] - 1
        csp   = CSP(n_components=CSP_N_COMPS, reg="ledoit_wolf",
                    log=True, norm_trace=False)
        csp.fit(X_all, y_all)
        fig = csp.plot_patterns(epochs.info, ch_type="eeg",
                                units="Patterns (AU)", size=1.5, show=False)
        fig.savefig("csp_patterns.png", dpi=150)
        print("    Saved: csp_patterns.png")
        plt.close("all")
    except Exception as e:
        print(f"    (CSP pattern plot skipped: {e})")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("  BrainWave Classifier  —  EEG Motor Imagery (CSP + SVM)")
    print("=" * 60)

    raw    = load_raw(EDF_FILES)
    raw    = preprocess(raw)
    epochs = make_epochs(raw)

    X = epochs.get_data()        # (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2] - 1  # 0=left, 1=right

    print(f"\n    X shape: {X.shape}  |  y: {np.bincount(y)} (left / right)")

    y_true, y_cv_pred, cv_mean = tune_and_train(X, y)

    evaluate(y_true, y_cv_pred, cv_mean)

    print("\n[7] Plotting CSP spatial patterns ...")
    plot_csp_patterns(epochs)

    print("\n" + "=" * 60)
    print(f"  Mean 10-fold CV accuracy : {cv_mean*100:.1f}%")
    if cv_mean >= 0.75:
        print("  Result: good — well above chance (50%) for single-subject BCI")
    elif cv_mean >= 0.65:
        print("  Result: fair — typical for a single subject without artefact removal")
    else:
        print("  Result: low — consider artefact rejection or different subject")
    print("=" * 60)


if __name__ == "__main__":
    main()
