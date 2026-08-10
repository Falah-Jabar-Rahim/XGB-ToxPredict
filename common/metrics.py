"""Metric computation shared by both single-stage (M1/M2) and hierarchical evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_ece(y_true, y_prob, n_bins: int) -> float:
    """Expected Calibration Error with uniform-width probability bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)

    ece = 0.0
    for i in range(1, n_bins + 1):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_confidence = np.mean(y_prob[bin_mask])
            bin_accuracy = np.mean(y_true[bin_mask])
            ece += (bin_size / len(y_true)) * abs(bin_confidence - bin_accuracy)
    return ece


def metrics_at_threshold(y_true, y_prob, n_bins: int, threshold: float = 0.5) -> dict:
    """Threshold a probability vector and compute a full metrics dict.

    (Renamed from the original ``metrics_at_05`` — the "05" in the name was
    misleading since it already accepted an arbitrary threshold.)
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "y_pred": y_pred,
        "cm": cm,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision),  # == PPV
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),  # sensitivity
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ECE": float(compute_ece(y_true, y_prob, n_bins=n_bins)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Specificity": float(specificity),
        "PPV": float(precision),
        "NPV": float(npv),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def compute_cascade_metrics(
    y_true,
    prob_final,
    prob_m1,
    y_pred_m1,
    prob_m2,
    y_pred_m2,
    sample_id,
    threshold: float = 0.5,
) -> dict:
    """Metrics for the M1 -> M2 hierarchical cascade's final prediction.

    Mirrors the original root ``test.py::compute_metrics`` 1:1, just moved
    into the shared metrics module and given a name that doesn't collide
    with ``metrics_at_threshold``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    prob_final = np.asarray(prob_final, dtype=np.float64).ravel()
    y_pred = (prob_final >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)

    return {
        "ID": sample_id,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob_pred": prob_final,
        "y_prob_pred_m1": prob_m1,
        "y_pred_m1": y_pred_m1,
        "y_prob_pred_m2": prob_m2,
        "y_pred_m2": y_pred_m2,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "PPV": precision,  # same as Precision
        "ROC_AUC": roc_auc_score(y_true, prob_final),
        "PR_AUC": average_precision_score(y_true, prob_final),
        "Brier": brier_score_loss(y_true, prob_final),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
