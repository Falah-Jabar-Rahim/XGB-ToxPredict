"""Plotting helpers shared by the M1/M2 training stages and the hierarchical
predictor.

Consolidates what used to be near-duplicate implementations:
- M1/util/visualize.py and M2/util/visualize.py were byte-identical.
- The root util/visualize.py had the same functions again, plus ~350 lines
  of RF/SVM/LogReg decision-boundary plots and EDA helpers that no
  train.py/test.py in the whole repo ever called (dead code, removed here).
- The root test.py additionally hand-rolled its own plot_roc_curve /
  plot_pr_curve / plot_calibration_curve_custom / plot_confusion_matrix
  instead of reusing the util versions. Those are folded in below as the
  "single-curve" variants, alongside the CV-fold "combined" variants.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as student_t
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from common.metrics import compute_ece

DEFAULT_FIGSIZE = (7, 6)
DEFAULT_DPI = 600


# --------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: Sequence[str] = ("Negative", "Positive"),
    figsize=DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    axis_fontsize: int = 12,
):
    """Plot a 2x2 confusion matrix (counts or fold-averaged floats)."""
    cm = np.asarray(cm)
    fmt = ".2f" if cm.dtype.kind == "f" else "d"

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues", cbar=False, square=True,
        xticklabels=class_names, yticklabels=class_names,
    )
    ax.set_xlabel("Predicted", fontsize=axis_fontsize)
    ax.set_ylabel("Actual", fontsize=axis_fontsize)
    plt.title("Confusion Matrix", fontsize=axis_fontsize)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# Single-curve plots (used by the hierarchical cascade, which has exactly
# one final prediction stream rather than several CV folds to combine)
# --------------------------------------------------------------------------
def plot_roc_curve(y_true, y_prob, save_path, figsize=(6, 6), dpi=DEFAULT_DPI):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_pr_curve(y_true, y_prob, save_path, figsize=(6, 6), dpi=DEFAULT_DPI):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(np.unique(y_true)) < 2:
        print("[WARN] PR curve cannot be computed (only one class present).")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_calibration_curve(
    y_true, y_prob, save_path, n_bins: int = 10, ece: Optional[float] = None,
    figsize=(6, 6), dpi=DEFAULT_DPI,
):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=figsize)
    label_model = f"Model (ECE = {ece:.3f})" if ece is not None else "Model"
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=label_model)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


# --------------------------------------------------------------------------
# Combined plots across one-or-more model results (used by the M1/M2 CV
# training stage). Each ``result`` dict is expected to have "model",
# "oof_y", "oof_prob".
# --------------------------------------------------------------------------
def bootstrap_metric_ci(y_true, y_prob, metric_fn, B: int, random_state: int):
    """Generic bootstrap 95% CI for an sklearn-style metric(y_true, y_prob) fn."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    values = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.min() == yt.max():
            continue  # degenerate resample, only one class present
        try:
            values.append(metric_fn(yt, yp))
        except ValueError:
            continue
    values = np.array(values)
    return values.mean(), np.percentile(values, 2.5), np.percentile(values, 97.5)


def plot_combined_roc(results, save_path, args):
    plt.figure(figsize=args.plots.figsize)
    for res in results:
        if res is None:
            continue
        fpr, tpr, _ = roc_curve(res["oof_y"], res["oof_prob"])
        mean_auc, lo, hi = bootstrap_metric_ci(
            res["oof_y"], res["oof_prob"], roc_auc_score,
            B=args.evaluation.bootstrap.B, random_state=args.evaluation.bootstrap.random_state,
        )
        plt.plot(fpr, tpr, lw=2, label=f'{res["model"]} (AUC={mean_auc:.2f} [{lo:.2f}\u2013{hi:.2f}])')
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=args.plots.axis_fontsize)
    plt.ylabel("True Positive Rate", fontsize=args.plots.axis_fontsize)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=args.plots.dpi, bbox_inches="tight")
    plt.close()


def plot_combined_pr(results, save_path, args):
    plt.figure(figsize=args.plots.figsize)
    for res in results:
        if res is None:
            continue
        y_true, y_prob = res["oof_y"], res["oof_prob"]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        mean_ap, lo, hi = bootstrap_metric_ci(
            y_true, y_prob, average_precision_score,
            B=args.evaluation.bootstrap.B, random_state=args.evaluation.bootstrap.random_state,
        )
        plt.plot(recall, precision, lw=2, label=f'{res["model"]} (AP={mean_ap:.2f} [{lo:.2f}\u2013{hi:.2f}])')
    plt.xlabel("Recall", fontsize=args.plots.axis_fontsize)
    plt.ylabel("Precision", fontsize=args.plots.axis_fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=args.plots.dpi, bbox_inches="tight")
    plt.close()


def plot_combined_calibration(results, save_path, args):
    n_bins = args.evaluation.ece.n_bins
    strategy = args.evaluation.ece.strategy

    plt.figure(figsize=args.plots.figsize)
    for res in results:
        if res is None:
            continue
        y_true = np.asarray(res["oof_y"]).ravel().astype(int)
        y_prob = np.asarray(res["oof_prob"]).ravel().astype(float)
        frac_pos, prob_mean = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
        ece = compute_ece(y_true, y_prob, n_bins=n_bins)
        plt.plot(prob_mean, frac_pos, markersize=4, marker="o", label=f'{res["model"]} (ECE={ece:.2f})')

    plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfect calibration")
    plt.xlabel("Predicted probability", fontsize=args.plots.axis_fontsize)
    plt.ylabel("Empirical probability", fontsize=args.plots.axis_fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=args.plots.dpi, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances_list, feature_names, k, save_path, dpi=DEFAULT_DPI):
    """Mean feature importance across CV folds, with a 95% CI errorbar."""
    if not importances_list:
        return

    imps = np.vstack(importances_list)  # (k folds, p features)
    imp_mean = imps.mean(axis=0)
    imp_std = imps.std(axis=0)
    half_width = (
        student_t.ppf(0.975, df=k - 1) * (imp_std / np.sqrt(k)) if k > 1 else np.zeros_like(imp_mean)
    )

    order = np.argsort(imp_mean)[::-1]
    feats_sorted = [feature_names[i] for i in order]
    mean_sorted = imp_mean[order]
    hw_sorted = half_width[order]

    plt.figure(figsize=(max(8, 0.35 * len(feats_sorted)), 5))
    x = np.arange(len(feats_sorted))
    plt.bar(x, mean_sorted, alpha=0.85)
    plt.errorbar(x, mean_sorted, yerr=hw_sorted, fmt="none", ecolor="black", capsize=3, lw=1)
    plt.xticks(x, feats_sorted, rotation=90)
    plt.ylabel("Importance score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# SHAP (XGBoost-only: the original supported a KernelExplainer branch for
# SVM, which is unreachable now that the non-XGBoost model code has been
# removed as dead code — see stages/train.py)
# --------------------------------------------------------------------------
def plot_shap_summary(model, X, feature_names, out_dir: str, dpi: int = DEFAULT_DPI, axis_fontsize: int = 12):
    """Beeswarm + bar SHAP summary plots for a tree model, plus a ranked table."""
    import shap  # local import: only stages/train.py (which fits models) needs this

    os.makedirs(out_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X))

    if shap_values.ndim == 3:  # (n_samples, n_features, n_classes) -> positive class
        shap_values = shap_values[:, :, 1]
    elif shap_values.ndim != 2:
        raise ValueError(f"Unexpected shap_values shape: {shap_values.shape}")

    n_features = len(feature_names)

    for plot_type, filename in (("dot", "shap_summary_beeswarm.png"), ("bar", "shap_summary_bar.png")):
        plt.figure(figsize=(10, max(6, 0.25 * n_features)))
        shap.summary_plot(
            shap_values, X, feature_names=feature_names,
            plot_type=None if plot_type == "dot" else plot_type,
            max_display=n_features, show=False,
        )
        plt.xlabel("SHAP value" if plot_type == "dot" else "mean(|SHAP value|)", fontsize=axis_fontsize)
        plt.ylabel("Feature", fontsize=axis_fontsize)
        plt.xticks(fontsize=axis_fontsize)
        plt.yticks(fontsize=axis_fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=dpi, bbox_inches="tight")
        plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    df_shap = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df_shap = df_shap.sort_values("mean_abs_shap", ascending=False)
    df_shap.to_excel(os.path.join(out_dir, "shap_mean_abs.xlsx"), index=False)

    print(f"[OK] Saved SHAP plots (all {n_features} features, dpi={dpi})")
