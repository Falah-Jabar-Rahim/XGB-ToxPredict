"""Evaluate a trained XGBoost stage (M1 or M2) on its held-out test set.

Replaces M1/test.py and M2/test.py (also byte-for-byte identical files).
Run:

    python -m stages.test --config configs/m1.yaml
    python -m stages.test --config configs/m2.yaml
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from common.config import load_config
from common.data import build_xy, load_dataset_for_config, to_numpy_xy
from common.metrics import metrics_at_threshold
from common.plotting import plot_combined_calibration, plot_combined_pr, plot_combined_roc, plot_confusion_matrix


def resolve_model_and_threshold(args) -> tuple[str, float]:
    """Pick the (possibly calibrated) model path + decision threshold.

    If calibration is enabled, use the "<name>_calibrated.joblib" model and
    the threshold stored in best_threshold.txt next to it; otherwise use the
    plain model path and the configured probability threshold.
    """
    model_path = args.evaluation.xgboost_model_path
    threshold = args.evaluation.prob_thr

    if not args.evaluation.do_calibration:
        return model_path, threshold

    model_dir = os.path.dirname(model_path)
    model_name, model_ext = os.path.splitext(os.path.basename(model_path))
    model_path = os.path.join(model_dir, f"{model_name}_calibrated{model_ext}")

    threshold_path = os.path.join(model_dir, "best_threshold.txt")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"[ERROR] Threshold file not found: {threshold_path}")
    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())

    print(f"[INFO] Using calibrated model: {model_path}")
    print(f"[INFO] Using calibrated threshold: {threshold:.4f}")
    return model_path, threshold


def evaluate(args, model_path, X_test, y_test, n_bins, threshold, output_dir, calibrated) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    model = load(model_path)
    y_prob = np.asarray(model.predict_proba(X_test)[:, 1], dtype=np.float64).ravel()
    y_test = np.asarray(y_test).ravel().astype(int)
    threshold = 0.5 if threshold is None else threshold

    m = metrics_at_threshold(y_test, y_prob, n_bins, threshold=threshold)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    metrics_row = {
        "fold": 1,
        "n_test": len(y_test),
        "pos_rate_test": np.mean(y_test),
        "AUROC": roc_auc_score(y_test, y_prob),
        "AUPRC": average_precision_score(y_test, y_prob),
        "threshold_used": threshold,
        "calibrated": calibrated,
        **{k: m[k] for k in ("Accuracy", "Precision", "Recall", "F1", "Specificity", "PPV", "NPV", "ECE")},
        "Brier": m["Brier"],
        "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
    }
    metrics_df = pd.DataFrame([metrics_row])
    predictions_df = pd.DataFrame({"TrueLabel": y_test, "PredProb": y_prob, "PredLabel": m["y_pred"]})

    plot_confusion_matrix(
        m["cm"], save_path=os.path.join(output_dir, "XGB_confusion_matrix.png"),
        class_names=args.plots.class_names, figsize=args.plots.figsize, dpi=args.plots.dpi,
        axis_fontsize=args.plots.axis_fontsize,
    )

    xlsx_path = os.path.join(output_dir, "XGB_Metrics_and_Preds.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)

    return {
        "model": "XGBoost", "oof_y": y_test, "oof_prob": y_prob,
        "metrics_table": metrics_df, "predictions": predictions_df,
        "fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single XGBoost stage on its test set.")
    parser.add_argument("--config_path", "--config", dest="config_path", type=str, required=True)
    cli_args = parser.parse_args()

    args = load_config(cli_args.config_path, extra_required=("models", "training"))

    output_dir = args.experiment.prediction_output_dir.replace("train", "test")
    os.makedirs(output_dir, exist_ok=True)
    n_bins = args.evaluation.ece.n_bins

    model_path, threshold = resolve_model_and_threshold(args)

    test_data = load_dataset_for_config(args.evaluation.test_dataset_path, args)
    X_test, y_test = build_xy(df_raw=test_data, id_col=args.data.id_col, target_col=args.data.target_col)
    X_test, y_test = to_numpy_xy(X_test, y_test)
    print(f"[INFO] X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")

    result = evaluate(
        args, model_path, X_test, y_test, n_bins, threshold, output_dir,
        calibrated=args.evaluation.do_calibration,
    )
    results = [result]

    print("[INFO] Generating evaluation plots...")
    roc_path = os.path.join(output_dir, "ROC_XGB.png")
    pr_path = os.path.join(output_dir, "PR_XGB.png")
    calibration_path = os.path.join(output_dir, "Calibration_XGB.png")

    plot_combined_roc(results, roc_path, args)
    plot_combined_pr(results, pr_path, args)
    plot_combined_calibration(results, calibration_path, args)

    print("\n[OK] Test evaluation completed.")
    print(f" - ROC:         {roc_path}")
    print(f" - PR:          {pr_path}")
    print(f" - Calibration: {calibration_path}")


if __name__ == "__main__":
    main()
