"""Hierarchical two-stage toxicity screen: M1 (any toxicity) -> M2 (severe
toxicity), matching the flowchart:

    Patients -> M1: any toxicity (Grade>0)? --No--> Negative (Grade<3)
                        |Yes
                        v
                M2: severe toxicity (Grade>=3)? --No--> Negative (Grade<3)
                        |Yes
                        v
                    Positive (Grade>=3)

M1 and M2 must already be trained (see stages/train.py) before running this.
Replaces the old root-level test.py, which duplicated most of its data
loading and plotting code from M1/util and M2/util verbatim.

Run:
    python -m hierarchical.predict --config configs/hierarchical.yaml
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix

from common.config import load_config
from common.data import build_xy, load_dataset_for_config
from common.metrics import compute_cascade_metrics, compute_ece
from common.plotting import plot_calibration_curve, plot_confusion_matrix, plot_pr_curve, plot_roc_curve


def predict_proba_positive(model_path: str, X: np.ndarray) -> np.ndarray:
    """Load a saved XGBoost model and return P(class=1) for each row."""
    model = load(model_path)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64).ravel()


def run_cascade(prob_m1: np.ndarray, prob_m2: np.ndarray, threshold: float):
    """Apply the M1 -> M2 decision cascade from the flowchart.

    - M1 predicts "no toxicity" (0) -> final prediction is negative,
      reported probability is M1's (there is nothing for M2 to refine).
    - M1 predicts "toxicity present" (1) -> defer to M2's severe-toxicity
      call for both the final label and the final probability.
    """
    y_pred_m1 = (prob_m1 >= threshold).astype(int)
    y_pred_m2 = (prob_m2 >= threshold).astype(int)

    y_pred_final = np.where(y_pred_m1 == 0, 0, y_pred_m2)
    prob_final = np.where(y_pred_m1 == 0, prob_m1, prob_m2)

    return y_pred_final, prob_final, y_pred_m1, y_pred_m2


def build_agreement_table(y_true, y_pred_m1, y_pred_m2) -> pd.DataFrame:
    """Where do M1 and M2's calls agree/disagree, broken down by ground truth?"""
    is_pos, is_neg = (y_true == 1), (y_true == 0)
    disagree = y_pred_m1 != y_pred_m2

    n_total = len(y_true)
    n_pos_total = int(np.sum(is_pos))
    n_neg_total = int(np.sum(is_neg))

    counts = {
        "GT=1 and M1=1 and M2=1": int(np.sum(is_pos & (y_pred_m1 == 1) & (y_pred_m2 == 1))),
        "GT=0 and M1=0 and M2=0": int(np.sum(is_neg & (y_pred_m1 == 0) & (y_pred_m2 == 0))),
        "GT=1 and models disagree": int(np.sum(is_pos & disagree)),
        "GT=0 and models disagree": int(np.sum(is_neg & disagree)),
        "GT=1 total": n_pos_total,
        "GT=0 total": n_neg_total,
        "Total samples": n_total,
    }
    denom_of_total = {k: n_total for k in counts}
    denom_within_gt = {
        "GT=1 and M1=1 and M2=1": n_pos_total, "GT=0 and M1=0 and M2=0": n_neg_total,
        "GT=1 and models disagree": n_pos_total, "GT=0 and models disagree": n_neg_total,
        "GT=1 total": n_pos_total, "GT=0 total": n_neg_total, "Total samples": None,
    }

    rows = []
    for metric, count in counts.items():
        within_gt_denom = denom_within_gt[metric]
        rows.append({
            "Metric": metric,
            "Count": count,
            "Percent_of_total": round(count / denom_of_total[metric], 4),
            "Percent_within_GT": round(count / within_gt_denom, 4) if within_gt_denom else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run the M1->M2 hierarchical toxicity cascade.")
    parser.add_argument("--config_path", "--config", dest="config_path", type=str, required=True)
    cli_args = parser.parse_args()

    args = load_config(cli_args.config_path)

    output_dir = args.experiment.prediction_output_dir
    os.makedirs(output_dir, exist_ok=True)
    threshold = args.evaluation.prob_thr

    # ---- Load data. NOTE: args.data.target_col must be the FINAL label
    # (0 = Grade<3, 1 = Grade>=3), not the M1 "any toxicity" label. ----
    data = load_dataset_for_config(args.evaluation.test_dataset_path, args)
    X, y, sample_id = build_xy(df_raw=data, id_col=args.data.id_col, target_col=args.data.target_col, return_id=True)
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_np = np.asarray(y).astype(int).ravel()

    prob_m1 = predict_proba_positive(args.evaluation.xgboost_model_path_m1, X_np)
    prob_m2 = predict_proba_positive(args.evaluation.xgboost_model_path_m2, X_np)
    y_pred_final, prob_final, y_pred_m1, y_pred_m2 = run_cascade(prob_m1, prob_m2, threshold)

    metrics = compute_cascade_metrics(
        y_np, prob_final, prob_m1, y_pred_m1, prob_m2, y_pred_m2, sample_id, threshold=threshold
    )

    plot_confusion_matrix(
        confusion_matrix(y_np, y_pred_final, labels=[0, 1]),
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
        class_names=args.plots.class_names, figsize=args.plots.figsize, dpi=args.plots.dpi,
        axis_fontsize=args.plots.axis_fontsize,
    )

    try:
        metrics["ECE"] = compute_ece(y_np, prob_final, n_bins=args.evaluation.ece.n_bins)
    except Exception as e:  # pragma: no cover - defensive, matches original behavior
        print(f"[WARN] ECE computation failed: {e}")

    df_predictions = pd.DataFrame({
        "ID": sample_id, "y_true": metrics["y_true"], "y_pred": metrics["y_pred"], "y_prob": metrics["y_prob_pred"],
        "y_prob_pred_m1": metrics["y_prob_pred_m1"], "y_pred_m1": metrics["y_pred_m1"],
        "y_prob_pred_m2": metrics["y_prob_pred_m2"], "y_pred_m2": metrics["y_pred_m2"],
    })
    array_valued_keys = {"ID", "y_true", "y_pred", "y_prob_pred", "y_prob_pred_m1", "y_prob_pred_m2",
                          "y_pred_m1", "y_pred_m2"}
    df_summary = pd.DataFrame([{k: v for k, v in metrics.items() if k not in array_valued_keys}])
    df_agreement = build_agreement_table(y_np, y_pred_m1, y_pred_m2)

    excel_path = os.path.join(output_dir, "metrics.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_predictions.to_excel(writer, sheet_name="Predictions", index=False)
        df_agreement.to_excel(writer, sheet_name="Agreement_Analysis", index=False)

    plot_roc_curve(y_np, prob_final, os.path.join(output_dir, "roc_hierarchical.png"))
    plot_pr_curve(y_np, prob_final, os.path.join(output_dir, "pr_hierarchical.png"))
    plot_calibration_curve(
        y_np, prob_final, os.path.join(output_dir, "calibration_hierarchical.png"),
        n_bins=args.evaluation.ece.n_bins, ece=metrics.get("ECE"),
    )

    print("\n=== Hierarchical Model Metrics ===")
    for k, v in metrics.items():
        if k not in array_valued_keys:
            print(f"{k}: {v}")
    print(f"\n[OK] Results written to: {output_dir}")


if __name__ == "__main__":
    main()
