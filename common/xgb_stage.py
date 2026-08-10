"""Train + cross-validate a single XGBoost stage (used for both M1 and M2 —
the two config files just point at different targets/datasets).

This is a cleaned-up version of the old ``util.util.XGB_model``. Modeling
behavior (CV scheme, hyperparameters, scale_pos_weight, final-fit-on-all-data)
is unchanged; what changed is organization:
  - no longer copy-pasted between M1/util/util.py and M2/util/util.py
  - dead RF/SVM/LogisticRegression training code (never called by any
    train.py/test.py in the repo) has been dropped
  - the "make a plot" and "compute a metric" concerns live in
    common/plotting.py and common/metrics.py rather than being inlined here
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from common.data import build_xy
from common.metrics import metrics_at_threshold
from common.plotting import plot_confusion_matrix, plot_shap_summary


def _make_xgb_classifier(params, random_state: int, scale_pos_weight: float) -> XGBClassifier:
    """Build an XGBClassifier from the config's `models.xgboost.params` block."""
    return XGBClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        eval_metric=params.eval_metric,
        n_jobs=params.n_jobs,
        min_child_weight=params.min_child_weight,
        reg_lambda=params.reg_lambda,
        reg_alpha=params.reg_alpha,
        gamma=params.gamma,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )


def train_xgb_with_cv(input_data: pd.DataFrame, args) -> dict:
    """Run stratified K-fold CV for reporting, then fit + save a final model
    on all the data. Returns pooled out-of-fold predictions plus the CV
    metrics table, matching the shape the plotting helpers expect.
    """
    output_dir = args.experiment.prediction_output_dir
    os.makedirs(output_dir, exist_ok=True)

    n_bins = args.evaluation.ece.n_bins
    xgb_params = args.models.xgboost.params

    X_all, y_all = build_xy(df_raw=input_data, id_col=args.data.id_col, target_col=args.data.target_col)
    feature_names = X_all.columns.tolist() if isinstance(X_all, pd.DataFrame) else [
        f"feat_{i}" for i in range(X_all.shape[1])
    ]
    X_all = X_all.to_numpy() if isinstance(X_all, pd.DataFrame) else np.asarray(X_all)
    y_all = np.asarray(y_all).ravel()

    skf = StratifiedKFold(
        n_splits=args.training.cv.n_splits,
        shuffle=args.training.cv.shuffle,
        random_state=args.experiment.random_seed,
    )

    fold_rows, all_preds = [], []
    cm_accum = None
    oof_y, oof_prob = [], []
    fold_id = 0

    for train_idx, test_idx in skf.split(X_all, y_all):
        fold_id += 1
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        scale_pos_weight = neg / max(pos, 1)

        model = _make_xgb_classifier(
            xgb_params, random_state=xgb_params.random_state + fold_id, scale_pos_weight=scale_pos_weight
        )
        model.fit(X_tr, y_tr)

        prob_te = model.predict_proba(X_te)[:, 1]
        fold_metrics = metrics_at_threshold(y_te, prob_te, n_bins)

        y_te = np.asarray(y_te).ravel()
        prob_te = np.asarray(prob_te).ravel()
        oof_y.append(y_te)
        oof_prob.append(prob_te)
        all_preds.append(pd.DataFrame({
            "fold": fold_id, "TrueLabel": y_te, "PredLabel": fold_metrics["y_pred"], "PredProb": prob_te,
        }))

        cm_accum = fold_metrics["cm"].astype(float) if cm_accum is None else cm_accum + fold_metrics["cm"]

        auroc = roc_auc_score(y_te, prob_te)
        ap = average_precision_score(y_te, prob_te)

        fold_rows.append({
            "fold": fold_id,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "pos_rate_test": float(y_te.mean()),
            "AUROC": auroc,
            "AUPRC": ap,
            "Accuracy@0.5": fold_metrics["Accuracy"],
            "Precision@0.5": fold_metrics["Precision"],
            "Recall@0.5": fold_metrics["Recall"],
            "F1@0.5": fold_metrics["F1"],
            "Specificity@0.5": fold_metrics["Specificity"],
            "PPV@0.5": fold_metrics["PPV"],
            "NPV@0.5": fold_metrics["NPV"],
            "ECE@0.5": fold_metrics["ECE"],
            "brier@0.5": fold_metrics["Brier"],
            "TN@0.5": fold_metrics["TN"], "FP@0.5": fold_metrics["FP"],
            "FN@0.5": fold_metrics["FN"], "TP@0.5": fold_metrics["TP"],
        })

    plot_confusion_matrix(
        cm_accum.astype(int),
        save_path=os.path.join(output_dir, "XGB_confusion_matrix.png"),
        class_names=args.plots.class_names, figsize=args.plots.figsize, dpi=args.plots.dpi,
        axis_fontsize=args.plots.axis_fontsize,
    )

    oof_y = np.concatenate(oof_y)
    oof_prob = np.concatenate(oof_prob)

    df_folds = pd.DataFrame(fold_rows)
    mean_row = df_folds.mean(numeric_only=True).astype(object)
    mean_row["fold"] = "mean"
    std_row = df_folds.std(numeric_only=True).astype(object)
    std_row["fold"] = "std"
    df_metrics_table = pd.concat([df_folds, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)

    out_path = os.path.join(output_dir, "XGB_Metrics_and_Preds.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_metrics_table.to_excel(writer, sheet_name="Metrics", index=False)
        pd.concat(all_preds, ignore_index=True).to_excel(writer, sheet_name="Predictions", index=False)

    # ---- Final model, trained on ALL data, is what gets deployed/saved ----
    print("[INFO] Training final XGBoost model on all data...")
    class_counts = Counter(y_all)
    scale_pos_weight_final = class_counts[0] / class_counts[1]
    print(f"[INFO] Class counts: {dict(class_counts)} -> scale_pos_weight={scale_pos_weight_final:.3f}")

    final_model = _make_xgb_classifier(
        xgb_params, random_state=xgb_params.random_state, scale_pos_weight=scale_pos_weight_final
    )
    final_model.fit(X_all, y_all)

    model_path = os.path.join(output_dir, "xgb_final_model.joblib")
    dump(final_model, model_path)
    print(f"[OK] Saved final XGB model to: {model_path}")

    plot_shap_summary(
        final_model, X_all, feature_names, output_dir,
        dpi=args.plots.dpi, axis_fontsize=args.plots.axis_fontsize,
    )

    return {
        "model": "XGBoost",
        "oof_y": oof_y,
        "oof_prob": oof_prob,
        "metrics_table": df_metrics_table,
        "predictions": pd.concat(all_preds, ignore_index=True),
    }
