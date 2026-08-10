import os
import shap
import pandas as pd
import numpy as np
from joblib import dump
from collections import Counter
from util.metrics import compute_ece
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from util.visualize import plot_conf_mat, feature_importance, shap_plot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve,
    roc_curve
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


def RF_model(input_data, args):
    # make a subfolder to save results
    subfolder = os.path.join(args.experiment.prediction_output_dir, "RF")
    os.makedirs(subfolder, exist_ok=True)

    n_bins = args.evaluation.ece.n_bins
    strategy = args.evaluation.ece.strategy

    X_all, y_all = build_xy(df_raw=input_data, id_col=args.data.id_col, target_col=args.data.target_col)

    # Save feature names before converting to numpy
    if isinstance(X_all, pd.DataFrame):
        feature_names = X_all.columns.tolist()
    else:
        feature_names = [f"feat_{i}" for i in range(X_all.shape[1])]

    # convert to numpy
    if isinstance(X_all, pd.DataFrame):
        X_all = X_all.to_numpy()
    if isinstance(y_all, (pd.Series, pd.DataFrame)):
        y_all = np.asarray(y_all).ravel()

    # CV procedure
    skf = StratifiedKFold(n_splits=args.training.cv.n_splits, shuffle=args.training.cv.shuffle,
                          random_state=args.experiment.random_seed)
    rows = []
    fold_id = 0
    # accumulator for confusion matrix over folds
    cm_accum = None
    # collectors for averaged curves & feature importance
    fprs, tprs, aucs = [], [], []
    recalls, precisions, aps = [], [], []
    calib_probs, calib_fracs, eces = [], [], []
    importances_list = []
    # collect pooled OOF (out of fold) predictions for bootstrap CIs
    oof_y = []
    oof_prob = []
    all_preds = []

    for tr_idx, te_idx in skf.split(X_all, y_all):
        fold_id += 1
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # RF Model
        rf = RandomForestClassifier(
            n_estimators=args.models.random_forest.params.n_estimators,
            max_depth=args.models.random_forest.params.max_depth,
            min_samples_leaf=args.models.random_forest.params.min_samples_leaf,
            max_features=args.models.random_forest.params.max_features,
            class_weight=args.models.random_forest.params.class_weight,
            n_jobs=args.models.random_forest.params.n_jobs,
            random_state=args.models.random_forest.params.random_state + fold_id
        )

        rf.fit(X_tr, y_tr)

        # store feature importance per fold
        if hasattr(rf, "feature_importances_"):
            importances_list.append(rf.feature_importances_)

        # Predictions
        prob_te = rf.predict_proba(X_te)[:, 1]
        m05 = metrics_at_05(y_te, prob_te, n_bins)
        # stash out-of-fold predictions
        oof_y.append(y_te)
        oof_prob.append(prob_te)

        # ensure 1-D
        y_te = np.asarray(y_te).ravel()
        prob_te = np.asarray(prob_te).ravel()

        y_pred = m05["y_pred"]
        all_preds.append(pd.DataFrame({
            "fold": fold_id,
            "TrueLabel": y_te,
            "PredLabel": y_pred,
            "PredProb": prob_te
        }))

        cm_fold = m05["cm"].astype(float)

        # Accumulate raw counts
        if cm_accum is None:
            cm_accum = cm_fold
        else:
            cm_accum += cm_fold

        # AUCs and curves
        auroc = roc_auc_score(y_te, prob_te)
        fpr, tpr, _ = roc_curve(y_te, prob_te)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auroc)

        prec, rec, _ = precision_recall_curve(y_te, prob_te)
        precisions.append(prec)
        recalls.append(rec)
        aps.append(average_precision_score(y_te, prob_te))

        frac_pos, prob_mean = calibration_curve(y_te, prob_te, n_bins=n_bins, strategy=strategy)
        calib_fracs.append(frac_pos)
        calib_probs.append(prob_mean)
        eces.append(compute_ece(y_te, prob_te, n_bins=n_bins))
        # Add metrics to results (kept as you had)
        rows.append({
            "fold": fold_id,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "pos_rate_test": float(y_te.mean()),
            "AUROC": auroc,
            "AUPRC": aps[-1],
            "Accuracy@0.5": m05["Accuracy"],
            "Precision@0.5": m05["Precision"],  # PPV
            "Recall@0.5": m05["Recall"],
            "F1@0.5": m05["F1"],
            "Specificity@0.5": m05["Specificity"],
            "PPV@0.5": m05["PPV"],
            "NPV@0.5": m05["NPV"],
            "ECE@0.5": m05["ECE"],
            "brier@0.5": m05["brier_rf"],
            "TN@0.5": m05["TN"], "FP@0.5": m05["FP"],
            "FN@0.5": m05["FN"], "TP@0.5": m05["TP"],

        })

    # save confusion matrix plot
    plot_conf_mat(
        cm_accum.astype(int),
        args, save_path=os.path.join(subfolder, "RF_confusion_matrix.png"))

    # pool OOF predictions across folds for bootstrap CIs
    oof_y = np.concatenate(oof_y, axis=0)
    oof_prob = np.concatenate(oof_prob, axis=0)
    # save feature importance plot: mean ± 95% CI
    feature_importance(importances_list, input_data, args, fold_id,
                       save_path=os.path.join(subfolder, "RF_feature_importance_avg.png"))

    # ===== Save fold table (mean/std at bottom) =====
    df_res = pd.DataFrame(rows)
    mean_row = df_res.mean(numeric_only=True).astype(object)
    mean_row["fold"] = "mean"
    std_row = df_res.std(numeric_only=True).astype(object)
    std_row["fold"] = "std"
    df_out = pd.concat([df_res, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)

    # ===== Save results and predictions =====
    df_preds = pd.concat(all_preds, ignore_index=True)
    out_path = os.path.join(subfolder, "RF_Metrics_and_Preds.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Metrics", index=False)
        df_preds.to_excel(writer, sheet_name="Predictions", index=False)
    print(f"[RF] Saved CV results and predictions to: {out_path}")

    # RETURN pooled OOF for combined plotting
    oof_y = _concat_1d(oof_y, dtype=int)
    oof_prob = _concat_1d(oof_prob, dtype=float)

    # ===== Train FINAL model on all data & save it =====
    print("Train FINAL RF model on all data & save it")

    # RF Model
    rf_final = RandomForestClassifier(
        n_estimators=args.models.random_forest.params.n_estimators,
        max_depth=args.models.random_forest.params.max_depth,
        min_samples_leaf=args.models.random_forest.params.min_samples_leaf,
        max_features=args.models.random_forest.params.max_features,
        class_weight=args.models.random_forest.params.class_weight,
        n_jobs=args.models.random_forest.params.n_jobs,
        random_state=args.models.random_forest.params.random_state
    )

    rf_final.fit(X_all, y_all)
    model_path = os.path.join(subfolder, "rf_final_model.joblib")
    dump(rf_final, model_path)
    print(f"[OK] Saved final rf model to: {model_path}")

    # shap features
    shap_plot(rf_final, X_all, feature_names, subfolder, args)

    # optional sanity checks
    assert oof_y.ndim == 1 and oof_prob.ndim == 1 and oof_y.shape[0] == oof_prob.shape[0], \
        f"Bad OOF shapes: y={oof_y.shape}, p={oof_prob.shape}"

    return {
        "model": "Random Forest",
        "oof_y": oof_y,
        "oof_prob": oof_prob,
        "metrics_table": df_out,
        "predictions": df_preds,
        "importances": np.vstack(importances_list) if len(importances_list) > 0 else None  # k x p
    }


import os
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from joblib import dump
import os
import numpy as np
import pandas as pd
from collections import Counter
from joblib import dump

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def _sigmoid(z):
    z = np.asarray(z)
    return 1.0 / (1.0 + np.exp(-z))



def plot_svm_decision_boundary_pca(model, X, y, save_path):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    svm_2d = SVC(C=model.C, kernel=model.kernel, gamma=model.gamma)
    svm_2d.fit(X_pca, y)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_2d.decision_function(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z > 0, alpha=0.2)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=40)

    plt.title("SVM Decision Boundary (PCA projection)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def SVM_model(input_data, args):
    subfolder = os.path.join(args.experiment.prediction_output_dir, "SVM")
    os.makedirs(subfolder, exist_ok=True)

    n_bins = args.evaluation.ece.n_bins
    strategy = args.evaluation.ece.strategy

    X_all, y_all = build_xy(df_raw=input_data, id_col=args.data.id_col, target_col=args.data.target_col)

    # Save feature names before converting to numpy
    if isinstance(X_all, pd.DataFrame):
        feature_names = X_all.columns.tolist()
        X_all = X_all.to_numpy()
    else:
        feature_names = [f"feat_{i}" for i in range(X_all.shape[1])]

    if isinstance(y_all, (pd.Series, pd.DataFrame)):
        y_all = np.asarray(y_all).ravel()

    skf = StratifiedKFold(
        n_splits=args.training.cv.n_splits,
        shuffle=args.training.cv.shuffle,
        random_state=args.experiment.random_seed
    )

    rows = []
    fold_id = 0
    cm_accum = None

    oof_y = []
    oof_prob = []
    all_preds = []

    for tr_idx, te_idx in skf.split(X_all, y_all):
        fold_id += 1
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # ---- imbalance handling (SVM equivalent of scale_pos_weight)
        if getattr(args.models.SVM, "class_weight", None) is None:
            neg = int((y_tr == 0).sum())
            pos = int((y_tr == 1).sum())
            w_pos = neg / max(pos, 1)
            class_weight = {0: 1.0, 1: float(w_pos)}
        else:
            class_weight = args.models.SVM.class_weight

        # ---- SVM with probabilities ON (Platt scaling inside SVC)
        svm = SVC(
            C=args.models.SVM.C,
            kernel=args.models.SVM.kernel,
            gamma=args.models.SVM.gamma,
            class_weight=class_weight,
            probability=False,
            random_state=(args.models.SVM.random_state + fold_id)
        )

        svm.fit(X_tr, y_tr)

        # Predictions (now real proba)
        scores_te = svm.decision_function(X_te)
        prob_te = _sigmoid(scores_te)  # your existing helper
        m05 = metrics_at_05(y_te, prob_te, n_bins)

        y_te_arr = np.asarray(y_te).ravel()
        prob_te_arr = np.asarray(prob_te).ravel()
        if y_te_arr.shape[0] != prob_te_arr.shape[0]:
            raise ValueError(
                f"Fold {fold_id}: y_te len {y_te_arr.shape[0]} != prob_te len {prob_te_arr.shape[0]}"
            )

        oof_y.append(y_te_arr)
        oof_prob.append(prob_te_arr)

        y_pred = m05["y_pred"]
        all_preds.append(pd.DataFrame({
            "fold": fold_id,
            "TrueLabel": y_te_arr,
            "PredLabel": y_pred,
            "PredProb": prob_te_arr
        }))

        cm_fold = m05["cm"].astype(float)
        cm_accum = cm_fold if cm_accum is None else (cm_accum + cm_fold)

        # Metrics
        auroc = roc_auc_score(y_te_arr, prob_te_arr)
        fpr, tpr, _ = roc_curve(y_te_arr, prob_te_arr)

        prec, rec, _ = precision_recall_curve(y_te_arr, prob_te_arr)
        ap = average_precision_score(y_te_arr, prob_te_arr)

        # Calibration curve + ECE (now meaningful because prob_te is calibrated-ish)
        frac_pos, prob_mean = calibration_curve(y_te_arr, prob_te_arr, n_bins=n_bins, strategy=strategy)

        rows.append({
            "fold": fold_id,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te_arr)),
            "pos_rate_test": float(y_te_arr.mean()),
            "AUROC": auroc,
            "AUPRC": ap,
            "Accuracy@0.5": m05["Accuracy"],
            "Precision@0.5": m05["Precision"],
            "Recall@0.5": m05["Recall"],
            "F1@0.5": m05["F1"],
            "Specificity@0.5": m05["Specificity"],
            "PPV@0.5": m05["PPV"],
            "NPV@0.5": m05["NPV"],
            "ECE@0.5": m05["ECE"],
            "brier@0.5": m05["brier_rf"],  # keep same key as your pipeline
            "TN@0.5": m05["TN"], "FP@0.5": m05["FP"],
            "FN@0.5": m05["FN"], "TP@0.5": m05["TP"],
        })

    # Confusion matrix plot
    plot_conf_mat(
        cm_accum.astype(int),
        args,
        save_path=os.path.join(subfolder, "SVM_confusion_matrix.png")
    )

    # Pool OOF
    oof_y = np.concatenate(oof_y, axis=0)
    oof_prob = np.concatenate(oof_prob, axis=0)

    # Save fold table (mean/std)
    df_res = pd.DataFrame(rows)
    mean_row = df_res.mean(numeric_only=True).astype(object);
    mean_row["fold"] = "mean"
    std_row = df_res.std(numeric_only=True).astype(object);
    std_row["fold"] = "std"
    df_out = pd.concat([df_res, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)

    # Save metrics + predictions
    out_path = os.path.join(subfolder, "SVM_Metrics_and_Preds.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Metrics", index=False)
        pd.concat(all_preds, ignore_index=True).to_excel(writer, sheet_name="Predictions", index=False)

    # RETURN pooled OOF for combined plotting
    oof_y = _concat_1d(oof_y, dtype=int)
    oof_prob = _concat_1d(oof_prob, dtype=float)

    # ===== Train FINAL model on all data & save it =====
    print("Train FINAL SVM model on all data & save it")

    class_counts = Counter(y_all)
    n_neg = class_counts.get(0, 0)
    n_pos = class_counts.get(1, 0)
    w_pos = n_neg / max(n_pos, 1)
    print("Class counts:", class_counts, " -> pos_weight =", w_pos)

    if getattr(args.models.SVM, "class_weight", None) is None:
        class_weight_final = {0: 1.0, 1: float(w_pos)}
    else:
        class_weight_final = args.models.SVM.class_weight

    svm_final = SVC(
        C=args.models.SVM.C,
        kernel=args.models.SVM.kernel,
        gamma=args.models.SVM.gamma,
        class_weight=class_weight_final,
        probability=False,
        random_state=getattr(args.models.SVM, "random_state", None)
    )

    svm_final.fit(X_all, y_all)
    model_path = os.path.join(subfolder, "svm_final_model.joblib")
    dump(svm_final, model_path)
    print(f"[OK] Saved final SVM model to: {model_path}")
    plot_svm_decision_boundary_pca(
        svm_final,
        X_all,
        y_all,
        os.path.join(subfolder, "svm_decision_boundary_pca.png")
    )
    # SHAP exactly like XGB call site
    shap_plot(svm_final, X_all, feature_names, subfolder, args)

    return {
        "model": "SVM",
        "oof_y": oof_y,
        "oof_prob": oof_prob,
        "metrics_table": df_out,
        "predictions": pd.concat(all_preds, ignore_index=True),
        "importances": None
    }


def LOGREG_model(input_data, args):

    subfolder = os.path.join(args.experiment.prediction_output_dir, "LogReg")
    os.makedirs(subfolder, exist_ok=True)

    n_bins = args.evaluation.ece.n_bins
    strategy = args.evaluation.ece.strategy

    X_all, y_all = build_xy(df_raw=input_data, id_col=args.data.id_col, target_col=args.data.target_col)

    # Save feature names before converting to numpy
    if isinstance(X_all, pd.DataFrame):
        feature_names = X_all.columns.tolist()
        X_all = X_all.to_numpy()
    else:
        feature_names = [f"feat_{i}" for i in range(X_all.shape[1])]

    if isinstance(y_all, (pd.Series, pd.DataFrame)):
        y_all = np.asarray(y_all).ravel()

    skf = StratifiedKFold(
        n_splits=args.training.cv.n_splits,
        shuffle=args.training.cv.shuffle,
        random_state=args.experiment.random_seed
    )

    rows = []
    fold_id = 0
    cm_accum = None

    oof_y = []
    oof_prob = []
    all_preds = []

    for tr_idx, te_idx in skf.split(X_all, y_all):
        fold_id += 1
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # ---- imbalance handling
        if getattr(args.models, "LOGREG", None) is not None and getattr(args.models.LOGREG, "class_weight",
                                                                        None) is not None:
            class_weight = args.models.LOGREG.class_weight
        else:
            neg = int((y_tr == 0).sum())
            pos = int((y_tr == 1).sum())
            w_pos = neg / max(pos, 1)
            class_weight = {0: 1.0, 1: float(w_pos)}

        # ---- Logistic Regression (input already normalized + one-hot)
        # ---- imbalance handling
        logreg_cfg = getattr(args.models, "LOGREG", None)

        # fallback defaults if LOGREG not present
        C = getattr(logreg_cfg, "C", 1.0)
        penalty = getattr(logreg_cfg, "penalty", "l2")
        solver = getattr(logreg_cfg, "solver", "liblinear")
        max_iter = getattr(logreg_cfg, "max_iter", 2000)
        tol = getattr(logreg_cfg, "tol", 1e-4)
        fit_intercept = getattr(logreg_cfg, "fit_intercept", True)
        warm_start = getattr(logreg_cfg, "warm_start", False)
        n_jobs = getattr(logreg_cfg, "n_jobs", None)
        l1_ratio = getattr(logreg_cfg, "l1_ratio", None)
        rs0 = getattr(logreg_cfg, "random_state", 42)

        logreg = LogisticRegression(
            C=float(C),
            penalty=str(penalty),
            solver=str(solver),
            class_weight=class_weight,  # you compute this per-fold or use "balanced"
            max_iter=int(max_iter),
            tol=float(tol),
            fit_intercept=bool(fit_intercept),
            warm_start=bool(warm_start),
            n_jobs=None if n_jobs is None else int(n_jobs),
            l1_ratio=None if l1_ratio is None else float(l1_ratio),
            random_state=int(rs0) + fold_id,
        )

        logreg.fit(X_tr, y_tr)

        # Probabilities
        prob_te = logreg.predict_proba(X_te)[:, 1]

        # your pipeline threshold metrics @0.5 (or whatever metrics_at_05 does)
        m05 = metrics_at_05(y_te, prob_te, n_bins)

        y_te_arr = np.asarray(y_te).ravel()
        prob_te_arr = np.asarray(prob_te).ravel()
        if y_te_arr.shape[0] != prob_te_arr.shape[0]:
            raise ValueError(
                f"Fold {fold_id}: y_te len {y_te_arr.shape[0]} != prob_te len {prob_te_arr.shape[0]}"
            )

        oof_y.append(y_te_arr)
        oof_prob.append(prob_te_arr)

        y_pred = m05["y_pred"]
        all_preds.append(pd.DataFrame({
            "fold": fold_id,
            "TrueLabel": y_te_arr,
            "PredLabel": y_pred,
            "PredProb": prob_te_arr
        }))

        cm_fold = m05["cm"].astype(float)
        cm_accum = cm_fold if cm_accum is None else (cm_accum + cm_fold)

        # Metrics
        auroc = roc_auc_score(y_te_arr, prob_te_arr)
        fpr, tpr, _ = roc_curve(y_te_arr, prob_te_arr)

        prec, rec, _ = precision_recall_curve(y_te_arr, prob_te_arr)
        ap = average_precision_score(y_te_arr, prob_te_arr)

        # Calibration curve + ECE (if you use it elsewhere)
        frac_pos, prob_mean = calibration_curve(y_te_arr, prob_te_arr, n_bins=n_bins, strategy=strategy)

        rows.append({
            "fold": fold_id,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te_arr)),
            "pos_rate_test": float(y_te_arr.mean()),
            "AUROC": auroc,
            "AUPRC": ap,
            "Accuracy@0.5": m05["Accuracy"],
            "Precision@0.5": m05["Precision"],
            "Recall@0.5": m05["Recall"],
            "F1@0.5": m05["F1"],
            "Specificity@0.5": m05["Specificity"],
            "PPV@0.5": m05["PPV"],
            "NPV@0.5": m05["NPV"],
            "ECE@0.5": m05["ECE"],
            "brier@0.5": m05["brier_rf"],  # keep same key as your pipeline
            "TN@0.5": m05["TN"], "FP@0.5": m05["FP"],
            "FN@0.5": m05["FN"], "TP@0.5": m05["TP"],
        })

    # Confusion matrix plot
    plot_conf_mat(
        cm_accum.astype(int),
        args,
        save_path=os.path.join(subfolder, "LogReg_confusion_matrix.png")
    )

    # Pool OOF
    oof_y = np.concatenate(oof_y, axis=0)
    oof_prob = np.concatenate(oof_prob, axis=0)

    # Save fold table (mean/std)
    df_res = pd.DataFrame(rows)
    mean_row = df_res.mean(numeric_only=True).astype(object)
    mean_row["fold"] = "mean"
    std_row = df_res.std(numeric_only=True).astype(object)
    std_row["fold"] = "std"
    df_out = pd.concat([df_res, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)

    # Save metrics + predictions
    out_path = os.path.join(subfolder, "LogReg_Metrics_and_Preds.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Metrics", index=False)
        pd.concat(all_preds, ignore_index=True).to_excel(writer, sheet_name="Predictions", index=False)

    # RETURN pooled OOF for combined plotting
    oof_y = _concat_1d(oof_y, dtype=int)
    oof_prob = _concat_1d(oof_prob, dtype=float)

    # ===== Train FINAL model on all data & save it =====
    print("Train FINAL Logistic Regression model on all data & save it")

    class_counts = Counter(y_all)
    n_neg = class_counts.get(0, 0)
    n_pos = class_counts.get(1, 0)
    w_pos = n_neg / max(n_pos, 1)
    print("Class counts:", class_counts, " -> pos_weight =", w_pos)

    if getattr(args.models, "LOGREG", None) is not None and getattr(args.models.LOGREG, "class_weight",
                                                                    None) is not None:
        class_weight_final = args.models.LOGREG.class_weight
    else:
        class_weight_final = {0: 1.0, 1: float(w_pos)}

    # fallback defaults if LOGREG not present
    C = getattr(logreg_cfg, "C", 1.0)
    penalty = getattr(logreg_cfg, "penalty", "l2")
    solver = getattr(logreg_cfg, "solver", "liblinear")
    max_iter = getattr(logreg_cfg, "max_iter", 2000)
    tol = getattr(logreg_cfg, "tol", 1e-4)
    fit_intercept = getattr(logreg_cfg, "fit_intercept", True)
    warm_start = getattr(logreg_cfg, "warm_start", False)
    n_jobs = getattr(logreg_cfg, "n_jobs", None)
    l1_ratio = getattr(logreg_cfg, "l1_ratio", None)
    rs0 = getattr(logreg_cfg, "random_state", 42)

    logreg_final = LogisticRegression(
            C=float(C),
            penalty=str(penalty),
            solver=str(solver),
            class_weight=class_weight_final,  # you compute this per-fold or use "balanced"
            max_iter=int(max_iter),
            tol=float(tol),
            fit_intercept=bool(fit_intercept),
            warm_start=bool(warm_start),
            n_jobs=None if n_jobs is None else int(n_jobs),
            l1_ratio=None if l1_ratio is None else float(l1_ratio),
            random_state=int(rs0) + fold_id,
        )
    logreg_final.fit(X_all, y_all)

    plot_logreg_decision_boundary_pca(
        logreg_final,
        X_all,
        y_all,
        os.path.join(subfolder, "logreg_decision_boundary_pca.png")
    )

    model_path = os.path.join(subfolder, "logreg_final_model.joblib")
    dump(logreg_final, model_path)
    print(f"[OK] Saved final LogReg model to: {model_path}")

    # SHAP (your existing call site)
    shap_plot(logreg_final, X_all, feature_names, subfolder, args)

    return {
        "model": "LogReg",
        "oof_y": oof_y,
        "oof_prob": oof_prob,
        "metrics_table": df_out,
        "predictions": pd.concat(all_preds, ignore_index=True),
        "importances": None
    }


def plot_logreg_decision_boundary_pca(model, X, y, save_path):
    """
    Plot 2D PCA projection of Logistic Regression decision boundary.
    """

    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Refit logistic regression in PCA space
    clf_2d = LogisticRegression(
        C=model.C,
        penalty=model.penalty,
        solver=model.solver,
        class_weight=model.class_weight,
        max_iter=model.max_iter,
        tol=model.tol,
        fit_intercept=model.fit_intercept,
        l1_ratio=model.l1_ratio,
        random_state=42,
    )
    clf_2d.fit(X_pca, y)

    # Create grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf_2d.predict_proba(grid)[:, 1]
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    # probability surface
    plt.contourf(xx, yy, Z, levels=25, alpha=0.35)

    # decision boundary at 0.5
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=2)

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="k",
        s=40
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Logistic Regression Decision Boundary (PCA)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def XGB_model(input_data, args):
    subfolder = os.path.join(args.experiment.prediction_output_dir, "XGB")
    os.makedirs(subfolder, exist_ok=True)

    n_bins = args.evaluation.ece.n_bins
    strategy = args.evaluation.ece.strategy

    X_all, y_all = build_xy(df_raw=input_data, id_col=args.data.id_col, target_col=args.data.target_col)

    # Save feature names before converting to numpy
    if isinstance(X_all, pd.DataFrame):
        feature_names = X_all.columns.tolist()
    else:
        feature_names = [f"feat_{i}" for i in range(X_all.shape[1])]
    # convert to numpy
    if isinstance(X_all, pd.DataFrame):
        X_all = X_all.to_numpy()
    if isinstance(y_all, (pd.Series, pd.DataFrame)):
        y_all = np.asarray(y_all).ravel()

    # CV procedure
    skf = StratifiedKFold(n_splits=args.training.cv.n_splits, shuffle=args.training.cv.shuffle,
                          random_state=args.experiment.random_seed)
    rows = []
    fold_id = 0
    # accumulator for confusion matrix over folds
    cm_accum = None
    # collectors for averaged curves & feature importance
    fprs, tprs, aucs = [], [], []
    recalls, precisions, aps = [], [], []
    calib_probs, calib_fracs, eces = [], [], []
    importances_list = []
    # collect pooled OOF predictions for bootstrap CIs
    oof_y = []
    oof_prob = []
    all_preds = []
    for tr_idx, te_idx in skf.split(X_all, y_all):
        fold_id += 1
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        # XGBoost Model
        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()
        spw = neg / max(pos, 1)

        xgb = XGBClassifier(
            n_estimators=args.models.xgboost.params.n_estimators,
            max_depth=args.models.xgboost.params.max_depth,
            learning_rate=args.models.xgboost.params.learning_rate,
            subsample=args.models.xgboost.params.subsample,
            colsample_bytree=args.models.xgboost.params.colsample_bytree,
            eval_metric=args.models.xgboost.params.eval_metric,
            n_jobs=args.models.xgboost.params.n_jobs,
            min_child_weight=args.models.xgboost.params.min_child_weight,
            reg_lambda=args.models.xgboost.params.reg_lambda,
            reg_alpha=args.models.xgboost.params.reg_alpha,
            gamma=args.models.xgboost.params.gamma,
            scale_pos_weight=spw,
            random_state=args.models.xgboost.params.random_state + fold_id,
        )
        xgb.fit(X_tr, y_tr)

        # store feature importance per fold
        # if hasattr(xgb, "feature_importances_"):
        #   importances_list.append(xgb.feature_importances_)

        # Predictions
        prob_te = xgb.predict_proba(X_te)[:, 1]
        m05 = metrics_at_05(y_te, prob_te, n_bins)
        # force 1-D + sanity checks ---
        y_te_arr = np.asarray(y_te).ravel()
        prob_te_arr = np.asarray(prob_te).ravel()
        if y_te_arr.shape[0] != prob_te_arr.shape[0]:
            raise ValueError(f"Fold {fold_id}: y_te len {y_te_arr.shape[0]} != prob_te len {prob_te_arr.shape[0]}")
        # stash OOF (out of fold)
        oof_y.append(y_te_arr)
        oof_prob.append(prob_te_arr)
        y_pred = m05["y_pred"]
        all_preds.append(pd.DataFrame({"fold": fold_id, "TrueLabel": y_te, "PredLabel": y_pred, "PredProb": prob_te}))

        cm_fold = m05["cm"].astype(float)

        # Accumulate raw counts
        if cm_accum is None:
            cm_accum = cm_fold
        else:
            cm_accum += cm_fold

        # AUCs and curves
        auroc = roc_auc_score(y_te, prob_te)
        fpr, tpr, _ = roc_curve(y_te, prob_te)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auroc)

        prec, rec, _ = precision_recall_curve(y_te, prob_te)
        precisions.append(prec)
        recalls.append(rec)
        aps.append(average_precision_score(y_te, prob_te))

        frac_pos, prob_mean = calibration_curve(y_te, prob_te, n_bins=n_bins, strategy=strategy)
        calib_fracs.append(frac_pos)
        calib_probs.append(prob_mean)
        eces.append(compute_ece(y_te, prob_te, n_bins=n_bins))
        # Add metrics to results
        rows.append({

            "fold": fold_id,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "pos_rate_test": float(y_te.mean()),
            "AUROC": auroc,
            "AUPRC": aps[-1],
            "Accuracy@0.5": m05["Accuracy"],
            "Precision@0.5": m05["Precision"],  # PPV
            "Recall@0.5": m05["Recall"],
            "F1@0.5": m05["F1"],
            "Specificity@0.5": m05["Specificity"],
            "PPV@0.5": m05["PPV"],
            "NPV@0.5": m05["NPV"],
            "ECE@0.5": m05["ECE"],
            "brier@0.5": m05["brier_rf"],
            "TN@0.5": m05["TN"], "FP@0.5": m05["FP"],
            "FN@0.5": m05["FN"], "TP@0.5": m05["TP"],
        })

    # save confusion matrix plot
    plot_conf_mat(
        cm_accum.astype(int),
        args, save_path=os.path.join(subfolder, "XGB_confusion_matrix.png"))

    # pool OOF predictions across folds for bootstrap CIs
    oof_y = np.concatenate(oof_y, axis=0)
    oof_prob = np.concatenate(oof_prob, axis=0)

    # save feature importance plot: mean ± 95% CI
    # feature_importance(importances_list, input_data, args, fold_id, save_path=os.path.join(subfolder, "XGB_feature_importance_avg.png"))

    # ===== Save fold table (mean/std at bottom) =====
    df_res = pd.DataFrame(rows)
    mean_row = df_res.mean(numeric_only=True).astype(object)
    mean_row["fold"] = "mean"
    std_row = df_res.std(numeric_only=True).astype(object)
    std_row["fold"] = "std"
    df_out = pd.concat([df_res, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)

    # save predictions
    out_path = os.path.join(subfolder, "XGB_Metrics_and_Preds.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Metrics", index=False)
        pd.concat(all_preds, ignore_index=True).to_excel(writer, sheet_name="Predictions", index=False)

    # RETURN pooled OOF for combined plotting
    oof_y = _concat_1d(oof_y, dtype=int)
    oof_prob = _concat_1d(oof_prob, dtype=float)

    # ===== Train FINAL model on all data & save it =====
    print("Train FINAL XGB model on all data & save it")
    # after X_all, y_all are defined:
    class_counts = Counter(y_all)
    n_neg = class_counts[0]
    n_pos = class_counts[1]
    scale_pos_weight = n_neg / n_pos
    print("Class counts:", class_counts, " -> scale_pos_weight =", scale_pos_weight)

    xgb_final = XGBClassifier(
        n_estimators=args.models.xgboost.params.n_estimators,
        max_depth=args.models.xgboost.params.max_depth,
        learning_rate=args.models.xgboost.params.learning_rate,
        subsample=args.models.xgboost.params.subsample,
        colsample_bytree=args.models.xgboost.params.colsample_bytree,
        eval_metric=args.models.xgboost.params.eval_metric,
        n_jobs=args.models.xgboost.params.n_jobs,
        min_child_weight=args.models.xgboost.params.min_child_weight,
        reg_lambda=args.models.xgboost.params.reg_lambda,
        reg_alpha=args.models.xgboost.params.reg_alpha,
        gamma=args.models.xgboost.params.gamma,
        random_state=args.models.xgboost.params.random_state,
        scale_pos_weight=scale_pos_weight,
    )
    xgb_final.fit(X_all, y_all)

    plot_xgb_decision_boundary_pca(
        xgb_final,
        X_all,
        y_all,
        os.path.join(subfolder, "xgb_decision_boundary_pca.png")
    )
    model_path = os.path.join(subfolder, "xgb_final_model.joblib")
    dump(xgb_final, model_path)
    print(f"[OK] Saved final XGB model to: {model_path}")

    # shap features
    shap_plot(xgb_final, X_all, feature_names, subfolder, args)

    return {
        "model": "XGBoost",
        "oof_y": oof_y,
        "oof_prob": oof_prob,
        "metrics_table": df_out,
        "predictions": pd.concat(all_preds, ignore_index=True),
        "importances": np.vstack(importances_list) if len(importances_list) > 0 else None  # k x p
    }


def plot_xgb_decision_boundary_pca(model, X, y, save_path):
    """
    Plot 2D PCA projection of XGBoost decision boundary.
    """

    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Refit XGB in PCA space
    xgb_2d = XGBClassifier(
        n_estimators=model.n_estimators,
        max_depth=model.max_depth,
        learning_rate=model.learning_rate,
        subsample=model.subsample,
        colsample_bytree=model.colsample_bytree,
        eval_metric=model.eval_metric,
        n_jobs=model.n_jobs,
        min_child_weight=model.min_child_weight,
        reg_lambda=model.reg_lambda,
        reg_alpha=model.reg_alpha,
        gamma=model.gamma,
        scale_pos_weight=model.scale_pos_weight,
        random_state=42,
    )

    xgb_2d.fit(X_pca, y)

    # Create grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = xgb_2d.predict_proba(grid)[:, 1]
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    # probability surface
    plt.contourf(xx, yy, Z, levels=30, alpha=0.35)

    # decision boundary at p=0.5
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=2)

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="k",
        s=40
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("XGBoost Decision Boundary (PCA)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def build_xy(df_raw: pd.DataFrame, id_col: str, target_col: str):
    """
    Build feature matrix X and target vector y.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Input dataframe containing ID, features, and target.
    id_col : str
        Name of patient ID column (to be excluded).
    target_col : str
        Name of binary target column (0/1).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (ID and target removed).
    y : pd.Series
        Target vector.
    """

    if target_col not in df_raw.columns:
        raise ValueError(f"[ERROR] target_col '{target_col}' not found in dataframe")

    if id_col not in df_raw.columns:
        raise ValueError(f"[ERROR] id_col '{id_col}' not found in dataframe")

    # Target
    y = df_raw[target_col].astype(int)

    # Features (drop ID + target)
    X = df_raw.drop(columns=[id_col, target_col])

    return X, y


def metrics_at_05(y_true, y_prob, n_bins, thr=0.5):
    """
    Threshold probabilities at 0.5, then compute a rich set of metrics.
    Returns a dict with Accuracy, Precision (PPV), Recall (Sensitivity),
    F1, Specificity, PPV, NPV, and TN/FP/FN/TP.
    """
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm = confusion_matrix(y_true, y_pred)

    # Core metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)  # PPV
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ece = compute_ece(y_true, y_prob, n_bins=n_bins)

    brier_rf = brier_score_loss(y_true, y_prob)

    # Derived metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = prec
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "y_pred": y_pred,
        "cm": cm,
        "Accuracy": float(acc),
        "Precision": float(prec),  # same as PPV
        "Recall": float(rec),  # Sensitivity
        "F1": float(f1),
        "ECE": float(ece),
        "brier_rf": float(brier_rf),
        "Specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }


def _concat_1d(arr_list, dtype=None):
    # ensure it's really a list
    if arr_list is None or len(arr_list) == 0:
        return np.array([], dtype=dtype if dtype is not None else float)
    return np.concatenate([np.asarray(a).ravel() for a in arr_list], axis=0)
