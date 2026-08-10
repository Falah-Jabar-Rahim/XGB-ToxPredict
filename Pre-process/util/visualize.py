
import re
import os
import shap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t
from util.metrics import compute_ece
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score



def group_ohe_mean_abs(df_shap: pd.DataFrame, categorical_cols):
    """
    df_shap: columns = ["feature", "mean_abs_shap"] (per OHE column)
    categorical_cols: list of original categorical feature names, e.g. ["Sex", "Cancer Category", "IO Category"]

    Returns:
      df_grouped: grouped mean(|SHAP|) with OHE columns summed into their parent.
      used_map: dict(parent -> list of matched OHE columns)
    """
    if categorical_cols is None:
        categorical_cols = []

    categorical_cols = [str(c).strip() for c in categorical_cols if str(c).strip()]
    feats = df_shap["feature"].astype(str)

    grouped_rows = []
    used_cols = set()
    used_map = {}

    for parent in categorical_cols:
        # Most common pandas.get_dummies naming: "Parent_value"
        # Also allow "Parent=value" (some encoders) and "Parent__value"
        pat = re.compile(rf"^{re.escape(parent)}(_|=|__)", re.IGNORECASE)

        mask = feats.str.match(pat)
        cols = df_shap.loc[mask, "feature"].tolist()

        if len(cols) == 0:
            continue

        used_map[parent] = cols
        used_cols.update(cols)

        grouped_rows.append({
            "feature": parent,
            "mean_abs_shap": float(df_shap.loc[mask, "mean_abs_shap"].sum())
        })

    # keep non-OHE features as-is
    mask_rest = ~df_shap["feature"].isin(list(used_cols))
    rest = df_shap.loc[mask_rest, ["feature", "mean_abs_shap"]].copy()

    df_grouped = pd.concat([pd.DataFrame(grouped_rows), rest], ignore_index=True)
    df_grouped = df_grouped.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return df_grouped, used_map

def shap_plot(model, X, feature_names, out_dir, args, max_bg=200, nsamples=100):
    """
    SHAP plotting:
    - Shows ALL features
    - High DPI
    - Adjustable axis font size
    - Saves beeswarm + bar plots
    """

    os.makedirs(out_dir, exist_ok=True)

    dpi = getattr(args.plots, "dpi", 600)
    axis_fs = getattr(args.plots, "axis_fontsize", 12)

    # -----------------------------
    # Subsample for SHAP stability & speed
    # -----------------------------
    if X.shape[0] > max_bg:
        idx = np.random.choice(X.shape[0], max_bg, replace=False)
        X_bg = X[idx]
    else:
        X_bg = X

    # -----------------------------
    # Choose explainer
    # -----------------------------
    if "XGB" in model.__class__.__name__:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

    else:
        # SVM / non-tree models (EXPENSIVE)
        f = model.decision_function  # instead of predict_proba
        explainer = shap.KernelExplainer(
            f,
            X_bg
        )
        shap_values = explainer.shap_values(X_bg, nsamples=nsamples)

        # for binary classification → use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        X = X_bg  # align shapes

    # -----------------------------
    # Convert to array if needed
    # -----------------------------
    shap_values = np.asarray(shap_values)

    # --- Ensure SHAP values are 2D: (n_samples, n_features) for POSITIVE class
    if isinstance(shap_values, list):
        # list of [class0, class1]
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        # common case: (n, p, 2) for predict_proba outputs -> take positive class
        shap_values = shap_values[:, :, 1]
    elif shap_values.ndim != 2:
        raise ValueError(f"Unexpected shap_values shape: {shap_values.shape}")

    n_features = len(feature_names)

    # -----------------------------
    # Beeswarm plot (ALL features)
    # -----------------------------
    plt.figure(figsize=(10, max(6, 0.25 * n_features)))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=n_features,   # <<< ALL FEATURES
        show=False
    )

    plt.xlabel("SHAP value", fontsize=axis_fs)
    plt.ylabel("Feature", fontsize=axis_fs)
    plt.xticks(fontsize=axis_fs)
    plt.yticks(fontsize=axis_fs)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "shap_summary_beeswarm.png"),
        dpi=dpi,
        bbox_inches="tight"
    )
    plt.close()

    # -----------------------------
    # Bar plot (ALL features)
    # -----------------------------
    plt.figure(figsize=(10, max(6, 0.25 * n_features)))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        max_display=n_features,   # <<< ALL FEATURES
        show=False
    )

    plt.xlabel("mean(|SHAP value|)", fontsize=axis_fs)
    plt.ylabel("Feature", fontsize=axis_fs)
    plt.xticks(fontsize=axis_fs)
    plt.yticks(fontsize=axis_fs)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "shap_summary_bar.png"),
        dpi=dpi,
        bbox_inches="tight"
    )
    plt.close()

    # -----------------------------
    # Save mean(|SHAP|)
    # -----------------------------
    mean_abs = np.abs(shap_values).mean(axis=0)
    df_shap = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    # -----------------------------
    # Group OHE categoricals back to parent feature (SVM only)
    # -----------------------------
    cat_cols = []
    if hasattr(args, "data") and hasattr(args.data, "categorical_cols"):
        cat_cols = args.data.categorical_cols
    elif hasattr(args, "categorical_cols"):
        cat_cols = args.categorical_cols

    # Only group for non-tree models (i.e., SVM branch)
    if "XGB" not in model.__class__.__name__:
        df_grouped, used_map = group_ohe_mean_abs(df_shap, cat_cols)

        # Save grouped table
        df_grouped.to_excel(
            os.path.join(out_dir, "shap_mean_abs_GROUPED.xlsx"),
            index=False
        )

        # Optional: save mapping for debugging/audit
        map_path = os.path.join(out_dir, "shap_ohe_grouping_map.txt")
        with open(map_path, "w") as f:
            for k, v in used_map.items():
                f.write(f"{k}:\n")
                for col in v:
                    f.write(f"  - {col}\n")
                f.write("\n")

        # Grouped bar plot (comparable to XGB feature-level)
        n_g = len(df_grouped)
        plt.figure(figsize=(10, max(6, 0.25 * n_g)))
        plt.barh(df_grouped["feature"][::-1], df_grouped["mean_abs_shap"][::-1])
        plt.xlabel("mean(|SHAP value|)", fontsize=axis_fs)
        plt.ylabel("Feature", fontsize=axis_fs)
        plt.xticks(fontsize=axis_fs)
        plt.yticks(fontsize=axis_fs)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, "shap_summary_bar_GROUPED.png"),
            dpi=dpi,
            bbox_inches="tight"
        )
        plt.close()

        print(f"[OK] Saved grouped SHAP outputs: shap_mean_abs_GROUPED.xlsx + shap_summary_bar_GROUPED.png")


    df_shap.to_excel(
        os.path.join(out_dir, "shap_mean_abs.xlsx"),
        index=False
    )

    print(f"[OK] Saved SHAP plots (ALL features, dpi={dpi})")

def plot_conf_mat(cm, args, save_path=None):
    """
    Plot a confusion matrix (counts or averaged floats).
    - cm: 2x2 numpy array
    - class_names: labels for [0, 1]
    - title: plot title
    - save_path: if provided, saves PNG
    """
    class_names = args.plots.class_names
    figsize = args.plots.figsize
    dpi = args.plots.dpi
    axis_fontsize = args.plots.axis_fontsize

    cm = np.array(cm)
    fmt = ".2f" if cm.dtype.kind == "f" else "d"

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues", cbar=False, square=True,
        xticklabels=class_names, yticklabels=class_names
    )
    ax.set_xlabel("Predicted", fontsize=axis_fontsize)
    ax.set_ylabel("Actual", fontsize=axis_fontsize)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    # plt.show()


def feature_importance(importances_list, input_data, args, k, save_path=None):
    dpi = args.plots.dpi
    if len(importances_list) > 0:
        imps = np.vstack(importances_list)  # k x p
        imp_mean = imps.mean(axis=0)
        imp_std = imps.std(axis=0)
        # vector half-width per feature
        imp_hw = t.ppf(0.975, df=k - 1) * (imp_std / np.sqrt(k)) if k > 1 else np.zeros_like(imp_mean)

        feat_names = input_data.columns[1:-1].tolist()
        order = np.argsort(imp_mean)[::-1]
        feats_sorted = [feat_names[i] for i in order]
        mean_sorted = imp_mean[order]
        hw_sorted = imp_hw[order]

        plt.figure(figsize=(max(8, 0.35 * len(feats_sorted)), 5))
        x = np.arange(len(feats_sorted))
        plt.bar(x, mean_sorted, alpha=0.85)
        plt.errorbar(x, mean_sorted, yerr=hw_sorted, fmt='none', ecolor='black', capsize=3, lw=1)
        plt.xticks(x, feats_sorted, rotation=90)
        plt.ylabel("Importance scores")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()


def plot_combined_roc(results, save_path, args):
    figsize = args.plots.figsize
    dpi = args.plots.dpi
    axis_fontsize = args.plots.axis_fontsize
    plt.figure(figsize=figsize)
    for res in results:

        if res is not None:
            fpr, tpr, _ = roc_curve(res["oof_y"], res["oof_prob"])
            mean_auc, lo, hi = bootstrap_auc_ci(res["oof_y"], res["oof_prob"], args)
            plt.plot(
                fpr, tpr, lw=2,
                label=f'{res["model"]} (AUC={mean_auc:.2f} [{lo:.2f}–{hi:.2f}])'
            )
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=axis_fontsize)
    plt.ylabel("True Positive Rate", fontsize=axis_fontsize)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def bootstrap_auc_ci(y_true, y_prob, args):
    """Bootstrap 95% CI for AUROC."""
    b = args.evaluation.bootstrap.B
    seed = args.evaluation.bootstrap.random_state
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(b):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        try:
            aucs.append(roc_auc_score(yt, yp))
        except ValueError:
            continue
    aucs = np.array(aucs)
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def plot_combined_pr(results, save_path, args):
    figsize = args.plots.figsize
    dpi = args.plots.dpi
    axis_fontsize = args.plots.axis_fontsize

    plt.figure(figsize=figsize)
    # prevalence baseline (same y across models)
    prev = float(np.mean(results[0]["oof_y"]))
    for res in results:

        if res is not None:
            y_true, y_prob = res["oof_y"], res["oof_prob"]
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap_mean, ap_lo, ap_hi = bootstrap_ap_ci(y_true, y_prob, args)
            plt.plot(rec, prec, lw=2,
                     label=f'{res["model"]} (AP={ap_mean:.2f} [{ap_lo:.2f}–{ap_hi:.2f}])')
    # plt.hlines(prev, 0, 1, linestyles="--", linewidth=1, label=f"Prevalence={prev:.2f}")
    plt.xlabel("Recall", fontsize=axis_fontsize)
    plt.ylabel("Precision", fontsize=axis_fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def bootstrap_ap_ci(y_true, y_prob, args):
    """Bootstrap 95% CI for Average Precision (area under PR)."""


    b = args.evaluation.bootstrap.B
    seed = args.evaluation.bootstrap.random_state
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    aps = []
    for _ in range(b):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        # skip degenerate resamples with a single class
        if yt.min() == yt.max():
            continue
        try:
            aps.append(average_precision_score(yt, yp))
        except ValueError:
            continue
    aps = np.array(aps)
    return aps.mean(), np.percentile(aps, 2.5), np.percentile(aps, 97.5)


def plot_combined_calibration(results, save_path, args):
    figsize = args.plots.figsize
    dpi = args.plots.dpi
    axis_fontsize = args.plots.axis_fontsize
    n_bins = args.evaluation.ece.n_bins
    strategy = args.evaluation.ece.strategy
    plt.figure(figsize=figsize)
    for res in results:
        if res is not None:

            y_true = np.asarray(res["oof_y"]).ravel().astype(int)
            y_prob = np.asarray(res["oof_prob"]).ravel().astype(float)

            frac_pos, prob_mean = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
            ece = compute_ece(y_true, y_prob, n_bins=n_bins)

            # curve with ECE in legend
            plt.plot(prob_mean, frac_pos, markersize=4, marker="o",
                     label=f'{res["model"]} (ECE={ece:.2f})')

    # perfect calibration line with legend
    plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfect calibration")

    plt.xlabel("Predicted probability", fontsize=axis_fontsize)
    plt.ylabel("Empirical probability", fontsize=axis_fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')  # optional: square grid for clarity
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

def compute_skewness(df: pd.DataFrame, output_dir: str, exclude_cols=None) -> pd.DataFrame:
    """1) Compute skewness for numeric features and save to Excel."""
    exclude_cols = set(exclude_cols or [])
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude_cols]

    skew_df = (
        df[num_cols]
        .skew(numeric_only=True)
        .sort_values(ascending=False)
        .rename("skewness")
        .to_frame()
    )

    out_path = os.path.join(output_dir, "skewness.xlsx")
    skew_df.to_excel(out_path)
    print(f"[INFO] Saved skewness table to: {out_path}")
    return skew_df


def flag_outliers_percentile(
    df: pd.DataFrame,
    output_dir: str,
    exclude_cols=None,
    upper_q: float = 0.99,
    lower_q: float = 0.01
) -> pd.DataFrame:
    """
    2) Flag outliers using percentile thresholds (default 1% and 99%).
    Saves a tidy table of outlier counts + thresholds.
    """
    exclude_cols = set(exclude_cols or [])
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude_cols]

    rows = []
    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        lo = s.quantile(lower_q)
        hi = s.quantile(upper_q)
        n_lo = (s < lo).sum()
        n_hi = (s > hi).sum()
        rows.append({
            "feature": col,
            f"q{int(lower_q*100)}": float(lo),
            f"q{int(upper_q*100)}": float(hi),
            "n_low_outliers": int(n_lo),
            "n_high_outliers": int(n_hi),
            "n_total": int(s.shape[0]),
            "pct_low_outliers": float(n_lo / s.shape[0]) if s.shape[0] else np.nan,
            "pct_high_outliers": float(n_hi / s.shape[0]) if s.shape[0] else np.nan,
        })

    out_df = pd.DataFrame(rows).sort_values("n_high_outliers", ascending=False)

    out_path = os.path.join(output_dir, "outliers_percentiles.xlsx")
    out_df.to_excel(out_path, index=False)
    print(f"[INFO] Saved outlier (percentile) report to: {out_path}")
    return out_df


def compare_by_target(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str,
    exclude_cols=None
) -> pd.DataFrame:
    """
    3) Compare numeric features by target classes (e.g., ADV_EVENT 0 vs 1).
    Produces a summary table (mean/median/std per group) and saves to Excel.
    """
    if target_col not in df.columns:
        print(f"[WARN] Target column '{target_col}' not found. Skipping target comparison.")
        return pd.DataFrame()

    exclude_cols = set(exclude_cols or [])
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in exclude_cols and c != target_col]

    # Group summary
    summary = (
        df.groupby(target_col)[num_cols]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )

    out_path = os.path.join(output_dir, f"group_summary_by_{target_col}.xlsx")
    summary.to_excel(out_path)
    print(f"[INFO] Saved group comparison summary to: {out_path}")

    return summary
def feature_plot(f,input_data, args, max_norm=10):
    """
    Plots violinplots of all features (except ID), scaling numeric features to [0, max_norm]
    if their max is > max_norm, and marks scaled ones with '*'.

    Also runs:
      1) skewness table
      2) outlier counts using percentiles
      3) group comparison by target (assumed last column after dropping ID)
    """

    dpi = args.plots.dpi
    fontsize = args.plots.axis_fontsize
    plt.rcParams['font.family'] = 'Times New Roman'
    renamed = {}

    out_dir = args.experiment.pre_process_output_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Keep a copy for EDA stats (raw, unscaled) ---
    raw = f.copy()

    # Drop ID (first column)
    id_col = raw.columns[0]
    raw_no_id = raw.drop(columns=[id_col], errors="ignore")

    # use target_col or target is the last column (after dropping ID)
    target_col = getattr(args.data, "target_col", raw_no_id.columns[-1])

    print(f"[INFO] Using target column (last column): {target_col}")

    # 1) skewness (exclude target)
    exclude = {target_col}
    compute_skewness(raw_no_id, out_dir, exclude_cols=exclude)

    # 2) percentile outliers (exclude target)
    flag_outliers_percentile(raw_no_id, out_dir, exclude_cols=exclude, upper_q=0.99, lower_q=0.01)

    # 3) compare by target
    compare_by_target(raw_no_id, target_col=target_col, output_dir=out_dir)

    # --- Now proceed with plotting dataframe (scaled) ---
    plot_df = raw_no_id.copy()

    # Scale numeric features except the target
    for col in plot_df.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue

        max_val = plot_df[col].max()
        if pd.notna(max_val) and max_val > max_norm:
            min_val = plot_df[col].min()
            denom = max_val - min_val
            if denom != 0:
                plot_df[col] = max_norm * (plot_df[col] - min_val) / denom
                renamed[col] = col + " *"

    plot_df.rename(columns=renamed, inplace=True)

    # Plot violin plots
    plt.figure(figsize=(max(8, len(plot_df.columns) * 0.6), 6), dpi=dpi)
    sns.violinplot(data=plot_df, inner="quartile", cut=0)
    plt.xticks(
        ticks=np.arange(len(plot_df.columns)),
        labels=plot_df.columns,
        rotation=90,
        fontsize=fontsize * 0.9
    )
    plt.ylabel("Feature Values", fontsize=fontsize)
    plt.xlabel("Features", fontsize=fontsize)
    plt.title(f"Feature Distribution (* = scaled to [0,{max_norm}])", fontsize=fontsize)
    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(out_dir, "Feature_plot_violin.png"))
    plt.close()
    # --- NEW: Cancer type vs target counts plot ---
    cancer_col = getattr(args.data, "cancer_col", "Cancer Category")  # optional config
    save_path = os.path.join(out_dir, "patients_by_cancer_and_event.png")

    if cancer_col in raw_no_id.columns and target_col in raw_no_id.columns:
        plot_event_by_cancer(df=input_data, args=args)
        print(f"[INFO] Saved cancer/outcome plot to: {save_path}")
    else:
        print(f"[WARN] Skipping cancer/outcome plot. Missing columns: "
              f"{[c for c in [cancer_col, target_col] if c not in raw_no_id.columns]}")


def plot_event_by_cancer(
    df: pd.DataFrame,
    args,
    save_path: str = None
):
    """
    Plot number of patients with target=0 vs target=1 for each cancer type,
    using column names and plot settings from args.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (should contain cancer + target columns)
    args : Namespace / config object
        Loaded config with args.data and args.plots
    save_path : str, optional
        Output image path (defaults to pre_process_output_dir)
    """

    cancer_col = getattr(args.data, "cancer_col", "Cancer Category")
    target_col = getattr(args.data, "target_col", df.columns[-1])

    if cancer_col not in df.columns or target_col not in df.columns:
        print(
            f"[WARN] Cannot plot cancer vs event. Missing columns: "
            f"{[c for c in [cancer_col, target_col] if c not in df.columns]}"
        )
        return

    # Count patients per (cancer type, target)
    count_df = (
        df.groupby([cancer_col, target_col])
        .size()
        .reset_index(name="count")
    )

    # Plot settings
    figsize = args.plots.figsize
    dpi = args.plots.dpi
    axis_fontsize = args.plots.axis_fontsize

    plt.figure(figsize=figsize)
    sns.barplot(
        data=count_df,
        x=cancer_col,
        y="count",
        hue=target_col,
    )

    plt.xlabel("Cancer Type", fontsize=axis_fontsize)
    plt.ylabel("Number of Patients", fontsize=axis_fontsize)
    plt.title("Patient Counts by Cancer Type and Outcome", fontsize=axis_fontsize)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=target_col)
    plt.tight_layout()

    # Default save location
    if save_path is None:
        save_path = os.path.join(
            args.experiment.pre_process_output_dir,
            "patients_by_cancer_and_event.png"
        )

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved cancer vs event plot to: {save_path}")
def visualize_separability(
        data: pd.DataFrame,
        method: str = "pca",  # "pca" or "tsne"
        title: str = "Separability visualization",
        standardize: bool = True,  # standardize features before embedding
        tsne_perplexity: float = 30.0,
        tsne_n_iter: int = 1000,
        random_state: int = 42,
        save_path: str = None  # optional save path
):
    """
        Visualize dataset separability in 2D using PCA or t-SNE.

        Assumptions:
          - First column = ID (excluded)
          - Last column = target
          - All other columns = numeric features
        """
    # Split into features + target
    id_col = data.columns[0]
    target_col = data.columns[-1]
    X = data.drop(columns=[id_col, target_col])
    y = data[target_col]

    # Standardization
    X_proc = X.values
    if standardize:
        X_proc = StandardScaler().fit_transform(X_proc)

    # Embedding
    if method.lower() == "pca":
        emb = PCA(n_components=2, random_state=random_state).fit_transform(X_proc)
        method_title = "PCA"
    elif method.lower() == "tsne":
        emb = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            n_iter=tsne_n_iter,
            random_state=random_state,
            init="pca",
            learning_rate="auto"
        ).fit_transform(X_proc)
        method_title = f"t-SNE (perp={tsne_perplexity}, iters={tsne_n_iter})"
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    df_vis = pd.DataFrame({
        "Dim1": emb[:, 0],
        "Dim2": emb[:, 1],
        "Target": y.astype(str).values
    })

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_vis, x="Dim1", y="Dim2", hue="Target", alpha=0.8, s=45)
    plt.title(f"{title} — {method_title}")
    plt.legend(title="Target", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return df_vis
