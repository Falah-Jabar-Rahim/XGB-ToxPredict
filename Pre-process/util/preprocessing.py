
import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def highlight_correlated_features(
        df: pd.DataFrame,
        output_dir: str,
        threshold: float = 0.9,
        method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute correlation matrix and extract highly correlated feature pairs.
    Save results to a single Excel file with two sheets:
      1) Correlation_Matrix
      2) Highly_Correlated_Features
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select numeric columns only
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        print("[WARN] Not enough numeric columns to compute correlations.")
        return pd.DataFrame(columns=["Feature_1", "Feature_2", "Correlation"])

    # Compute correlation matrix
    corr_table = numeric_df.corr(method=method).round(3)

    # Upper triangle mask (avoid duplicates & self-correlation)
    upper_tri = corr_table.abs().where(
        np.triu(np.ones(corr_table.shape), k=1).astype(bool)
    )

    # Extract highly correlated pairs
    high_corr_pairs = (
        upper_tri.stack()
        .reset_index()
        .rename(
            columns={
                "level_0": "Feature_1",
                "level_1": "Feature_2",
                0: "Correlation",
            }
        )
        .query("Correlation >= @threshold")
        .sort_values("Correlation", ascending=False)
    )

    # Save to Excel
    excel_path = os.path.join(output_dir, "correlation_analysis.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        corr_table.to_excel(writer, sheet_name="Correlation_Matrix")
        high_corr_pairs.to_excel(
            writer, sheet_name="Highly_Correlated_Features", index=False
        )

    print(f"[INFO] Correlation analysis saved to: {excel_path}")
    print(f"[INFO] Threshold used: |corr| ≥ {threshold}")


def preprocess_data(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Steps
    -----
    1) Clean inequalities & coerce numerics
    2) Drop patients with high row-wise missingness
    3) Ordinal-encode specified categorical columns
    4) Impute missing values (MICE or MissForest)
    5) Apply log1p AFTER imputation (strict: raise error if any negatives in log cols)
    """
    out_dir = args.experiment.pre_process_output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Optional per-column eps overrides for inequality cleaning
    EPS_OVERRIDES = {
        # "column_name": 0.2,
    }
    _inequal_re = re.compile(r"^\s*([<>])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")

    id_col = args.data.id_col
    df[id_col] = df[id_col].astype(str)

    # 1) Clean inequalities & numeric coercion
    df = clean_inequalities_and_numeric(df, args.preprocessing.imputation.eps, _inequal_re, EPS_OVERRIDES, exclude_cols=[args.data.id_col])
    df.to_excel(os.path.join(out_dir, "patient_inequalities_cleaned.xlsx"), index=False)

    # 2) Drop patients with excessive missingness
    df_row_kept, patient_missing_report, dropped_patients = drop_high_row_missing(
        df,
        args.preprocessing.imputation.drop_missing,
        args.data.id_col,
    )
    patient_missing_report.to_excel(os.path.join(out_dir, "patient_missing_report.xlsx"), index=False)

    if not dropped_patients.empty:
        print(
            f"[INFO] Dropped {len(dropped_patients)} patients with >= "
            f"{args.preprocessing.imputation.drop_missing:.0%} missingness (row-wise)."
        )

    df = df_row_kept

    # 3) Ordinal-encode categorical columns
    df_enc, mappings = ordinal_encode(df, include_cols=args.data.categorical_cols)
    df_enc.to_excel(os.path.join(out_dir, "patient_ordinal_encoding.xlsx"), index=False)

    with open(os.path.join(out_dir, "encoding_mappings.json"), "w") as f:
        json.dump(mappings, f, indent=4)

    # 4) Imputation
    # --- identify ID and target columns ---
    target_col = get_target_col(df_enc, args)
    # --- keep copies ---
    id_series = df_enc[id_col]
    target_series = df_enc[target_col]

    # --- drop ID + target before imputation ---
    df_features = df_enc.drop(columns=[id_col, target_col]).copy()
    # --- drop ID + target before imputation ---

    # IMPORTANT: convert pandas nullable dtypes / pd.NA to sklearn-friendly float matrix
    df_features = df_features.replace({pd.NA: np.nan})
    df_features = df_features.apply(pd.to_numeric, errors="coerce")
    df_features = df_features.astype(float)

    print(df_features.dtypes)

    method = str(getattr(args.preprocessing.imputation, "imp_method", "MICE")).strip().lower()
    # --- run imputation on FEATURES ONLY ---
    if method == "mice":
        df_imp_feat = mice_impute(df_features, args)
        method_tag = "MICE"

    elif method in {"missforest", "miss_forest", "miss-forest"}:
        df_imp_feat = missforest_like_impute(df_features, args)
        method_tag = "MissForest"

    else:
        raise ValueError(f"Unknown imputation method '{method}'")

    # --- reattach ID + target (unchanged) ---
    df_imputed = df_imp_feat.copy()
    df_imputed[id_col] = id_series.values


    df_imputed[target_col] = target_series.values

    # --- restore original column order ---
    df_imputed = df_imputed[[id_col] + df_features.columns.tolist() + [target_col]]

    cat_cols = args.data.int_cols

    for col in cat_cols:
        df_imputed[col] = np.round(df_imputed[col]).astype(int)

    # --- save imputed dataset (NO log yet) ---
    imputed_path = os.path.join(
        out_dir, f"patient_imputed_final_{method_tag}.xlsx"
    )
    df_imputed.to_excel(imputed_path, index=False)

    print(f"[INFO] Saved imputed dataset to: {imputed_path}")

    # Evaluate imputation methods on df_features (still has NaNs)
    eval_enabled = bool(getattr(args.preprocessing.imputation, "eval_enabled", False))
    if eval_enabled:
        from eval_imputation import evaluate_imputation_methods
        evaluate_imputation_methods(
            df_enc=df_features,
            args=args,
            out_dir=out_dir,
            methods=("mice", "missforest"),
            mask_frac=float(getattr(args.preprocessing.imputation, "eval_mask_frac", 0.10)),
            repeats=int(getattr(args.preprocessing.imputation, "eval_repeats", 5)),
            random_state=int(getattr(args.experiment, "random_seed", 42)),
        )

    # 5) apply Log(x+1)
    log_cfg = getattr(args.preprocessing, "log_transform", None)
    log_enabled = bool(getattr(log_cfg, "enabled", False))
    exclude_from_log = list(getattr(log_cfg, "exclude_from_log", []) or []) if log_cfg is not None else []

    df_final = df_imputed  # default: no log transform

    if log_enabled:
        target_col = get_target_col(df_imputed, args)
        exclude_cols = [args.data.id_col, target_col] + exclude_from_log
        log_cols = detect_log_features(df_imputed, exclude_cols=exclude_cols)

        # Strict negative check (raise error if any negatives exist)
        neg_report = check_negative_values(df_imputed, log_cols, output_dir=out_dir)
        if not neg_report.empty:
            raise ValueError(
                "[ERROR] Negative values detected in columns selected for log1p.\n"
                "See 'negative_value_report.xlsx' for details.\n"
                "Negative values may have been introduced after data treatment.\n"
            )

        df_final = apply_log1p(df_imputed, log_cols)

        # Save metadata + final dataset
        pd.DataFrame({"log1p_features": log_cols}).to_excel(
            os.path.join(out_dir, "log1p_features.xlsx"),
            index=False,
        )

        final_path = os.path.join(out_dir, f"patient_imputed_log1p_{method}.xlsx")
        df_final.to_excel(final_path, index=False)
        print(f"[INFO] Saved final dataset (imputed + log1p) to: {final_path}")

    else:
        # Save final dataset without log (still imputed)
        final_path = os.path.join(out_dir, f"patient_imputed_{method}.xlsx")
        df_final.to_excel(final_path, index=False)
        print(f"[INFO] log1p disabled. Saved final dataset (imputed) to: {final_path}")

    return df_final


def _coerce_numeric_with_inequalities(series, f_name, eps, _inequal_re):
    """
    If a series has any '<' or '>' patterns, convert them using eps,
    else try regular numeric coercion. Returns numeric series.
    """
    matched_any = False
    converted = []
    for v in series:
        nv, matched = _parse_inequality(v, eps, _inequal_re, f_name)
        matched_any = matched_any or matched
        converted.append(nv)
    s = pd.Series(converted, index=series.index)

    # If not matched_any, still try tocoerce numeric (for strings like '12.3')
    if not matched_any:
        s = pd.to_numeric(series, errors="coerce")
        # If all became NaN and original had non-strings, fall back
        if s.isna().all():
            return series  # keep as-is (likely categorical text)
    else:
        # Ensure numeric dtype
        s = pd.to_numeric(s, errors="coerce")
    return s


def _parse_inequality(val, eps, _inequal_re, f_name=None):
    """
    Convert strings like '<0.2' or '> 60' to numeric by applying +/- eps.
    Returns (numeric_value, matched_bool)
    """
    if isinstance(val, str):
        m = _inequal_re.match(val)
        if m:
            sign, num = m.groups()
            x = float(num)

            y = x - eps if sign == "<" else x + eps
            if sign == "<" and y < 0:
                warnings.warn(
                    f"[WARNING⚠️⚠️] ' in feature {f_name} value {val}' with eps={eps} >>> {y} (negative value produced: check {f_name} values the dataset or reduce eps)"
                )

            return (x - eps) if sign == "<" else (x + eps), True
    return val, False


def drop_high_row_missing(df, threshold, patient_id_col):
    row_missing_rate = df.isna().mean(axis=1)
    row_missing_count = df.isna().sum(axis=1)
    # Build report with ID (if present) or index
    if patient_id_col and patient_id_col in df.columns:
        ids = df[patient_id_col].astype(str)
    else:
        ids = df.index.astype(str)

    report = pd.DataFrame({
        "patient_id": ids,
        "missing_count": row_missing_count,
        "missing_rate": row_missing_rate
    })

    # Drop rows with too much missingness
    keep_mask = row_missing_rate < threshold
    df_kept = df.loc[keep_mask].copy()
    dropped_rows = report.loc[~keep_mask].sort_values("missing_rate", ascending=False)

    return df_kept, report, dropped_rows


def ordinal_encode(df, include_cols=None, start_index=1):
    """
    Replace categorical values in include_cols with integer codes (1..N).
    Preserves original column locations.
    """
    include_cols = include_cols or []
    out = df.copy()
    mappings = {}

    for col in include_cols:
        if col in out.columns:
            uniques = pd.Series(out[col].dropna().unique()).sort_values().tolist()
            mapping = {val: i + start_index for i, val in enumerate(uniques)}
            out[col] = out[col].map(mapping).astype("Int64")  # nullable integer type
            mappings[col] = mapping

    return out, mappings


def missforest_like_impute(df: pd.DataFrame, args) -> pd.DataFrame:
    cfg = getattr(args.preprocessing.imputation, "MissForest", None)
    if cfg is None:
        raise ValueError("[ERROR] Missing YAML config: preprocessing.imputation.MissForest")

    rf_cfg = getattr(cfg, "rf", None)
    it_cfg = getattr(cfg, "iterative_imputer", None)
    if rf_cfg is None or it_cfg is None:
        raise ValueError("[ERROR] Missing YAML blocks: MissForest.rf and/or MissForest.iterative_imputer")

    # Random state handling
    rf_random_state = getattr(rf_cfg, "random_state", None)
    it_random_state = getattr(it_cfg, "random_state", None)

    if rf_random_state == "use_experiment_seed":
        rf_random_state = getattr(args.experiment, "random_seed", None)

    if it_random_state == "use_experiment_seed":
        it_random_state = getattr(args.experiment, "random_seed", None)

    rf = RandomForestRegressor(
        n_estimators=int(getattr(rf_cfg, "n_estimators", 100)),
        criterion=str(getattr(rf_cfg, "criterion", "squared_error")),
        max_depth=getattr(rf_cfg, "max_depth", None),
        min_samples_split=int(getattr(rf_cfg, "min_samples_split", 2)),
        min_samples_leaf=int(getattr(rf_cfg, "min_samples_leaf", 1)),
        min_weight_fraction_leaf=float(getattr(rf_cfg, "min_weight_fraction_leaf", 0.0)),
        max_features=getattr(rf_cfg, "max_features", 1.0),
        max_leaf_nodes=getattr(rf_cfg, "max_leaf_nodes", None),
        min_impurity_decrease=float(getattr(rf_cfg, "min_impurity_decrease", 0.0)),
        bootstrap=bool(getattr(rf_cfg, "bootstrap", True)),
        oob_score=bool(getattr(rf_cfg, "oob_score", False)),
        n_jobs=int(getattr(rf_cfg, "n_jobs", -1)),
        random_state=rf_random_state,
        verbose=int(getattr(rf_cfg, "verbose", 0)),
        warm_start=bool(getattr(rf_cfg, "warm_start", False)),
    )

    imp = IterativeImputer(
        estimator=rf,
        max_iter=int(getattr(it_cfg, "max_iter", 10)),
        initial_strategy=str(getattr(it_cfg, "initial_strategy", "median")),
        random_state=it_random_state,
    )

    X_imp = imp.fit_transform(df.to_numpy())

    return pd.DataFrame(X_imp, columns=df.columns, index=df.index)


def mice_impute(df, p):
    imputer = IterativeImputer(
        estimator=p.preprocessing.imputation.mice.estimator,
        max_iter=p.preprocessing.imputation.mice.max_iter,
        tol=p.preprocessing.imputation.mice.tol,
        imputation_order=p.preprocessing.imputation.mice.imputation_order,
        initial_strategy=p.preprocessing.imputation.mice.initial_strategy,
        n_nearest_features=p.preprocessing.imputation.mice.n_nearest_features,
        sample_posterior=p.preprocessing.imputation.mice.sample_posterior,
        min_value=p.preprocessing.imputation.mice.min_value,
        max_value=p.preprocessing.imputation.mice.max_value,
        add_indicator=p.preprocessing.imputation.mice.add_indicator,
        random_state=p.preprocessing.imputation.mice.random_state,
    )

    arr = imputer.fit_transform(df.values.astype(float))
    imputed = pd.DataFrame(arr, index=df.index, columns=list(df.columns) + (
        [f"{c}_was_missing" for c in df.columns if
         p.preprocessing.imputation.mice.add_indicator] if p.preprocessing.imputation.mice.add_indicator else []
    ))
    if p.preprocessing.imputation.mice.preserve_observed:
        imputed.loc[:, df.columns] = df.where(~df.isna(), imputed.loc[:, df.columns])
    return imputed


def check_negative_values(df: pd.DataFrame, log_cols, output_dir: str = None) -> pd.DataFrame:
    """
    Check negative values in features before applying log1p.

    Returns
    -------
    neg_df : pd.DataFrame
        Summary of negative values per feature.
    """
    records = []
    for col in log_cols:
        s = df[col]
        neg_mask = s < 0
        n_neg = int(neg_mask.sum())
        if n_neg > 0:
            denom = int(s.notna().sum())
            records.append(
                {
                    "feature": col,
                    "n_negative": n_neg,
                    "min_value": float(s.min(skipna=True)),
                    "neg_fraction": float(n_neg / denom) if denom > 0 else np.nan,
                }
            )

    if records:
        neg_df = (
            pd.DataFrame(records)
            .sort_values(["n_negative", "min_value"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        neg_df = pd.DataFrame(columns=["feature", "n_negative", "min_value", "neg_fraction"])

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        neg_df.to_excel(os.path.join(output_dir, "negative_value_report.xlsx"), index=False)

    return neg_df


def get_target_col(df: pd.DataFrame, args) -> str:
    """Target column from config if present; fallback to last column."""
    t = getattr(args.data, "target_col", None)
    if t is not None and t in df.columns:
        return t
    return df.columns[-1]


def detect_log_features(df: pd.DataFrame, exclude_cols=None):
    """
    Select numeric columns eligible for log1p transformation, excluding specified columns.
    Note: Negativity is checked separately (strict mode).
    """
    exclude_cols = set(exclude_cols or [])
    log_cols = []

    for col in df.select_dtypes(include="number").columns:
        if col in exclude_cols:
            continue
        log_cols.append(col)

    return log_cols


def apply_log1p(df: pd.DataFrame, log_cols):
    """Apply log(1+x) to selected columns (assumes no negative values in those columns)."""
    df = df.copy()
    for col in log_cols:
        df[col] = np.log1p(df[col])
    return df


def clean_inequalities_and_numeric(df, default_eps, _inequal_re, eps_overrides=None, exclude_cols=None):
    """
    For each column, try to parse inequality strings and coerce numeric.
    Leaves true categorical text as-is.
    """
    eps_overrides = eps_overrides or {}
    out = df.copy()
    for col in out.columns:

        # ---- skip excluded columns (ID, etc.) ----
        if col in exclude_cols:
            continue
        eps = eps_overrides.get(col, default_eps)
        # Only attempt on object/string-like columns or mixed
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = _coerce_numeric_with_inequalities(out[col], col, eps, _inequal_re)
        # If still object but looks numeric-ish, try once more
        if out[col].dtype == object:
            # Try general numeric coercion without inequality rule
            maybe_num = pd.to_numeric(out[col], errors="coerce")
            # Heuristic: if this yields some non-NaNs, use it
            if maybe_num.notna().sum() > 0 and (maybe_num.notna().sum() >= len(out[col]) * 0.5):
                out[col] = maybe_num
    return out
