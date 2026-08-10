"""Dataset loading and feature/target splitting.

Replaces the load_xgboost_data / load_XGB_data functions that were
duplicated (with tiny inconsistencies) across M1/train.py, M1/test.py,
M2/train.py, M2/test.py, and the root test.py.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_tabular_dataset(path: str, drop_columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Load a CSV/Excel dataset and optionally drop excluded feature columns.

    Args:
        path: Path to a .csv, .xlsx, or .xls file.
        drop_columns: Column names to drop if present (missing ones are
            silently ignored, matching the original pipeline's behavior).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension == ".csv":
        data = pd.read_csv(path)
    elif extension in SUPPORTED_EXTENSIONS:
        data = pd.read_excel(path)
    else:
        raise ValueError(
            f"[ERROR] Unsupported file type: {extension}. Use CSV or Excel."
        )

    if drop_columns:
        data = data.drop(columns=drop_columns, errors="ignore")

    print(f"[INFO] Loaded dataset: {path}")
    print(f"[INFO] Data shape: {data.shape}")
    return data


def load_dataset_for_config(path: str, args) -> pd.DataFrame:
    """Convenience wrapper: apply a config's drop_feature settings while loading."""
    drop_columns = None
    if getattr(args.data, "drop_feature_enable", False):
        drop_columns = getattr(args.models.xgboost, "drop_feature", None)
    return load_tabular_dataset(path, drop_columns=drop_columns)


def build_xy(
    df_raw: pd.DataFrame,
    id_col: str,
    target_col: str,
    return_id: bool = False,
):
    """Split a raw dataframe into features (X), target (y), and optionally ID.

    Args:
        df_raw: Input dataframe containing ID, features, and target columns.
        id_col: Name of the patient/sample identifier column (excluded from X).
        target_col: Name of the binary (0/1) target column.
        return_id: If True, also return the ID column as a third value.

    Returns:
        (X, y) by default, or (X, y, id) if ``return_id=True``.
    """
    if target_col not in df_raw.columns:
        raise ValueError(f"[ERROR] target_col '{target_col}' not found in dataframe")
    if id_col not in df_raw.columns:
        raise ValueError(f"[ERROR] id_col '{id_col}' not found in dataframe")

    y = df_raw[target_col].astype(int)
    X = df_raw.drop(columns=[id_col, target_col])

    if return_id:
        return X, y, df_raw[id_col]
    return X, y


def to_numpy_xy(X, y):
    """Coerce X/y (possibly pandas objects) into plain numpy arrays."""
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_np = np.asarray(y).astype(int).ravel()
    return X_np, y_np
