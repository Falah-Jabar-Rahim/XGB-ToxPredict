# ==============================================================================
# Author: Falah Jabar
# Affiliation: University Hospital of North Norway (UNN)
# Project: Machine Learning–Based Toxicity Prediction
# Description: Preprocess and explore the clinical toxicity dataset.
# ==============================================================================

import argparse
import os
from pathlib import Path

import pandas as pd

from config.config_loader import load_config
from typing import Union, Optional, List, Dict
from util.preprocessing import highlight_correlated_features, preprocess_data
from util.visualize import feature_plot


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve relative project paths independently of the terminal working directory."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load a CSV or Excel clinical dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] Input file not found: {input_path}")

    extension = input_path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(input_path)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)

    raise ValueError(
        f"[ERROR] Unsupported file type: {extension}. "
        "Supported formats are CSV (.csv) and Excel (.xlsx, .xls)."
    )


def save_dataset_summary(data: pd.DataFrame, output_dir: Path) -> Path:
    """Save descriptive statistics and column data types to Excel."""
    describe_df = data.describe(include="all").T
    dtypes_df = pd.DataFrame(
        {
            "column": data.columns,
            "dtype": data.dtypes.astype(str).values,
        }
    )

    output_path = output_dir / "describe_all.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        describe_df.to_excel(writer, sheet_name="Describe_All")
        dtypes_df.to_excel(writer, sheet_name="Column_Dtypes", index=False)

    return output_path


def print_dataset_overview(data: pd.DataFrame) -> None:
    """Print a concise overview of the loaded dataset."""
    print("\n=== Dataset Dimensions ===")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    print("\n=== Columns and Data Types ===")
    for column, dtype in data.dtypes.items():
        print(f"{column}: {dtype}")


def run_preprocessing(config) -> None:
    """Run the complete preprocessing and exploratory-analysis pipeline."""
    # Resolve config paths relative to this project, not the current terminal directory.
    input_path = resolve_path(config.data.data_path)
    output_dir = resolve_path(config.experiment.pre_process_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store resolved paths because downstream utilities read them from config.
    config.data.data_path = str(input_path)
    config.experiment.pre_process_output_dir = str(output_dir)

    print(f"[INFO] Input dataset: {input_path}")
    print(f"[INFO] Output directory: {output_dir}")

    # 1. Load dataset
    input_data = load_dataset(input_path)
    print_dataset_overview(input_data)

    # 2. Exploratory summary
    summary_path = save_dataset_summary(input_data, output_dir)
    print(f"[INFO] Saved dataset summary: {summary_path}")

    # 3. Correlation analysis
    highlight_correlated_features(
        df=input_data,
        output_dir=str(output_dir),
        threshold=config.data.corr_threshold,
        method=config.data.corr_method,
    )

    # 4. Preprocessing
    preprocessed_data = preprocess_data(input_data.copy(), config)

    # 5. Feature visualization
    feature_plot(preprocessed_data, input_data, config)

    print("\n[OK] Preprocessing completed successfully.")
    print(f"[OK] Results saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess and explore the clinical toxicity dataset."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "Path to the YAML configuration file "
            f"(default: {DEFAULT_CONFIG_PATH})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    cli_args = parse_args()
    config_path = resolve_path(cli_args.config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

    print(f"[INFO] Loading configuration: {config_path}")
    config = load_config(str(config_path))
    run_preprocessing(config)


if __name__ == "__main__":
    main()
