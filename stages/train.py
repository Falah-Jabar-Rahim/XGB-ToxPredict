"""Train an XGBoost toxicity-screening stage (M1 or M2 — pick with --config).

Replaces M1/train.py and M2/train.py, which were byte-for-byte identical
files. Run:

    python -m stages.train --config configs/m1.yaml
    python -m stages.train --config configs/m2.yaml
"""

from __future__ import annotations

import argparse
import os

from common.config import load_config
from common.data import load_dataset_for_config
from common.plotting import plot_combined_calibration, plot_combined_pr, plot_combined_roc
from common.xgb_stage import train_xgb_with_cv


def run(args) -> None:
    output_dir = args.experiment.prediction_output_dir
    os.makedirs(output_dir, exist_ok=True)

    data = load_dataset_for_config(args.data.data_path_clean, args)

    print("[INFO] Training XGBoost...")
    result = train_xgb_with_cv(data, args)
    print("[OK] XGBoost training finished.")

    results = [result]
    roc_path = os.path.join(output_dir, "ROC_XGB.png")
    pr_path = os.path.join(output_dir, "PR_XGB.png")
    calibration_path = os.path.join(output_dir, "Calibration_XGB.png")

    plot_combined_roc(results, roc_path, args)
    plot_combined_pr(results, pr_path, args)
    plot_combined_calibration(results, calibration_path, args)

    print("\n[OK] Training completed.")
    print(f" - ROC:         {roc_path}")
    print(f" - PR:          {pr_path}")
    print(f" - Calibration: {calibration_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a single XGBoost stage.")
    parser.add_argument("--config_path", "--config", dest="config_path", type=str, required=True,
                         help="Path to the stage's YAML config (e.g. configs/m1.yaml).")
    cli_args = parser.parse_args()

    config = load_config(cli_args.config_path, extra_required=("models", "training"))
    run(config)


if __name__ == "__main__":
    main()
