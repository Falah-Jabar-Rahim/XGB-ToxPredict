"""YAML configuration loading.

A single, shared implementation used by the M1 stage, the M2 stage, and the
hierarchical predictor. Previously this ~20-line file was copy-pasted three
times (root/config, M1/config, M2/config) and had drifted slightly (the root
copy validated a different set of required keys). There is now one copy.
"""

from __future__ import annotations

import argparse
import pprint
from types import SimpleNamespace
from typing import Any

import yaml

# Keys every config must define. Individual stages can require additional
# keys (e.g. the training stages need "models"/"training") by passing
# `extra_required` to load_config.
BASE_REQUIRED_KEYS = ("experiment", "data", "evaluation")


def _dict_to_namespace(value: Any) -> Any:
    """Recursively convert nested dicts/lists into dot-accessible objects."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_dict_to_namespace(item) for item in value]
    return value


def load_config(path: str, extra_required: tuple[str, ...] = ()) -> SimpleNamespace:
    """Load a YAML config file and return a dot-accessible namespace.

    Args:
        path: Path to the YAML file.
        extra_required: Additional top-level keys that must be present,
            beyond the always-required ``BASE_REQUIRED_KEYS``.
    """
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    required_keys = (*BASE_REQUIRED_KEYS, *extra_required)
    missing = [key for key in required_keys if key not in raw_config]
    if missing:
        raise ValueError(f"Config '{path}' is missing required section(s): {missing}")

    return _dict_to_namespace(raw_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a resolved config file.")
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    config = load_config(cli_args.config)
    pprint.pp(config.__dict__)
