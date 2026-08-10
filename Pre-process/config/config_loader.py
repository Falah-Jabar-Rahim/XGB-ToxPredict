import os
import yaml
from types import SimpleNamespace

def _dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [ _dict_to_ns(x) for x in d ]
    else:
        return d

def load_config(path: str):
    """Load a YAML config and return a dot-accessible namespace."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Minimal validation
    required = ['experiment', 'data']
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing top-level section: {key}")
    # Create output dir
    return _dict_to_ns(cfg)

if __name__ == "__main__":
    import argparse, pprint
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    args = ap.parse_args()
    C = load_config(args.config)
    pprint.pp(C.__dict__)
