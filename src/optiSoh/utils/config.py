"""Configuration loader with deep-merge support for defaults and experiment overrides."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_yaml_config(path: Path) -> dict:
    """Load a raw YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_experiment_config(
    exp_path: Path,
    base_default_path: Optional[Path] = None,
) -> dict:
    """Load an experiment configuration, inheriting and overriding defaults from default.yaml."""
    exp_dict = load_yaml_config(exp_path)
    if base_default_path is None:
        base_default_path = exp_path.parent.parent / "default.yaml"

    if base_default_path.exists():
        base_dict = load_yaml_config(base_default_path)
        return _deep_merge(base_dict, exp_dict)
    return exp_dict
