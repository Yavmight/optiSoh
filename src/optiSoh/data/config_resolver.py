"""Resolves dataset specifications from configs/datasets.yaml."""

from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.config import load_yaml_config

_DEFAULT_DATASETS_CONFIG = Path("configs/datasets.yaml")


def get_dataset_spec(
    dataset_key: str,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Retrieve specifications for a given dataset (nasa, oxford, sandia) from config."""
    cfg_file = config_path or _DEFAULT_DATASETS_CONFIG
    if not cfg_file.exists():
        # Fallback if invoked from a different subdirectory
        alt_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "datasets.yaml"
        if alt_path.exists():
            cfg_file = alt_path

    if cfg_file.exists():
        all_specs = load_yaml_config(cfg_file)
        if dataset_key in all_specs:
            return all_specs[dataset_key]

    # Graceful fallback defaults if config is absent
    defaults = {
        "nasa": {"nominal_capacity": 2.0, "temperature_c": 24.0},
        "oxford": {"nominal_capacity": 0.74, "temperature_c": 40.0},
        "sandia": {
            "nominal_capacity_nca": 3.2,
            "nominal_capacity_nmc": 3.0,
            "temperature_c": 25.0,
        },
    }
    return defaults.get(dataset_key, {})
