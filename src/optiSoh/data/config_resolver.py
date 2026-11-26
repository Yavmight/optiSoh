"""Resolves dataset and preprocessing specifications from configs/."""

from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.config import load_yaml_config

_DEFAULT_DATASETS_CONFIG = Path("configs/datasets.yaml")
_DEFAULT_GLOBAL_CONFIG = Path("configs/default.yaml")

_PREPROCESSING_FALLBACK: Dict[str, Any] = {
    "cc_current_tolerance": 0.05,
    "cc_min_points": 20,
    "cc_min_voltage_span": 0.15,
    "cc_v_min_valid": 2.0,
    "cc_v_max_valid": 4.5,
    "savgol_time_window": 11,
    "savgol_grid_window": 11,
    "savgol_thermal_window": 35,
    "savgol_polyorder": 2,
    "regeneration_jump_pct": 1.0,
    "label_outlier_window": 5,
    "label_outlier_threshold_pct": 2.0,
}


def _resolve_config_file(default_rel: Path, config_path: Optional[Path] = None) -> Path:
    cfg_file = config_path or default_rel
    if not cfg_file.exists():
        alt_path = Path(__file__).resolve().parent.parent.parent.parent / default_rel
        if alt_path.exists():
            cfg_file = alt_path
    return cfg_file


def get_dataset_spec(
    dataset_key: str,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Retrieve specifications for a given dataset (nasa, oxford, sandia) from config."""
    cfg_file = _resolve_config_file(_DEFAULT_DATASETS_CONFIG, config_path)
    if cfg_file.exists():
        all_specs = load_yaml_config(cfg_file)
        if dataset_key in all_specs:
            return all_specs[dataset_key]

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


def get_preprocessing_spec(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Retrieve fixed a-priori preprocessing constants from configs/default.yaml."""
    cfg_file = _resolve_config_file(_DEFAULT_GLOBAL_CONFIG, config_path)
    merged = dict(_PREPROCESSING_FALLBACK)
    if cfg_file.exists():
        cfg = load_yaml_config(cfg_file)
        if isinstance(cfg, dict) and isinstance(cfg.get("preprocessing"), dict):
            merged.update(cfg["preprocessing"])
    return merged
