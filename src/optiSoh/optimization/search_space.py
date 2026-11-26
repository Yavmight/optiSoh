"""Decision vector encoding, decoding, unified hyperparameter bounds, and repair operators."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from ..contracts.constants import (
    WINDOW_END_MAX,
    WINDOW_END_MIN,
    WINDOW_MIN_WIDTH,
    WINDOW_START_MAX,
    WINDOW_START_MIN,
)

# Single source of truth for hyperparameter search bounds shared across
# RandomSearchOptimizer, OptunaTPEOptimizer, and NSGA2Runner (budget-matched parity).
SVR_C_BOUNDS: Tuple[float, float] = (0.1, 100.0)  # log-uniform
RF_N_ESTIMATORS_CHOICES: Tuple[int, ...] = (50, 100, 150)
RF_MAX_DEPTH_BOUNDS: Tuple[int, int] = (2, 8)  # inclusive [2, 8]
XGB_LR_BOUNDS: Tuple[float, float] = (0.01, 0.2)  # uniform

HYPERPARAM_BOUNDS: Dict[str, Any] = {
    "svr_c": SVR_C_BOUNDS,
    "rf_n_estimators": RF_N_ESTIMATORS_CHOICES,
    "rf_max_depth": RF_MAX_DEPTH_BOUNDS,
    "xgb_learning_rate": XGB_LR_BOUNDS,
}


@dataclass
class CoOptDecisionVector:
    v_start: float
    v_end: float
    feature_mask: np.ndarray  # Boolean array of length D
    hyperparams: dict


def repair_decision_vector(
    v_start: float,
    v_end: float,
    feature_mask: np.ndarray,
    min_width: float = WINDOW_MIN_WIDTH,
):
    """Enforce minimum window width and ensure at least one feature is active."""
    v_start = float(np.clip(v_start, WINDOW_START_MIN, WINDOW_START_MAX))
    v_end = float(np.clip(v_end, WINDOW_END_MIN, WINDOW_END_MAX))
    if v_end - v_start < min_width:
        v_end = min(WINDOW_END_MAX, v_start + min_width)
    if not np.any(feature_mask):
        feature_mask[0] = True
    return v_start, v_end, feature_mask


def sample_random_hyperparams(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample hyperparameters uniformly/log-uniformly from the canonical HYPERPARAM_BOUNDS."""
    log_c_lo, log_c_hi = np.log10(SVR_C_BOUNDS[0]), np.log10(SVR_C_BOUNDS[1])
    depth_lo, depth_hi = RF_MAX_DEPTH_BOUNDS
    lr_lo, lr_hi = XGB_LR_BOUNDS
    return {
        "svr_c": float(10.0 ** rng.uniform(log_c_lo, log_c_hi)),
        "rf_n_estimators": int(rng.choice(RF_N_ESTIMATORS_CHOICES)),
        "rf_max_depth": int(rng.integers(depth_lo, depth_hi, endpoint=True)),
        "xgb_learning_rate": float(rng.uniform(lr_lo, lr_hi)),
    }


def suggest_optuna_hyperparams(trial: Any) -> Dict[str, Any]:
    """Suggest hyperparameters for an Optuna trial from the canonical HYPERPARAM_BOUNDS."""
    return {
        "svr_c": float(trial.suggest_float("svr_c", SVR_C_BOUNDS[0], SVR_C_BOUNDS[1], log=True)),
        "rf_n_estimators": int(trial.suggest_categorical("rf_n_estimators", list(RF_N_ESTIMATORS_CHOICES))),
        "rf_max_depth": int(trial.suggest_int("rf_max_depth", RF_MAX_DEPTH_BOUNDS[0], RF_MAX_DEPTH_BOUNDS[1])),
        "xgb_learning_rate": float(trial.suggest_float("xgb_learning_rate", XGB_LR_BOUNDS[0], XGB_LR_BOUNDS[1])),
    }
