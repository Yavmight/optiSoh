"""Normalized model complexity metric (Objective f4)."""

import numpy as np


def compute_model_complexity(model) -> float:
    """Compute normalized complexity in [0, 1] across SVR support vectors and tree depths."""
    svr_sv_fraction = getattr(model.svr, "n_support_", np.array([50]))[0] / 200.0
    rf_depth_norm = model.rf_max_depth / 10.0
    xgb_depth_norm = model.xgb_max_depth / 10.0
    comp = 0.4 * svr_sv_fraction + 0.3 * rf_depth_norm + 0.3 * xgb_depth_norm
    return float(np.clip(comp, 0.0, 1.0))
