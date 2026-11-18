"""Normalized model complexity metric (Objective f4)."""

from typing import Optional

import numpy as np


def compute_model_complexity(model, n_train_samples: Optional[int] = None) -> float:
    """
    Compute normalized complexity in [0, 1] across SVR support vectors and tree depths.
    Support vector count is normalized by actual training sample count rather than a hardcoded magic number.
    """
    svr = getattr(model, "svr_", getattr(model, "svr", None))

    # 1. Determine number of support vectors in SVR
    if svr is not None and hasattr(svr, "support_"):
        n_sv = float(len(svr.support_))
    elif svr is not None and hasattr(svr, "n_support_"):
        n_sv = float(np.sum(svr.n_support_))
    else:
        n_sv = 50.0

    # 2. Determine sample count normalization denominator
    if n_train_samples is not None and n_train_samples > 0:
        denom = float(n_train_samples)
    elif hasattr(model, "n_samples_fit_") and model.n_samples_fit_ > 0:
        denom = float(model.n_samples_fit_)
    elif svr is not None and hasattr(svr, "shape_fit_") and svr.shape_fit_[0] > 0:
        denom = float(svr.shape_fit_[0])
    else:
        # Fallback if training size cannot be inferred
        denom = max(n_sv, 100.0)

    svr_sv_fraction = n_sv / max(denom, 1.0)
    rf_depth_norm = getattr(model, "rf_max_depth", 5) / 10.0
    xgb_depth_norm = getattr(model, "xgb_max_depth", 4) / 10.0

    comp = 0.4 * svr_sv_fraction + 0.3 * rf_depth_norm + 0.3 * xgb_depth_norm
    return float(np.clip(comp, 0.0, 1.0))
