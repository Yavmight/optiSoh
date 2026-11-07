"""Incremental Capacity (dQ/dV) feature extraction."""

import numpy as np

# Compatible with NumPy 1.x and NumPy 2.x
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def extract_ic_features(v_sub: np.ndarray, dq_sub: np.ndarray) -> dict:
    """Extract 12 features from partial IC curve: peaks, sub-window areas, moments."""
    return {
        "ic_peak_height": float(np.max(dq_sub)) if len(dq_sub) else 0.0,
        "ic_peak_voltage": float(v_sub[np.argmax(dq_sub)]) if len(dq_sub) else 0.0,
        "ic_area_total": float(_trapezoid(dq_sub, v_sub)) if len(dq_sub) > 1 else 0.0,
    }
