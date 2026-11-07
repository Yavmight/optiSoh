"""Differential Temperature (dT/dV) feature extraction."""

from typing import Dict

import numpy as np
from scipy import stats

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def extract_thermal_features(
    v_sub: np.ndarray,
    dt_sub: np.ndarray,
    temp_sub: np.ndarray,
) -> Dict[str, float]:
    """Extract 10 thermal features: max dT/dV, voltage location, temperature rise, moments."""
    n_pts = len(dt_sub)
    if n_pts == 0:
        return {
            k: 0.0
            for k in [
                "dt_max",
                "dt_voltage_at_max",
                "dt_mean",
                "dt_std",
                "delta_temp_rise",
                "dt_area",
                "dt_skew",
                "dt_kurtosis",
                "temp_ratio_end_start",
                "temp_voltage_slope",
            ]
        }

    max_idx = int(np.argmax(dt_sub))
    t_start = float(temp_sub[0]) if len(temp_sub) > 0 else 1.0
    t_end = float(temp_sub[-1]) if len(temp_sub) > 0 else 1.0
    delta_rise = t_end - t_start

    ratio = t_end / max(abs(t_start), 1e-4)
    v_span = max(float(v_sub[-1] - v_sub[0]), 1e-4) if n_pts > 1 else 1.0
    slope = delta_rise / v_span

    return {
        "dt_max": float(dt_sub[max_idx]),
        "dt_voltage_at_max": float(v_sub[max_idx]),
        "dt_mean": float(np.mean(dt_sub)),
        "dt_std": float(np.std(dt_sub)),
        "delta_temp_rise": delta_rise,
        "dt_area": float(_trapezoid(dt_sub, v_sub)) if n_pts > 1 else 0.0,
        "dt_skew": float(stats.skew(dt_sub)) if n_pts > 2 else 0.0,
        "dt_kurtosis": float(stats.kurtosis(dt_sub)) if n_pts > 3 else 0.0,
        "temp_ratio_end_start": float(ratio),
        "temp_voltage_slope": float(slope),
    }
