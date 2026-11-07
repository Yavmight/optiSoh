"""Incremental Capacity (dQ/dV) feature extraction."""

from typing import Dict

import numpy as np
from scipy import stats

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def extract_ic_features(v_sub: np.ndarray, dq_sub: np.ndarray) -> Dict[str, float]:
    """Extract 12 features from partial IC curve: peaks, sub-window areas, moments."""
    n_pts = len(dq_sub)
    if n_pts == 0:
        return {
            k: 0.0
            for k in [
                "ic_peak_height",
                "ic_peak_voltage",
                "ic_peak_prominence",
                "ic_peak_fwhm",
                "ic_area_total",
                "ic_area_sub1",
                "ic_area_sub2",
                "ic_area_sub3",
                "ic_mean",
                "ic_std",
                "ic_skew",
                "ic_kurtosis",
            ]
        }

    max_idx = int(np.argmax(dq_sub))
    peak_h = float(dq_sub[max_idx])
    peak_v = float(v_sub[max_idx])
    min_val = float(np.min(dq_sub))
    prominence = peak_h - min_val

    # FWHM approximation
    half_max = min_val + prominence / 2.0
    above_half = np.where(dq_sub >= half_max)[0]
    if len(above_half) > 1:
        fwhm = float(v_sub[above_half[-1]] - v_sub[above_half[0]])
    else:
        fwhm = 0.0

    # Total and 3 sub-window areas
    area_tot = float(_trapezoid(dq_sub, v_sub)) if n_pts > 1 else 0.0
    k = n_pts // 3
    if k > 0:
        a1 = float(_trapezoid(dq_sub[:k], v_sub[:k])) if k > 1 else 0.0
        a2 = float(_trapezoid(dq_sub[k : 2 * k], v_sub[k : 2 * k])) if k > 1 else 0.0
        a3 = float(_trapezoid(dq_sub[2 * k :], v_sub[2 * k :])) if len(dq_sub[2 * k :]) > 1 else 0.0
    else:
        a1 = a2 = a3 = area_tot / 3.0

    return {
        "ic_peak_height": peak_h,
        "ic_peak_voltage": peak_v,
        "ic_peak_prominence": prominence,
        "ic_peak_fwhm": fwhm,
        "ic_area_total": area_tot,
        "ic_area_sub1": a1,
        "ic_area_sub2": a2,
        "ic_area_sub3": a3,
        "ic_mean": float(np.mean(dq_sub)),
        "ic_std": float(np.std(dq_sub)),
        "ic_skew": float(stats.skew(dq_sub)) if n_pts > 2 else 0.0,
        "ic_kurtosis": float(stats.kurtosis(dq_sub)) if n_pts > 3 else 0.0,
    }
