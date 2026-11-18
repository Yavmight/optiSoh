"""Cross-channel electrical-thermal coupling features."""

from typing import Dict

import numpy as np
from scipy import stats


def extract_coupling_features(
    v_sub: np.ndarray,
    q_sub: np.ndarray,
    dq_sub: np.ndarray,
    dt_sub: np.ndarray,
) -> Dict[str, float]:
    """Extract 4 electrical-thermal coupling, correlation, and curve arc length features."""
    n_pts = len(dq_sub)
    if n_pts < 3:
        return {
            "ic_dt_pearson": 0.0,
            "ic_dt_spearman": 0.0,
            "vq_arc_length": 0.0,
            "peak_voltage_ratio_ic_dt": 1.0,
        }

    # Pearson & Spearman correlation between dQ/dV and dT/dV
    std_dq = np.std(dq_sub)
    std_dt = np.std(dt_sub)
    if std_dq > 1e-8 and std_dt > 1e-8:
        pearson_r, _ = stats.pearsonr(dq_sub, dt_sub)
        spearman_r, _ = stats.spearmanr(dq_sub, dt_sub)
    else:
        pearson_r, spearman_r = 0.0, 0.0

    # Curve arc length in normalized (V, Q) space
    dv = np.diff(v_sub)
    dq = np.diff(q_sub) if len(q_sub) == n_pts else np.zeros_like(dv)
    arc_length = float(np.sum(np.sqrt(dv**2 + dq**2)))

    # Peak voltage location ratio
    v_peak_ic = float(v_sub[np.argmax(dq_sub)])
    v_peak_dt = float(v_sub[np.argmax(dt_sub)])
    peak_ratio = v_peak_ic / max(v_peak_dt, 1e-4)

    return {
        "ic_dt_pearson": float(pearson_r),
        "ic_dt_spearman": float(spearman_r),
        "vq_arc_length": arc_length,
        "peak_voltage_ratio_ic_dt": float(peak_ratio),
    }
