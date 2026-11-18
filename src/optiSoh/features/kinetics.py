"""Charging kinetics and energy feature extraction."""

from typing import Dict

import numpy as np

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def extract_kinetics_features(
    v_sub: np.ndarray,
    tau_sub: np.ndarray,
    q_sub: np.ndarray,
) -> Dict[str, float]:
    """Extract 8 charging kinetics, duration, and partial energy features."""
    n_pts = len(v_sub)
    if n_pts < 3:
        return {
            k: 0.0
            for k in [
                "tau_duration",
                "dv_dtau_mean",
                "dv_dtau_max",
                "dv_dtau_min",
                "curvature_mean",
                "energy_partial",
                "time_to_midpoint_ratio",
                "quartile_skewness",
            ]
        }

    duration = float(tau_sub[-1] - tau_sub[0])
    dt = np.gradient(tau_sub)
    dt = np.where(np.abs(dt) < 1e-6, 1.0, dt)
    dv_dtau = np.gradient(v_sub) / dt
    d2v_dtau2 = np.gradient(dv_dtau) / dt

    energy = float(_trapezoid(v_sub, q_sub)) if len(q_sub) == n_pts else 0.0

    # Time to midpoint voltage
    v_mid = (v_sub[0] + v_sub[-1]) / 2.0
    mid_idx = int(np.argmin(np.abs(v_sub - v_mid)))
    mid_ratio = float((tau_sub[mid_idx] - tau_sub[0]) / max(duration, 1e-4))

    # Quartile skewness of voltage profile
    q25, q50, q75 = np.percentile(v_sub, [25, 50, 75])
    denom = max(q75 - q25, 1e-6)
    q_skew = float(((q75 - q50) - (q50 - q25)) / denom)

    return {
        "tau_duration": duration,
        "dv_dtau_mean": float(np.mean(dv_dtau)),
        "dv_dtau_max": float(np.max(dv_dtau)),
        "dv_dtau_min": float(np.min(dv_dtau)),
        "curvature_mean": float(np.mean(np.abs(d2v_dtau2))),
        "energy_partial": energy,
        "time_to_midpoint_ratio": mid_ratio,
        "quartile_skewness": q_skew,
    }
