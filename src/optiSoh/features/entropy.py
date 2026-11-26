"""Numba-accelerated Sample Entropy and signal complexity metrics."""

from typing import Dict, Optional

import numpy as np
from numba import njit


@njit(fastmath=True)
def sample_entropy_numba(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Fast Numba implementation of Sample Entropy."""
    n_pts = len(signal)
    if n_pts <= m + 1:
        return 0.0
    r_thresh = r * np.std(signal)
    if r_thresh <= 1e-8:
        return 0.0

    count_m = 0
    count_m1 = 0
    for i in range(n_pts - m):
        for j in range(i + 1, n_pts - m):
            match_m = True
            for k in range(m):
                if abs(signal[i + k] - signal[j + k]) > r_thresh:
                    match_m = False
                    break
            if match_m:
                count_m += 1
                if abs(signal[i + m] - signal[j + m]) <= r_thresh:
                    count_m1 += 1

    if count_m == 0 or count_m1 == 0:
        return 0.0
    return float(-np.log(count_m1 / count_m))


def extract_entropy_features(
    v_sub: np.ndarray,
    dq_sub: np.ndarray,
    tau_sub: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Extract 8 complexity and entropy features on cycle-varying signals.

    Because v_sub lies on a fixed uniform 2 mV voltage grid (identical across cycles),
    voltage-domain entropy metrics reconstruct the true uniform-time voltage curve V(tau)
    and voltage rate dV/dtau from (v_sub, tau_sub), while spectral and Hjorth metrics
    operate on the non-monotonic incremental capacity curve dQ/dV.
    """
    n_pts = len(v_sub)
    if n_pts < 5:
        return {
            k: 0.0
            for k in [
                "sampen_voltage",
                "sampen_dqdv",
                "perm_entropy_voltage",
                "shannon_entropy_voltage",
                "fft_dominant_ratio",
                "hjorth_activity",
                "hjorth_mobility",
                "hjorth_complexity",
            ]
        }

    # Reconstruct uniform-time V(tau) and dV/dtau from (v_sub, tau_sub)
    if tau_sub is not None and len(tau_sub) == n_pts and float(tau_sub[-1] - tau_sub[0]) > 1e-9:
        tau_uniform = np.linspace(float(tau_sub[0]), float(tau_sub[-1]), n_pts)
        v_of_tau = np.interp(tau_uniform, tau_sub, v_sub)
        dv_dtau = np.gradient(v_of_tau, tau_uniform)
    else:
        v_of_tau = np.asarray(v_sub, dtype=float)
        dv_dtau = np.asarray(dq_sub, dtype=float)

    dq_arr = np.asarray(dq_sub, dtype=float)

    se_v = sample_entropy_numba(v_of_tau)
    se_dq = sample_entropy_numba(dq_arr)

    # Discrete Shannon entropy of uniform-time V(tau) occupancy across voltage bins
    counts, _ = np.histogram(v_of_tau, bins=min(10, n_pts // 2))
    counts = counts[counts > 0]
    if len(counts) > 0:
        probs_h = counts / np.sum(counts)
        shannon = float(-np.sum(probs_h * np.log(probs_h)))
    else:
        shannon = 0.0

    # FFT dominant frequency ratio on non-monotonic dQ/dV curve
    fft_vals = np.abs(np.fft.rfft(dq_arr - np.mean(dq_arr)))
    if len(fft_vals) > 1 and np.sum(fft_vals[1:]) > 1e-8:
        fft_ratio = float(np.max(fft_vals[1:]) / np.sum(fft_vals[1:]))
    else:
        fft_ratio = 1.0

    # Permutation entropy (order 3) on non-monotonic voltage rate dV/dtau
    perm_ent = 0.0
    if n_pts >= 4:
        patterns = [tuple(np.argsort(dv_dtau[i : i + 3])) for i in range(n_pts - 2)]
        _, p_counts = np.unique(patterns, axis=0, return_counts=True)
        probs = p_counts / np.sum(p_counts)
        perm_ent = float(-np.sum(probs * np.log(probs)))

    # Hjorth parameters on non-monotonic dQ/dV curve
    diff1 = np.diff(dq_arr)
    diff2 = np.diff(diff1)
    var0 = float(np.var(dq_arr))
    var1 = float(np.var(diff1)) if len(diff1) > 0 else 0.0
    var2 = float(np.var(diff2)) if len(diff2) > 0 else 0.0

    activity = var0
    mobility = float(np.sqrt(var1 / max(var0, 1e-12))) if var0 > 1e-12 else 0.0
    complexity = (
        float(np.sqrt(var2 / max(var1, 1e-12)) / max(mobility, 1e-12)) if (var1 > 1e-12 and mobility > 1e-12) else 0.0
    )

    return {
        "sampen_voltage": se_v,
        "sampen_dqdv": se_dq,
        "perm_entropy_voltage": perm_ent,
        "shannon_entropy_voltage": shannon,
        "fft_dominant_ratio": fft_ratio,
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }
