"""Numba-accelerated Sample Entropy and signal complexity metrics."""

from typing import Dict

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


def extract_entropy_features(v_sub: np.ndarray, dq_sub: np.ndarray) -> Dict[str, float]:
    """Extract 8 complexity and entropy features."""
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

    se_v = sample_entropy_numba(v_sub)
    se_dq = sample_entropy_numba(dq_sub)

    # Shannon histogram entropy
    hist, _ = np.histogram(v_sub, bins=min(10, n_pts // 2), density=True)
    hist = hist[hist > 0]
    shannon = float(-np.sum(hist * np.log(hist))) if len(hist) > 0 else 0.0

    # FFT dominant ratio
    fft_vals = np.abs(np.fft.rfft(v_sub - np.mean(v_sub)))
    if len(fft_vals) > 1 and np.sum(fft_vals) > 1e-8:
        fft_ratio = float(np.max(fft_vals[1:]) / np.sum(fft_vals[1:]))
    else:
        fft_ratio = 1.0

    # Permutation entropy (order 3)
    perm_ent = 0.0
    if n_pts >= 4:
        patterns = []
        for i in range(n_pts - 2):
            patterns.append(tuple(np.argsort(v_sub[i : i + 3])))
        _, counts = np.unique(patterns, axis=0, return_counts=True)
        probs = counts / np.sum(counts)
        perm_ent = float(-np.sum(probs * np.log(probs)))

    # Hjorth parameters
    diff1 = np.diff(v_sub)
    diff2 = np.diff(diff1)
    var0 = np.var(v_sub)
    var1 = np.var(diff1) if len(diff1) > 0 else 1e-8
    var2 = np.var(diff2) if len(diff2) > 0 else 1e-8

    activity = float(var0)
    mobility = float(np.sqrt(max(var1 / max(var0, 1e-8), 0.0)))
    complexity = float(np.sqrt(max((var2 / max(var1, 1e-8)), 0.0)) / max(mobility, 1e-8))

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
