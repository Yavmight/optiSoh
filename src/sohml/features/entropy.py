"""Numba-accelerated Sample Entropy and signal complexity metrics."""
import numpy as np
from numba import njit

@njit(fastmath=True)
def sample_entropy_numba(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Fast Numba implementation of Sample Entropy."""
    N = len(signal)
    if N <= m + 1:
        return 0.0
    r_thresh = r * np.std(signal)
    if r_thresh <= 1e-8:
        return 0.0

    count_m = 0
    count_m1 = 0
    for i in range(N - m):
        for j in range(i + 1, N - m):
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
    return -np.log(count_m1 / count_m)
