"""Harmonizes raw CC curves onto canonical 2 mV uniform voltage grid."""

from typing import Dict, Optional

import numpy as np
from scipy.signal import savgol_filter

from ..contracts.constants import DEFAULT_V_GRID_MAX, DEFAULT_V_GRID_MIN, DEFAULT_V_GRID_STEP
from .config_resolver import get_preprocessing_spec


def build_canonical_v_grid(
    v_min: float = DEFAULT_V_GRID_MIN,
    v_max: float = DEFAULT_V_GRID_MAX,
    step: float = DEFAULT_V_GRID_STEP,
) -> np.ndarray:
    """Construct the canonical uniform 2 mV voltage grid."""
    n_steps = int(round((v_max - v_min) / step))
    return np.round(v_min + np.arange(n_steps + 1) * step, 6)


def _safe_savgol(arr: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing safely, adapting window size for short arrays."""
    n = len(arr)
    if n < polyorder + 2:
        return arr.copy()
    w = min(window_length, n)
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        return arr.copy()
    return savgol_filter(arr, window_length=w, polyorder=polyorder)


def resample_to_grid(
    voltage: np.ndarray,
    capacity: np.ndarray,
    temp: np.ndarray,
    time: np.ndarray,
    nominal_capacity: float,
    v_grid: Optional[np.ndarray] = None,
    c_rate: float = 1.0,
    time_smooth_window: Optional[int] = None,
    grid_smooth_window: Optional[int] = None,
    thermal_smooth_window: Optional[int] = None,
    polyorder: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Smooth raw CC charging curves, normalize by nominal capacity (never Q_0),
    and interpolate strictly within observed voltage bounds onto v_grid (NaN outside).
    Omitted smoothing parameters default to the fixed a-priori values in `configs/default.yaml`.
    """
    if nominal_capacity <= 0:
        raise ValueError(f"nominal_capacity must be positive, got {nominal_capacity}")

    spec = get_preprocessing_spec()
    t_win = int(spec["savgol_time_window"] if time_smooth_window is None else time_smooth_window)
    g_win = int(spec["savgol_grid_window"] if grid_smooth_window is None else grid_smooth_window)
    th_win = int(spec["savgol_thermal_window"] if thermal_smooth_window is None else thermal_smooth_window)
    poly = int(spec["savgol_polyorder"] if polyorder is None else polyorder)

    v = np.asarray(voltage, dtype=float)
    q = np.asarray(capacity, dtype=float)
    temp_arr = np.asarray(temp, dtype=float)
    t = np.asarray(time, dtype=float)

    if v_grid is None:
        v_grid = build_canonical_v_grid()
    else:
        v_grid = np.asarray(v_grid, dtype=float)

    # Filter finite samples
    valid = np.isfinite(v) & np.isfinite(q) & np.isfinite(temp_arr) & np.isfinite(t)
    v, q, temp_arr, t = v[valid], q[valid], temp_arr[valid], t[valid]

    if len(v) < 5:
        nan_arr = np.full_like(v_grid, np.nan, dtype=float)
        return {
            "v_grid": v_grid.copy(),
            "q_norm": nan_arr.copy(),
            "dq_dv": nan_arr.copy(),
            "delta_t": nan_arr.copy(),
            "dt_dv": nan_arr.copy(),
            "tau": nan_arr.copy(),
        }

    # Physics normalization: Q_norm = (Q - Q_start) / Q_nominal (strictly NO Q_0 leakage)
    q_norm_raw = (q - q[0]) / float(nominal_capacity)
    delta_t_raw = temp_arr - temp_arr[0]
    tau_raw = (t - t[0]) * float(c_rate)

    # Light time-domain smoothing to suppress ADC voltage/temperature quantization noise
    v_smooth = _safe_savgol(v, window_length=t_win, polyorder=poly)
    q_smooth = _safe_savgol(q_norm_raw, window_length=t_win, polyorder=poly)
    dt_smooth = _safe_savgol(delta_t_raw, window_length=t_win, polyorder=poly)

    # Enforce strict voltage monotonicity by retaining strictly advancing voltage samples
    keep_idx = [0]
    v_max_so_far = v_smooth[0]
    for idx in range(1, len(v_smooth)):
        if v_smooth[idx] > v_max_so_far + 1e-6:
            keep_idx.append(idx)
            v_max_so_far = v_smooth[idx]

    keep = np.array(keep_idx, dtype=int)
    v_mono = v_smooth[keep]
    q_mono = q_smooth[keep]
    t_mono = dt_smooth[keep]
    tau_mono = tau_raw[keep]

    if len(v_mono) < 5:
        nan_arr = np.full_like(v_grid, np.nan, dtype=float)
        return {
            "v_grid": v_grid.copy(),
            "q_norm": nan_arr.copy(),
            "dq_dv": nan_arr.copy(),
            "delta_t": nan_arr.copy(),
            "dt_dv": nan_arr.copy(),
            "tau": nan_arr.copy(),
        }

    # Interpolate strictly within observed [v_mono[0], v_mono[-1]] (NO extrapolation!)
    q_grid = np.interp(v_grid, v_mono, q_mono, left=np.nan, right=np.nan)
    delta_t_grid = np.interp(v_grid, v_mono, t_mono, left=np.nan, right=np.nan)
    tau_grid = np.interp(v_grid, v_mono, tau_mono, left=np.nan, right=np.nan)

    # Compute smooth derivatives dQ_norm/dV and dT/dV strictly on valid finite grid region
    dq_dv_grid = np.full_like(v_grid, np.nan, dtype=float)
    dt_dv_grid = np.full_like(v_grid, np.nan, dtype=float)

    finite_mask = np.isfinite(q_grid) & np.isfinite(delta_t_grid)
    if int(np.sum(finite_mask)) >= 5:
        v_fin = v_grid[finite_mask]
        q_fin_s = _safe_savgol(q_grid[finite_mask], window_length=g_win, polyorder=poly)
        t_fin_s = _safe_savgol(delta_t_grid[finite_mask], window_length=th_win, polyorder=poly)

        q_grid[finite_mask] = q_fin_s
        delta_t_grid[finite_mask] = t_fin_s
        dq_dv_grid[finite_mask] = np.gradient(q_fin_s, v_fin)
        dt_dv_grid[finite_mask] = np.gradient(t_fin_s, v_fin)

    return {
        "v_grid": v_grid.copy(),
        "q_norm": q_grid,
        "dq_dv": dq_dv_grid,
        "delta_t": delta_t_grid,
        "dt_dv": dt_dv_grid,
        "tau": tau_grid,
    }
