"""Catalog of all 42 candidate health indicators and unified partial-window extractor."""

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..contracts.schemas import CanonicalCycleGrid, FeatureMatrixBundle, WindowQuery
from .coupling import extract_coupling_features
from .differential_thermal import extract_thermal_features
from .entropy import extract_entropy_features
from .incremental_capacity import extract_ic_features
from .kinetics import extract_kinetics_features

FEATURE_REGISTRY: List[str] = [
    # IC features (12)
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
    # Thermal features (10)
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
    # Entropy & Complexity features (8)
    "sampen_voltage",
    "sampen_dqdv",
    "perm_entropy_voltage",
    "shannon_entropy_voltage",
    "fft_dominant_ratio",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    # Kinetics features (8)
    "tau_duration",
    "dv_dtau_mean",
    "dv_dtau_max",
    "dv_dtau_min",
    "curvature_mean",
    "energy_partial",
    "time_to_midpoint_ratio",
    "quartile_skewness",
    # Coupling features (4)
    "ic_dt_pearson",
    "ic_dt_spearman",
    "vq_arc_length",
    "peak_voltage_ratio_ic_dt",
]


def extract_all_window_features(
    v_sub: np.ndarray,
    q_sub: np.ndarray,
    dq_sub: np.ndarray,
    delta_t_sub: np.ndarray,
    dt_sub: np.ndarray,
    tau_sub: np.ndarray,
    rezero: bool = True,
) -> Dict[str, float]:
    """Extract all 42 candidate health indicators for a specified window slice.

    When rezero=True (default), cumulative channels (q_sub, delta_t_sub, tau_sub) are
    re-zeroed at the start of the window so no pre-v_start charge, thermal, or duration
    history leaks into partial-window features. Raises ValueError on non-finite inputs.
    """
    v_arr = np.asarray(v_sub, dtype=float)
    q_arr = np.asarray(q_sub, dtype=float)
    dq_arr = np.asarray(dq_sub, dtype=float)
    delta_t_arr = np.asarray(delta_t_sub, dtype=float)
    dt_arr = np.asarray(dt_sub, dtype=float)
    tau_arr = np.asarray(tau_sub, dtype=float)

    for name, arr in (
        ("v_sub", v_arr),
        ("q_sub", q_arr),
        ("dq_sub", dq_arr),
        ("delta_t_sub", delta_t_arr),
        ("dt_sub", dt_arr),
        ("tau_sub", tau_arr),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values found in {name}; filter incomplete cycles before feature extraction.")

    if rezero and len(v_arr) > 0:
        if len(q_arr) > 0:
            q_arr = q_arr - q_arr[0]
        if len(delta_t_arr) > 0:
            delta_t_arr = delta_t_arr - delta_t_arr[0]
        if len(tau_arr) > 0:
            tau_arr = tau_arr - tau_arr[0]

    feats: Dict[str, float] = {}
    feats.update(extract_ic_features(v_arr, dq_arr))
    feats.update(extract_thermal_features(v_arr, dt_arr, delta_t_arr))
    feats.update(extract_entropy_features(v_arr, dq_arr, tau_sub=tau_arr))
    feats.update(extract_kinetics_features(v_arr, tau_arr, q_arr))
    feats.update(extract_coupling_features(v_arr, q_arr, dq_arr, dt_arr))

    # Guarantee all registry features exist and are finite
    for k in FEATURE_REGISTRY:
        if k not in feats or not np.isfinite(feats[k]):
            feats[k] = 0.0
    return feats


def extract_cycle_window_features(
    cycle: CanonicalCycleGrid,
    v_start: float,
    v_end: float,
) -> Optional[Dict[str, float]]:
    """Slice [v_start, v_end] from a CanonicalCycleGrid, re-zero cumulative channels, and extract 42 features.

    Returns None if any point inside [v_start, v_end] is non-finite in any of the 5 channels
    (strict zero-extrapolation integrity).
    """
    mask = (cycle.v_grid >= v_start - 1e-6) & (cycle.v_grid <= v_end + 1e-6)
    if not np.any(mask):
        return None

    v_sub = cycle.v_grid[mask]
    q_sub = cycle.q_norm[mask]
    dq_sub = cycle.dq_dv[mask]
    delta_t_sub = cycle.delta_t[mask]
    dt_sub = cycle.dt_dv[mask]
    tau_sub = cycle.tau[mask]

    if not (
        np.all(np.isfinite(q_sub))
        and np.all(np.isfinite(dq_sub))
        and np.all(np.isfinite(delta_t_sub))
        and np.all(np.isfinite(dt_sub))
        and np.all(np.isfinite(tau_sub))
    ):
        return None

    return extract_all_window_features(
        v_sub=v_sub,
        q_sub=q_sub,
        dq_sub=dq_sub,
        delta_t_sub=delta_t_sub,
        dt_sub=dt_sub,
        tau_sub=tau_sub,
        rezero=True,
    )


def extract_feature_matrix_bundle(
    cycles: Sequence[CanonicalCycleGrid],
    v_start: float,
    v_end: float,
) -> FeatureMatrixBundle:
    """Extract a strictly finite, re-zeroed FeatureMatrixBundle across all valid cycles in [v_start, v_end].

    Cycles with missing (NaN) points inside [v_start, v_end] are excluded while keeping
    X, y_soh, cell_indices, and cycle_indices strictly row-aligned.
    """
    window = WindowQuery(v_start=float(v_start), v_end=float(v_end))
    rows: List[List[float]] = []
    sohs: List[float] = []
    cell_ids: List[str] = []
    cyc_idxs: List[int] = []

    for cycle in cycles:
        feats = extract_cycle_window_features(cycle, window.v_start, window.v_end)
        if feats is None:
            continue
        rows.append([feats[name] for name in FEATURE_REGISTRY])
        sohs.append(float(cycle.soh_true))
        cell_ids.append(cycle.cell_id)
        cyc_idxs.append(int(cycle.cycle_index))

    if rows:
        X = np.asarray(rows, dtype=np.float64)
        y_soh = np.asarray(sohs, dtype=np.float64)
        cell_indices = np.asarray(cell_ids)
        cycle_indices = np.asarray(cyc_idxs, dtype=np.int64)
    else:
        X = np.empty((0, len(FEATURE_REGISTRY)), dtype=np.float64)
        y_soh = np.empty((0,), dtype=np.float64)
        cell_indices = np.empty((0,), dtype=str)
        cycle_indices = np.empty((0,), dtype=np.int64)

    return FeatureMatrixBundle(
        window=window,
        feature_names=list(FEATURE_REGISTRY),
        X=X,
        y_soh=y_soh,
        cell_indices=cell_indices,
        cycle_indices=cycle_indices,
    )
