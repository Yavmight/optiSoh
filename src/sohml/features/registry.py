"""Catalog of all 42 candidate health indicators and unified extractor."""

from typing import Dict, List

import numpy as np

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
) -> Dict[str, float]:
    """Extract all 42 candidate health indicators for a specified window slice."""
    feats: Dict[str, float] = {}
    feats.update(extract_ic_features(v_sub, dq_sub))
    feats.update(extract_thermal_features(v_sub, dt_sub, delta_t_sub))
    feats.update(extract_entropy_features(v_sub, dq_sub))
    feats.update(extract_kinetics_features(v_sub, tau_sub, q_sub))
    feats.update(extract_coupling_features(v_sub, q_sub, dq_sub, dt_sub))

    # Guarantee all registry features exist
    for k in FEATURE_REGISTRY:
        if k not in feats:
            feats[k] = 0.0
    return feats
