"""Catalog of all 42+ candidate health indicators."""
from typing import List

FEATURE_REGISTRY: List[str] = [
    # IC features (12)
    "ic_peak_height", "ic_peak_voltage", "ic_peak_prominence", "ic_peak_fwhm",
    "ic_area_total", "ic_area_sub1", "ic_area_sub2", "ic_area_sub3",
    "ic_mean", "ic_std", "ic_skew", "ic_kurtosis",
    # Thermal features (10)
    "dt_max", "dt_voltage_at_max", "dt_mean", "dt_std", "delta_temp_rise",
    "dt_area", "dt_skew", "dt_kurtosis", "temp_ratio_end_start", "temp_voltage_slope",
    # Entropy features (8)
    "sampen_voltage", "sampen_dqdv", "perm_entropy_voltage", "shannon_entropy_voltage",
    "fft_dominant_ratio", "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    # Kinetics features (8)
    "tau_duration", "dv_dtau_mean", "dv_dtau_max", "dv_dtau_min",
    "curvature_mean", "energy_partial", "time_to_midpoint_ratio", "quartile_skewness",
    # Coupling features (4)
    "ic_dt_pearson", "ic_dt_spearman", "vq_arc_length", "peak_voltage_ratio_ic_dt"
]
