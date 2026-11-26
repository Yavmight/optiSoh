"""Tests for CC segment extraction, SOH labeling, and 2 mV grid harmonization."""

import numpy as np
import pytest
from optiSoh.contracts.constants import DEFAULT_V_GRID_MAX, DEFAULT_V_GRID_MIN, DEFAULT_V_GRID_STEP
from optiSoh.data.cc_segment import extract_cc_segment
from optiSoh.data.config_resolver import get_preprocessing_spec
from optiSoh.data.harmonize import build_canonical_v_grid, resample_to_grid
from optiSoh.data.labels import compute_soh_labels, detect_capacity_regeneration, detect_label_outliers


def test_v_grid_resolution():
    v_grid = build_canonical_v_grid()
    step = np.diff(v_grid)
    assert np.allclose(step, DEFAULT_V_GRID_STEP)
    assert np.isclose(v_grid[0], DEFAULT_V_GRID_MIN)
    assert np.isclose(v_grid[-1], DEFAULT_V_GRID_MAX)


def test_synthetic_bundle_grid_integrity(synthetic_bundle):
    """Verify synthetic bundle cycles adhere to the canonical 2 mV grid constants."""
    assert len(synthetic_bundle.cycles) > 0
    cycle = synthetic_bundle.cycles[0]
    assert np.allclose(np.diff(cycle.v_grid), DEFAULT_V_GRID_STEP)
    assert cycle.v_grid[0] >= DEFAULT_V_GRID_MIN
    assert cycle.v_grid[-1] <= DEFAULT_V_GRID_MAX + 1e-4


def test_extract_cc_segment_and_glitch_rejection():
    # Simulated charge curve: brief ramp -> 100 pts at 1.5 A (3.5 to 4.2 V) -> CV taper
    v = np.concatenate([np.linspace(3.4, 3.5, 5), np.linspace(3.5, 4.2, 100), np.full(30, 4.2)])
    cur = np.concatenate([np.linspace(0.0, 1.5, 5), np.full(100, 1.5), np.linspace(1.4, 0.05, 30)])
    sl = extract_cc_segment(v, cur, target_current=1.5)
    assert sl is not None
    assert sl.start >= 4 and sl.stop <= 110
    assert v[sl][-1] - v[sl][0] >= 0.65

    # Reject corrupted 8 V NASA sensor glitch
    v_glitch = np.linspace(3.8, 8.3, 100)
    cur_glitch = np.full(100, 0.2)
    assert extract_cc_segment(v_glitch, cur_glitch, target_current=1.5) is None


def test_compute_soh_labels():
    caps = [1.85, 1.80, 1.48]
    soh, q0 = compute_soh_labels(caps, initial_capacity_rule="first_valid")
    assert np.isclose(q0, 1.85)
    assert np.isclose(soh[0], 100.0)
    assert np.isclose(soh[-1], 100.0 * 1.48 / 1.85)

    with pytest.raises(ValueError):
        compute_soh_labels([np.nan, 0.0, -1.0])


def test_detect_capacity_regeneration():
    # Genuine sustained jump (+3.5%) after gradual fade is flagged
    soh_regen = [85.0, 84.5, 84.0, 87.5, 87.0, 86.5]
    flags = detect_capacity_regeneration(soh_regen, jump_threshold_pct=1.0)
    assert flags.tolist() == [False, False, False, True, False, False]

    # Recovery after an isolated outlier dip (76.5%) must NOT be flagged as regeneration
    soh_dip = [85.0, 84.8, 76.5, 85.1, 84.9]
    flags_dip = detect_capacity_regeneration(soh_dip, jump_threshold_pct=1.0)
    assert flags_dip.tolist() == [False, False, False, False, False]


def test_detect_label_outliers():
    # Isolated mid-trajectory dip (-7.6%) and sharp end drop (-20%)
    soh = [85.0, 84.5, 76.5, 84.0, 83.5, 83.0, 62.0]
    flags = detect_label_outliers(soh, window=5, threshold_pct=2.0)
    assert flags.tolist() == [False, False, True, False, False, False, True]


def test_preprocessing_spec_loaded_from_default_yaml():
    spec = get_preprocessing_spec()
    for key in (
        "cc_current_tolerance",
        "cc_min_points",
        "cc_min_voltage_span",
        "cc_v_min_valid",
        "cc_v_max_valid",
        "savgol_time_window",
        "savgol_grid_window",
        "savgol_thermal_window",
        "savgol_polyorder",
        "regeneration_jump_pct",
        "label_outlier_window",
        "label_outlier_threshold_pct",
    ):
        assert key in spec
    assert spec["cc_current_tolerance"] == 0.05
    assert spec["savgol_time_window"] == 11
    assert spec["savgol_grid_window"] == 11
    assert spec["savgol_thermal_window"] == 35


def test_resample_to_grid_no_extrapolation():
    # Observed CC voltage only spans [3.70, 4.12] V
    n = 200
    v = np.linspace(3.70, 4.12, n)
    q = np.linspace(0.0, 1.2, n)
    temp = np.linspace(24.0, 27.0, n)
    time = np.linspace(0.0, 2880.0, n)

    res = resample_to_grid(v, q, temp, time, nominal_capacity=2.0, c_rate=0.75)
    vg = res["v_grid"]

    # Strictly NaN outside [3.70, 4.12] V (no extrapolation)
    assert np.all(np.isnan(res["q_norm"][vg < 3.699]))
    assert np.all(np.isnan(res["q_norm"][vg > 4.121]))

    # Strictly finite inside [3.71, 4.11] V
    inside = (vg >= 3.71) & (vg <= 4.11)
    assert np.all(np.isfinite(res["q_norm"][inside]))
    assert np.all(np.isfinite(res["dq_dv"][inside]))
    assert np.all(np.isfinite(res["dt_dv"][inside]))
    assert np.all(np.isfinite(res["tau"][inside]))
    # Nominal capacity normalization: max q_norm <= 1.2 / 2.0 = 0.60
    assert float(np.nanmax(res["q_norm"])) <= 0.61
