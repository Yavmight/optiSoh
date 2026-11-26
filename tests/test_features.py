"""Tests for 42 candidate health indicators and feature hygiene."""

import numpy as np
import pytest
from optiSoh.contracts.feature_rules import FORBIDDEN_FEATURES, validate_feature_names
from optiSoh.features.registry import (
    FEATURE_REGISTRY,
    extract_all_window_features,
    extract_cycle_window_features,
    extract_feature_matrix_bundle,
)


def test_forbidden_features():
    for f in FORBIDDEN_FEATURES:
        with pytest.raises(ValueError):
            validate_feature_names(["valid_feature", f])


def test_feature_registry_exact_match():
    # Test feature extraction on synthetic window slices
    n_pts = 50
    v = np.linspace(3.6, 4.0, n_pts)
    q = np.linspace(0.1, 0.8, n_pts)
    dq = np.sin(np.linspace(0, np.pi, n_pts)) + 0.1
    delta_t = np.linspace(0, 3.0, n_pts)
    dt = np.gradient(delta_t, v)
    tau = np.linspace(0, 1800, n_pts)

    feats = extract_all_window_features(v, q, dq, delta_t, dt, tau)

    assert len(FEATURE_REGISTRY) == 42
    assert set(feats.keys()) == set(FEATURE_REGISTRY)
    assert len(feats) == 42
    for val in feats.values():
        assert np.isfinite(val)


def test_feature_extraction_on_synthetic_cycle(synthetic_bundle):
    """Test feature extraction on a slice from synthetic bundle fixture."""
    cycle = synthetic_bundle.cycles[0]
    mask = (cycle.v_grid >= 3.65) & (cycle.v_grid <= 4.05)
    feats = extract_all_window_features(
        cycle.v_grid[mask],
        cycle.q_norm[mask],
        cycle.dq_dv[mask],
        cycle.delta_t[mask],
        cycle.dt_dv[mask],
        cycle.tau[mask],
    )
    assert len(feats) == 42
    for val in feats.values():
        assert np.isfinite(val)
    assert feats["shannon_entropy_voltage"] >= 0.0


def test_p2_1_previously_dead_features_vary_across_aging():
    """Verify the 8 entropy/kinetics features are non-constant on a fixed uniform voltage grid."""
    v = np.arange(3.60, 4.101, 0.002)
    rows = []
    # Simulate 8 aging states with shifting IC peak and nonlinear charging time tau(V)
    for step in range(8):
        peak_v = 3.74 + 0.015 * step
        peak_h = 2.0 - 0.12 * step
        dq = peak_h * np.exp(-0.5 * ((v - peak_v) / (0.04 + 0.005 * step)) ** 2) + 0.35 + 0.02 * np.sin(25 * v + step)
        q = np.cumsum(dq) * 0.002
        q -= q[0]
        tau = q * (1.2 + 0.1 * step) + 0.05 * (v - v[0]) ** 2
        delta_t = 1.5 * q + 0.2 * (v - v[0])
        dt = np.gradient(delta_t, v)
        rows.append(extract_all_window_features(v, q, dq, delta_t, dt, tau))

    target_feats = [
        "sampen_voltage",
        "perm_entropy_voltage",
        "shannon_entropy_voltage",
        "fft_dominant_ratio",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "quartile_skewness",
    ]
    for name in target_feats:
        vals = np.array([r[name] for r in rows], dtype=float)
        assert np.all(np.isfinite(vals)), f"{name} produced non-finite value"
        assert np.std(vals) > 1e-6, f"{name} is still constant across cycles (std={np.std(vals)})"


def test_p2_2_temp_ratio_bounded_and_shift_invariant():
    """Verify temp_ratio_end_start stays bounded in [-1, 1] when delta_t starts at 0.0."""
    v = np.linspace(3.60, 4.10, 100)
    q = np.linspace(0.0, 0.7, 100)
    dq = np.gradient(q, v)
    tau = np.linspace(0.0, 1.0, 100)
    # Re-zeroed delta_t starting at exact 0.0 (previously caused 1.5 / 1e-4 = 15,000 explosion)
    delta_t = 1.5 * ((v - v[0]) / (v[-1] - v[0])) ** 2
    dt = np.gradient(delta_t, v)

    feats_zeroed = extract_all_window_features(v, q, dq, delta_t, dt, tau)
    feats_shifted = extract_all_window_features(v, q, dq, delta_t + 25.0, dt, tau)

    r0 = feats_zeroed["temp_ratio_end_start"]
    r_shift = feats_shifted["temp_ratio_end_start"]
    assert -1.0 <= r0 <= 1.0
    assert np.isclose(r0, r_shift, atol=1e-9)


def test_p2_3_rezeroing_eliminates_pre_window_offset_leakage():
    """Verify adding arbitrary pre-v_start offsets to q, delta_t, tau does not change any feature."""
    v = np.arange(3.75, 4.051, 0.002)
    dq = 1.6 * np.exp(-0.5 * ((v - 3.85) / 0.05) ** 2) + 0.3
    q = np.cumsum(dq) * 0.002
    q -= q[0]
    delta_t = 2.0 * q + 0.1 * (v - v[0])
    dt = np.gradient(delta_t, v)
    tau = 1.5 * q

    f_base = extract_all_window_features(v, q, dq, delta_t, dt, tau)
    f_offset = extract_all_window_features(v, q + 0.45, dq, delta_t + 3.7, dt, tau + 500.0)

    for k in FEATURE_REGISTRY:
        assert np.isclose(f_base[k], f_offset[k], rtol=1e-9, atol=1e-9), f"{k} leaked pre-window offset!"

    # Non-finite input must raise ValueError rather than silently producing NaNs
    q_bad = q.copy()
    q_bad[5] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        extract_all_window_features(v, q_bad, dq, delta_t, dt, tau)


def test_p2_3_extract_feature_matrix_bundle_filters_nans_and_aligns_metadata(synthetic_bundle):
    """Verify NaN cycles are excluded while X, y_soh, cell_indices, and cycle_indices stay aligned."""
    # Corrupt first 5 cycles below 3.85 V (simulating aged-cell ohmic rebound)
    for c in synthetic_bundle.cycles[:5]:
        low_mask = c.v_grid < 3.85 - 1e-6
        c.q_norm[low_mask] = np.nan
        c.dq_dv[low_mask] = np.nan

    assert extract_cycle_window_features(synthetic_bundle.cycles[0], 3.60, 4.10) is None
    assert extract_cycle_window_features(synthetic_bundle.cycles[0], 3.95, 4.10) is not None

    bundle_360 = extract_feature_matrix_bundle(synthetic_bundle.cycles, 3.60, 4.10)
    expected_n = len(synthetic_bundle.cycles) - 5
    assert bundle_360.X.shape == (expected_n, 42)
    assert np.all(np.isfinite(bundle_360.X))
    assert bundle_360.y_soh.shape == (expected_n,)
    assert bundle_360.cell_indices.shape == (expected_n,)
    assert bundle_360.cycle_indices.shape == (expected_n,)

    # Exact 150 mV boundary [3.95, 4.10] V must keep all cycles (testing IEEE-754 width tolerance too)
    bundle_395 = extract_feature_matrix_bundle(synthetic_bundle.cycles, 3.95, 4.10)
    assert bundle_395.X.shape == (len(synthetic_bundle.cycles), 42)
    assert np.all(np.isfinite(bundle_395.X))
