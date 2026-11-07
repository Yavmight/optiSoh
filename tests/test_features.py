"""Tests for 42 candidate health indicators and feature hygiene."""

import numpy as np
import pytest
from sohml.contracts.feature_rules import FORBIDDEN_FEATURES, validate_feature_names
from sohml.features.registry import FEATURE_REGISTRY, extract_all_window_features


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
