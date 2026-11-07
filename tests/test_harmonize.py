"""Tests for 2 mV grid harmonization and normalization."""

import numpy as np
from sohml.contracts.constants import DEFAULT_V_GRID_MAX, DEFAULT_V_GRID_MIN, DEFAULT_V_GRID_STEP


def test_v_grid_resolution():
    v_grid = np.arange(
        DEFAULT_V_GRID_MIN,
        DEFAULT_V_GRID_MAX + DEFAULT_V_GRID_STEP / 2,
        DEFAULT_V_GRID_STEP,
    )
    step = np.diff(v_grid)
    assert np.allclose(step, DEFAULT_V_GRID_STEP)


def test_synthetic_bundle_grid_integrity(synthetic_bundle):
    """Verify synthetic bundle cycles adhere to the canonical 2 mV grid constants."""
    assert len(synthetic_bundle.cycles) > 0
    cycle = synthetic_bundle.cycles[0]
    assert np.allclose(np.diff(cycle.v_grid), DEFAULT_V_GRID_STEP)
    assert cycle.v_grid[0] >= DEFAULT_V_GRID_MIN
    assert cycle.v_grid[-1] <= DEFAULT_V_GRID_MAX + 1e-4
