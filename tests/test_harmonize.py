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
