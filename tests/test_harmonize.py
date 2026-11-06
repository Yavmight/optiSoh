"""Tests for 2 mV grid harmonization and normalization."""
import numpy as np

def test_v_grid_resolution():
    v_grid = np.arange(3.50, 4.182, 0.002)
    step = np.diff(v_grid)
    assert np.allclose(step, 0.002)
