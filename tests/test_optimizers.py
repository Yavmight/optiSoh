"""Tests for Pareto knee selection and optimizers."""
import numpy as np
from sohml.optimization.knee import select_guarded_knee_point

def test_knee_selection():
    pareto_costs = np.array([
        [1.0, 2.0, 0.1, 0.2],
        [0.8, 1.5, 0.3, 0.4],
        [0.7, 1.2, 0.8, 0.9],
    ])
    idx = select_guarded_knee_point(pareto_costs)
    assert 0 <= idx < 3
