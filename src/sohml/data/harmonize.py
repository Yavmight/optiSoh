"""Harmonizes raw CC curves onto canonical 2 mV uniform voltage grid."""

import numpy as np


def resample_to_grid(
    voltage: np.ndarray,
    capacity: np.ndarray,
    temp: np.ndarray,
    time: np.ndarray,
    nominal_capacity: float,
    v_grid: np.ndarray,
):
    """Smooth, normalize by nominal capacity, and interpolate onto v_grid."""
    pass
