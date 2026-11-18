"""The 4 minimized objectives for co-optimization."""

import numpy as np


def compute_objectives(cell_rmses: dict, feature_mask: np.ndarray, model_complexity: float):
    """
    f1: mean cross-cell RMSE
    f2: worst-cell RMSE
    f3: feature fraction N_selected / N_total
    f4: model complexity
    """
    rmses = list(cell_rmses.values())
    f1 = float(np.mean(rmses))
    f2 = float(np.max(rmses))
    f3 = float(np.mean(feature_mask.astype(float)))
    f4 = float(model_complexity)
    return np.array([f1, f2, f3, f4])
