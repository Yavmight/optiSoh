"""Accuracy-guarded Utopia Knee-Point selector on the Pareto front."""
import numpy as np

def select_guarded_knee_point(pareto_costs: np.ndarray, guard_factor: float = 1.25) -> int:
    """
    Select knee point minimizing distance to normalized utopia point,
    excluding degenerately underfit solutions with RMSE > guard_factor * median front RMSE.
    """
    n_points, n_obj = pareto_costs.shape
    if n_points == 1:
        return 0

    f1_vals = pareto_costs[:, 0]
    threshold = np.median(f1_vals) * guard_factor
    valid_idx = np.where(f1_vals <= threshold)[0]
    if len(valid_idx) == 0:
        valid_idx = np.arange(n_points)

    sub_costs = pareto_costs[valid_idx]
    c_min = np.min(sub_costs, axis=0)
    c_max = np.max(sub_costs, axis=0)
    denom = np.where(c_max - c_min < 1e-8, 1.0, c_max - c_min)
    norm_costs = (sub_costs - c_min) / denom
    distances = np.linalg.norm(norm_costs, axis=1)
    best_sub = np.argmin(distances)
    return int(valid_idx[best_sub])
