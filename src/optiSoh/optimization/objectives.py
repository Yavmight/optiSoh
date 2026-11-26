"""The 4 minimized objectives and per-cell window coverage guard for co-optimization."""

from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..contracts.schemas import CanonicalCycleGrid

DEFAULT_MIN_CELL_COVERAGE: float = 0.95
INFEASIBLE_BASE_PENALTY: float = 100.0


def cycle_covers_window(cycle: CanonicalCycleGrid, v_start: float, v_end: float) -> bool:
    """Return True iff all 5 canonical channels are finite across [v_start, v_end] with zero extrapolation."""
    mask = (cycle.v_grid >= v_start - 1e-6) & (cycle.v_grid <= v_end + 1e-6)
    if not np.any(mask):
        return False
    return bool(
        np.all(np.isfinite(cycle.q_norm[mask]))
        and np.all(np.isfinite(cycle.dq_dv[mask]))
        and np.all(np.isfinite(cycle.delta_t[mask]))
        and np.all(np.isfinite(cycle.dt_dv[mask]))
        and np.all(np.isfinite(cycle.tau[mask]))
    )


def compute_per_cell_coverage(
    cycles: Sequence[CanonicalCycleGrid],
    v_start: float,
    v_end: float,
    cell_ids: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compute the fraction of cycles per cell that fully cover [v_start, v_end]."""
    totals: Dict[str, int] = defaultdict(int)
    covered: Dict[str, int] = defaultdict(int)
    allowed = set(cell_ids) if cell_ids is not None else None

    for c in cycles:
        if allowed is not None and c.cell_id not in allowed:
            continue
        totals[c.cell_id] += 1
        if cycle_covers_window(c, v_start, v_end):
            covered[c.cell_id] += 1

    target_ids = list(cell_ids) if cell_ids is not None else sorted(totals.keys())
    return {cid: (covered[cid] / totals[cid] if totals[cid] > 0 else 0.0) for cid in target_ids}


def check_window_coverage(
    cycles: Sequence[CanonicalCycleGrid],
    v_start: float,
    v_end: float,
    min_cell_coverage: float = DEFAULT_MIN_CELL_COVERAGE,
    cell_ids: Optional[Sequence[str]] = None,
) -> Tuple[bool, float, Dict[str, float]]:
    """Validate whether every cell meets the minimum per-cell cycle coverage threshold."""
    per_cell = compute_per_cell_coverage(cycles, v_start, v_end, cell_ids=cell_ids)
    if not per_cell:
        return False, 0.0, {}
    min_cov = float(min(per_cell.values()))
    is_feasible = (min_cov + 1e-12) >= float(min_cell_coverage)
    return is_feasible, min_cov, per_cell


def penalized_infeasible_objectives(
    min_cell_coverage: float,
    target_coverage: float = DEFAULT_MIN_CELL_COVERAGE,
    feature_mask: Optional[np.ndarray] = None,
    model_complexity: float = 1.0,
    base_penalty: float = INFEASIBLE_BASE_PENALTY,
) -> np.ndarray:
    """
    Return a smooth penalty vector strictly dominated by every feasible 4D objective vector.

    Because f3 and f4 are shifted above 1.0 (+ shortfall), low-coverage windows with few
    features cannot pollute the Pareto front, while higher-coverage infeasible windows still
    dominate lower-coverage ones to guide NSGA-II / Optuna toward the feasible region.
    """
    shortfall = max(0.0, float(target_coverage) - float(min_cell_coverage))
    rmse_pen = float(base_penalty) * (1.0 + shortfall)
    f3_raw = (
        float(np.mean(np.asarray(feature_mask, dtype=float)))
        if feature_mask is not None and len(feature_mask) > 0
        else 1.0
    )
    f4_raw = max(0.0, float(model_complexity))
    # Add tiny epsilon (1e-4) so even shortfall -> 0 stays strictly > 1.0
    f3_pen = 1.0 + shortfall + 0.01 * f3_raw + 1e-4
    f4_pen = 1.0 + shortfall + 0.01 * f4_raw + 1e-4
    return np.array([rmse_pen, rmse_pen, f3_pen, f4_pen], dtype=float)


def compute_objectives(
    cell_rmses: dict,
    feature_mask: np.ndarray,
    model_complexity: float,
    min_cell_coverage: Optional[float] = None,
    target_coverage: float = DEFAULT_MIN_CELL_COVERAGE,
) -> np.ndarray:
    """
    Compute the 4 minimized objectives for co-optimization:
      f1: mean cross-cell RMSE
      f2: worst-cell RMSE
      f3: feature fraction N_selected / N_total
      f4: model complexity

    If min_cell_coverage is provided and falls below target_coverage (default 0.95),
    returns penalized_infeasible_objectives(...) to prevent aged-cycle selection bias.
    """
    if min_cell_coverage is not None and (float(min_cell_coverage) + 1e-12) < float(target_coverage):
        return penalized_infeasible_objectives(
            min_cell_coverage=min_cell_coverage,
            target_coverage=target_coverage,
            feature_mask=feature_mask,
            model_complexity=model_complexity,
        )

    rmses = list(cell_rmses.values())
    f1 = float(np.mean(rmses))
    f2 = float(np.max(rmses))
    f3 = float(np.mean(np.asarray(feature_mask, dtype=float)))
    f4 = float(model_complexity)
    return np.array([f1, f2, f3, f4], dtype=float)
