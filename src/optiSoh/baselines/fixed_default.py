"""Fixed-window baseline utilities with training-derived coverage-safe reference window selection."""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts.constants import (
    WINDOW_MIN_WIDTH,
    WINDOW_QUANTIZATION_STEP_MV,
    WINDOW_V_MAX,
    WINDOW_V_MIN,
)
from ..contracts.schemas import CanonicalCycleGrid
from ..optimization.objectives import DEFAULT_MIN_CELL_COVERAGE, cycle_covers_window


def filter_cycles_for_window(
    cycles: Sequence[CanonicalCycleGrid],
    v_start: float,
    v_end: float,
) -> List[CanonicalCycleGrid]:
    """Keep only cycles with complete finite coverage across [v_start, v_end] (zero extrapolation)."""
    return [c for c in cycles if cycle_covers_window(c, v_start, v_end)]


def select_reference_window(
    train_cycles: Sequence[CanonicalCycleGrid],
    min_cell_coverage: float = DEFAULT_MIN_CELL_COVERAGE,
    v_min: float = WINDOW_V_MIN,
    v_max: float = WINDOW_V_MAX,
    min_width: float = WINDOW_MIN_WIDTH,
    step_mv: int = WINDOW_QUANTIZATION_STEP_MV,
    fallback_to_max_coverage: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Select the widest quantized [v_start, v_end] window achieving >= min_cell_coverage
    across every cell present in train_cycles (strictly training-only to prevent test leakage).

    Returns (v_start, v_end, per_cell_coverage_dict).
    """
    if not train_cycles:
        raise ValueError("train_cycles must be non-empty to select a reference window.")

    # Group finite-point masks per cell once for fast vectorized evaluation across all candidate windows
    ref_grid = train_cycles[0].v_grid
    by_cell_masks: Dict[str, List[np.ndarray]] = defaultdict(list)
    for c in train_cycles:
        finite_all = (
            np.isfinite(c.q_norm)
            & np.isfinite(c.dq_dv)
            & np.isfinite(c.delta_t)
            & np.isfinite(c.dt_dv)
            & np.isfinite(c.tau)
        )
        by_cell_masks[c.cell_id].append(finite_all)

    cell_ids = sorted(by_cell_masks.keys())
    cell_mask_mats: Dict[str, np.ndarray] = {cid: np.vstack(by_cell_masks[cid]) for cid in cell_ids}

    start_mv_lo = int(round(v_min * 1000))
    end_mv_hi = int(round(v_max * 1000))
    min_width_mv = int(round(min_width * 1000))

    best_feasible: Optional[Tuple[int, float, float, int, float, float, Dict[str, float]]] = None
    best_fallback: Optional[Tuple[float, int, float, int, float, float, Dict[str, float]]] = None

    for s_mv in range(start_mv_lo, end_mv_hi - min_width_mv + 1, step_mv):
        v_s = s_mv / 1000.0
        for e_mv in range(s_mv + min_width_mv, end_mv_hi + 1, step_mv):
            v_e = e_mv / 1000.0
            grid_mask = (ref_grid >= v_s - 1e-6) & (ref_grid <= v_e + 1e-6)
            if not np.any(grid_mask):
                continue

            per_cell_cov = {
                cid: float(np.mean(np.all(mat[:, grid_mask], axis=1))) for cid, mat in cell_mask_mats.items()
            }
            min_cov = min(per_cell_cov.values())
            mean_cov = float(np.mean(list(per_cell_cov.values())))
            width_mv = e_mv - s_mv

            if (min_cov + 1e-12) >= min_cell_coverage:
                # Prefer widest window, then highest min_cov, highest mean_cov, lowest v_start
                key = (width_mv, min_cov, mean_cov, -s_mv, v_s, v_e, per_cell_cov)
                if best_feasible is None or key[:4] > best_feasible[:4]:
                    best_feasible = key
            else:
                fb_key = (min_cov, width_mv, mean_cov, -s_mv, v_s, v_e, per_cell_cov)
                if best_fallback is None or fb_key[:4] > best_fallback[:4]:
                    best_fallback = fb_key

    if best_feasible is not None:
        return best_feasible[4], best_feasible[5], best_feasible[6]

    if fallback_to_max_coverage and best_fallback is not None:
        return best_fallback[4], best_fallback[5], best_fallback[6]

    raise ValueError(
        f"No window of width >= {min_width:.2f} V in [{v_min:.2f}, {v_max:.2f}] V "
        f"achieved >= {100.0 * min_cell_coverage:.1f}% per-cell coverage."
    )


def run_fixed_default_baseline(
    train_cycles: Optional[Sequence[CanonicalCycleGrid]] = None,
    min_cell_coverage: float = DEFAULT_MIN_CELL_COVERAGE,
):
    """Resolve the training-only reference window for fixed-window default baseline evaluation."""
    if train_cycles is None:
        return None
    return select_reference_window(train_cycles, min_cell_coverage=min_cell_coverage)
