"""Tests for Pareto knee selection, coverage guards, and optimizers."""

import numpy as np
from optiSoh.optimization.knee import select_guarded_knee_point
from optiSoh.optimization.objectives import (
    check_window_coverage,
    compute_objectives,
    penalized_infeasible_objectives,
)


def test_knee_selection():
    pareto_costs = np.array(
        [
            [1.0, 2.0, 0.1, 0.2],
            [0.8, 1.5, 0.3, 0.4],
            [0.7, 1.2, 0.8, 0.9],
        ]
    )
    idx = select_guarded_knee_point(pareto_costs)
    assert 0 <= idx < 3


def test_compute_objectives_coverage_guard_and_pareto_dominance():
    mask_many = np.ones(10, dtype=bool)
    mask_one = np.zeros(10, dtype=bool)
    mask_one[0] = True

    feasible = compute_objectives({"C1": 1.2, "C2": 2.1}, mask_many, model_complexity=0.8, min_cell_coverage=0.98)
    infeasible_small_mask = compute_objectives(
        {"C1": 0.1, "C2": 0.2}, mask_one, model_complexity=0.05, min_cell_coverage=0.50
    )
    # Every feasible solution must strictly dominate an infeasible low-coverage solution across all 4 objectives
    assert np.all(feasible < infeasible_small_mask)

    # Smooth gradient: higher-coverage infeasible window strictly dominates lower-coverage infeasible window
    pen_80 = penalized_infeasible_objectives(min_cell_coverage=0.80, target_coverage=0.95, feature_mask=mask_one)
    pen_30 = penalized_infeasible_objectives(min_cell_coverage=0.30, target_coverage=0.95, feature_mask=mask_one)
    assert np.all(pen_80 < pen_30)


def test_check_window_coverage_rejects_ohmic_rebound_window(synthetic_bundle):
    # Simulate aged-cell ohmic rebound on cell 1: half its cycles lack data below 3.85 V
    target_cid = synthetic_bundle.cell_ids[0]
    cell_cycles = [c for c in synthetic_bundle.cycles if c.cell_id == target_cid]
    for c in cell_cycles[len(cell_cycles) // 2 :]:
        low_mask = c.v_grid < 3.85
        c.q_norm[low_mask] = np.nan
        c.dq_dv[low_mask] = np.nan

    ok_360, min_360, _ = check_window_coverage(synthetic_bundle.cycles, 3.60, 4.10, min_cell_coverage=0.95)
    ok_390, min_390, _ = check_window_coverage(synthetic_bundle.cycles, 3.90, 4.10, min_cell_coverage=0.95)
    assert ok_360 is False
    assert min_360 < 0.60
    assert ok_390 is True
    assert min_390 == 1.0
