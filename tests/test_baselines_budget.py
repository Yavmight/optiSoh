"""Tests verifying strict evaluation budget parity across methods and baseline models."""

import numpy as np
import pytest
from optiSoh.baselines.fixed_default import filter_cycles_for_window, select_reference_window
from optiSoh.baselines.optuna_tpe import OptunaTPEOptimizer
from optiSoh.baselines.random_search import RandomSearchOptimizer
from optiSoh.baselines.two_stage import TwoStageFeatureSelectionBaseline
from optiSoh.optimization.base import EvaluationBudgetCounter
from optiSoh.optimization.search_space import RF_MAX_DEPTH_BOUNDS, sample_random_hyperparams


def test_budget_counter():
    counter = EvaluationBudgetCounter(max_evaluations=100)
    assert counter.step(50) is True
    assert counter.step(51) is False
    assert counter.remaining == 50


def test_random_search_budget_enforcement():
    counter = EvaluationBudgetCounter(max_evaluations=10)
    opt = RandomSearchOptimizer(budget_counter=counter, n_features=5, seed=42)

    def dummy_fitness(cand):
        return np.array([0.5, 0.6, 0.2, 0.1])

    res = opt.optimize(dummy_fitness)
    assert res["evaluations"] == 10
    assert len(res["history"]) == 10
    assert counter.remaining == 0


def test_optuna_tpe_budget_enforcement():
    counter = EvaluationBudgetCounter(max_evaluations=8)
    opt = OptunaTPEOptimizer(budget_counter=counter, n_features=5, seed=42)

    def dummy_fitness(cand):
        return [0.5, 0.6, 0.2, 0.1]

    res = opt.optimize(dummy_fitness)
    assert res["evaluations"] == 8
    assert len(res["history"]) == 8
    assert counter.remaining == 0


def test_two_stage_baseline():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 10)
    y = np.linspace(100, 75, 30)
    cells = np.repeat(["C1", "C2", "C3"], 10)

    # Strict default: omitting cell_indices must raise ValueError
    with pytest.raises(ValueError, match="cell_indices"):
        TwoStageFeatureSelectionBaseline(selector="lasso", n_features_to_select=4).fit(X, y)

    for selector in ["lasso", "mi", "shap"]:
        baseline = TwoStageFeatureSelectionBaseline(selector=selector, n_features_to_select=4, random_state=42)
        baseline.fit(X, y, cell_indices=cells)
        preds = baseline.predict(X)
        assert preds.shape == (30,)
        assert len(baseline.selected_indices_) == 4


def test_hyperparam_search_space_inclusive_rf_max_depth():
    rng = np.random.default_rng(42)
    depths = {sample_random_hyperparams(rng)["rf_max_depth"] for _ in range(300)}
    lo, hi = RF_MAX_DEPTH_BOUNDS
    assert lo in depths and hi in depths
    assert min(depths) == lo and max(depths) == hi


def test_select_reference_window_training_only(synthetic_bundle):
    # Full coverage case -> widest window [3.60, 4.10] V
    vs, ve, covs = select_reference_window(synthetic_bundle.cycles, min_cell_coverage=0.95)
    assert (vs, ve) == (3.60, 4.10)
    assert all(v == 1.0 for v in covs.values())

    # Simulate aged-cell ohmic rebound on first training cell (missing < 3.84 V on 40% of cycles)
    target_cid = synthetic_bundle.cell_ids[0]
    cell_cycles = [c for c in synthetic_bundle.cycles if c.cell_id == target_cid]
    for c in cell_cycles[int(0.6 * len(cell_cycles)) :]:
        low_mask = c.v_grid < 3.84 - 1e-6
        c.q_norm[low_mask] = np.nan
        c.dq_dv[low_mask] = np.nan

    vs2, ve2, covs2 = select_reference_window(synthetic_bundle.cycles, min_cell_coverage=0.95)
    assert (vs2, ve2) == (3.84, 4.10)
    assert min(covs2.values()) >= 0.95
    kept = filter_cycles_for_window(synthetic_bundle.cycles, vs2, ve2)
    assert len(kept) == len(synthetic_bundle.cycles)
