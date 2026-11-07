"""Tests verifying strict evaluation budget parity across methods and baseline models."""

import numpy as np
from sohml.baselines.optuna_tpe import OptunaTPEOptimizer
from sohml.baselines.random_search import RandomSearchOptimizer
from sohml.baselines.two_stage import TwoStageFeatureSelectionBaseline
from sohml.optimization.base import EvaluationBudgetCounter


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
    X = np.random.randn(25, 10)
    y = np.linspace(100, 75, 25)
    baseline = TwoStageFeatureSelectionBaseline(selector="mi", n_features_to_select=4)
    baseline.fit(X, y)
    preds = baseline.predict(X)
    assert preds.shape == (25,)
    assert len(baseline.selected_indices_) == 4
