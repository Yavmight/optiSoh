"""Tests verifying strict evaluation budget parity across methods."""
from sohml.optimization.base import EvaluationBudgetCounter

def test_budget_counter():
    counter = EvaluationBudgetCounter(max_evaluations=100)
    assert counter.step(50) is True
    assert counter.step(51) is False
    assert counter.remaining == 50
