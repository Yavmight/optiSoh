"""Common optimizer interface and strict evaluation budget counter."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class EvaluationBudgetCounter:
    """Tracks fitness evaluations to guarantee exact evaluation budget parity across all methods."""

    def __init__(self, max_evaluations: int):
        self.max_evaluations = max_evaluations
        self.current_count = 0

    def step(self, batch_size: int = 1) -> bool:
        if self.current_count + batch_size > self.max_evaluations:
            return False
        self.current_count += batch_size
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.max_evaluations - self.current_count)


class BaseOptimizer(ABC):
    """Abstract base class for all co-optimizers and baselines."""

    def __init__(self, budget_counter: EvaluationBudgetCounter, seed: int = 42):
        self.budget_counter = budget_counter
        self.seed = seed

    @abstractmethod
    def optimize(self, fitness_fn) -> Dict[str, Any]:
        """Run optimization and return Pareto front / best candidate."""
        pass
