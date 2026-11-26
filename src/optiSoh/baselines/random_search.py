"""Joint Random Search baseline under exact evaluation budget B."""

from typing import Any, Callable, Dict, List

import numpy as np

from ..contracts.constants import WINDOW_END_MAX, WINDOW_END_MIN, WINDOW_START_MAX, WINDOW_START_MIN
from ..optimization.base import BaseOptimizer, EvaluationBudgetCounter
from ..optimization.search_space import repair_decision_vector, sample_random_hyperparams


class RandomSearchOptimizer(BaseOptimizer):
    """Joint random search over voltage window, feature mask, and hyperparameters."""

    def __init__(self, budget_counter: EvaluationBudgetCounter, n_features: int = 42, seed: int = 42):
        super().__init__(budget_counter, seed)
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)

    def optimize(self, fitness_fn: Callable) -> Dict[str, Any]:
        """Sample random candidates until budget is exhausted."""
        history: List[Dict[str, Any]] = []

        while self.budget_counter.step(1):
            v_start = float(self.rng.uniform(WINDOW_START_MIN, WINDOW_START_MAX))
            v_end = float(self.rng.uniform(WINDOW_END_MIN, WINDOW_END_MAX))
            mask = self.rng.random(self.n_features) > 0.5
            v_start, v_end, mask = repair_decision_vector(v_start, v_end, mask)

            hyperparams = sample_random_hyperparams(self.rng)

            candidate = {
                "v_start": v_start,
                "v_end": v_end,
                "feature_mask": mask,
                "hyperparams": hyperparams,
            }
            costs = fitness_fn(candidate)
            history.append({"candidate": candidate, "costs": costs})

        return {"history": history, "evaluations": self.budget_counter.current_count}
