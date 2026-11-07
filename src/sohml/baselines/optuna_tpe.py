"""Joint Optuna TPE / MOTPE baseline under exact evaluation budget B."""

from typing import Any, Callable, Dict, List

import numpy as np
import optuna

from ..optimization.base import BaseOptimizer, EvaluationBudgetCounter
from ..optimization.search_space import repair_decision_vector

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTPEOptimizer(BaseOptimizer):
    """Joint Multi-Objective TPE search over window, features, and hyperparameters."""

    def __init__(self, budget_counter: EvaluationBudgetCounter, n_features: int = 42, seed: int = 42):
        super().__init__(budget_counter, seed)
        self.n_features = n_features

    def optimize(self, fitness_fn: Callable) -> Dict[str, Any]:
        """Run Optuna TPE until the budget counter is exhausted."""
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"], sampler=sampler)
        history: List[Dict[str, Any]] = []

        def objective(trial: optuna.Trial):
            if not self.budget_counter.step(1):
                raise optuna.exceptions.TrialPruned()

            v_start = trial.suggest_float("v_start", 3.60, 3.95)
            v_end = trial.suggest_float("v_end", 3.75, 4.10)
            mask = np.array([trial.suggest_categorical(f"feat_{i}", [True, False]) for i in range(self.n_features)])
            v_start, v_end, mask = repair_decision_vector(v_start, v_end, mask)

            hyperparams = {
                "svr_c": trial.suggest_float("svr_c", 0.1, 100.0, log=True),
                "rf_n_estimators": trial.suggest_categorical("rf_n_estimators", [50, 100, 150]),
                "rf_max_depth": trial.suggest_int("rf_max_depth", 2, 8),
                "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2),
            }

            candidate = {
                "v_start": v_start,
                "v_end": v_end,
                "feature_mask": mask,
                "hyperparams": hyperparams,
            }
            costs = fitness_fn(candidate)
            history.append({"candidate": candidate, "costs": costs})
            return tuple(costs)

        try:
            study.optimize(objective, n_trials=self.budget_counter.remaining)
        except optuna.exceptions.TrialPruned:
            pass

        return {"study": study, "history": history, "evaluations": self.budget_counter.current_count}
