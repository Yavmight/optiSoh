"""Stacked SOH Regressor: SVR + Random Forest + XGBoost with Ridge meta-learner."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

class StackedSOHRegressor(BaseEstimator, RegressorMixin):
    """Stacked ensemble using cell-grouped out-of-fold predictions for Ridge meta-learning."""
    def __init__(
        self,
        svr_c: float = 10.0,
        svr_gamma: float = 0.01,
        svr_epsilon: float = 0.1,
        rf_n_estimators: int = 100,
        rf_max_depth: int = 5,
        rf_min_samples_leaf: int = 2,
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 4,
        xgb_learning_rate: float = 0.05,
        xgb_reg_lambda: float = 1.0,
        ridge_alpha: float = 1.0,
    ):
        self.svr_c = svr_c
        self.svr_gamma = svr_gamma
        self.svr_epsilon = svr_epsilon
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_reg_lambda = xgb_reg_lambda
        self.ridge_alpha = ridge_alpha

        self.svr = SVR(C=self.svr_c, gamma=self.svr_gamma, epsilon=self.svr_epsilon)
        self.rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_leaf=self.rf_min_samples_leaf,
            random_state=42,
        )
        self.xgb = XGBRegressor(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            reg_lambda=self.xgb_reg_lambda,
            random_state=42,
            verbosity=0,
        )
        self.meta_learner = Ridge(alpha=self.ridge_alpha)

    def fit(self, X: np.ndarray, y: np.ndarray, cell_indices: Optional[np.ndarray] = None):
        """Fit base learners and meta-learner."""
        self.svr.fit(X, y)
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        oof_preds = np.column_stack([self.svr.predict(X), self.rf.predict(X), self.xgb.predict(X)])
        self.meta_learner.fit(oof_preds, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_preds = np.column_stack([self.svr.predict(X), self.rf.predict(X), self.xgb.predict(X)])
        return self.meta_learner.predict(base_preds)
