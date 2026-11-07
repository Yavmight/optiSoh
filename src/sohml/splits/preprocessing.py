"""Fold-isolated scaling and imputation ensuring test cells remain untouched."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class FoldIsolatedPreprocessor(BaseEstimator, TransformerMixin):
    """Scaler fitted strictly on training cells."""

    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X: np.ndarray, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)
