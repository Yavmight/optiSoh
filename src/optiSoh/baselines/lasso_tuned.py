"""Two-stage baseline: LASSO L1 feature selection + tuned stack."""

from .two_stage import TwoStageFeatureSelectionBaseline


def create_lasso_baseline(n_features: int = 12, random_state: int = 42) -> TwoStageFeatureSelectionBaseline:
    """Create a LASSO feature selection + tuned stacked ensemble baseline."""
    return TwoStageFeatureSelectionBaseline(
        selector="lasso",
        n_features_to_select=n_features,
        random_state=random_state,
    )
