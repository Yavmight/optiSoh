"""Two-stage baseline: Mutual Information feature selection + tuned stack."""

from .two_stage import TwoStageFeatureSelectionBaseline


def create_mi_baseline(n_features: int = 12, random_state: int = 42) -> TwoStageFeatureSelectionBaseline:
    """Create a Mutual Information feature selection + tuned stacked ensemble baseline."""
    return TwoStageFeatureSelectionBaseline(
        selector="mi",
        n_features_to_select=n_features,
        random_state=random_state,
    )
