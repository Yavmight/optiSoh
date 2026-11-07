"""Two-stage baseline: SHAP Recursive Feature Elimination + tuned stack."""

from .two_stage import TwoStageFeatureSelectionBaseline


def create_shap_baseline(n_features: int = 12, random_state: int = 42) -> TwoStageFeatureSelectionBaseline:
    """Create a SHAP-based feature selection + tuned stacked ensemble baseline."""
    return TwoStageFeatureSelectionBaseline(
        selector="shap",
        n_features_to_select=n_features,
        random_state=random_state,
    )
