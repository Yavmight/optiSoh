"""Tests for 40+ candidate health indicators."""
from sohml.contracts.feature_rules import validate_feature_names, FORBIDDEN_FEATURES
import pytest

def test_forbidden_features():
    for f in FORBIDDEN_FEATURES:
        with pytest.raises(ValueError):
            validate_feature_names(["valid_feature", f])
