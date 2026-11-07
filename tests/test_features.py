"""Tests for 40+ candidate health indicators."""

import pytest
from sohml.contracts.feature_rules import FORBIDDEN_FEATURES, validate_feature_names


def test_forbidden_features():
    for f in FORBIDDEN_FEATURES:
        with pytest.raises(ValueError):
            validate_feature_names(["valid_feature", f])
