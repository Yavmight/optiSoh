"""Pytest configuration and shared fixtures."""
import pytest
from sohml.contracts.synthetic import generate_synthetic_battery_dataset

@pytest.fixture
def synthetic_bundle():
    return generate_synthetic_battery_dataset(n_cells=4, cycles_per_cell=20, seed=42)
