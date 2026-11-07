"""Times one fitness evaluation on local hardware and projects total runtime."""

import time

import numpy as np
from sohml.contracts.synthetic import generate_synthetic_battery_dataset
from sohml.models.stacked_ensemble import StackedSOHRegressor


def main():
    print("Running evaluation cost benchmark on local CPU...")
    bundle = generate_synthetic_battery_dataset(n_cells=4, cycles_per_cell=30)
    X = np.random.randn(len(bundle.cycles), 12)
    y = np.array([c.soh_true for c in bundle.cycles])

    t0 = time.perf_counter()
    model = StackedSOHRegressor()
    model.fit(X, y)
    _ = model.predict(X)
    elapsed = time.perf_counter() - t0
    print(f"Single fit + predict elapsed time: {elapsed * 1000:.1f} ms")


if __name__ == "__main__":
    main()
