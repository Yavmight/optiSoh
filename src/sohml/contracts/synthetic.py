"""Deterministic synthetic battery dataset generator for zero-blocking parallel development."""
import numpy as np
from typing import List
from .schemas import CanonicalCycleGrid, DatasetBundle

def generate_synthetic_battery_dataset(
    n_cells: int = 4,
    cycles_per_cell: int = 30,
    seed: int = 42,
    nominal_capacity: float = 2.0,
) -> DatasetBundle:
    """Generate realistic synthetic CC battery curves and aging trajectories."""
    rng = np.random.default_rng(seed)
    v_grid = np.arange(3.50, 4.182, 0.002)
    n_pts = len(v_grid)
    cycles_list: List[CanonicalCycleGrid] = []
    cell_names = [f"SYN_{i+1:02d}" for i in range(n_cells)]

    for cell_idx, cell_id in enumerate(cell_names):
        fade_rate = rng.uniform(0.004, 0.008)
        initial_capacity = nominal_capacity * rng.uniform(0.98, 1.02)

        for cyc in range(cycles_per_cell):
            soh = 100.0 * np.exp(-fade_rate * cyc) + rng.normal(0, 0.2)
            soh = float(np.clip(soh, 60.0, 102.0))
            meas_cap = (soh / 100.0) * initial_capacity

            # Synthesize IC peak shrinking and shifting right with degradation
            peak_v = 3.90 + 0.05 * (1.0 - soh / 100.0) + rng.normal(0, 0.005)
            peak_h = 3.0 * (soh / 100.0)
            dq_dv = peak_h * np.exp(-0.5 * ((v_grid - peak_v) / 0.04) ** 2) + rng.normal(0, 0.05, n_pts)
            dq_dv = np.maximum(dq_dv, 0.01)

            # Cumulative Q_norm integral
            q_norm = np.cumsum(dq_dv) * 0.002
            q_norm = (q_norm / q_norm[-1]) * (meas_cap / nominal_capacity)

            # Thermal curve
            delta_t = 3.0 * (1.0 + (100.0 - soh) * 0.02) * ((v_grid - v_grid[0]) / (v_grid[-1] - v_grid[0]))
            delta_t += rng.normal(0, 0.05, n_pts)
            dt_dv = np.gradient(delta_t, v_grid)

            tau = np.linspace(0, 3600 * (soh / 100.0), n_pts)

            cycles_list.append(
                CanonicalCycleGrid(
                    cell_id=cell_id,
                    cycle_index=cyc,
                    v_grid=v_grid,
                    q_norm=q_norm,
                    dq_dv=dq_dv,
                    delta_t=delta_t,
                    dt_dv=dt_dv,
                    tau=tau,
                    soh_true=soh,
                    discharge_capacity=float(meas_cap),
                    nominal_capacity=nominal_capacity,
                )
            )

    return DatasetBundle(
        name="synthetic_dataset",
        cell_ids=cell_names,
        cycles=cycles_list,
        nominal_capacity=nominal_capacity,
        ambient_temp_c=25.0,
    )
