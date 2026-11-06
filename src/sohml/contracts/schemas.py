"""Core data schemas exchanged across DE1, DE2, and ML1."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

@dataclass
class CanonicalCycleGrid:
    """Uniformly resampled constant-current charging curve for a single cycle."""
    cell_id: str
    cycle_index: int
    v_grid: np.ndarray          # 1D array of voltages (e.g. 3.500 to 4.180 V, step 2 mV)
    q_norm: np.ndarray          # Normalized charge Q / Q_nominal
    dq_dv: np.ndarray           # Differential capacity d(Q_norm)/dV
    delta_t: np.ndarray         # Relative temperature T(V) - T_start
    dt_dv: np.ndarray           # Differential temperature dT/dV
    tau: np.ndarray             # C-rate normalized charging duration
    soh_true: float             # Ground truth SOH (%) = 100 * (Capacity / Initial_Capacity)
    discharge_capacity: float   # Measured discharge capacity (Ah)
    nominal_capacity: float     # Rated nominal capacity (Ah)

@dataclass
class DatasetBundle:
    """Collection of canonical cycles belonging to a dataset."""
    name: str
    cell_ids: List[str]
    cycles: List[CanonicalCycleGrid]
    nominal_capacity: float
    ambient_temp_c: float
    metadata: Dict = field(default_factory=dict)

@dataclass
class WindowQuery:
    """Partial-charging window query."""
    v_start: float
    v_end: float

    def __post_init__(self):
        if self.v_end <= self.v_start:
            raise ValueError(f"v_end ({self.v_end}) must be greater than v_start ({self.v_start})")
        if (self.v_end - self.v_start) < 0.10:
            raise ValueError(f"Window width ({self.v_end - self.v_start:.3f} V) is below minimum (0.10 V)")

@dataclass
class FeatureMatrixBundle:
    """Feature matrix extracted for a specific window query."""
    window: WindowQuery
    feature_names: List[str]
    X: np.ndarray               # Shape: (N_cycles, D_features)
    y_soh: np.ndarray           # Shape: (N_cycles,) SOH values
    cell_indices: np.ndarray    # Shape: (N_cycles,) Cell ID for each row
    cycle_indices: np.ndarray   # Shape: (N_cycles,) Cycle index for tracking
