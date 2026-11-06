"""Quantized 10 mV window feature cache for O(1) metaheuristic evaluation lookups."""
from pathlib import Path
import numpy as np
from typing import Dict, Tuple

class WindowFeatureCache:
    """Caches precomputed (N_cycles, D_features) matrices for discrete (V_start, V_end) pairs."""
    def __init__(self, cache_dir: Path, step_mv: int = 10):
        self.cache_dir = Path(cache_dir)
        self.step_mv = step_mv
        self.memory_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def quantize(self, v_start: float, v_end: float) -> Tuple[int, int]:
        """Convert continuous voltages to integer millivolts on the quantized grid."""
        s = int(round(v_start * 1000 / self.step_mv) * self.step_mv)
        e = int(round(v_end * 1000 / self.step_mv) * self.step_mv)
        return s, e

    def get(self, v_start: float, v_end: float) -> np.ndarray:
        """Lookup feature matrix for the specified window query."""
        key = self.quantize(v_start, v_end)
        return self.memory_cache.get(key)
