"""Quantized 10 mV window feature cache for O(1) metaheuristic evaluation lookups."""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..contracts.constants import WINDOW_QUANTIZATION_STEP_MV


class WindowFeatureCache:
    """Caches precomputed (N_cycles, D_features) matrices for discrete (V_start, V_end) pairs.

    Supports O(1) in-memory lookup, on-the-fly computation fallback, and persistent disk caching.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        step_mv: int = WINDOW_QUANTIZATION_STEP_MV,
        enable_disk: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.step_mv = step_mv
        self.enable_disk = enable_disk and (self.cache_dir is not None)
        self.memory_cache: Dict[Tuple[int, int], np.ndarray] = {}

        if self.enable_disk and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def quantize(self, v_start: float, v_end: float) -> Tuple[int, int]:
        """Convert continuous voltages to integer millivolts on the quantized grid."""
        s = int(round(v_start * 1000 / self.step_mv) * self.step_mv)
        e = int(round(v_end * 1000 / self.step_mv) * self.step_mv)
        return s, e

    def _disk_path(self, s: int, e: int) -> Optional[Path]:
        if not self.enable_disk or self.cache_dir is None:
            return None
        return self.cache_dir / f"window_{s}_{e}.npy"

    def put(self, v_start: float, v_end: float, matrix: np.ndarray) -> Tuple[int, int]:
        """Store feature matrix into in-memory cache and optional disk cache."""
        key = self.quantize(v_start, v_end)
        self.memory_cache[key] = matrix

        if self.enable_disk:
            disk_p = self._disk_path(*key)
            if disk_p is not None:
                np.save(disk_p, matrix)
        return key

    def get(self, v_start: float, v_end: float) -> Optional[np.ndarray]:
        """Lookup feature matrix for the specified window query from memory, then disk."""
        key = self.quantize(v_start, v_end)

        # 1. In-memory check
        if key in self.memory_cache:
            return self.memory_cache[key]

        # 2. Disk persistence check
        if self.enable_disk:
            disk_p = self._disk_path(*key)
            if disk_p is not None and disk_p.exists():
                matrix = np.load(disk_p)
                self.memory_cache[key] = matrix
                return matrix

        return None

    def get_or_compute(
        self,
        v_start: float,
        v_end: float,
        compute_fn: Callable[[float, float], np.ndarray],
    ) -> np.ndarray:
        """Lookup cached matrix or compute it on miss, persisting the result."""
        cached = self.get(v_start, v_end)
        if cached is not None:
            return cached
        matrix = compute_fn(v_start, v_end)
        self.put(v_start, v_end, matrix)
        return matrix

    def contains(self, v_start: float, v_end: float) -> bool:
        """Check if window query is present in memory or on disk."""
        key = self.quantize(v_start, v_end)
        if key in self.memory_cache:
            return True
        if self.enable_disk:
            disk_p = self._disk_path(*key)
            return disk_p is not None and disk_p.exists()
        return False

    def clear(self, clear_disk: bool = False):
        """Clear memory cache, and optionally delete on-disk cached files."""
        self.memory_cache.clear()
        if clear_disk and self.enable_disk and self.cache_dir is not None and self.cache_dir.exists():
            for f in self.cache_dir.glob("window_*.npy"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def __len__(self) -> int:
        return len(self.memory_cache)
