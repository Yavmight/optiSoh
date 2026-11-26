"""Quantized 10 mV window feature cache for O(1) metaheuristic evaluation lookups."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..contracts.constants import (
    WINDOW_MIN_WIDTH,
    WINDOW_QUANTIZATION_STEP_MV,
    WINDOW_V_MAX,
    WINDOW_V_MIN,
)
from ..contracts.schemas import CanonicalCycleGrid, FeatureMatrixBundle, WindowQuery
from .registry import extract_feature_matrix_bundle

CacheValue = Union[FeatureMatrixBundle, np.ndarray]


def enumerate_quantized_windows(
    v_min: float = WINDOW_V_MIN,
    v_max: float = WINDOW_V_MAX,
    min_width: float = WINDOW_MIN_WIDTH,
    step_mv: int = WINDOW_QUANTIZATION_STEP_MV,
) -> List[Tuple[float, float]]:
    """Return all discrete (v_start, v_end) pairs on the quantized millivolt grid."""
    start_lo = int(round(v_min * 1000))
    end_hi = int(round(v_max * 1000))
    min_w = int(round(min_width * 1000))
    windows: List[Tuple[float, float]] = []
    for s_mv in range(start_lo, end_hi - min_w + 1, step_mv):
        for e_mv in range(s_mv + min_w, end_hi + 1, step_mv):
            windows.append((s_mv / 1000.0, e_mv / 1000.0))
    return windows


class WindowFeatureCache:
    """Caches precomputed FeatureMatrixBundle (or legacy ndarray) objects for discrete (V_start, V_end) pairs.

    Supports O(1) in-memory lookup, on-the-fly computation fallback, and persistent disk caching
    (.npz for FeatureMatrixBundle with X, y_soh, cell_indices, cycle_indices; .npy for raw arrays).
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
        self.memory_cache: Dict[Tuple[int, int], CacheValue] = {}

        if self.enable_disk and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def quantize(self, v_start: float, v_end: float) -> Tuple[int, int]:
        """Convert continuous voltages to integer millivolts on the quantized grid."""
        s = int(round(v_start * 1000 / self.step_mv) * self.step_mv)
        e = int(round(v_end * 1000 / self.step_mv) * self.step_mv)
        return s, e

    def _bundle_disk_path(self, s: int, e: int) -> Optional[Path]:
        if not self.enable_disk or self.cache_dir is None:
            return None
        return self.cache_dir / f"window_{s}_{e}.npz"

    def _array_disk_path(self, s: int, e: int) -> Optional[Path]:
        if not self.enable_disk or self.cache_dir is None:
            return None
        return self.cache_dir / f"window_{s}_{e}.npy"

    def _save_bundle_npz(self, path: Path, bundle: FeatureMatrixBundle) -> None:
        np.savez_compressed(
            path,
            X=np.asarray(bundle.X, dtype=np.float64),
            y_soh=np.asarray(bundle.y_soh, dtype=np.float64),
            cell_indices=np.asarray(bundle.cell_indices, dtype=str),
            cycle_indices=np.asarray(bundle.cycle_indices, dtype=np.int64),
            feature_names=np.asarray(bundle.feature_names, dtype=str),
            v_start=np.array([bundle.window.v_start], dtype=np.float64),
            v_end=np.array([bundle.window.v_end], dtype=np.float64),
        )

    def _load_bundle_npz(self, path: Path) -> FeatureMatrixBundle:
        with np.load(path, allow_pickle=False) as data:
            v_start = float(data["v_start"][0])
            v_end = float(data["v_end"][0])
            return FeatureMatrixBundle(
                window=WindowQuery(v_start=v_start, v_end=v_end),
                feature_names=[str(x) for x in data["feature_names"].tolist()],
                X=np.asarray(data["X"], dtype=np.float64),
                y_soh=np.asarray(data["y_soh"], dtype=np.float64),
                cell_indices=np.asarray(data["cell_indices"], dtype=str),
                cycle_indices=np.asarray(data["cycle_indices"], dtype=np.int64),
            )

    def put(self, v_start: float, v_end: float, value: CacheValue) -> Tuple[int, int]:
        """Store FeatureMatrixBundle or raw ndarray into memory cache and optional disk cache."""
        key = self.quantize(v_start, v_end)
        self.memory_cache[key] = value

        if self.enable_disk:
            if isinstance(value, FeatureMatrixBundle):
                npz_p = self._bundle_disk_path(*key)
                npy_p = self._array_disk_path(*key)
                if npz_p is not None:
                    self._save_bundle_npz(npz_p, value)
                if npy_p is not None and npy_p.exists():
                    npy_p.unlink()
            else:
                npy_p = self._array_disk_path(*key)
                npz_p = self._bundle_disk_path(*key)
                if npy_p is not None:
                    np.save(npy_p, value)
                if npz_p is not None and npz_p.exists():
                    npz_p.unlink()
        return key

    def get(self, v_start: float, v_end: float) -> Optional[CacheValue]:
        """Lookup cached FeatureMatrixBundle or ndarray from memory, then disk."""
        key = self.quantize(v_start, v_end)

        # 1. In-memory check
        if key in self.memory_cache:
            return self.memory_cache[key]

        # 2. Disk persistence check (.npz FeatureMatrixBundle preferred, then legacy .npy)
        if self.enable_disk:
            npz_p = self._bundle_disk_path(*key)
            if npz_p is not None and npz_p.exists():
                bundle = self._load_bundle_npz(npz_p)
                self.memory_cache[key] = bundle
                return bundle

            npy_p = self._array_disk_path(*key)
            if npy_p is not None and npy_p.exists():
                matrix = np.load(npy_p)
                self.memory_cache[key] = matrix
                return matrix

        return None

    def get_or_compute(
        self,
        v_start: float,
        v_end: float,
        compute_fn: Callable[[float, float], CacheValue],
    ) -> CacheValue:
        """Lookup cached entry or compute on miss using quantized voltages, persisting the result."""
        cached = self.get(v_start, v_end)
        if cached is not None:
            return cached
        s_mv, e_mv = self.quantize(v_start, v_end)
        value = compute_fn(s_mv / 1000.0, e_mv / 1000.0)
        self.put(v_start, v_end, value)
        return value

    def get_or_compute_bundle(
        self,
        v_start: float,
        v_end: float,
        cycles: Sequence[CanonicalCycleGrid],
    ) -> FeatureMatrixBundle:
        """Lookup or compute a re-zeroed, NaN-filtered FeatureMatrixBundle for [v_start, v_end]."""
        result = self.get_or_compute(
            v_start,
            v_end,
            lambda vs, ve: extract_feature_matrix_bundle(cycles, vs, ve),
        )
        if not isinstance(result, FeatureMatrixBundle):
            raise TypeError(f"Expected FeatureMatrixBundle from cache, got {type(result).__name__}")
        return result

    def contains(self, v_start: float, v_end: float) -> bool:
        """Check if window query is present in memory or on disk."""
        key = self.quantize(v_start, v_end)
        if key in self.memory_cache:
            return True
        if self.enable_disk:
            npz_p = self._bundle_disk_path(*key)
            npy_p = self._array_disk_path(*key)
            return bool((npz_p is not None and npz_p.exists()) or (npy_p is not None and npy_p.exists()))
        return False

    def clear(self, clear_disk: bool = False) -> None:
        """Clear memory cache, and optionally delete on-disk cached files."""
        self.memory_cache.clear()
        if clear_disk and self.enable_disk and self.cache_dir is not None and self.cache_dir.exists():
            for pattern in ("window_*.npz", "window_*.npy"):
                for f in self.cache_dir.glob(pattern):
                    try:
                        f.unlink()
                    except OSError:
                        pass

    def __len__(self) -> int:
        return len(self.memory_cache)
