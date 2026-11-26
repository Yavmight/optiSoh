"""Precomputes 10 mV quantized FeatureMatrixBundle window caches for NASA, Oxford, and Sandia."""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple

from joblib import Parallel, delayed
from optiSoh.contracts.constants import (
    WINDOW_MIN_WIDTH,
    WINDOW_QUANTIZATION_STEP_MV,
    WINDOW_V_MAX,
    WINDOW_V_MIN,
)
from optiSoh.features.registry import extract_feature_matrix_bundle
from optiSoh.features.window_cache import WindowFeatureCache, enumerate_quantized_windows

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][build_cache][%(levelname)s] %(message)s")
logger = logging.getLogger("build_cache")


def _compute_and_store_window(cycles, cache_dir: Path, step_mv: int, v_start: float, v_end: float) -> Tuple[int, int]:
    """Worker computing one quantized FeatureMatrixBundle and persisting it to disk."""
    bundle = extract_feature_matrix_bundle(cycles, v_start, v_end)
    cache = WindowFeatureCache(cache_dir=cache_dir, step_mv=step_mv, enable_disk=True)
    cache.put(v_start, v_end, bundle)
    return int(bundle.X.shape[0]), int(bundle.X.shape[1])


def build_dataset_cache(
    dataset_name: str,
    processed_dir: Path,
    cache_root: Path,
    step_mv: int = WINDOW_QUANTIZATION_STEP_MV,
    v_min: float = WINDOW_V_MIN,
    v_max: float = WINDOW_V_MAX,
    min_width: float = WINDOW_MIN_WIDTH,
    n_jobs: int = -1,
    force: bool = False,
) -> int:
    """Build all quantized FeatureMatrixBundle .npz files for one processed dataset bundle."""
    pkl_path = processed_dir / f"{dataset_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset {pkl_path}. Run: python scripts/build_processed.py --dataset {dataset_name}"
        )

    with open(pkl_path, "rb") as f:
        dataset_bundle = pickle.load(f)

    ds_cache_dir = cache_root / dataset_name
    cache = WindowFeatureCache(cache_dir=ds_cache_dir, step_mv=step_mv, enable_disk=True)
    if force:
        cache.clear(clear_disk=True)

    all_windows = enumerate_quantized_windows(v_min=v_min, v_max=v_max, min_width=min_width, step_mv=step_mv)
    pending: List[Tuple[float, float]] = [(vs, ve) for (vs, ve) in all_windows if force or not cache.contains(vs, ve)]

    logger.info(
        f"[{dataset_name}] {len(dataset_bundle.cycles)} cycles across {len(dataset_bundle.cell_ids)} cells | "
        f"{len(pending)}/{len(all_windows)} quantized windows to compute -> {ds_cache_dir}"
    )
    if not pending:
        logger.info(f"[{dataset_name}] Cache already complete ({len(all_windows)} windows).")
        return 0

    t0 = time.perf_counter()
    if n_jobs == 1 or len(pending) <= 2:
        counts = [_compute_and_store_window(dataset_bundle.cycles, ds_cache_dir, step_mv, vs, ve) for vs, ve in pending]
    else:
        counts = Parallel(n_jobs=n_jobs)(
            delayed(_compute_and_store_window)(dataset_bundle.cycles, ds_cache_dir, step_mv, vs, ve)
            for vs, ve in pending
        )
    elapsed = time.perf_counter() - t0
    non_empty = sum(1 for n_rows, _ in counts if n_rows > 0)
    logger.info(
        f"[{dataset_name}] Cached {len(pending)} windows ({non_empty} non-empty) in {elapsed:.1f}s -> {ds_cache_dir}"
    )
    return len(pending)


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute 10 mV quantized FeatureMatrixBundle window caches.")
    parser.add_argument("--dataset", choices=["nasa", "oxford", "sandia", "all"], default="all")
    parser.add_argument(
        "--processed-dir", default="data/processed", help="Directory containing processed .pkl bundles."
    )
    parser.add_argument("--cache-dir", default="data/cache", help="Root directory for persisted .npz window caches.")
    parser.add_argument("--step-mv", type=int, default=WINDOW_QUANTIZATION_STEP_MV, help="Quantization step in mV.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel worker count (-1 uses all CPU cores).")
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite existing cached windows.")
    args = parser.parse_args()

    targets = ["nasa", "oxford", "sandia"] if args.dataset == "all" else [args.dataset]
    for ds in targets:
        build_dataset_cache(
            dataset_name=ds,
            processed_dir=Path(args.processed_dir),
            cache_root=Path(args.cache_dir),
            step_mv=args.step_mv,
            n_jobs=args.n_jobs,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
