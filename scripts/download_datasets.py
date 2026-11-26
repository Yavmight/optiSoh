"""Dataset downloader and integrity checker for NASA, Oxford, and Sandia."""

import argparse
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][download_datasets][%(levelname)s] %(message)s")
logger = logging.getLogger("download_datasets")

NASA_ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
NASA_REQUIRED_CORE_CELLS = ("B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat")

OXFORD_FILES: List[Tuple[str, str]] = [
    (
        "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/m5ac36a1e2073852e4f1f7dee647909a7",
        "Oxford_Battery_Degradation_Dataset_1.mat",
    ),
    (
        "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/m43cc05e7c5f1245f4895d9dbd495e52f",
        "Readme.txt",
    ),
    (
        "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/me9fc40a60ac98708f1b73f3a836548e9",
        "ExampleDC_C1.mat",
    ),
]
OXFORD_MIN_MAT_BYTES = 200 * 1024 * 1024  # ~253.8 MB

# Open Zenodo mirror of BatteryLife_Processed Sandia (SNL Preger 2020) pickle files
SANDIA_ZIP_URL = "https://zenodo.org/api/records/19688272/files/SNL.zip/content"
SANDIA_MIN_FULL_RANGE_CELLS = 39


def _download_stream(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a remote file atomically via a .part temporary file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    logger.info(f"Downloading {dest.name} from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "optiSoh-dataset-downloader/1.0"})
    downloaded = 0
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
    tmp_path.replace(dest)
    if total > 0:
        logger.info(
            f"Finished {dest.name}: {downloaded / (1024 * 1024):.2f} MB (expected {total / (1024 * 1024):.2f} MB)"
        )
    else:
        logger.info(f"Finished {dest.name}: {downloaded / (1024 * 1024):.2f} MB")


def verify_nasa(raw_dir: Path) -> Tuple[bool, str]:
    nasa_dir = raw_dir / "nasa"
    mat_files = {p.name: p for p in nasa_dir.rglob("*.mat")} if nasa_dir.exists() else {}
    missing = [name for name in NASA_REQUIRED_CORE_CELLS if name not in mat_files]
    if missing:
        return False, f"Missing core NASA .mat files in {nasa_dir}: {missing} (found {len(mat_files)} .mat files)"
    return True, f"NASA OK: {len(mat_files)} .mat files present (including all 4 core cells) in {nasa_dir}"


def verify_oxford(raw_dir: Path) -> Tuple[bool, str]:
    ox_dir = raw_dir / "oxford"
    mat_path = ox_dir / "Oxford_Battery_Degradation_Dataset_1.mat"
    if not mat_path.exists():
        return False, f"Missing {mat_path}"
    size = mat_path.stat().st_size
    if size < OXFORD_MIN_MAT_BYTES:
        return False, f"{mat_path} looks truncated ({size / (1024 * 1024):.2f} MB < 200 MB)"
    return True, f"Oxford OK: {mat_path.name} ({size / (1024 * 1024):.2f} MB) in {ox_dir}"


def verify_sandia(raw_dir: Path) -> Tuple[bool, str]:
    snl_dir = raw_dir / "sandia"
    pkl_files = list(snl_dir.rglob("*.pkl")) if snl_dir.exists() else []
    full_range = [p for p in pkl_files if "_0-100_" in p.stem and ("_NCA_" in p.stem or "_NMC_" in p.stem)]
    if len(full_range) < SANDIA_MIN_FULL_RANGE_CELLS:
        return (
            False,
            f"Expected >= {SANDIA_MIN_FULL_RANGE_CELLS} full-range NCA/NMC .pkl files in {snl_dir}, "
            f"found {len(full_range)} ({len(pkl_files)} total .pkl files)",
        )
    return (
        True,
        f"Sandia OK: {len(full_range)} full-range NCA/NMC cells ({len(pkl_files)} total .pkl files) in {snl_dir}",
    )


def download_nasa(raw_dir: Path, force: bool = False) -> None:
    ok, msg = verify_nasa(raw_dir)
    if ok and not force:
        logger.info(f"Skipping NASA download ({msg}). Use --force to re-download.")
        return

    nasa_dir = raw_dir / "nasa"
    nasa_zip = nasa_dir / "5_Battery_Data_Set.zip"
    _download_stream(NASA_ZIP_URL, nasa_zip)

    logger.info("Extracting NASA archive (including nested .zip bundles)...")
    with zipfile.ZipFile(nasa_zip, "r") as zf:
        zf.extractall(nasa_dir)
    nasa_zip.unlink()

    for subzip in sorted(nasa_dir.rglob("*.zip")):
        logger.info(f"Extracting nested archive {subzip.name}...")
        with zipfile.ZipFile(subzip, "r") as zf:
            zf.extractall(subzip.parent)
        subzip.unlink()

    ok, msg = verify_nasa(raw_dir)
    if not ok:
        raise RuntimeError(f"NASA integrity check failed after extraction: {msg}")
    logger.info(msg)


def download_oxford(raw_dir: Path, force: bool = False) -> None:
    ok, msg = verify_oxford(raw_dir)
    if ok and not force:
        logger.info(f"Skipping Oxford download ({msg}). Use --force to re-download.")
        return

    ox_dir = raw_dir / "oxford"
    for url, fname in OXFORD_FILES:
        dest = ox_dir / fname
        if dest.exists() and not force:
            continue
        _download_stream(url, dest)

    ok, msg = verify_oxford(raw_dir)
    if not ok:
        raise RuntimeError(f"Oxford integrity check failed after download: {msg}")
    logger.info(msg)


def download_sandia(raw_dir: Path, force: bool = False) -> None:
    ok, msg = verify_sandia(raw_dir)
    if ok and not force:
        logger.info(f"Skipping Sandia download ({msg}). Use --force to re-download.")
        return

    snl_dir = raw_dir / "sandia"
    snl_zip = snl_dir / "SNL.zip"
    _download_stream(SANDIA_ZIP_URL, snl_zip)

    logger.info("Extracting Sandia SNL.zip...")
    with zipfile.ZipFile(snl_zip, "r") as zf:
        zf.extractall(snl_dir)
    snl_zip.unlink()

    ok, msg = verify_sandia(raw_dir)
    if not ok:
        raise RuntimeError(f"Sandia integrity check failed after extraction: {msg}")
    logger.info(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify raw NASA, Oxford, and Sandia battery datasets.")
    parser.add_argument("--dataset", choices=["nasa", "oxford", "sandia", "all"], default="all")
    parser.add_argument("--raw-dir", default="data/raw", help="Base raw data directory (default: data/raw).")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already pass verification.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files without downloading.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    targets = ["nasa", "oxford", "sandia"] if args.dataset == "all" else [args.dataset]
    verifiers: Dict[str, object] = {"nasa": verify_nasa, "oxford": verify_oxford, "sandia": verify_sandia}
    downloaders: Dict[str, object] = {"nasa": download_nasa, "oxford": download_oxford, "sandia": download_sandia}

    if args.verify_only:
        all_ok = True
        for ds in targets:
            ok, msg = verifiers[ds](raw_dir)
            if ok:
                logger.info(msg)
            else:
                logger.error(msg)
                all_ok = False
        return 0 if all_ok else 1

    for ds in targets:
        downloaders[ds](raw_dir, force=args.force)

    logger.info("All requested datasets are ready. Next step: python scripts/build_processed.py --dataset all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
