"""Converts raw battery time series to canonical 2 mV uniform voltage grids."""

import argparse
import pickle
from pathlib import Path

from optiSoh.data.nasa_audit import audit_nasa_archive
from optiSoh.data.nasa_loader import NasaBatteryLoader
from optiSoh.data.oxford_loader import OxfordBatteryLoader
from optiSoh.data.qc_report import generate_qc_markdown_report
from optiSoh.data.sandia_loader import SandiaBatteryLoader
from optiSoh.utils.logging import setup_logger

logger = setup_logger("build_processed")


def build_nasa(raw_root: Path, out_dir: Path, run_audit: bool = True) -> Path:
    """Preprocess NASA PCoE core cohort into data/processed/nasa.pkl + QC report."""
    nasa_raw = raw_root / "nasa"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading NASA core cohort from {nasa_raw} ...")
    loader = NasaBatteryLoader(raw_dir=nasa_raw)
    bundle = loader.load_cohort()

    out_pkl = out_dir / "nasa.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved {len(bundle.cycles)} harmonized NASA cycles across {bundle.cell_ids} -> {out_pkl}")

    qc_path = out_dir / "nasa_qc_report.md"
    report = generate_qc_markdown_report(bundle, qc_path)
    logger.info(f"Wrote NASA QC report -> {qc_path}")
    print("\n" + report)

    if run_audit:
        logger.info("Running full 34-cell NASA archive audit...")
        audit_df = audit_nasa_archive(nasa_raw)
        audit_csv = out_dir / "nasa_audit.csv"
        audit_df.to_csv(audit_csv, index=False)
        n_core = int(audit_df["core_grade"].sum())
        logger.info(f"Saved 34-cell audit ({n_core} core-grade cells) -> {audit_csv}")

    return out_pkl


def build_oxford(raw_root: Path, out_dir: Path) -> Path:
    """Preprocess Oxford Degradation Dataset 1 into data/processed/oxford.pkl + QC report."""
    ox_raw = raw_root / "oxford"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Oxford cells from {ox_raw} ...")
    loader = OxfordBatteryLoader(mat_path=ox_raw)
    bundle = loader.load()

    out_pkl = out_dir / "oxford.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved {len(bundle.cycles)} harmonized Oxford cycles across {bundle.cell_ids} -> {out_pkl}")

    qc_path = out_dir / "oxford_qc_report.md"
    report = generate_qc_markdown_report(bundle, qc_path)
    logger.info(f"Wrote Oxford QC report -> {qc_path}")
    print("\n" + report)
    return out_pkl


def build_sandia(raw_root: Path, out_dir: Path) -> Path:
    """Preprocess Sandia Preger 2020 (0-100% NCA & NMC) into data/processed/sandia.pkl + QC report."""
    snl_raw = raw_root / "sandia"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Sandia NCA & NMC (0-100% SOC) cohort from {snl_raw} ...")
    loader = SandiaBatteryLoader(raw_dir=snl_raw)
    bundle = loader.load_cohort()

    out_pkl = out_dir / "sandia.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_dedup = sum(m.get("n_deduplicated_cycles", 0) for m in bundle.metadata["per_cell_stats"].values())
    total_bad_t = sum(m.get("dropped_bad_temp_cycles", 0) for m in bundle.metadata["per_cell_stats"].values())
    logger.info(
        f"Saved {len(bundle.cycles)} harmonized Sandia cycles across {len(bundle.cell_ids)} cells "
        f"(deduplicated={total_dedup}, dropped_bad_temp={total_bad_t}) -> {out_pkl}"
    )

    qc_path = out_dir / "sandia_qc_report.md"
    report = generate_qc_markdown_report(bundle, qc_path)
    logger.info(f"Wrote Sandia QC report -> {qc_path}")
    print("\n" + report)
    return out_pkl


def main():
    parser = argparse.ArgumentParser(description="Build canonical 2 mV DatasetBundles in data/processed/.")
    parser.add_argument("--dataset", choices=["nasa", "oxford", "sandia", "all"], default="nasa")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-audit", action="store_true", help="Skip scanning all 34 NASA .mat files.")
    args = parser.parse_args()

    if args.dataset in ("nasa", "all"):
        build_nasa(args.raw_dir, args.out_dir, run_audit=not args.skip_audit)
    if args.dataset in ("oxford", "all"):
        build_oxford(args.raw_dir, args.out_dir)
    if args.dataset in ("sandia", "all"):
        build_sandia(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
