"""CLI entrypoint for running smoke, loco, and transfer experiments."""

import argparse

from optiSoh.contracts.synthetic import generate_synthetic_battery_dataset
from optiSoh.utils.logging import setup_logger

logger = setup_logger("runner")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "loco", "transfer"], default="smoke")
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    logger.info(f"Starting optiSoh runner | mode: {args.mode} | budget: {args.budget} | seeds: {args.seeds}")
    if args.mode == "smoke":
        bundle = generate_synthetic_battery_dataset(n_cells=4, cycles_per_cell=20)
        logger.info(f"Synthetic smoke dataset ready with {len(bundle.cycles)} cycles.")
        logger.info("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
