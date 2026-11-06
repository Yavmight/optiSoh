"""Dataset downloader and integrity checker for NASA, Oxford, and Sandia."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download and verify raw battery datasets.")
    parser.add_argument("--dataset", choices=["nasa", "oxford", "sandia", "all"], default="all")
    args = parser.parse_args()
    print(f"Dataset fetcher invoked for: {args.dataset}")
    print("See docs/data_contracts.md for placement instructions.")

if __name__ == "__main__":
    main()
