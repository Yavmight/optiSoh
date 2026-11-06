"""Parquet-backed results and metrics store."""
from pathlib import Path
import pandas as pd

class ResultsStore:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
