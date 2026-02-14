#!/usr/bin/env python3
"""Bill outcome prediction (automated, no interaction).

Usage:
    python scripts/ml_predict.py        # Train + score all bills
    make ml-predict                     # Same via Makefile

First run ``make ml-pipeline`` to generate the parquet files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

PROCESSED_DIR = Path("processed")


def main() -> None:
    if not (PROCESSED_DIR / "dim_bills.parquet").exists():
        print("ERROR: Parquet files not found. Run 'make ml-pipeline' first.")
        sys.exit(1)

    from ilga_graph.ml.bill_predictor import run_auto

    run_auto()


if __name__ == "__main__":
    main()
