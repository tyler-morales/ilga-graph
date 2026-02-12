#!/usr/bin/env python3
"""Run the ML data pipeline: cache/*.json -> processed/*.parquet.

Usage:
    python scripts/ml_pipeline.py
    make ml-pipeline
"""

from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    from ilga_graph.ml.pipeline import run_pipeline

    t0 = time.perf_counter()
    results = run_pipeline()
    elapsed = time.perf_counter() - t0

    print(f"\nPipeline completed in {elapsed:.1f}s")
    print("Tables written to processed/:")
    for name, df in results.items():
        print(f"  {name:30s} {len(df):>8,} rows")


if __name__ == "__main__":
    sys.exit(main() or 0)
