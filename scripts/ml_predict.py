#!/usr/bin/env python3
"""Bill outcome prediction with interactive training.

Usage:
    python scripts/ml_predict.py              # Train + interactive review
    python scripts/ml_predict.py --train      # Train and evaluate only
    python scripts/ml_predict.py --evaluate   # Show metrics of saved model

First run ``make ml-pipeline`` to generate the parquet files.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Bill outcome predictor")
    parser.add_argument("--train", action="store_true", help="Train and evaluate only")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved model")
    args = parser.parse_args()

    # Check parquet files exist
    if not (PROCESSED_DIR / "dim_bills.parquet").exists():
        print("ERROR: Parquet files not found. Run 'make ml-pipeline' first.")
        sys.exit(1)

    from ilga_graph.ml.bill_predictor import (
        display_metrics,
        evaluate_model,
        load_model,
        run_interactive,
        run_train_and_evaluate,
    )
    from ilga_graph.ml.features import build_feature_matrix

    if args.evaluate:
        model = load_model()
        if model is None:
            print("ERROR: No saved model. Run with --train first.")
            sys.exit(1)
        (_, X_test, _, y_test, _, _, _, _) = build_feature_matrix()
        metrics = evaluate_model(model, X_test, y_test)
        display_metrics(metrics)
    elif args.train:
        run_train_and_evaluate()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
