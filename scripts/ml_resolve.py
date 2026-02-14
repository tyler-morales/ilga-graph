#!/usr/bin/env python3
"""Interactive entity resolution: link vote names to member IDs.

Usage:
    python scripts/ml_resolve.py              # Interactive human-in-the-loop
    python scripts/ml_resolve.py --auto       # Auto-resolve (no user input)
    python scripts/ml_resolve.py --stats      # Show resolution stats only
    python scripts/ml_resolve.py --batch 20   # Set batch size for interactive

First run ``make ml-pipeline`` to generate the parquet files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

PROCESSED_DIR = Path("processed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entity resolution for vote names")
    parser.add_argument("--auto", action="store_true", help="Auto-resolve only (no user input)")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    parser.add_argument("--batch", type=int, default=50, help="Batch size for interactive mode")
    parser.add_argument(
        "--threshold",
        type=float,
        default=95.0,
        help="Fuzzy auto-resolve threshold (0-100)",
    )
    args = parser.parse_args()

    # Check parquet files exist
    casts_path = PROCESSED_DIR / "fact_vote_casts_raw.parquet"
    members_path = PROCESSED_DIR / "dim_members.parquet"

    if not casts_path.exists() or not members_path.exists():
        print("ERROR: Parquet files not found. Run 'make ml-pipeline' first.")
        sys.exit(1)

    df_casts = pl.read_parquet(casts_path)
    df_members = pl.read_parquet(members_path)

    print(f"Loaded {len(df_casts):,} vote casts, {len(df_members)} members")

    from ilga_graph.ml.active_learner import auto_resolve, display_stats, interactive_session
    from ilga_graph.ml.entity_resolution import resolve_all_names

    if args.stats:
        report = resolve_all_names(df_casts, df_members, fuzzy_auto_threshold=args.threshold)
        display_stats(report)
        # Show some unresolved examples
        unresolved = [r for r in report.results if r.method == "unresolved"]
        if unresolved:
            print("\nTop 20 unresolved names (by frequency):")
            for r in sorted(unresolved, key=lambda x: -x.occurrence_count)[:20]:
                print(f"  {r.raw_name:30s} ({r.chamber}, {r.occurrence_count} votes)")
    elif args.auto:
        auto_resolve(df_casts, df_members, fuzzy_threshold=args.threshold)
    else:
        interactive_session(df_casts, df_members, batch_size=args.batch)


if __name__ == "__main__":
    main()
