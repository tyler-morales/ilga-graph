#!/usr/bin/env python3
"""Run the full ML intelligence pipeline end-to-end.

Single command, no interaction. Produces enriched analytical data:

    0. Backtest previous predictions (self-correcting feedback loop)
    1. Star schema (cache/*.json -> processed/*.parquet)
    2. Entity resolution (vote names -> member IDs)
    3. Bill outcome scoring (every bill gets a probability)
    4. Coalition discovery (voting blocs via graph embeddings)
    5. Anomaly detection (astroturfing signals in witness slips)
    6. Snapshot predictions (for next run's backtest)

Usage:
    python scripts/ml_run.py          # Full pipeline
    make ml-run                       # Same via Makefile
"""

from __future__ import annotations

import logging
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

console = Console()


def main() -> None:
    t_total = time.perf_counter()

    console.print(
        Panel(
            "[bold]Legislative Intelligence Engine[/]\n"
            "Fully automated ML pipeline -- no interaction required.\n"
            "Self-correcting: backtests previous predictions on each run.",
            border_style="cyan",
        )
    )

    # ── Step 0: Backtest previous predictions ─────────────────────────
    console.print("\n[bold cyan]== Step 0/6: Backtest Previous Predictions ==[/]")
    t0 = time.perf_counter()

    backtest_result = None
    try:
        from ilga_graph.ml.backtester import backtest_previous_run

        backtest_result = backtest_previous_run()
    except Exception as e:
        console.print(f"[dim]Backtest skipped: {e}[/]")
    t_backtest = time.perf_counter() - t0

    # ── Step 1: Data Pipeline ────────────────────────────────────────
    console.print("\n[bold cyan]== Step 1/6: Data Pipeline ==[/]")
    t0 = time.perf_counter()

    from ilga_graph.ml.pipeline import run_pipeline

    tables = run_pipeline()
    t_pipeline = time.perf_counter() - t0

    # ── Step 2: Entity Resolution ────────────────────────────────────
    console.print("\n[bold cyan]== Step 2/6: Entity Resolution ==[/]")
    t0 = time.perf_counter()

    import polars as pl

    from ilga_graph.ml.active_learner import auto_resolve

    df_casts = pl.read_parquet("processed/fact_vote_casts_raw.parquet")
    df_members = pl.read_parquet("processed/dim_members.parquet")
    report = auto_resolve(df_casts, df_members, fuzzy_threshold=90.0)
    t_resolve = time.perf_counter() - t0

    # ── Step 3: Bill Outcome Scoring ─────────────────────────────────
    console.print("\n[bold cyan]== Step 3/6: Bill Outcome Prediction ==[/]")
    t0 = time.perf_counter()

    from ilga_graph.ml.bill_predictor import run_auto

    df_scores = run_auto()
    t_predict = time.perf_counter() - t0

    # ── Step 4: Coalition Discovery ──────────────────────────────────
    console.print("\n[bold cyan]== Step 4/6: Coalition Discovery ==[/]")
    t0 = time.perf_counter()

    from ilga_graph.ml.coalitions import run_coalition_discovery

    df_coalitions = run_coalition_discovery()
    t_coalitions = time.perf_counter() - t0

    # ── Step 5: Anomaly Detection ────────────────────────────────────
    console.print("\n[bold cyan]== Step 5/6: Anomaly Detection ==[/]")
    t0 = time.perf_counter()

    from ilga_graph.ml.anomaly_detection import run_anomaly_detection

    df_anomalies = run_anomaly_detection()
    t_anomalies = time.perf_counter() - t0

    # ── Step 6: Snapshot predictions for next backtest ───────────────
    console.print("\n[bold cyan]== Step 6/6: Snapshot Predictions ==[/]")
    t0 = time.perf_counter()

    from ilga_graph.ml.backtester import snapshot_current_predictions

    snapshot_current_predictions()
    t_snapshot = time.perf_counter() - t0

    # ── Summary ──────────────────────────────────────────────────────
    t_total_elapsed = time.perf_counter() - t_total

    console.print()
    summary = Table(
        title="Pipeline Complete",
        show_lines=True,
        title_style="bold green",
    )
    summary.add_column("Step", style="bold")
    summary.add_column("Output", style="dim")
    summary.add_column("Time", justify="right")

    backtest_desc = "no previous predictions"
    if backtest_result:
        backtest_desc = (
            f"{backtest_result.accuracy:.1%} accuracy "
            f"({backtest_result.correct}/{backtest_result.total_testable})"
        )
    summary.add_row("Backtest", backtest_desc, f"{t_backtest:.1f}s")
    summary.add_row(
        "Data Pipeline",
        (f"{sum(len(df) for df in tables.values()):,} rows across {len(tables)} tables"),
        f"{t_pipeline:.1f}s",
    )
    summary.add_row(
        "Entity Resolution",
        (
            f"{report.resolution_rate:.1%} resolved "
            f"({report.total_resolved}/{report.total_unique_names})"
        ),
        f"{t_resolve:.1f}s",
    )
    summary.add_row(
        "Bill Scoring",
        f"{len(df_scores):,} bills scored",
        f"{t_predict:.1f}s",
    )
    summary.add_row(
        "Coalitions",
        (f"{len(df_coalitions)} members clustered" if len(df_coalitions) > 0 else "skipped"),
        f"{t_coalitions:.1f}s",
    )
    anomaly_count = len(df_anomalies.filter(pl.col("is_anomaly"))) if len(df_anomalies) > 0 else 0
    summary.add_row(
        "Anomaly Detection",
        (f"{anomaly_count} bills flagged" if len(df_anomalies) > 0 else "skipped"),
        f"{t_anomalies:.1f}s",
    )
    summary.add_row(
        "Snapshot",
        "saved for next backtest",
        f"{t_snapshot:.1f}s",
    )
    summary.add_row(
        "[bold]Total[/]",
        "",
        f"[bold]{t_total_elapsed:.1f}s[/]",
    )

    console.print(summary)
    console.print("\n[dim]All outputs in processed/. Query with: polars, DuckDB, or pandas.[/]")


if __name__ == "__main__":
    sys.exit(main() or 0)
