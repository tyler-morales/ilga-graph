"""Prediction backtesting and accuracy tracking.

On each pipeline run:
1. Load previous predictions (if they exist)
2. Compare them against current actual outcomes
3. Grade accuracy overall and by confidence bucket
4. Save the report to processed/accuracy_history.json
5. Snapshot current predictions for the next run

This creates a self-correcting feedback loop: every scrape cycle,
the system checks how well it did and you can watch accuracy trends.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")
HISTORY_DIR = PROCESSED_DIR / "history"
HISTORY_FILE = PROCESSED_DIR / "accuracy_history.json"
SCORES_FILE = PROCESSED_DIR / "bill_scores.parquet"
MAX_SNAPSHOTS = 20

console = Console()


@dataclass
class ConfidenceBucket:
    bucket: str  # e.g. "80-90%"
    total: int
    correct: int
    accuracy: float


@dataclass
class PredictionMiss:
    bill_number: str
    description: str
    predicted: str  # "ADVANCE" or "STUCK"
    actual: str
    confidence: float
    prob_advance: float


@dataclass
class BacktestResult:
    run_date: str
    snapshot_date: str
    days_elapsed: int
    total_testable: int
    correct: int
    accuracy: float
    precision_advance: float  # of predicted ADVANCE, how many did
    recall_advance: float  # of actually advanced, how many predicted
    f1_advance: float
    confidence_buckets: list[ConfidenceBucket] = field(default_factory=list)
    biggest_misses: list[PredictionMiss] = field(default_factory=list)
    model_version: str = ""


def _load_history() -> list[dict]:
    """Load existing accuracy history."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            data = json.load(f)
            return data.get("runs", [])
    return []


def _save_history(runs: list[dict]) -> None:
    """Save accuracy history."""
    PROCESSED_DIR.mkdir(exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump({"runs": runs}, f, indent=2)


def snapshot_current_predictions() -> None:
    """Copy current bill_scores.parquet to history/ with timestamp."""
    if not SCORES_FILE.exists():
        return

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = HISTORY_DIR / f"{timestamp}_scores.parquet"
    shutil.copy2(SCORES_FILE, dest)
    LOGGER.info("Snapshot saved: %s", dest)

    # Prune old snapshots
    snapshots = sorted(HISTORY_DIR.glob("*_scores.parquet"))
    if len(snapshots) > MAX_SNAPSHOTS:
        for old in snapshots[: len(snapshots) - MAX_SNAPSHOTS]:
            old.unlink()
            LOGGER.info("Pruned old snapshot: %s", old.name)


def _find_latest_snapshot() -> Path | None:
    """Find the most recent prediction snapshot."""
    if not HISTORY_DIR.exists():
        return None
    snapshots = sorted(HISTORY_DIR.glob("*_scores.parquet"))
    return snapshots[-1] if snapshots else None


def backtest_previous_run() -> BacktestResult | None:
    """Compare previous predictions against current actual outcomes.

    Returns a BacktestResult, or None if no previous snapshot exists.
    """
    snapshot_path = _find_latest_snapshot()
    if snapshot_path is None:
        console.print("[dim]No previous predictions to backtest (first run).[/]")
        return None

    console.print(f"\n[bold]Backtesting predictions from {snapshot_path.name}...[/]")

    # Load old predictions
    df_old = pl.read_parquet(snapshot_path)

    # Build current actual outcomes from the latest action data
    actions_path = PROCESSED_DIR / "fact_bill_actions.parquet"
    bills_path = PROCESSED_DIR / "dim_bills.parquet"
    if not actions_path.exists() or not bills_path.exists():
        # Need fresh data to compare against -- will be available
        # after the data pipeline step runs
        console.print("[dim]No current data to compare against yet.[/]")
        return None

    # Build current labels from actions
    from .features import build_bill_labels

    df_bills = pl.read_parquet(bills_path)
    df_actions = pl.read_parquet(actions_path)
    df_labels = build_bill_labels(
        df_bills.filter(pl.col("bill_type").is_in(["SB", "HB"])),
        df_actions,
    )

    # Join old predictions with current labels
    current_labels = {row["bill_id"]: row["target_advanced"] for row in df_labels.to_dicts()}

    # Only test bills that:
    # 1. Were in the old predictions
    # 2. Had reliable labels at prediction time (mature)
    # 3. Have current labels available
    old_preds = df_old.filter(pl.col("label_reliable")).to_dicts()

    testable = []
    for pred in old_preds:
        bid = pred.get("bill_id")
        if bid and bid in current_labels:
            current_outcome = "ADVANCE" if current_labels[bid] == 1 else "STUCK"
            testable.append(
                {
                    **pred,
                    "current_actual": current_outcome,
                }
            )

    if not testable:
        console.print("[dim]No testable predictions found.[/]")
        return None

    # Compute accuracy
    correct = sum(1 for t in testable if t["predicted_outcome"] == t["current_actual"])
    total = len(testable)
    accuracy = correct / total if total > 0 else 0.0

    # Precision / recall for ADVANCE class
    tp = sum(
        1
        for t in testable
        if t["predicted_outcome"] == "ADVANCE" and t["current_actual"] == "ADVANCE"
    )
    fp = sum(
        1
        for t in testable
        if t["predicted_outcome"] == "ADVANCE" and t["current_actual"] != "ADVANCE"
    )
    fn = sum(
        1
        for t in testable
        if t["predicted_outcome"] != "ADVANCE" and t["current_actual"] == "ADVANCE"
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Confidence buckets
    buckets_def = [
        ("50-60%", 0.5, 0.6),
        ("60-70%", 0.6, 0.7),
        ("70-80%", 0.7, 0.8),
        ("80-90%", 0.8, 0.9),
        ("90-100%", 0.9, 1.01),
    ]
    confidence_buckets = []
    for label, lo, hi in buckets_def:
        in_bucket = [t for t in testable if lo <= t.get("confidence", 0) < hi]
        if in_bucket:
            b_correct = sum(1 for t in in_bucket if t["predicted_outcome"] == t["current_actual"])
            confidence_buckets.append(
                ConfidenceBucket(
                    bucket=label,
                    total=len(in_bucket),
                    correct=b_correct,
                    accuracy=(b_correct / len(in_bucket) if len(in_bucket) > 0 else 0.0),
                )
            )

    # Biggest misses: wrong predictions with highest confidence
    misses = [t for t in testable if t["predicted_outcome"] != t["current_actual"]]
    misses.sort(key=lambda t: -t.get("confidence", 0))
    biggest_misses = [
        PredictionMiss(
            bill_number=m.get("bill_number_raw", ""),
            description=(m.get("description", "") or "")[:60],
            predicted=m["predicted_outcome"],
            actual=m["current_actual"],
            confidence=m.get("confidence", 0),
            prob_advance=m.get("prob_advance", 0),
        )
        for m in misses[:10]
    ]

    # Determine snapshot date from filename
    snapshot_name = snapshot_path.stem.replace("_scores", "")
    now = datetime.now().strftime("%Y-%m-%d")
    try:
        snap_dt = datetime.strptime(snapshot_name.split("_")[0], "%Y-%m-%d")
        days_elapsed = (datetime.now() - snap_dt).days
    except ValueError:
        days_elapsed = 0

    # Load model version if available
    quality_path = PROCESSED_DIR / "model_quality.json"
    model_version = ""
    if quality_path.exists():
        with open(quality_path) as f:
            q = json.load(f)
            model_version = q.get("model_selected", "")

    result = BacktestResult(
        run_date=now,
        snapshot_date=snapshot_name.split("_")[0],
        days_elapsed=days_elapsed,
        total_testable=total,
        correct=correct,
        accuracy=round(accuracy, 4),
        precision_advance=round(precision, 4),
        recall_advance=round(recall, 4),
        f1_advance=round(f1, 4),
        confidence_buckets=confidence_buckets,
        biggest_misses=biggest_misses,
        model_version=model_version,
    )

    # Save to history
    runs = _load_history()
    runs.append(asdict(result))
    _save_history(runs)
    LOGGER.info("Backtest saved to %s", HISTORY_FILE)

    # Display
    _display_backtest(result)

    return result


def _display_backtest(result: BacktestResult) -> None:
    """Show backtest results."""
    console.print()
    table = Table(
        title="Backtest: Previous Predictions vs Reality",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Predictions tested", str(result.total_testable))
    table.add_row(
        "Correct",
        f"{result.correct} ({result.accuracy:.1%})",
    )
    table.add_row(
        "Precision (ADVANCE)",
        f"{result.precision_advance:.1%}",
    )
    table.add_row(
        "Recall (ADVANCE)",
        f"{result.recall_advance:.1%}",
    )
    table.add_row("F1 (ADVANCE)", f"{result.f1_advance:.3f}")
    table.add_row(
        "Days since prediction",
        str(result.days_elapsed),
    )
    console.print(table)

    # Confidence calibration
    if result.confidence_buckets:
        console.print()
        cal = Table(
            title="Confidence Calibration",
            show_lines=True,
        )
        cal.add_column("Bucket", style="bold")
        cal.add_column("Total", justify="right")
        cal.add_column("Correct", justify="right")
        cal.add_column("Accuracy", justify="right")

        for b in result.confidence_buckets:
            cal.add_row(
                b.bucket,
                str(b.total),
                str(b.correct),
                f"{b.accuracy:.1%}",
            )
        console.print(cal)

    # Biggest misses
    if result.biggest_misses:
        console.print()
        misses = Table(
            title="Biggest Misses (confidently wrong)",
            show_lines=True,
        )
        misses.add_column("Bill", style="bold")
        misses.add_column("Predicted")
        misses.add_column("Actual")
        misses.add_column("Confidence", justify="right")

        for m in result.biggest_misses[:5]:
            misses.add_row(
                m.bill_number,
                m.predicted,
                f"[red]{m.actual}[/]",
                f"{m.confidence:.1%}",
            )
        console.print(misses)
