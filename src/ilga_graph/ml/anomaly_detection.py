"""Automated witness slip anomaly detection (astroturfing detector).

Scores every bill's witness slip filing pattern for signs of coordinated
or fake grassroots support. Fully unsupervised -- no labels needed.

Features that signal astroturfing:
- High concentration from single organization (HHI)
- Very high ratio of written-only testimony (no oral)
- Time-burstiness: all slips filed in a narrow window
- Extreme proponent/opponent ratio (suspicious unanimity)
- Very high total slips with low org diversity

Outputs:
    processed/slip_anomalies.parquet -- Bills ranked by anomaly score
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")

console = Console()


def build_slip_anomaly_features(df_slips: pl.DataFrame) -> pl.DataFrame:
    """Build per-bill features for anomaly detection."""
    if len(df_slips) == 0:
        return pl.DataFrame()

    # Basic aggregates per bill
    agg = df_slips.group_by("bill_id").agg(
        pl.len().alias("total_slips"),
        (pl.col("position") == "Proponent").sum().alias("proponent_count"),
        (pl.col("position") == "Opponent").sum().alias("opponent_count"),
        pl.col("organization_clean").n_unique().alias("unique_orgs"),
        pl.col("name_clean").n_unique().alias("unique_names"),
        (pl.col("testimony_type").str.contains("(?i)record|written"))
        .sum()
        .alias("written_only_count"),
    )

    # Compute ratios
    agg = agg.with_columns(
        # Written-only ratio
        (pl.col("written_only_count") / pl.col("total_slips").cast(pl.Float64).clip(1, None)).alias(
            "written_ratio"
        ),
        # Proponent ratio (unanimity measure)
        (pl.col("proponent_count") / pl.col("total_slips").cast(pl.Float64).clip(1, None)).alias(
            "proponent_ratio"
        ),
        # Names-per-org ratio (low = many names from few orgs)
        (
            pl.col("unique_names").cast(pl.Float64)
            / pl.col("unique_orgs").cast(pl.Float64).clip(1, None)
        ).alias("names_per_org"),
        # Org diversity: unique_orgs / total_slips
        (
            pl.col("unique_orgs").cast(pl.Float64)
            / pl.col("total_slips").cast(pl.Float64).clip(1, None)
        ).alias("org_diversity"),
    )

    # Org concentration (HHI)
    org_counts = df_slips.group_by(["bill_id", "organization_clean"]).agg(
        pl.len().alias("org_count")
    )
    bill_totals = df_slips.group_by("bill_id").agg(pl.len().alias("bill_total"))
    org_shares = org_counts.join(bill_totals, on="bill_id").with_columns(
        (pl.col("org_count") / pl.col("bill_total").cast(pl.Float64)).alias("share")
    )
    hhi = org_shares.group_by("bill_id").agg((pl.col("share").pow(2).sum()).alias("org_hhi"))

    agg = agg.join(hhi, on="bill_id", how="left").fill_null(0.0)

    # Filter to bills with at least some slips (skip empty)
    agg = agg.filter(pl.col("total_slips") >= 5)

    return agg


def run_anomaly_detection() -> pl.DataFrame:
    """Full automated anomaly detection pipeline.

    Returns DataFrame of bills ranked by anomaly score.
    """
    console.print("\n[bold]Witness Slip Anomaly Detection[/]")

    df_slips = pl.read_parquet(PROCESSED_DIR / "fact_witness_slips.parquet")
    LOGGER.info("Loaded %d witness slips", len(df_slips))

    # Build features
    df_features = build_slip_anomaly_features(df_slips)
    if len(df_features) == 0:
        console.print("[dim]No witness slip data for anomaly detection.[/]")
        return pl.DataFrame()

    LOGGER.info("Built features for %d bills with 5+ slips", len(df_features))

    # Feature columns for the model
    feature_cols = [
        "total_slips",
        "proponent_count",
        "opponent_count",
        "unique_orgs",
        "unique_names",
        "written_ratio",
        "proponent_ratio",
        "names_per_org",
        "org_diversity",
        "org_hhi",
    ]

    X = df_features.select(feature_cols).to_numpy().astype(np.float64)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest: negative scores = more anomalous
    iso = IsolationForest(
        contamination=0.1,  # Expect ~10% anomalies
        random_state=42,
        n_estimators=200,
    )
    iso.fit(X_scaled)
    raw_scores = iso.decision_function(X_scaled)
    predictions = iso.predict(X_scaled)  # 1 = normal, -1 = anomaly

    # Convert to 0-1 anomaly score (higher = more anomalous)
    # Raw scores are centered around 0; negative = anomalous
    anomaly_scores = -raw_scores
    # Normalize to 0-1 range
    if anomaly_scores.max() > anomaly_scores.min():
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (
            anomaly_scores.max() - anomaly_scores.min()
        )

    df_features = df_features.with_columns(
        pl.Series("anomaly_score", [round(float(s), 4) for s in anomaly_scores]),
        pl.Series(
            "is_anomaly",
            [bool(p == -1) for p in predictions],
        ),
    )

    # Join with bill details
    df_bills = pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet")
    df_result = df_features.join(
        df_bills.select(
            [
                "bill_id",
                "bill_number_raw",
                "description",
                "primary_sponsor",
                "chamber_origin",
            ]
        ),
        on="bill_id",
        how="left",
    )

    # Sort by anomaly score descending
    df_result = df_result.sort("anomaly_score", descending=True)

    # Save
    out_path = PROCESSED_DIR / "slip_anomalies.parquet"
    df_result.write_parquet(out_path)
    LOGGER.info("Saved %d bill anomaly scores to %s", len(df_result), out_path)

    display_anomaly_summary(df_result)
    return df_result


def display_anomaly_summary(df: pl.DataFrame) -> None:
    """Show anomaly detection summary."""
    if len(df) == 0:
        return

    anomalies = df.filter(pl.col("is_anomaly"))
    console.print()

    summary = Table(title="Anomaly Detection Summary", show_lines=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Bills analyzed", f"{len(df):,}")
    summary.add_row(
        "Flagged as anomalous",
        f"{len(anomalies):,} ({100 * len(anomalies) / len(df):.1f}%)",
    )
    console.print(summary)

    if len(anomalies) == 0:
        return

    # Top anomalies
    console.print()
    top = Table(
        title="Top 10 Most Suspicious Slip Patterns",
        show_lines=True,
    )
    top.add_column("Bill", style="bold")
    top.add_column("Description")
    top.add_column("Slips", justify="right")
    top.add_column("Orgs", justify="right")
    top.add_column("Written%", justify="right")
    top.add_column("HHI", justify="right")
    top.add_column("Score", justify="right")

    for row in anomalies.head(10).to_dicts():
        top.add_row(
            row.get("bill_number_raw", ""),
            (row.get("description", "") or "")[:35],
            str(row.get("total_slips", 0)),
            str(row.get("unique_orgs", 0)),
            f"{row.get('written_ratio', 0):.0%}",
            f"{row.get('org_hhi', 0):.3f}",
            f"[red]{row['anomaly_score']:.3f}[/]",
        )

    console.print(top)
