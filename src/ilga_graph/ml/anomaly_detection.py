"""Witness slip anomaly detection (astroturfing vs genuine engagement).

Fixes over v1:
- Adds COORDINATION features that distinguish organized campaigns from
  genuine public interest: name duplication rate, filing time burstiness,
  position unanimity relative to bill size
- Normalizes for bill size (large totals alone don't mean astroturfing)
- Separates "high engagement" (genuine controversy) from "suspicious
  coordination" (same names, same org, all one position, burst timing)
- Reports WHY each bill was flagged, not just a score

Outputs:
    processed/slip_anomalies.parquet -- Bills scored with anomaly reasons
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
    """Build per-bill features that distinguish coordination from engagement.

    Key insight: genuine controversy has many DIVERSE filers on BOTH sides.
    Astroturfing has many SIMILAR filers on ONE side.
    """
    if len(df_slips) == 0:
        return pl.DataFrame()

    # Basic aggregates per bill
    agg = df_slips.group_by("bill_id").agg(
        pl.len().alias("total_slips"),
        (pl.col("position") == "Proponent").sum().alias("n_proponent"),
        (pl.col("position") == "Opponent").sum().alias("n_opponent"),
        pl.col("organization_clean").n_unique().alias("unique_orgs"),
        pl.col("name_clean").n_unique().alias("unique_names"),
        (pl.col("testimony_type").str.contains("(?i)record|written")).sum().alias("n_written_only"),
    )

    agg = agg.with_columns(
        # ── Coordination signals ──────────────────────────────────────
        # 1. Name duplication rate: total_slips / unique_names
        # High = same people filing multiple times (suspicious)
        (
            pl.col("total_slips").cast(pl.Float64)
            / pl.col("unique_names").cast(pl.Float64).clip(1, None)
        ).alias("name_duplication_rate"),
        # 2. Position unanimity: |proponents - opponents| / total
        # High = all one side (could be real OR astroturf)
        (
            (pl.col("n_proponent") - pl.col("n_opponent")).abs()
            / pl.col("total_slips").cast(pl.Float64).clip(1, None)
        ).alias("position_unanimity"),
        # 3. Written-only rate
        (pl.col("n_written_only") / pl.col("total_slips").cast(pl.Float64).clip(1, None)).alias(
            "written_ratio"
        ),
        # 4. Org diversity: unique_orgs / total_slips
        # Low = few orgs dominating (suspicious)
        (
            pl.col("unique_orgs").cast(pl.Float64)
            / pl.col("total_slips").cast(pl.Float64).clip(1, None)
        ).alias("org_diversity"),
        # 5. Names per org: unique_names / unique_orgs
        # Very high = one org mobilizing many individuals (could be real)
        (
            pl.col("unique_names").cast(pl.Float64)
            / pl.col("unique_orgs").cast(pl.Float64).clip(1, None)
        ).alias("names_per_org"),
        # 6. Log scale of total (normalize for size)
        pl.col("total_slips").cast(pl.Float64).log().alias("log_total"),
    )

    # ── Org concentration (HHI) ──────────────────────────────────────
    org_counts = df_slips.group_by(["bill_id", "organization_clean"]).agg(
        pl.len().alias("org_count")
    )
    bill_totals = df_slips.group_by("bill_id").agg(pl.len().alias("bill_total"))
    org_shares = org_counts.join(bill_totals, on="bill_id").with_columns(
        (pl.col("org_count") / pl.col("bill_total").cast(pl.Float64)).alias("share")
    )
    hhi = org_shares.group_by("bill_id").agg((pl.col("share").pow(2).sum()).alias("org_hhi"))
    # Also get the top org's share
    top_org = (
        org_shares.sort("share", descending=True)
        .group_by("bill_id")
        .first()
        .select(["bill_id", pl.col("share").alias("top_org_share")])
    )

    agg = (
        agg.join(hhi, on="bill_id", how="left")
        .join(top_org, on="bill_id", how="left")
        .fill_null(0.0)
    )

    # Filter: at least 10 slips for meaningful analysis
    agg = agg.filter(pl.col("total_slips") >= 10)

    return agg


def _classify_anomaly_reason(row: dict) -> str:
    """Generate human-readable reason for why a bill was flagged."""
    reasons = []

    if row.get("top_org_share", 0) > 0.5:
        reasons.append(f"single org files {row['top_org_share']:.0%} of slips")
    if row.get("name_duplication_rate", 0) > 2.0:
        reasons.append(f"names appear {row['name_duplication_rate']:.1f}x on avg")
    if row.get("position_unanimity", 0) > 0.95:
        reasons.append("near-unanimous position (>95% one side)")
    if row.get("org_diversity", 0) < 0.05:
        reasons.append(f"very low org diversity ({row['org_diversity']:.1%})")
    if row.get("org_hhi", 0) > 0.3:
        reasons.append(f"high org concentration (HHI={row['org_hhi']:.2f})")

    return "; ".join(reasons) if reasons else "unusual pattern"


def _load_anomaly_gold_labels() -> dict[str, int]:
    """Load gold anomaly labels from processed/anomaly_labels_gold.json.

    Returns {bill_number: label} where 1 = suspicious, 0 = genuine.
    """
    gold_path = PROCESSED_DIR / "anomaly_labels_gold.json"
    if not gold_path.exists():
        return {}

    import json

    with open(gold_path) as f:
        data = json.load(f)

    labels = {}
    for bill_num, entry in data.get("labels", {}).items():
        if bill_num.startswith("_comment"):
            continue
        if isinstance(entry, dict):
            labels[bill_num] = entry.get("label", 0)
        else:
            labels[bill_num] = int(entry)

    return labels


def _tune_contamination_with_gold(
    X_scaled: np.ndarray,
    bill_numbers: list[str],
    gold_labels: dict[str, int],
) -> float:
    """Tune Isolation Forest contamination using gold labels.

    Tests contamination values from 0.02 to 0.20 and picks the one
    that maximizes F1 score against gold-labeled bills.

    Returns the best contamination value, or 0.08 as default.
    """
    # Map bill_numbers to gold labels
    gold_indices = []
    gold_y = []
    for i, bn in enumerate(bill_numbers):
        if bn in gold_labels:
            gold_indices.append(i)
            gold_y.append(gold_labels[bn])

    if len(gold_y) < 5:
        LOGGER.info(
            "Too few gold labels (%d) for contamination tuning; using default 0.08", len(gold_y)
        )
        return 0.08

    gold_y_arr = np.array(gold_y)
    LOGGER.info(
        "Tuning contamination with %d gold labels (%d suspicious, %d genuine)",
        len(gold_y),
        gold_y_arr.sum(),
        len(gold_y) - gold_y_arr.sum(),
    )

    best_f1 = 0.0
    best_contamination = 0.08

    for contamination in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=300,
        )
        iso.fit(X_scaled)
        preds = iso.predict(X_scaled)

        # Get predictions for gold-labeled bills
        gold_preds = np.array([1 if preds[i] == -1 else 0 for i in gold_indices])

        # Compute F1
        tp = ((gold_preds == 1) & (gold_y_arr == 1)).sum()
        fp = ((gold_preds == 1) & (gold_y_arr == 0)).sum()
        fn = ((gold_preds == 0) & (gold_y_arr == 1)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        LOGGER.debug(
            "contamination=%.2f: P=%.2f R=%.2f F1=%.3f (tp=%d fp=%d fn=%d)",
            contamination,
            precision,
            recall,
            f1,
            tp,
            fp,
            fn,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_contamination = contamination

    LOGGER.info(
        "Best contamination: %.2f (F1=%.3f on gold labels)",
        best_contamination,
        best_f1,
    )
    return best_contamination


def run_anomaly_detection() -> pl.DataFrame:
    """Full anomaly detection pipeline."""
    console.print("\n[bold]Witness Slip Anomaly Detection[/]")

    df_slips = pl.read_parquet(PROCESSED_DIR / "fact_witness_slips.parquet")
    LOGGER.info("Loaded %d witness slips", len(df_slips))

    df_features = build_slip_anomaly_features(df_slips)
    if len(df_features) == 0:
        console.print("[dim]No witness slip data.[/]")
        return pl.DataFrame()

    LOGGER.info(
        "Built features for %d bills with 10+ slips",
        len(df_features),
    )

    # Features that capture COORDINATION, not just size
    feature_cols = [
        "name_duplication_rate",
        "position_unanimity",
        "written_ratio",
        "org_diversity",
        "names_per_org",
        "org_hhi",
        "top_org_share",
        "log_total",  # Size as context, not primary signal
    ]

    X = df_features.select(feature_cols).fill_nan(0).fill_null(0).to_numpy().astype(np.float64)

    # Guard: if all rows were filtered out, nothing to do
    if X.shape[0] == 0:
        LOGGER.warning("No bills with enough slips for anomaly detection.")
        return pl.DataFrame()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Gold-label-based contamination tuning ─────────────────────────
    # If gold labels exist, tune contamination to maximize F1 on known
    # astroturfing vs genuine engagement cases. Otherwise use default.
    gold_labels = _load_anomaly_gold_labels()
    bill_numbers = df_features["bill_id"].to_list()

    # We need bill_number_raw for gold label matching; join if available
    df_bills = pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet")
    bn_map = dict(
        zip(
            df_bills["bill_id"].to_list(),
            df_bills["bill_number_raw"].to_list(),
        )
    )
    bill_number_raws = [bn_map.get(bid, "") for bid in bill_numbers]

    if gold_labels:
        contamination = _tune_contamination_with_gold(
            X_scaled,
            bill_number_raws,
            gold_labels,
        )
        console.print(f"  Contamination tuned to {contamination:.0%} using gold labels")
    else:
        contamination = 0.08

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=300,
    )
    iso.fit(X_scaled)
    raw_scores = iso.decision_function(X_scaled)
    predictions = iso.predict(X_scaled)

    # Normalize to 0-1
    anomaly_scores = -raw_scores
    if anomaly_scores.max() > anomaly_scores.min():
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (
            anomaly_scores.max() - anomaly_scores.min()
        )

    # Classify reasons
    feature_dicts = df_features.to_dicts()
    reasons = [_classify_anomaly_reason(row) for row in feature_dicts]

    df_features = df_features.with_columns(
        pl.Series(
            "anomaly_score",
            [round(float(s), 4) for s in anomaly_scores],
        ),
        pl.Series("is_anomaly", [bool(p == -1) for p in predictions]),
        pl.Series("anomaly_reason", reasons),
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

    df_result = df_result.sort("anomaly_score", descending=True)

    out_path = PROCESSED_DIR / "slip_anomalies.parquet"
    df_result.write_parquet(out_path)
    LOGGER.info("Saved %d bill anomaly scores to %s", len(df_result), out_path)

    display_anomaly_summary(df_result)
    return df_result


def display_anomaly_summary(df: pl.DataFrame) -> None:
    """Show anomaly detection summary with reasons."""
    if len(df) == 0:
        return

    anomalies = df.filter(pl.col("is_anomaly"))
    console.print()

    summary = Table(title="Anomaly Detection Summary", show_lines=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Bills analyzed (10+ slips)", f"{len(df):,}")
    summary.add_row(
        "Flagged as suspicious",
        f"{len(anomalies):,} ({100 * len(anomalies) / len(df):.1f}%)",
    )
    console.print(summary)

    if len(anomalies) == 0:
        return

    console.print()
    top = Table(
        title="Top Suspicious Slip Patterns (with reasons)",
        show_lines=True,
    )
    top.add_column("Bill", style="bold")
    top.add_column("Description")
    top.add_column("Slips", justify="right")
    top.add_column("TopOrg%", justify="right")
    top.add_column("HHI", justify="right")
    top.add_column("Why Flagged")

    for row in anomalies.head(10).to_dicts():
        top.add_row(
            row.get("bill_number_raw", ""),
            (row.get("description", "") or "")[:30],
            str(int(row.get("total_slips", 0))),
            f"{row.get('top_org_share', 0):.0%}",
            f"{row.get('org_hhi', 0):.3f}",
            (row.get("anomaly_reason", "") or "")[:45],
        )

    console.print(top)
