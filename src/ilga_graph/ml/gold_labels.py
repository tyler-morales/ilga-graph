"""Gold label set generation and management.

Generates a stratified sample of bills with verified outcome labels
for model validation. The gold set is designed to:

1. Include both ADVANCE and STUCK examples (balanced sampling)
2. Stratify by chamber (Senate vs House) and bill type
3. Include metadata (bill_number, description, chamber) for auditability
4. Use the action classifier's labels as baseline truth

The gold set is saved to ``processed/bill_labels_gold.json`` and can be
used for:
- Evaluation: report accuracy on gold set in model_quality.json
- Override: use gold labels for those bills in the target vector
- Monitoring: track model drift over time

Usage::

    from ilga_graph.ml.gold_labels import generate_gold_labels
    generate_gold_labels()  # Writes processed/bill_labels_gold.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")
GOLD_LABELS_PATH = PROCESSED_DIR / "bill_labels_gold.json"


def generate_gold_labels(
    *,
    target_size: int = 400,
    balance_ratio: float = 0.5,
    seed: int = 42,
) -> dict:
    """Generate a stratified gold label set from current data.

    Samples mature bills (120+ days old) stratified by:
    - Outcome (ADVANCE vs STUCK) — balanced to ``balance_ratio``
    - Chamber (Senate vs House) — proportional
    - Bill type (SB vs HB) — proportional

    Parameters
    ----------
    target_size:
        Total number of gold labels to generate.
    balance_ratio:
        Fraction of positive (ADVANCE) examples. 0.5 = balanced.
    seed:
        Random seed for reproducible sampling.

    Returns
    -------
    dict with the gold label set and metadata.
    """
    from .features import build_bill_labels

    bills_path = PROCESSED_DIR / "dim_bills.parquet"
    actions_path = PROCESSED_DIR / "fact_bill_actions.parquet"

    if not bills_path.exists() or not actions_path.exists():
        LOGGER.warning("Cannot generate gold labels: parquet files not found.")
        return {}

    df_bills = pl.read_parquet(bills_path)
    df_actions = pl.read_parquet(actions_path)

    # Only substantive bills
    df_sub = df_bills.filter(pl.col("bill_type").is_in(["SB", "HB"]))
    df_labels = build_bill_labels(df_sub, df_actions)

    # Only mature bills (reliable labels)
    df_mature = df_labels.filter(pl.col("is_mature"))

    # Join with bill metadata
    df = df_mature.join(
        df_sub.select(
            [
                "bill_id",
                "bill_number_raw",
                "description",
                "chamber_origin",
                "bill_type",
                "primary_sponsor",
                "introduction_date",
            ]
        ),
        on="bill_id",
        how="left",
    )

    # Split into positive and negative
    df_pos = df.filter(pl.col("target_advanced") == 1)
    df_neg = df.filter(pl.col("target_advanced") == 0)

    n_pos = min(int(target_size * balance_ratio), len(df_pos))
    n_neg = min(target_size - n_pos, len(df_neg))

    LOGGER.info(
        "Gold label pool: %d advanced, %d stuck. Sampling %d + %d = %d.",
        len(df_pos),
        len(df_neg),
        n_pos,
        n_neg,
        n_pos + n_neg,
    )

    # Stratified sample
    sampled_pos = df_pos.sample(n=n_pos, seed=seed) if n_pos > 0 else df_pos.head(0)
    sampled_neg = df_neg.sample(n=n_neg, seed=seed) if n_neg > 0 else df_neg.head(0)

    df_gold = pl.concat([sampled_pos, sampled_neg]).sort("introduction_date")

    # Build gold label dict with metadata
    gold_entries = []
    for row in df_gold.to_dicts():
        gold_entries.append(
            {
                "bill_id": row["bill_id"],
                "bill_number": row.get("bill_number_raw", ""),
                "description": row.get("description", ""),
                "chamber": row.get("chamber_origin", ""),
                "bill_type": row.get("bill_type", ""),
                "sponsor": row.get("primary_sponsor", ""),
                "introduction_date": row.get("introduction_date", ""),
                "label": int(row["target_advanced"]),
                "label_law": int(row.get("target_law", 0)),
                "source": "action_classifier",
            }
        )

    # Also build the simple {leg_id: label} format for backward compatibility
    simple_labels = {entry["bill_id"]: entry["label"] for entry in gold_entries}

    result = {
        "version": 2,
        "generated_date": _today(),
        "total_labels": len(gold_entries),
        "positive_count": n_pos,
        "negative_count": n_neg,
        "balance_ratio": round(n_pos / max(n_pos + n_neg, 1), 3),
        "source_pool": {
            "total_mature_bills": len(df_mature),
            "total_advanced": len(df_pos),
            "total_stuck": len(df_neg),
        },
        "labels": simple_labels,
        "entries": gold_entries,
    }

    # Save
    PROCESSED_DIR.mkdir(exist_ok=True)
    with open(GOLD_LABELS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    LOGGER.info(
        "Gold labels saved: %d entries (%d advanced, %d stuck) to %s",
        len(gold_entries),
        n_pos,
        n_neg,
        GOLD_LABELS_PATH,
    )

    return result


def load_gold_labels() -> dict[str, int]:
    """Load gold labels as {bill_id: label} dict.

    Handles both v1 (simple dict) and v2 (structured) formats.
    """
    if not GOLD_LABELS_PATH.exists():
        return {}

    with open(GOLD_LABELS_PATH) as f:
        data = json.load(f)

    # v2 format: has "labels" key
    if isinstance(data, dict) and "labels" in data:
        return {str(k): int(v) for k, v in data["labels"].items()}

    # v1 format: simple {leg_id: label}
    if isinstance(data, dict):
        return {str(k): int(v) for k, v in data.items()}

    return {}


def _today() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")
