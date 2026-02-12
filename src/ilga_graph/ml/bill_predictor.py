"""Fully automated bill outcome prediction.

Trains a GradientBoosting classifier on bills with KNOWN outcomes (old enough
to have completed their lifecycle), then scores every bill in the dataset with
a probability of advancement. No human interaction required.

Outputs:
    processed/bill_scores.parquet  -- Every bill with predicted probability
    processed/bill_predictor.pkl   -- Trained model for reuse
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)

from .features import build_feature_matrix

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")
MODEL_PATH = PROCESSED_DIR / "bill_predictor.pkl"

console = Console()


# ── Model training ───────────────────────────────────────────────────────────


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
) -> GradientBoostingClassifier:
    """Train a GradientBoosting classifier with class-weight handling."""
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    if pos_count > 0 and neg_count > 0:
        weight_pos = len(y_train) / (2 * pos_count)
        weight_neg = len(y_train) / (2 * neg_count)
        sample_weights = np.where(y_train == 1, weight_pos, weight_neg)
    else:
        sample_weights = np.ones(len(y_train))

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def evaluate_model(
    model: GradientBoostingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "roc_auc": auc,
        "accuracy": report.get("accuracy", 0),
        "precision_pos": report.get("1", {}).get("precision", 0),
        "recall_pos": report.get("1", {}).get("recall", 0),
        "f1_pos": report.get("1", {}).get("f1-score", 0),
        "precision_neg": report.get("0", {}).get("precision", 0),
        "recall_neg": report.get("0", {}).get("recall", 0),
        "f1_neg": report.get("0", {}).get("f1-score", 0),
        "support_pos": int(y_test.sum()),
        "support_neg": int(len(y_test) - y_test.sum()),
    }


def display_metrics(metrics: dict) -> None:
    """Show a rich table of model metrics."""
    table = Table(
        title="Bill Outcome Predictor -- Training Metrics",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    auc = metrics.get("roc_auc")
    table.add_row(
        "ROC-AUC",
        f"{auc:.4f}" if auc is not None else "N/A (single class in test)",
    )
    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("", "")
    table.add_row("[green]Advanced (positive)[/]", "")
    table.add_row("  Precision", f"{metrics['precision_pos']:.4f}")
    table.add_row("  Recall", f"{metrics['recall_pos']:.4f}")
    table.add_row("  F1", f"{metrics['f1_pos']:.4f}")
    table.add_row("  Support", str(metrics["support_pos"]))
    table.add_row("", "")
    table.add_row("[red]Stuck/Dead (negative)[/]", "")
    table.add_row("  Precision", f"{metrics['precision_neg']:.4f}")
    table.add_row("  Recall", f"{metrics['recall_neg']:.4f}")
    table.add_row("  F1", f"{metrics['f1_neg']:.4f}")
    table.add_row("  Support", str(metrics["support_neg"]))

    console.print()
    console.print(table)


def display_top_features(
    model: GradientBoostingClassifier,
    feature_names: list[str],
    top_n: int = 15,
) -> None:
    """Show top feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    table = Table(
        title=f"Top {top_n} Feature Importances",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Feature", style="bold")
    table.add_column("Importance", justify="right")

    for rank, idx in enumerate(indices, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        table.add_row(str(rank), name, f"{importances[idx]:.4f}")

    console.print(table)


def save_model(model: GradientBoostingClassifier) -> None:
    """Save trained model to disk."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    LOGGER.info("Model saved to %s", MODEL_PATH)


# ── Scoring: predict on ALL bills ────────────────────────────────────────────


def score_all_bills(
    model: GradientBoostingClassifier,
    X_all: np.ndarray,
    bill_ids_all: list[str],
    y_all: np.ndarray,
) -> pl.DataFrame:
    """Score every bill with probability of advancement.

    Returns a DataFrame: bill_id, probability_advance, predicted_label,
    actual_label (from data), confidence.
    """
    y_proba = model.predict_proba(X_all)[:, 1]
    y_pred = model.predict(X_all)

    df_scores = pl.DataFrame(
        {
            "bill_id": bill_ids_all,
            "prob_advance": [round(float(p), 4) for p in y_proba],
            "predicted_outcome": ["ADVANCE" if p == 1 else "STUCK" for p in y_pred],
            "actual_outcome": ["ADVANCE" if y == 1 else "STUCK" for y in y_all],
            "confidence": [round(float(max(p, 1 - p)), 4) for p in y_proba],
        }
    )

    # Join with bill details for a useful output
    df_bills = pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet")
    df_scores = df_scores.join(
        df_bills.select(
            [
                "bill_id",
                "bill_number_raw",
                "bill_type",
                "description",
                "primary_sponsor",
                "chamber_origin",
                "introduction_date",
            ]
        ),
        on="bill_id",
        how="left",
    )

    # Sort by probability descending (most likely to advance first)
    df_scores = df_scores.sort("prob_advance", descending=True)

    out_path = PROCESSED_DIR / "bill_scores.parquet"
    df_scores.write_parquet(out_path)
    LOGGER.info("Saved %d bill scores to %s", len(df_scores), out_path)

    return df_scores


def display_score_summary(df_scores: pl.DataFrame) -> None:
    """Show summary of bill scores."""
    total = len(df_scores)
    predicted_advance = df_scores.filter(pl.col("predicted_outcome") == "ADVANCE")
    high_conf = df_scores.filter(pl.col("confidence") >= 0.8)

    console.print()
    table = Table(title="Bill Score Summary", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total bills scored", f"{total:,}")
    table.add_row(
        "Predicted to ADVANCE",
        f"{len(predicted_advance):,} ({100 * len(predicted_advance) / total:.1f}%)",
    )
    table.add_row(
        "High confidence (>= 80%)",
        f"{len(high_conf):,}",
    )

    console.print(table)

    # Top 10 most likely to advance
    console.print()
    top_table = Table(
        title="Top 15 Bills Most Likely to Advance",
        show_lines=True,
    )
    top_table.add_column("Bill", style="bold")
    top_table.add_column("Description")
    top_table.add_column("Sponsor")
    top_table.add_column("P(advance)", justify="right")
    top_table.add_column("Actual", justify="center")

    for row in df_scores.head(15).to_dicts():
        actual_color = "green" if row["actual_outcome"] == "ADVANCE" else "red"
        top_table.add_row(
            row.get("bill_number_raw", ""),
            (row.get("description", "") or "")[:40],
            (row.get("primary_sponsor", "") or "")[:20],
            f"{row['prob_advance']:.1%}",
            f"[{actual_color}]{row['actual_outcome']}[/{actual_color}]",
        )

    console.print(top_table)


# ── Main automated pipeline ──────────────────────────────────────────────────


def run_auto() -> pl.DataFrame:
    """Fully automated: train, evaluate, score all bills.

    No interaction. Returns the scored bill DataFrame.
    """
    console.print("\n[bold]Step 1: Building features...[/]")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        ids_train,
        ids_test,
        vectorizer,
        feature_names,
    ) = build_feature_matrix()

    console.print("[bold]Step 2: Training model...[/]")
    model = train_model(X_train, y_train)

    console.print("[bold]Step 3: Evaluating on held-out test set...[/]")
    metrics = evaluate_model(model, X_test, y_test)
    display_metrics(metrics)
    display_top_features(model, feature_names)

    save_model(model)

    # Score ALL bills (train + test combined)
    from scipy.sparse import vstack as sparse_vstack

    X_all = sparse_vstack([X_train, X_test])
    bill_ids_all = ids_train + ids_test
    y_all = np.concatenate([y_train, y_test])

    console.print("\n[bold]Step 4: Scoring all bills...[/]")
    df_scores = score_all_bills(model, X_all, bill_ids_all, y_all)
    display_score_summary(df_scores)

    return df_scores
