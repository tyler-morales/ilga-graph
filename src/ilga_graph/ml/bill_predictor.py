"""Bill outcome prediction with interactive human-in-the-loop training.

Trains a GradientBoosting classifier to predict whether a bill will advance
past committee. Supports an interactive loop where the user reviews predictions
on specific bills and provides corrections, which are used to retrain.

The model learns from:
1. Historical bill data (action history, sponsors, witness slips)
2. User corrections during interactive sessions

Usage::

    python scripts/ml_predict.py              # Train + interactive review
    python scripts/ml_predict.py --train      # Train only (no interaction)
    python scripts/ml_predict.py --evaluate   # Show metrics only
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.panel import Panel
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
LABELS_GOLD_PATH = PROCESSED_DIR / "bill_labels_gold.json"

console = Console()


# ── Gold label persistence ───────────────────────────────────────────────────


def load_gold_labels() -> dict[str, int]:
    """Load user-corrected bill labels: {bill_id: label}."""
    if not LABELS_GOLD_PATH.exists():
        return {}
    with open(LABELS_GOLD_PATH) as f:
        return json.load(f)


def save_gold_labels(labels: dict[str, int]) -> None:
    """Persist user-corrected labels."""
    with open(LABELS_GOLD_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    LOGGER.info("Saved %d gold labels to %s", len(labels), LABELS_GOLD_PATH)


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
    # Compute sample weights for class imbalance
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

    # ROC-AUC (handle edge case of single class)
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
    table = Table(title="Bill Outcome Predictor Metrics", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    auc = metrics.get("roc_auc")
    table.add_row("ROC-AUC", f"{auc:.4f}" if auc is not None else "N/A (single class)")
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
    console.print()


def display_top_features(
    model: GradientBoostingClassifier,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    """Show top feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    table = Table(title=f"Top {top_n} Feature Importances", show_lines=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Feature", style="bold")
    table.add_column("Importance", justify="right")

    for rank, idx in enumerate(indices, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        table.add_row(str(rank), name, f"{importances[idx]:.4f}")

    console.print(table)
    console.print()


def save_model(model: GradientBoostingClassifier) -> None:
    """Save trained model to disk."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    LOGGER.info("Model saved to %s", MODEL_PATH)


def load_model() -> GradientBoostingClassifier | None:
    """Load previously trained model."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Interactive prediction review ────────────────────────────────────────────


def interactive_review(
    model: GradientBoostingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bill_ids_test: list[str],
    feature_names: list[str],
    *,
    batch_size: int = 20,
) -> GradientBoostingClassifier:
    """Interactive loop: review predictions, correct labels, retrain.

    Strategy: present the bills where the model is LEAST confident first
    (closest to 0.5 probability). These are the highest-information cases
    for active learning.
    """
    df_bills = pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet")
    bill_lookup = {b["bill_id"]: b for b in df_bills.to_dicts()}

    gold_labels = load_gold_labels()
    corrections_this_session = 0
    total_reviewed = 0

    # Also load train data for retraining
    (X_train_full, _, y_train_full, _, ids_train_full, _, _, _) = build_feature_matrix()

    while True:
        # Get predictions and uncertainty
        y_proba = model.predict_proba(X_test)[:, 1]
        uncertainty = np.abs(y_proba - 0.5)

        # Sort by uncertainty (most uncertain first = active learning)
        order = np.argsort(uncertainty)

        # Find bills we haven't reviewed yet
        reviewed_ids = set(gold_labels.keys())
        unreviewed = [i for i in order if bill_ids_test[i] not in reviewed_ids]

        if not unreviewed:
            console.print("[bold green]All test bills reviewed![/]")
            break

        console.print(
            f"\n[bold]{len(unreviewed)} bills remaining[/] "
            f"({corrections_this_session} corrections this session)\n"
        )

        batch = unreviewed[:batch_size]
        for idx in batch:
            bill_id = bill_ids_test[idx]
            bill = bill_lookup.get(bill_id, {})
            prob = y_proba[idx]
            pred_label = "ADVANCE" if prob >= 0.5 else "STUCK"
            actual_label = "ADVANCE" if y_test[idx] == 1 else "STUCK"

            # Display bill info
            bill_num = bill.get("bill_number_raw", "???")
            desc = bill.get("description", "")
            synopsis = bill.get("synopsis_text", "")[:200]
            sponsor = bill.get("primary_sponsor", "Unknown")
            chamber = bill.get("chamber_origin", "")

            pred_color = "green" if pred_label == "ADVANCE" else "red"
            conf_pct = max(prob, 1 - prob) * 100

            console.print(
                Panel(
                    f"[bold]{bill_num}[/] {desc}\n"
                    f"[dim]{synopsis}"
                    f"{'...' if len(bill.get('synopsis_text', '')) > 200 else ''}[/]\n\n"
                    f"Sponsor: {sponsor} ({chamber})\n"
                    f"Model predicts: [{pred_color}]{pred_label}[/{pred_color}] "
                    f"({conf_pct:.0f}% confidence)\n"
                    f"Current label: {actual_label}",
                    title=f"[{total_reviewed + 1}] Bill Review",
                    border_style="cyan",
                )
            )

            console.print(
                "  [y] Prediction correct  "
                "[n] Prediction wrong  "
                "[a] Mark as ADVANCE  "
                "[s] Mark as STUCK  "
                "[skip] Skip  "
                "[q] Quit & retrain"
            )

            while True:
                choice = console.input("\n  Your choice: ").strip().lower()

                if choice == "q":
                    break
                elif choice == "skip":
                    total_reviewed += 1
                    break
                elif choice == "y":
                    # Prediction was correct -- confirm label
                    gold_labels[bill_id] = int(y_test[idx])
                    total_reviewed += 1
                    break
                elif choice == "n":
                    # Prediction was wrong -- flip label
                    corrected = 1 - int(y_test[idx])
                    gold_labels[bill_id] = corrected
                    corrections_this_session += 1
                    total_reviewed += 1
                    console.print(
                        f"  [yellow]Corrected: {bill_num} -> "
                        f"{'ADVANCE' if corrected == 1 else 'STUCK'}[/]"
                    )
                    break
                elif choice == "a":
                    gold_labels[bill_id] = 1
                    if y_test[idx] != 1:
                        corrections_this_session += 1
                    total_reviewed += 1
                    break
                elif choice == "s":
                    gold_labels[bill_id] = 0
                    if y_test[idx] != 0:
                        corrections_this_session += 1
                    total_reviewed += 1
                    break
                else:
                    console.print("  [red]Invalid choice.[/]")

            if choice == "q":
                break

            console.print()

        # Save gold labels after each batch
        save_gold_labels(gold_labels)

        if choice == "q" or not unreviewed:
            break

        # Retrain if we have corrections
        if corrections_this_session > 0:
            console.print(
                f"\n[bold yellow]Retraining with {corrections_this_session} corrections...[/]"
            )

            # Apply gold labels to training data
            y_train_corrected = y_train_full.copy()
            for i, bid in enumerate(ids_train_full):
                if bid in gold_labels:
                    y_train_corrected[i] = gold_labels[bid]

            # Also apply gold labels to test data
            y_test_corrected = y_test.copy()
            for i, bid in enumerate(bill_ids_test):
                if bid in gold_labels:
                    y_test_corrected[i] = gold_labels[bid]

            model = train_model(X_train_full, y_train_corrected)
            metrics = evaluate_model(model, X_test, y_test_corrected)
            display_metrics(metrics)

            # Update test labels for next round
            y_test = y_test_corrected

        cont = console.input("\nContinue reviewing? [Y/n]: ").strip().lower()
        if cont == "n":
            break

    # Final save
    save_gold_labels(gold_labels)
    save_model(model)

    console.print(
        f"\n[bold green]Session complete.[/] "
        f"Reviewed: {total_reviewed}, "
        f"Corrections: {corrections_this_session}"
    )

    return model


# ── Full pipeline ────────────────────────────────────────────────────────────


def run_train_and_evaluate() -> tuple[GradientBoostingClassifier, dict]:
    """Train model and evaluate. Returns (model, metrics)."""
    console.print("[bold]Building features...[/]")
    (X_train, X_test, y_train, y_test, bill_ids_train, bill_ids_test, vectorizer, feature_names) = (
        build_feature_matrix()
    )

    # Apply any existing gold labels
    gold_labels = load_gold_labels()
    if gold_labels:
        console.print(f"[dim]Applying {len(gold_labels)} gold labels...[/]")
        for i, bid in enumerate(bill_ids_train):
            if bid in gold_labels:
                y_train[i] = gold_labels[bid]
        for i, bid in enumerate(bill_ids_test):
            if bid in gold_labels:
                y_test[i] = gold_labels[bid]

    console.print("[bold]Training model...[/]")
    model = train_model(X_train, y_train)

    console.print("[bold]Evaluating...[/]")
    metrics = evaluate_model(model, X_test, y_test)
    display_metrics(metrics)
    display_top_features(model, feature_names)

    save_model(model)

    return model, metrics


def run_interactive() -> None:
    """Full pipeline: build features, train, then interactive review."""
    console.print("[bold]Building features...[/]")
    (X_train, X_test, y_train, y_test, bill_ids_train, bill_ids_test, vectorizer, feature_names) = (
        build_feature_matrix()
    )

    # Apply existing gold labels
    gold_labels = load_gold_labels()
    if gold_labels:
        console.print(f"[dim]Applying {len(gold_labels)} gold labels...[/]")
        for i, bid in enumerate(bill_ids_train):
            if bid in gold_labels:
                y_train[i] = gold_labels[bid]
        for i, bid in enumerate(bill_ids_test):
            if bid in gold_labels:
                y_test[i] = gold_labels[bid]

    console.print("[bold]Training initial model...[/]")
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    display_metrics(metrics)
    display_top_features(model, feature_names)

    console.print("[bold]Starting interactive review...[/]")
    model = interactive_review(model, X_test, y_test, bill_ids_test, feature_names)
