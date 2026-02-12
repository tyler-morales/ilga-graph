"""Robust bill outcome prediction with proper model selection.

Fixes over v1:
- Only evaluates on "mature" bills (120+ days old) so labels are reliable
- Cross-validates with StratifiedKFold (not just one split)
- Compares 4 algorithms, picks the best automatically
- Hyperparameter search via RandomizedSearchCV
- Reports confidence intervals, not just point estimates
- Scores immature/new bills separately as "predictions"

Outputs:
    processed/bill_scores.parquet     -- Every bill with predicted probability
    processed/bill_predictor.pkl      -- Best trained model
    processed/model_quality.json      -- Quality report for trust assessment
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from scipy.sparse import vstack as sparse_vstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from .features import build_feature_matrix

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")
MODEL_PATH = PROCESSED_DIR / "bill_predictor.pkl"
QUALITY_PATH = PROCESSED_DIR / "model_quality.json"

console = Console()


# ── Model comparison ─────────────────────────────────────────────────────────


def _compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_folds: int = 5,
) -> list[dict]:
    """Compare multiple algorithms using stratified k-fold cross-validation.

    Returns a list of {name, model, mean_auc, std_auc, scores} dicts,
    sorted by mean_auc descending.
    """
    console.print(f"\n[bold]Model comparison ({n_folds}-fold stratified cross-validation)...[/]")

    # Compute sample weights for class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    if pos_count > 0 and neg_count > 0:
        w_pos = len(y_train) / (2 * pos_count)
        w_neg = len(y_train) / (2 * neg_count)
    else:
        w_pos = w_neg = 1.0

    candidates = [
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "LogisticRegression",
            LogisticRegression(
                C=1.0,
                max_iter=5000,
                solver="saga",
                class_weight="balanced",
                random_state=42,
            ),
        ),
        (
            "AdaBoost",
            AdaBoostClassifier(
                n_estimators=150,
                learning_rate=0.1,
                random_state=42,
            ),
        ),
    ]

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    for name, model in candidates:
        console.print(f"  Testing {name}...", end=" ")

        # Use sample_weight for GBC/AdaBoost via fit_params
        sw = np.where(y_train == 1, w_pos, w_neg)

        try:
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                fit_params={"sample_weight": sw},
                n_jobs=1,  # Some models don't support parallel CV
            )
        except TypeError:
            # LogisticRegression doesn't use sample_weight in fit_params
            # It uses class_weight="balanced" instead
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )

        mean_auc = scores.mean()
        std_auc = scores.std()
        console.print(
            f"ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f} "
            f"(folds: {', '.join(f'{s:.3f}' for s in scores)})"
        )

        results.append(
            {
                "name": name,
                "model": model,
                "mean_auc": float(mean_auc),
                "std_auc": float(std_auc),
                "fold_scores": [float(s) for s in scores],
            }
        )

    results.sort(key=lambda r: -r["mean_auc"])
    return results


# ── Hyperparameter tuning ────────────────────────────────────────────────────


def _tune_best_model(
    best_result: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> object:
    """Fine-tune the best model with RandomizedSearchCV."""
    name = best_result["name"]
    console.print(f"\n[bold]Tuning {name} hyperparameters...[/]")

    # Define parameter spaces per model type
    param_spaces = {
        "GradientBoosting": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "min_samples_leaf": [5, 10, 20],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", "log2", 0.3],
        },
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
        },
        "AdaBoost": {
            "n_estimators": [50, 100, 150, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
    }

    param_space = param_spaces.get(name)
    if param_space is None:
        # No tuning defined; just refit on full training data
        best_result["model"].fit(X_train, y_train)
        return best_result["model"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        best_result["model"],
        param_space,
        n_iter=40,
        cv=cv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    # Compute sample weights
    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos > 0 and neg > 0:
        sw = np.where(
            y_train == 1,
            len(y_train) / (2 * pos),
            len(y_train) / (2 * neg),
        )
    else:
        sw = None

    try:
        search.fit(X_train, y_train, sample_weight=sw)
    except TypeError:
        search.fit(X_train, y_train)

    console.print(f"  Best params: {search.best_params_}")
    console.print(f"  Best CV AUC: {search.best_score_:.4f}")

    return search.best_estimator_


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate model on held-out test set."""
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


def display_metrics(metrics: dict, *, title: str = "") -> None:
    """Show evaluation metrics table."""
    table = Table(
        title=title or "Test Set Evaluation",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    auc = metrics.get("roc_auc")
    table.add_row(
        "ROC-AUC",
        f"{auc:.4f}" if auc is not None else "N/A",
    )
    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("", "")
    table.add_row("[green]Advanced (positive)[/]", "")
    table.add_row("  Precision", f"{metrics['precision_pos']:.4f}")
    table.add_row("  Recall", f"{metrics['recall_pos']:.4f}")
    table.add_row("  F1", f"{metrics['f1_pos']:.4f}")
    table.add_row(
        "  Support",
        str(metrics["support_pos"]),
    )
    table.add_row("", "")
    table.add_row("[red]Stuck/Dead (negative)[/]", "")
    table.add_row("  Precision", f"{metrics['precision_neg']:.4f}")
    table.add_row("  Recall", f"{metrics['recall_neg']:.4f}")
    table.add_row("  F1", f"{metrics['f1_neg']:.4f}")
    table.add_row("  Support", str(metrics["support_neg"]))

    console.print()
    console.print(table)


def display_top_features(
    model,
    feature_names: list[str],
    top_n: int = 15,
) -> None:
    """Show top feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        return

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


# ── Scoring ──────────────────────────────────────────────────────────────────


def score_all_bills(
    model,
    X_all: np.ndarray,
    bill_ids_all: list[str],
    y_all: np.ndarray,
    *,
    immature_ids: set[str] | None = None,
) -> pl.DataFrame:
    """Score every bill with probability of advancement."""
    y_proba = model.predict_proba(X_all)[:, 1]
    y_pred = model.predict(X_all)
    immature_ids = immature_ids or set()

    df_scores = pl.DataFrame(
        {
            "bill_id": bill_ids_all,
            "prob_advance": [round(float(p), 4) for p in y_proba],
            "predicted_outcome": ["ADVANCE" if p == 1 else "STUCK" for p in y_pred],
            "actual_outcome": ["ADVANCE" if y == 1 else "STUCK" for y in y_all],
            "confidence": [round(float(max(p, 1 - p)), 4) for p in y_proba],
            "label_reliable": [bid not in immature_ids for bid in bill_ids_all],
        }
    )

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

    df_scores = df_scores.sort("prob_advance", descending=True)

    out_path = PROCESSED_DIR / "bill_scores.parquet"
    df_scores.write_parquet(out_path)
    LOGGER.info("Saved %d bill scores to %s", len(df_scores), out_path)

    return df_scores


def display_score_summary(df_scores: pl.DataFrame) -> None:
    """Show summary of bill scores."""
    total = len(df_scores)
    reliable = df_scores.filter(pl.col("label_reliable"))
    unreliable = df_scores.filter(~pl.col("label_reliable"))
    predicted_advance = df_scores.filter(pl.col("predicted_outcome") == "ADVANCE")

    console.print()
    table = Table(title="Bill Score Summary", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total bills scored", f"{total:,}")
    table.add_row(
        "  Mature (reliable labels)",
        f"{len(reliable):,}",
    )
    table.add_row(
        "  Immature (predictions only)",
        f"{len(unreliable):,}",
    )
    table.add_row(
        "Predicted to ADVANCE",
        f"{len(predicted_advance):,} ({100 * len(predicted_advance) / total:.1f}%)",
    )

    # Accuracy on mature bills only
    mature_correct = reliable.filter(pl.col("predicted_outcome") == pl.col("actual_outcome"))
    if len(reliable) > 0:
        table.add_row(
            "Accuracy on mature bills",
            f"{100 * len(mature_correct) / len(reliable):.1f}%",
        )

    console.print(table)

    # Top predictions on IMMATURE bills (the actual forecasts)
    if len(unreliable) > 0:
        console.print()
        forecast = Table(
            title="Top 15 Forecasts (new bills, outcome unknown)",
            show_lines=True,
        )
        forecast.add_column("Bill", style="bold")
        forecast.add_column("Description")
        forecast.add_column("Sponsor")
        forecast.add_column("P(advance)", justify="right")

        for row in unreliable.sort("prob_advance", descending=True).head(15).to_dicts():
            forecast.add_row(
                row.get("bill_number_raw", ""),
                (row.get("description", "") or "")[:40],
                (row.get("primary_sponsor", "") or "")[:20],
                f"{row['prob_advance']:.1%}",
            )
        console.print(forecast)


# ── Quality report ───────────────────────────────────────────────────────────


def save_quality_report(
    comparison: list[dict],
    best_name: str,
    test_metrics: dict,
    feature_names: list[str],
    model,
    n_mature: int,
    n_immature: int,
) -> dict:
    """Save a human-readable quality report."""
    report = {
        "model_selected": best_name,
        "why": (f"Best cross-validated ROC-AUC across 5 folds out of {len(comparison)} candidates"),
        "model_comparison": [
            {
                "name": r["name"],
                "cv_auc_mean": round(r["mean_auc"], 4),
                "cv_auc_std": round(r["std_auc"], 4),
                "fold_scores": [round(s, 4) for s in r["fold_scores"]],
            }
            for r in comparison
        ],
        "test_set_metrics": {
            k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items()
        },
        "data_split": {
            "mature_bills": n_mature,
            "immature_bills": n_immature,
            "maturity_threshold_days": 120,
            "train_test_split": "70/30 time-based on mature bills",
        },
        "trust_assessment": _trust_assessment(test_metrics),
    }

    # Feature importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]
        report["top_features"] = [
            {
                "name": (feature_names[i] if i < len(feature_names) else f"feature_{i}"),
                "importance": round(float(importances[i]), 4),
            }
            for i in top_idx
        ]

    with open(QUALITY_PATH, "w") as f:
        json.dump(report, f, indent=2)

    LOGGER.info("Quality report saved to %s", QUALITY_PATH)
    return report


def _trust_assessment(metrics: dict) -> dict:
    """Generate plain-language trust assessment."""
    auc = metrics.get("roc_auc")
    f1_pos = metrics.get("f1_pos", 0)
    f1_neg = metrics.get("f1_neg", 0)
    support_pos = metrics.get("support_pos", 0)

    issues = []
    strengths = []

    if auc is None:
        issues.append("Could not compute ROC-AUC (single class in test set)")
    elif auc >= 0.9:
        strengths.append(f"Excellent discrimination (AUC={auc:.3f})")
    elif auc >= 0.8:
        strengths.append(f"Good discrimination (AUC={auc:.3f})")
    elif auc >= 0.7:
        issues.append(
            f"Moderate discrimination (AUC={auc:.3f}) -- "
            "predictions are directionally useful but not precise"
        )
    else:
        issues.append(f"Weak discrimination (AUC={auc:.3f}) -- predictions may not be reliable")

    if support_pos < 50:
        issues.append(
            f"Low positive support in test ({support_pos}) -- "
            "positive class metrics may be unstable"
        )
    if f1_pos > 0 and f1_pos < 0.5:
        issues.append(
            f"Low F1 on advanced bills ({f1_pos:.3f}) -- "
            "model misses many bills that actually advance"
        )
    if f1_pos >= 0.7:
        strengths.append(f"Good at identifying advancing bills (F1={f1_pos:.3f})")
    if f1_neg >= 0.8:
        strengths.append(f"Good at identifying stuck bills (F1={f1_neg:.3f})")

    overall = (
        "RELIABLE"
        if len(issues) <= 1 and auc and auc >= 0.8
        else ("MODERATE" if auc and auc >= 0.7 else "USE WITH CAUTION")
    )

    return {
        "overall": overall,
        "strengths": strengths,
        "issues": issues,
    }


def display_quality_summary(report: dict) -> None:
    """Show human-readable quality assessment."""
    trust = report.get("trust_assessment", {})
    overall = trust.get("overall", "UNKNOWN")
    color = "green" if overall == "RELIABLE" else "yellow" if overall == "MODERATE" else "red"

    console.print()
    table = Table(
        title=f"Model Quality: [{color}]{overall}[/{color}]",
        show_lines=True,
    )
    table.add_column("", style="bold")
    table.add_column("Detail")

    table.add_row("Model", report.get("model_selected", ""))

    for s in trust.get("strengths", []):
        table.add_row("[green]Strength[/]", s)
    for i in trust.get("issues", []):
        table.add_row("[red]Issue[/]", i)

    ds = report.get("data_split", {})
    table.add_row(
        "Data",
        f"{ds.get('mature_bills', 0):,} mature + {ds.get('immature_bills', 0):,} forecast-only",
    )

    console.print(table)


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_auto() -> pl.DataFrame:
    """Fully automated: compare models, tune, evaluate, score all bills."""
    console.print("\n[bold]Step 1: Building features...[/]")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        ids_train,
        ids_test,
        X_immature,
        y_immature,
        ids_immature,
        vectorizer,
        feature_names,
    ) = build_feature_matrix()

    # Step 2: Compare models with cross-validation
    console.print("[bold]Step 2: Comparing models...[/]")
    comparison = _compare_models(X_train, y_train, n_folds=5)

    best = comparison[0]
    console.print(
        f"\n  [bold green]Winner: {best['name']}[/] "
        f"(CV AUC: {best['mean_auc']:.4f} +/- {best['std_auc']:.4f})"
    )

    # Step 3: Tune hyperparameters of best model
    console.print("[bold]Step 3: Tuning hyperparameters...[/]")
    tuned_model = _tune_best_model(best, X_train, y_train)

    # Step 4: Calibrate probabilities
    console.print("[bold]Step 4: Calibrating probabilities...[/]")
    calibrated = CalibratedClassifierCV(tuned_model, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)

    # Step 5: Evaluate on held-out test set
    console.print("[bold]Step 5: Evaluating on held-out test...[/]")
    test_metrics = evaluate_model(calibrated, X_test, y_test)
    display_metrics(
        test_metrics,
        title=f"Test Evaluation ({best['name']}, calibrated)",
    )

    # Show feature importances from the uncalibrated model
    display_top_features(tuned_model, feature_names)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(calibrated, f)
    LOGGER.info("Model saved to %s", MODEL_PATH)

    # Step 6: Score ALL bills
    console.print("\n[bold]Step 6: Scoring all bills...[/]")
    parts = [X_train, X_test]
    ids_parts = ids_train + ids_test
    y_parts = [y_train, y_test]
    if X_immature is not None and X_immature.shape[0] > 0:
        parts.append(X_immature)
        ids_parts += ids_immature
        y_parts.append(y_immature)

    X_all = sparse_vstack(parts)
    y_all = np.concatenate(y_parts)

    df_scores = score_all_bills(
        calibrated,
        X_all,
        ids_parts,
        y_all,
        immature_ids=set(ids_immature),
    )
    display_score_summary(df_scores)

    # Step 7: Quality report
    n_mature = len(ids_train) + len(ids_test)
    n_immature = len(ids_immature)
    report = save_quality_report(
        comparison,
        best["name"],
        test_metrics,
        feature_names,
        tuned_model,
        n_mature,
        n_immature,
    )
    display_quality_summary(report)

    return df_scores
