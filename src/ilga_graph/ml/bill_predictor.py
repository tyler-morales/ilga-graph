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
import os
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

from .features import (
    FORECAST_DROP_COLUMNS,
    build_feature_matrix,
    build_panel_feature_matrix,
)

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")
MODEL_PATH = PROCESSED_DIR / "bill_predictor.pkl"
RAW_MODEL_PATH = PROCESSED_DIR / "bill_predictor_raw.pkl"
SHAP_MATRIX_PATH = PROCESSED_DIR / "shap_feature_matrix.npz"
SHAP_META_PATH = PROCESSED_DIR / "shap_feature_meta.json"
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

    # Define parameter spaces per model type (kept small so tuning finishes in ~5–10 min)
    param_spaces = {
        "GradientBoosting": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "min_samples_leaf": [5, 10],
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
        n_iter=20,
        cv=cv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=2,
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


def _compute_predicted_destination(
    prob_advance: float,
    prob_law: float,
    lifecycle: str,
    current_stage: str,
) -> str:
    """Derive a human-readable predicted destination for a bill.

    For terminal bills, returns the actual outcome.
    For open bills, combines P(advance) and P(law) into a destination:
        → Law        P(law) >= 0.5 OR already at PASSED_BOTH/GOVERNOR with P(law) >= 0.3
        → Governor   Already past both chambers, likely to reach governor
        → Floor      P(advance) >= 0.5 but P(law) < 0.5
        Stuck        P(advance) < 0.5
    """
    if lifecycle == "PASSED":
        return "Became Law"
    if lifecycle == "VETOED":
        return "Vetoed"

    # Open bills — use model probabilities + current stage
    high_stages = ("PASSED_BOTH", "GOVERNOR", "SIGNED")
    floor_stages = ("FLOOR_VOTE", "CROSSED_CHAMBERS")

    # Already very far along — lower threshold for "→ Law"
    if current_stage in high_stages and prob_law >= 0.3:
        return "→ Law"

    if prob_law >= 0.5:
        return "→ Law"

    # Already past committee or on the floor
    if current_stage in floor_stages and prob_advance >= 0.5:
        return "→ Passed"

    if prob_advance >= 0.5:
        return "→ Floor"

    return "Stuck"


def score_all_bills(
    model,
    X_all: np.ndarray,
    bill_ids_all: list[str],
    y_all: np.ndarray,
    *,
    immature_ids: set[str] | None = None,
    law_model=None,
    forecast_scores: dict[str, float] | None = None,
    forecast_confidences: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Score every bill with probability of advancement and law.

    Also computes pipeline stage, staleness, stuck sub-status,
    and predicted destination (→ Law, → Floor, Stuck, etc.).

    Parameters
    ----------
    forecast_scores:
        Optional ``{bill_id: P(law)}`` from the Forecast model.
        Merged into the output as ``forecast_score`` column.
    forecast_confidences:
        Optional ``{bill_id: "Low"|"Medium"|"High"}`` confidence labels.
    """
    from datetime import datetime as _dt

    from .features import (
        _bill_lifecycle_status,
        _stage_label,
        classify_stuck_status,
        compute_bill_stage,
    )

    y_proba = model.predict_proba(X_all)[:, 1]
    y_pred = model.predict(X_all)
    immature_ids = immature_ids or set()

    # Law model probabilities (P(becomes law))
    if law_model is not None and hasattr(law_model, "predict_proba"):
        y_proba_law = law_model.predict_proba(X_all)[:, 1]
    else:
        # Fallback: estimate from advance probability (less accurate)
        y_proba_law = y_proba * 0.3  # rough heuristic
        LOGGER.warning("No law model available — using heuristic P(law) = P(advance) * 0.3")

    # Cap confidence at 99% for robustness (avoid false certainty)
    CONFIDENCE_CAP = 0.99

    df_scores = pl.DataFrame(
        {
            "bill_id": bill_ids_all,
            "prob_advance": [round(float(p), 4) for p in y_proba],
            "prob_law": [round(float(p), 4) for p in y_proba_law],
            "predicted_outcome": ["ADVANCE" if p == 1 else "STUCK" for p in y_pred],
            "confidence": [round(float(min(max(p, 1 - p), CONFIDENCE_CAP)), 4) for p in y_proba],
            "label_reliable": [bid not in immature_ids for bid in bill_ids_all],
        }
    )

    df_bills = pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet")
    df_actions = pl.read_parquet(PROCESSED_DIR / "fact_bill_actions.parquet")
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
                "last_action",
                "last_action_date",
            ]
        ),
        on="bill_id",
        how="left",
    )

    # Group actions per bill for stage computation (chronological order so
    # rollbacks like Rule 19(b) after PASSED_BOTH are applied correctly).
    bill_action_texts = (
        df_actions.sort("date")
        .group_by("bill_id")
        .agg(pl.col("action_text").alias("actions"))
        .to_dicts()
    )
    action_map = {row["bill_id"]: row["actions"] for row in bill_action_texts}

    now = _dt.now()

    # Compute stage, stuck status, lifecycle for each bill
    stages = []
    stage_progresses = []
    days_since_actions = []
    stuck_statuses = []
    stuck_reasons = []
    stage_labels = []
    lifecycle_statuses = []
    # Override lists — we'll replace predictions for terminal bills
    override_outcomes = []
    override_proba = []
    override_proba_law = []
    override_confidence = []
    predicted_destinations = []
    rule_contexts = []  # Rule citation for current stage / next step

    # Try to load rule engine for stage context
    try:
        from ilga_graph.ml.rule_engine import (
            get_rule_tooltip,
            votes_required_for_override,
            votes_required_for_passage,
        )

        _has_rule_engine = True
    except Exception:
        _has_rule_engine = False

    for row in df_scores.to_dicts():
        bid = row["bill_id"]
        actions = action_map.get(bid, [])

        # Pipeline stage
        stage, progress = compute_bill_stage(actions)
        stages.append(stage)
        stage_progresses.append(round(progress, 2))
        stage_labels.append(_stage_label(stage))

        # Lifecycle status (OPEN / PASSED / VETOED); VETOED = governor veto only
        lifecycle = _bill_lifecycle_status(actions)
        lifecycle_statuses.append(lifecycle)

        # Days since last action
        last_date_str = row.get("last_action_date")
        days_inactive = 0
        if last_date_str:
            try:
                last_dt = _dt.strptime(last_date_str, "%Y-%m-%d")
                days_inactive = (now - last_dt).days
            except ValueError:
                try:
                    last_dt = _dt.strptime(last_date_str, "%m/%d/%Y")
                    days_inactive = (now - last_dt).days
                except ValueError:
                    LOGGER.debug("Unparseable last_action_date: %r for %s", last_date_str, bid)
        days_since_actions.append(days_inactive)

        # Days since introduction
        intro_str = row.get("introduction_date")
        days_since_intro = 0
        if intro_str:
            try:
                intro_dt = _dt.strptime(intro_str, "%Y-%m-%d")
                days_since_intro = (now - intro_dt).days
            except ValueError:
                LOGGER.debug("Unparseable introduction_date: %r for %s", intro_str, bid)

        # ── Override predictions for terminal bills ──
        # Vetoed/tabled/dead bills should show STUCK, not ADVANCE.
        # Passed/signed bills should show ADVANCE with 100% confidence.
        if lifecycle == "PASSED":
            override_outcomes.append("ADVANCE")
            override_proba.append(1.0)
            override_proba_law.append(1.0)
            override_confidence.append(1.0)
        elif lifecycle in ("VETOED", "DEAD"):
            override_outcomes.append("STUCK")
            override_proba.append(0.0)
            override_proba_law.append(0.0)
            override_confidence.append(1.0)
        else:
            # OPEN — keep the model's prediction
            override_outcomes.append(row["predicted_outcome"])
            override_proba.append(row["prob_advance"])
            override_proba_law.append(row["prob_law"])
            override_confidence.append(row["confidence"])

        # Compute predicted destination
        dest = _compute_predicted_destination(
            override_proba[-1],
            override_proba_law[-1],
            lifecycle,
            stage,
        )
        predicted_destinations.append(dest)

        # Rule context: cite the rule governing the current stage and next step
        if _has_rule_engine:
            rule_tip = get_rule_tooltip(stage) or ""
            # Detect origin chamber from bill_id prefix
            _chamber = "house" if str(bid).upper().startswith("HB") else "senate"
            if stage == "FLOOR_VOTE":
                rule_tip += f" Needs {votes_required_for_passage(_chamber)} votes for passage."
            elif stage == "VETOED":
                rule_tip += f" Override requires {votes_required_for_override(_chamber)} votes."
            rule_contexts.append(rule_tip)
        else:
            rule_contexts.append("")

        # Stuck sub-status (only for non-advanced bills)
        predicted = override_outcomes[-1]
        if lifecycle == "PASSED":
            stuck_statuses.append("")
            stuck_reasons.append("")
        elif stage in ("SIGNED", "PASSED_BOTH", "CROSSED_CHAMBERS"):
            stuck_statuses.append("")
            stuck_reasons.append("")
        elif predicted == "STUCK" or progress <= 0.40:
            ss, sr = classify_stuck_status(stage, days_inactive, days_since_intro, actions)
            stuck_statuses.append(ss)
            stuck_reasons.append(sr)
        else:
            stuck_statuses.append("")
            stuck_reasons.append("")

    df_scores = df_scores.with_columns(
        pl.Series("current_stage", stages, dtype=pl.Utf8),
        pl.Series("stage_progress", stage_progresses, dtype=pl.Float64),
        pl.Series("stage_label", stage_labels, dtype=pl.Utf8),
        pl.Series(
            "days_since_action",
            days_since_actions,
            dtype=pl.Int32,
        ),
        pl.Series("stuck_status", stuck_statuses, dtype=pl.Utf8),
        pl.Series("stuck_reason", stuck_reasons, dtype=pl.Utf8),
        pl.Series("lifecycle_status", lifecycle_statuses, dtype=pl.Utf8),
        # Apply overrides for terminal bills
        pl.Series("predicted_outcome", override_outcomes, dtype=pl.Utf8),
        pl.Series("prob_advance", override_proba, dtype=pl.Float64),
        pl.Series("prob_law", override_proba_law, dtype=pl.Float64),
        pl.Series("confidence", override_confidence, dtype=pl.Float64),
        pl.Series("predicted_destination", predicted_destinations, dtype=pl.Utf8),
        pl.Series("rule_context", rule_contexts, dtype=pl.Utf8),
    )

    # ── Forecast model columns ──────────────────────────────────────────
    forecast_scores = forecast_scores or {}
    forecast_confidences = forecast_confidences or {}
    df_scores = df_scores.with_columns(
        pl.Series(
            "forecast_score",
            [forecast_scores.get(bid, 0.0) for bid in df_scores["bill_id"].to_list()],
            dtype=pl.Float64,
        ),
        pl.Series(
            "forecast_confidence",
            [forecast_confidences.get(bid, "") for bid in df_scores["bill_id"].to_list()],
            dtype=pl.Utf8,
        ),
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

    # Stage distribution on scored bills
    if "current_stage" in df_scores.columns:
        signed = len(df_scores.filter(pl.col("current_stage") == "SIGNED"))
        in_committee = len(df_scores.filter(pl.col("current_stage") == "IN_COMMITTEE"))
        stagnant = len(df_scores.filter(pl.col("stuck_status") == "STAGNANT"))
        dead = len(df_scores.filter(pl.col("stuck_status") == "DEAD"))
        table.add_row("Signed into law", f"{signed:,}")
        table.add_row("In committee", f"{in_committee:,}")
        table.add_row("Stagnant (180+ days)", f"{stagnant:,}")
        table.add_row("Dead (vetoed/tabled)", f"{dead:,}")

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
    *,
    panel_mode: bool = False,
) -> dict:
    """Save a human-readable quality report."""
    split_desc = (
        "70/30 time-based on panel rows (snapshot_date order)"
        if panel_mode
        else "70/30 time-based on mature bills"
    )
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
            "training_rows": n_mature,
            "immature_bills": n_immature,
            "maturity_threshold_days": 120,
            "train_test_split": split_desc,
            "panel_mode": panel_mode,
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


# ── Forecast model ("Truth" model — intrinsic features only) ────────────────


LAW_MODEL_PATH = PROCESSED_DIR / "bill_law_predictor.pkl"
FORECAST_MODEL_PATH = PROCESSED_DIR / "forecast_predictor.pkl"
FORECAST_QUALITY_PATH = PROCESSED_DIR / "forecast_quality.json"


def _forecast_confidence_label(prob: float) -> str:
    """Convert a raw probability into a human-readable confidence label."""
    if prob >= 0.6:
        return "High"
    elif prob >= 0.35:
        return "Medium"
    return "Low"


def run_forecast_model() -> tuple[dict[str, float], dict[str, str]]:
    """Train and score the Forecast ("Truth") model.

    Uses only intrinsic/Day-0 features (no staleness, slips, action
    counts, or rule-derived features) to predict P(becomes law).

    Returns:
        (forecast_scores, forecast_confidences)
        Both are ``{bill_id: value}`` dicts that get merged into
        ``bill_scores.parquet`` by the caller.
    """
    console.print("\n[bold cyan]── Forecast Model (intrinsic features only) ──[/]")

    # ── Step F1: Build panel feature matrix in forecast mode ──────────
    console.print("[bold]F1: Building forecast features (panel, Day-0 + Day-30)...[/]")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        ids_train,
        ids_test,
        _X_imm,
        _y_imm,
        _ids_imm,
        _vectorizer,
        feature_names,
        y_train_law,
        y_test_law,
        _y_imm_law,
    ) = build_panel_feature_matrix(mode="forecast")

    # Use target_law as the label (not target_advanced) — the Forecast
    # model predicts "will this bill become law?"
    y_train_target = y_train_law
    y_test_target = y_test_law

    pos_train = int(y_train_target.sum())
    neg_train = len(y_train_target) - pos_train
    console.print(
        f"  Train: {len(y_train_target):,} rows "
        f"({pos_train} law / {neg_train} not, {100 * pos_train / max(len(y_train_target), 1):.1f}%)"
    )
    console.print(f"  Test:  {len(y_test_target):,} rows")
    console.print(f"  Features: {len(feature_names)} (no staleness/slips/actions)")

    if pos_train < 10:
        LOGGER.warning(
            "Very few positive law examples (%d) for Forecast model — results may be unreliable.",
            pos_train,
        )

    # ── Step F2: Train GradientBoosting with richer hyperparams ──────
    console.print("[bold]F2: Training Forecast GradientBoosting...[/]")
    base_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        min_samples_leaf=8,
        random_state=42,
    )

    # Sample weights for class imbalance
    if pos_train > 0 and neg_train > 0:
        sw = np.where(
            y_train_target == 1,
            len(y_train_target) / (2 * pos_train),
            len(y_train_target) / (2 * neg_train),
        )
    else:
        sw = None

    try:
        base_model.fit(X_train, y_train_target, sample_weight=sw)
    except TypeError:
        base_model.fit(X_train, y_train_target)

    # ── Step F3: Calibrate probabilities ─────────────────────────────
    console.print("[bold]F3: Calibrating forecast probabilities...[/]")
    n_splits = min(5, pos_train) if pos_train >= 2 else 2
    if n_splits >= 2:
        calibrated = CalibratedClassifierCV(base_model, cv=n_splits, method="isotonic")
        calibrated.fit(X_train, y_train_target)
    else:
        calibrated = base_model

    # ── Step F4: Evaluate on test set ────────────────────────────────
    console.print("[bold]F4: Evaluating forecast model...[/]")
    test_metrics = evaluate_model(calibrated, X_test, y_test_target)
    display_metrics(
        test_metrics,
        title="Forecast Model Test Evaluation (P(law), intrinsic features)",
    )
    display_top_features(base_model, feature_names)

    # ── Step F5: Save model + quality report ─────────────────────────
    with open(FORECAST_MODEL_PATH, "wb") as f:
        pickle.dump(calibrated, f)
    LOGGER.info("Forecast model saved to %s", FORECAST_MODEL_PATH)

    # Quality report
    forecast_report = {
        "model": "GradientBoosting (Forecast)",
        "target": "P(becomes law) — intrinsic features only",
        "features_excluded": sorted(FORECAST_DROP_COLUMNS),
        "test_set_metrics": {
            k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items()
        },
        "trust_assessment": _trust_assessment(test_metrics),
    }
    if hasattr(base_model, "feature_importances_"):
        importances = base_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]
        forecast_report["top_features"] = [
            {
                "name": (feature_names[i] if i < len(feature_names) else f"feature_{i}"),
                "importance": round(float(importances[i]), 4),
            }
            for i in top_idx
        ]
    with open(FORECAST_QUALITY_PATH, "w") as f:
        json.dump(forecast_report, f, indent=2)
    LOGGER.info("Forecast quality report saved to %s", FORECAST_QUALITY_PATH)

    # ── Step F6: Score ALL bills with forecast model ─────────────────
    console.print("[bold]F6: Scoring all bills with Forecast model...[/]")
    (
        X_score_train,
        X_score_test,
        _yt,
        _ytt,
        ids_score_train,
        ids_score_test,
        X_score_imm,
        _yi,
        ids_score_imm,
        _vec2,
        _fn2,
        _yl1,
        _yl2,
        _yl3,
    ) = build_feature_matrix(mode="forecast")

    # Combine all splits into one scoring set
    parts = [X_score_train, X_score_test]
    ids_parts = ids_score_train + ids_score_test
    if X_score_imm is not None and X_score_imm.shape[0] > 0:
        parts.append(X_score_imm)
        ids_parts += ids_score_imm

    X_all = sparse_vstack(parts)
    forecast_proba = calibrated.predict_proba(X_all)[:, 1]

    forecast_scores: dict[str, float] = {}
    forecast_confidences: dict[str, str] = {}
    for bid, prob in zip(ids_parts, forecast_proba):
        p = round(float(prob), 4)
        forecast_scores[bid] = p
        forecast_confidences[bid] = _forecast_confidence_label(p)

    console.print(
        f"  Scored {len(forecast_scores):,} bills "
        f"(mean forecast P(law)={np.mean(forecast_proba):.3f})"
    )

    return forecast_scores, forecast_confidences


# ── Main pipeline ────────────────────────────────────────────────────────────


def _train_law_model(
    X_train: np.ndarray,
    y_train_law: np.ndarray,
) -> object:
    """Train a calibrated model for P(becomes law).

    Uses GradientBoosting directly (skip full comparison since the advance
    model already validated algorithm choice) with calibration.
    """
    pos = int(y_train_law.sum())
    neg = len(y_train_law) - pos
    LOGGER.info(
        "Law model training set: %d became law / %d did not (%.1f%%)",
        pos,
        neg,
        100 * pos / max(len(y_train_law), 1),
    )

    if pos < 10:
        LOGGER.warning("Very few positive law examples (%d) — law model may be unreliable", pos)

    base = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )

    # Calibrate for reliable probabilities
    n_splits = min(5, pos) if pos >= 2 else 2
    if n_splits < 2:
        # Not enough positives to calibrate — just fit directly
        base.fit(X_train, y_train_law)
        return base

    calibrated = CalibratedClassifierCV(base, cv=n_splits, method="isotonic")
    calibrated.fit(X_train, y_train_law)
    return calibrated


def run_auto(*, use_panel: bool | None = None) -> pl.DataFrame:
    """Fully automated: compare models, tune, evaluate, score all bills.

    Parameters
    ----------
    use_panel:
        If True, use the time-sliced panel dataset for training (multiple
        rows per bill at different snapshot dates → larger training set).
        If None, reads the ``ILGA_ML_PANEL`` environment variable
        (``"1"`` = panel mode, anything else = standard single-row mode).
    """

    if use_panel is None:
        # Panel mode is the default — uses time-sliced snapshots for a
        # richer training set.  Set ILGA_ML_PANEL=0 to revert to the
        # legacy single-row-per-bill training mode.
        use_panel = os.getenv("ILGA_ML_PANEL", "1").strip() != "0"

    if use_panel:
        console.print("\n[bold]Step 1: Building features (PANEL mode — time-sliced dataset)...[/]")
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
            y_train_law,
            y_test_law,
            y_immature_law,
        ) = build_panel_feature_matrix()
    else:
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
            y_train_law,
            y_test_law,
            y_immature_law,
        ) = build_feature_matrix()

    # Step 2: Compare models with cross-validation
    console.print("[bold]Step 2: Comparing models (advance)...[/]")
    comparison = _compare_models(X_train, y_train, n_folds=5)

    best = comparison[0]
    console.print(
        f"\n  [bold green]Winner: {best['name']}[/] "
        f"(CV AUC: {best['mean_auc']:.4f} +/- {best['std_auc']:.4f})"
    )

    # Step 3: Tune hyperparameters of best model (or skip if ILGA_ML_SKIP_TUNE=1)
    console.print("[bold]Step 3: Tuning hyperparameters (advance)...[/]")
    if os.environ.get("ILGA_ML_SKIP_TUNE") == "1":
        console.print("  [dim]Skipping (ILGA_ML_SKIP_TUNE=1); fitting best model on full train.[/]")
        tuned_model = best["model"]
        tuned_model.fit(X_train, y_train)
    else:
        tuned_model = _tune_best_model(best, X_train, y_train)

    # Step 3b: Save raw (uncalibrated) model for SHAP explanations
    with open(RAW_MODEL_PATH, "wb") as f:
        pickle.dump(tuned_model, f)
    LOGGER.info("Raw model saved to %s (for SHAP explainer)", RAW_MODEL_PATH)

    # Step 4: Calibrate probabilities
    console.print("[bold]Step 4: Calibrating probabilities (advance)...[/]")
    calibrated = CalibratedClassifierCV(tuned_model, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)

    # Step 5: Evaluate on held-out test set
    console.print("[bold]Step 5: Evaluating advance model on held-out test...[/]")
    test_metrics = evaluate_model(calibrated, X_test, y_test)
    display_metrics(
        test_metrics,
        title=f"Test Evaluation ({best['name']}, calibrated)",
    )

    # Show feature importances from the uncalibrated model
    display_top_features(tuned_model, feature_names)

    # Save advance model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(calibrated, f)
    LOGGER.info("Advance model saved to %s", MODEL_PATH)

    # Step 5b: Train law prediction model (P(becomes law))
    console.print("\n[bold]Step 5b: Training law prediction model...[/]")
    law_model = _train_law_model(X_train, y_train_law)
    with open(LAW_MODEL_PATH, "wb") as f:
        pickle.dump(law_model, f)
    LOGGER.info("Law model saved to %s", LAW_MODEL_PATH)

    # Evaluate law model on test set
    if hasattr(law_model, "predict_proba"):
        law_test_proba = law_model.predict_proba(X_test)[:, 1]
        try:
            law_auc = roc_auc_score(y_test_law, law_test_proba)
            console.print(f"  Law model test ROC-AUC: [bold]{law_auc:.4f}[/]")
        except ValueError:
            console.print("  Law model test ROC-AUC: N/A (single class in test)")

    # Step 5c: Train Forecast model (intrinsic features only)
    console.print("\n[bold]Step 5c: Training Forecast model (intrinsic features)...[/]")
    forecast_scores_map: dict[str, float] = {}
    forecast_conf_map: dict[str, str] = {}
    try:
        forecast_scores_map, forecast_conf_map = run_forecast_model()
    except Exception as e:
        LOGGER.warning("Forecast model training failed: %s", e)
        console.print(f"  [yellow]Forecast model skipped: {e}[/]")

    # Step 6: Score ALL bills
    # In panel mode, training used time-sliced rows (panel_ids like
    # "155711_d30") but scoring still needs one row per bill with
    # features "as of now".  We always build the single-row feature
    # matrix for scoring so bill_scores.parquet has one row per bill.
    console.print("\n[bold]Step 6: Scoring all bills...[/]")

    if use_panel:
        console.print("  (Panel mode: building single-row features for scoring)")
        (
            X_score_train,
            X_score_test,
            y_score_train,
            y_score_test,
            ids_score_train,
            ids_score_test,
            X_score_imm,
            y_score_imm,
            ids_score_imm,
            _vec,
            _fnames,
            _y_law_train,
            _y_law_test,
            _y_law_imm,
        ) = build_feature_matrix()

        parts = [X_score_train, X_score_test]
        ids_parts = ids_score_train + ids_score_test
        y_parts = [y_score_train, y_score_test]
        if X_score_imm is not None and X_score_imm.shape[0] > 0:
            parts.append(X_score_imm)
            ids_parts += ids_score_imm
            y_parts.append(y_score_imm)
        immature_ids_set = set(ids_score_imm)
    else:
        parts = [X_train, X_test]
        ids_parts = ids_train + ids_test
        y_parts = [y_train, y_test]
        if X_immature is not None and X_immature.shape[0] > 0:
            parts.append(X_immature)
            ids_parts += ids_immature
            y_parts.append(y_immature)
        immature_ids_set = set(ids_immature)

    X_all = sparse_vstack(parts)
    y_all = np.concatenate(y_parts)

    # Save feature matrix + metadata for SHAP explanations at runtime
    # Use the scoring feature names (single-row mode) since that's what
    # the runtime explainer will use.
    from scipy import sparse as sp

    _shap_fnames = _fnames if use_panel else feature_names
    try:
        sp.save_npz(SHAP_MATRIX_PATH, X_all.tocsr())
        with open(SHAP_META_PATH, "w") as _meta_f:
            json.dump(
                {"bill_ids": ids_parts, "feature_names": list(_shap_fnames)},
                _meta_f,
            )
        LOGGER.info(
            "SHAP artifacts saved: %s (%d bills x %d features)",
            SHAP_MATRIX_PATH,
            X_all.shape[0],
            X_all.shape[1],
        )
    except Exception as _shap_err:
        LOGGER.warning("Failed to save SHAP artifacts: %s", _shap_err)

    df_scores = score_all_bills(
        calibrated,
        X_all,
        ids_parts,
        y_all,
        immature_ids=immature_ids_set,
        law_model=law_model,
        forecast_scores=forecast_scores_map,
        forecast_confidences=forecast_conf_map,
    )
    display_score_summary(df_scores)

    # Step 7: Quality report
    n_train_rows = len(ids_train)
    n_test_rows = len(ids_test)
    n_immature = len(ids_immature) if not use_panel else 0
    report = save_quality_report(
        comparison,
        best["name"],
        test_metrics,
        feature_names,
        tuned_model,
        n_train_rows + n_test_rows,
        n_immature,
        panel_mode=use_panel,
    )
    display_quality_summary(report)

    # Step 8: Generate/refresh gold labels for validation
    console.print("\n[bold]Step 8: Generating gold label set...[/]")
    try:
        from .gold_labels import generate_gold_labels

        gold = generate_gold_labels(target_size=400, balance_ratio=0.5)
        if gold:
            console.print(
                f"  Gold set: {gold['total_labels']} labels "
                f"({gold['positive_count']} advanced, {gold['negative_count']} stuck)"
            )
    except Exception as e:
        LOGGER.warning("Gold label generation failed: %s", e)

    return df_scores
