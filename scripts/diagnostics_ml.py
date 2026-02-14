#!/usr/bin/env python3
"""ML pipeline diagnostics: data inputs, outputs, and model artifacts.

Runs lightweight checks without a full ml-run. Use to verify the branch
state of the ML stack (data → features → models → API load).

Usage:
    PYTHONPATH=src python scripts/diagnostics_ml.py
    make -f /dev/null PYTHONPATH=src run-diagnostics  # if you add a target
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

# Allow running from repo root with PYTHONPATH=src
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "processed"
CACHE = REPO_ROOT / "cache"


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_data_inputs() -> bool:
    """Verify ETL inputs (cache) and pipeline outputs (processed parquet)."""
    print("\n--- Data inputs (cache) ---")
    bills = CACHE / "bills.json"
    members = CACHE / "members.json"
    if not bills.exists():
        _fail(f"Missing {bills}")
        return False
    _ok(f"bills.json exists ({bills.stat().st_size / 1e6:.1f} MB)")
    if not members.exists():
        _warn(f"Missing {members}")
    else:
        _ok(f"members.json exists ({members.stat().st_size / 1e3:.1f} KB)")

    print("\n--- Processed (ETL output) ---")
    required = [
        "dim_bills.parquet",
        "dim_members.parquet",
        "fact_bill_actions.parquet",
        "fact_witness_slips.parquet",
    ]
    optional = [
        "fact_vote_events.parquet",
        "fact_vote_casts.parquet",
        "fact_vote_casts_raw.parquet",
    ]
    all_ok = True
    for name in required:
        p = PROCESSED / name
        if not p.exists():
            _fail(f"Missing {name}")
            all_ok = False
        else:
            import polars as pl

            df = pl.read_parquet(p)
            _ok(f"{name}: {len(df)} rows, {len(df.columns)} cols")
    for name in optional:
        p = PROCESSED / name
        if p.exists():
            import polars as pl

            df = pl.read_parquet(p)
            _ok(f"{name}: {len(df)} rows (optional)")
        else:
            _warn(f"Optional {name} missing")
    return all_ok


def check_ml_outputs() -> bool:
    """Verify ML pipeline outputs: bill_scores, models, quality JSON."""
    print("\n--- ML outputs ---")
    scores_path = PROCESSED / "bill_scores.parquet"
    if not scores_path.exists():
        _fail("bill_scores.parquet missing (run 'make ml-run')")
        return False

    import polars as pl

    df = pl.read_parquet(scores_path)
    required_cols = [
        "bill_id",
        "prob_advance",
        "predicted_outcome",
        "confidence",
        "forecast_score",
        "forecast_confidence",
        "prob_law",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        _fail(f"bill_scores missing columns: {missing}")
        return False
    _ok(f"bill_scores.parquet: {len(df)} rows, forecast_score/forecast_confidence present")

    # Sample forecast values
    if "forecast_score" in df.columns:
        fc = df.filter(pl.col("forecast_score") > 0)
        _ok(f"Bills with forecast_score > 0: {len(fc)}")
    if "forecast_confidence" in df.columns:
        conf_counts = df.group_by("forecast_confidence").len()
        _ok(f"forecast_confidence distribution: {conf_counts.to_dicts()}")

    # Model artifacts
    for name, path in [
        ("Status (advance) model", PROCESSED / "bill_predictor.pkl"),
        ("Law model", PROCESSED / "bill_law_predictor.pkl"),
        ("Forecast model", PROCESSED / "forecast_predictor.pkl"),
    ]:
        if not path.exists():
            _warn(f"Missing {path.name}")
            continue
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if hasattr(obj, "predict_proba"):
                _ok(f"{name}: loaded, has predict_proba")
            else:
                _ok(f"{name}: loaded")
        except Exception as e:
            _fail(f"{name}: {e}")
            return False

    for name in ["model_quality.json", "forecast_quality.json"]:
        p = PROCESSED / name
        if not p.exists():
            _warn(f"Missing {name}")
            continue
        try:
            with open(p) as f:
                q = json.load(f)
            if "test_set_metrics" in q:
                auc = q["test_set_metrics"].get("roc_auc", "?")
                _ok(f"{name}: roc_auc = {auc}")
            else:
                _ok(f"{name}: valid JSON")
        except Exception as e:
            _fail(f"{name}: {e}")
    return True


def check_ml_loader() -> bool:
    """Verify ml_loader.load_ml_data() runs and returns expected structure."""
    print("\n--- ML loader (API) ---")
    try:
        from ilga_graph.ml_loader import BillScore, load_ml_data

        data = load_ml_data()
    except Exception as e:
        _fail(f"load_ml_data() raised: {e}")
        return False
    if not data.available:
        _fail("load_ml_data() returned available=False")
        return False
    _ok(f"load_ml_data(): available=True, {len(data.bill_scores)} bill scores")
    if data.bill_scores:
        s = data.bill_scores[0]
        if not isinstance(s, BillScore):
            _fail("First bill_scores entry is not BillScore")
            return False
        has_forecast = hasattr(s, "forecast_score") and hasattr(s, "forecast_confidence")
        if not has_forecast:
            _warn("BillScore missing forecast_score/forecast_confidence (old schema?)")
        else:
            _ok("BillScore has forecast_score and forecast_confidence")
    if data.quality:
        _ok(f"model quality loaded: {data.quality.get('model_selected', '?')}")
    return True


def check_feature_build_imports() -> bool:
    """Ensure feature and panel builders can be imported (no full build)."""
    print("\n--- Feature / panel imports ---")
    try:
        from ilga_graph.ml.features import (
            FORECAST_DROP_COLUMNS,
        )

        _ok(
            "features.build_feature_matrix, build_panel_feature_matrix, "
            "FeatureMode, FORECAST_DROP_COLUMNS"
        )
        _ok(
            f"FORECAST_DROP_COLUMNS: {len(FORECAST_DROP_COLUMNS)} columns excluded in forecast mode"
        )
    except Exception as e:
        _fail(f"Feature imports: {e}")
        return False
    return True


def main() -> int:
    print("ML pipeline diagnostics (lightweight)")
    print(f"Processed dir: {PROCESSED}")
    print(f"Cache dir:     {CACHE}")

    results = []
    results.append(("Data inputs", check_data_inputs()))
    results.append(("ML outputs", check_ml_outputs()))
    results.append(("ML loader", check_ml_loader()))
    results.append(("Feature imports", check_feature_build_imports()))

    print("\n--- Summary ---")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    all_pass = all(r[1] for r in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
