"""Load ML pipeline outputs for the API and web dashboard.

Reads parquet files from processed/ into typed dataclass containers.
Degrades gracefully if ML pipeline hasn't been run (available=False).
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")


@dataclass
class BillScore:
    bill_id: str
    bill_number: str
    description: str
    sponsor: str
    prob_advance: float
    predicted_outcome: str
    confidence: float
    label_reliable: bool
    chamber_origin: str
    introduction_date: str
    # Pipeline stage fields (v4)
    current_stage: str = "FILED"
    stage_progress: float = 0.1
    stage_label: str = "Filed"
    days_since_action: int = 0
    last_action_text: str = ""
    last_action_date: str = ""
    # Stuck analysis fields (v4)
    stuck_status: str = ""
    stuck_reason: str = ""
    # Lifecycle status (v5): OPEN / PASSED / VETOED (governor veto only)
    lifecycle_status: str = "OPEN"
    # Law prediction (v6): P(becomes law) + predicted destination
    prob_law: float = 0.0
    predicted_destination: str = "Stuck"
    # Rule context (v7): Senate Rule citation for current stage / next step
    rule_context: str = ""
    # Forecast model (v8): intrinsic-only P(law) — "Truth" model
    forecast_score: float = 0.0
    forecast_confidence: str = ""  # "Low" | "Medium" | "High"


@dataclass
class CoalitionMember:
    member_id: str
    name: str
    party: str
    chamber: str
    district: str
    coalition_id: int
    coalition_name: str = ""
    coalition_focus: str = ""


@dataclass
class CoalitionProfile:
    coalition_id: int
    name: str
    focus_areas: list[str] = field(default_factory=list)
    size: int = 0
    dem_count: int = 0
    rep_count: int = 0
    yes_rate: float = 0.0
    cohesion: float = 0.0
    total_votes: int = 0
    signature_bills: list[dict] = field(default_factory=list)


@dataclass
class SlipAnomaly:
    bill_id: str
    bill_number: str
    description: str
    total_slips: int
    anomaly_score: float
    is_anomaly: bool
    anomaly_reason: str
    top_org_share: float
    org_hhi: float
    position_unanimity: float
    n_proponent: int
    n_opponent: int
    unique_orgs: int


@dataclass
class AccuracyRun:
    run_date: str
    snapshot_date: str
    days_elapsed: int
    total_testable: int
    correct: int
    accuracy: float
    precision_advance: float
    recall_advance: float
    f1_advance: float
    confidence_buckets: list[dict] = field(default_factory=list)
    biggest_misses: list[dict] = field(default_factory=list)
    model_version: str = ""


@dataclass
class MLData:
    bill_scores: list[BillScore] = field(default_factory=list)
    coalitions: list[CoalitionMember] = field(default_factory=list)
    coalition_profiles: list[CoalitionProfile] = field(default_factory=list)
    anomalies: list[SlipAnomaly] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    accuracy_history: list[AccuracyRun] = field(default_factory=list)
    last_run_date: str = ""
    available: bool = False
    # SHAP explanation support (v9)
    explainer: Any = None  # SHAPExplainer | None
    feature_matrix: Any = None  # sparse matrix | None
    feature_bill_ids: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    _bill_id_to_row: dict[str, int] = field(default_factory=dict)


def load_ml_data() -> MLData:
    """Load all ML outputs from processed/.

    Returns MLData with available=False if the pipeline hasn't been run.
    """
    scores_path = PROCESSED_DIR / "bill_scores.parquet"
    if not scores_path.exists():
        LOGGER.info("ML data not found (run 'make ml-run' to generate).")
        return MLData()

    try:
        import polars as pl
    except ImportError:
        LOGGER.warning(
            "polars not installed -- ML data will not be loaded. "
            "Run 'make ml-setup' to install ML dependencies."
        )
        return MLData()

    data = MLData(available=True)

    # ── Bill scores ──────────────────────────────────────────────────
    try:
        df = pl.read_parquet(scores_path)
        data.bill_scores = [
            BillScore(
                bill_id=r.get("bill_id", ""),
                bill_number=r.get("bill_number_raw", ""),
                description=r.get("description", "") or "",
                sponsor=r.get("primary_sponsor", "") or "",
                prob_advance=r.get("prob_advance", 0.0),
                predicted_outcome=r.get("predicted_outcome", ""),
                confidence=r.get("confidence", 0.0),
                label_reliable=bool(r.get("label_reliable", True)),
                chamber_origin=r.get("chamber_origin", ""),
                introduction_date=r.get("introduction_date", ""),
                current_stage=r.get("current_stage", "FILED"),
                stage_progress=r.get("stage_progress", 0.1),
                stage_label=r.get("stage_label", "Filed"),
                days_since_action=int(r.get("days_since_action", 0)),
                last_action_text=r.get("last_action", "") or "",
                last_action_date=r.get("last_action_date", "") or "",
                stuck_status=r.get("stuck_status", "") or "",
                stuck_reason=r.get("stuck_reason", "") or "",
                lifecycle_status=r.get("lifecycle_status", "OPEN") or "OPEN",
                prob_law=r.get("prob_law", 0.0) or 0.0,
                predicted_destination=r.get("predicted_destination", "Stuck") or "Stuck",
                rule_context=r.get("rule_context", "") or "",
                forecast_score=r.get("forecast_score", 0.0) or 0.0,
                forecast_confidence=r.get("forecast_confidence", "") or "",
            )
            for r in df.to_dicts()
        ]
        LOGGER.info("Loaded %d bill scores", len(data.bill_scores))
    except Exception:
        LOGGER.exception("Failed to load bill scores")

    # ── Coalitions ───────────────────────────────────────────────────
    coalitions_path = PROCESSED_DIR / "coalitions.parquet"
    if coalitions_path.exists():
        try:
            df = pl.read_parquet(coalitions_path)
            data.coalitions = [
                CoalitionMember(
                    member_id=r.get("member_id", ""),
                    name=r.get("name", ""),
                    party=r.get("party", ""),
                    chamber=r.get("chamber", ""),
                    district=str(r.get("district", "")),
                    coalition_id=int(r.get("coalition_id", -1)),
                    coalition_name=r.get("coalition_name", "") or "",
                    coalition_focus=r.get("coalition_focus", "") or "",
                )
                for r in df.to_dicts()
            ]
            LOGGER.info(
                "Loaded %d coalition members",
                len(data.coalitions),
            )
        except Exception:
            LOGGER.exception("Failed to load coalitions")

    # ── Coalition profiles ────────────────────────────────────────────
    profiles_path = PROCESSED_DIR / "coalition_profiles.json"
    if profiles_path.exists():
        try:
            with open(profiles_path) as f:
                raw_profiles = json.load(f)
            data.coalition_profiles = [
                CoalitionProfile(
                    coalition_id=p.get("coalition_id", -1),
                    name=p.get("name", ""),
                    focus_areas=p.get("focus_areas", []),
                    size=p.get("size", 0),
                    dem_count=p.get("dem_count", 0),
                    rep_count=p.get("rep_count", 0),
                    yes_rate=p.get("yes_rate", 0.0),
                    cohesion=p.get("cohesion", 0.0),
                    total_votes=p.get("total_votes", 0),
                    signature_bills=p.get("signature_bills", []),
                )
                for p in raw_profiles
            ]
        except Exception:
            LOGGER.exception("Failed to load coalition profiles")

    # ── Anomalies ────────────────────────────────────────────────────
    anomalies_path = PROCESSED_DIR / "slip_anomalies.parquet"
    if anomalies_path.exists():
        try:
            df = pl.read_parquet(anomalies_path)
            data.anomalies = [
                SlipAnomaly(
                    bill_id=r.get("bill_id", ""),
                    bill_number=r.get("bill_number_raw", ""),
                    description=r.get("description", "") or "",
                    total_slips=int(r.get("total_slips", 0)),
                    anomaly_score=r.get("anomaly_score", 0.0),
                    is_anomaly=bool(r.get("is_anomaly", False)),
                    anomaly_reason=r.get("anomaly_reason", "") or "",
                    top_org_share=r.get("top_org_share", 0.0),
                    org_hhi=r.get("org_hhi", 0.0),
                    position_unanimity=r.get("position_unanimity", 0.0),
                    n_proponent=int(r.get("n_proponent", 0)),
                    n_opponent=int(r.get("n_opponent", 0)),
                    unique_orgs=int(r.get("unique_orgs", 0)),
                )
                for r in df.to_dicts()
            ]
            LOGGER.info("Loaded %d anomaly scores", len(data.anomalies))
        except Exception:
            LOGGER.exception("Failed to load anomalies")

    # ── Model quality ────────────────────────────────────────────────
    quality_path = PROCESSED_DIR / "model_quality.json"
    if quality_path.exists():
        try:
            with open(quality_path) as f:
                data.quality = json.load(f)
        except Exception:
            LOGGER.exception("Failed to load model quality")

    # ── Accuracy history ─────────────────────────────────────────────
    history_path = PROCESSED_DIR / "accuracy_history.json"
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
            data.accuracy_history = [
                AccuracyRun(
                    run_date=r.get("run_date", ""),
                    snapshot_date=r.get("snapshot_date", ""),
                    days_elapsed=r.get("days_elapsed", 0),
                    total_testable=r.get("total_testable", 0),
                    correct=r.get("correct", 0),
                    accuracy=r.get("accuracy", 0.0),
                    precision_advance=r.get("precision_advance", 0.0),
                    recall_advance=r.get("recall_advance", 0.0),
                    f1_advance=r.get("f1_advance", 0.0),
                    confidence_buckets=r.get("confidence_buckets", []),
                    biggest_misses=r.get("biggest_misses", []),
                    model_version=r.get("model_version", ""),
                )
                for r in history.get("runs", [])
            ]
        except Exception:
            LOGGER.exception("Failed to load accuracy history")

    # ── SHAP explainer artifacts ────────────────────────────────────
    raw_model_path = PROCESSED_DIR / "bill_predictor_raw.pkl"
    shap_matrix_path = PROCESSED_DIR / "shap_feature_matrix.npz"
    shap_meta_path = PROCESSED_DIR / "shap_feature_meta.json"

    if raw_model_path.exists() and shap_matrix_path.exists() and shap_meta_path.exists():
        try:
            from scipy import sparse as sp

            from .ml.explainer import SHAPExplainer

            with open(raw_model_path, "rb") as f:
                raw_model = pickle.load(f)

            data.feature_matrix = sp.load_npz(shap_matrix_path)

            with open(shap_meta_path) as f:
                meta = json.load(f)
            data.feature_bill_ids = meta.get("bill_ids", [])
            data.feature_names = meta.get("feature_names", [])
            data._bill_id_to_row = {bid: idx for idx, bid in enumerate(data.feature_bill_ids)}

            data.explainer = SHAPExplainer(raw_model)
            LOGGER.info(
                "SHAP explainer loaded: %d bills, %d features",
                len(data.feature_bill_ids),
                len(data.feature_names),
            )
        except Exception:
            LOGGER.exception("Failed to load SHAP explainer (explanations will be unavailable)")
    else:
        LOGGER.info(
            "SHAP artifacts not found -- run 'make ml-run' to generate. "
            "Prediction explanations will be unavailable."
        )

    # ── Last run date ────────────────────────────────────────────────
    import os

    try:
        mtime = os.path.getmtime(scores_path)
        from datetime import datetime

        data.last_run_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    return data
