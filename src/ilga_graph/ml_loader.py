"""Load ML pipeline outputs for the API and web dashboard.

Reads parquet files from processed/ into typed dataclass containers.
Degrades gracefully if ML pipeline hasn't been run (available=False).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

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
    actual_outcome: str
    confidence: float
    label_reliable: bool
    chamber_origin: str
    introduction_date: str


@dataclass
class CoalitionMember:
    member_id: str
    name: str
    party: str
    chamber: str
    district: str
    coalition_id: int


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
    anomalies: list[SlipAnomaly] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    accuracy_history: list[AccuracyRun] = field(default_factory=list)
    last_run_date: str = ""
    available: bool = False


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
                actual_outcome=r.get("actual_outcome", ""),
                confidence=r.get("confidence", 0.0),
                label_reliable=bool(r.get("label_reliable", True)),
                chamber_origin=r.get("chamber_origin", ""),
                introduction_date=r.get("introduction_date", ""),
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
                )
                for r in df.to_dicts()
            ]
            LOGGER.info("Loaded %d coalition members", len(data.coalitions))
        except Exception:
            LOGGER.exception("Failed to load coalitions")

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

    # ── Last run date ────────────────────────────────────────────────
    import os

    try:
        mtime = os.path.getmtime(scores_path)
        from datetime import datetime

        data.last_run_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    return data
