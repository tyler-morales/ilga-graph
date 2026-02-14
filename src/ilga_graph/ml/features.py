"""Feature engineering for bill outcome prediction.

Builds a feature matrix from the normalized Parquet tables where each row
represents one bill, and features capture:

- **Sponsor features**: party, chamber, historical passage rate, sponsor count
- **Text features**: TF-IDF of synopsis text (top N terms)
- **Slip features**: proponent/opponent counts, ratios, org concentration
- **Temporal features**: month, day of session, is_lame_duck
- **Action features**: time-capped action counts (30/60/90d), speed of early actions

Target: Binary -- 1 if bill passed committee (advanced), 0 if stuck/dead.

**Key design decisions:**
- Only "mature" bills (introduced 90+ days ago) are used for train/test
  to avoid labeling bills as "stuck" when they simply haven't had time yet.
- Time-based train/test split to prevent leakage.
- Stratified k-fold cross-validation within training for model selection.

**Panel (time-slice) dataset (optional, ``ILGA_ML_PANEL=1``):**
    Instead of one row per bill, creates multiple rows per bill at snapshot
    dates (e.g. 30, 60, 90 days after introduction).  For each snapshot row:
    - **Features** use only data up to the snapshot date (no future info).
    - **Label** = "did this bill advance AFTER this snapshot date?"
    - Only included when we have observed long enough after the snapshot
      (``OBSERVATION_DAYS_AFTER_SNAPSHOT``) to assign a reliable label.
    This yields 2-3x more training examples from the same set of bills and
    includes bills at earlier stages (e.g. "in committee" at day 30) that
    would otherwise be excluded for immaturity.  Inference still produces
    one row per bill using features "as of now".
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from datetime import timedelta as _timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .action_classifier import (
    bill_outcome_from_actions,
    classify_action_history,
)

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")

# Bills must be at least this many days old to have a reliable label.
# Bills introduced less than this many days ago are scored but not
# used for training or evaluation.
MATURITY_DAYS = 120

# ── Panel (time-slice) dataset configuration ─────────────────────────────────
# Snapshot dates: build one training row per bill per snapshot day.
SNAPSHOT_DAYS_AFTER_INTRO = [30, 60, 90]

# After each snapshot we must observe this many more days to assign a
# reliable label.  E.g. snapshot at day 60 + 90 observation = bill must
# be at least 150 days old for this snapshot row to be usable.
OBSERVATION_DAYS_AFTER_SNAPSHOT = 90

# ── Full-text feature caps (keeps signal, limits matrix size and runtime) ──────
# Override with ILGA_ML_FULLTEXT_MAX_FEATURES and ILGA_ML_FULLTEXT_MAX_TOKENS.
# Lower values = faster training; higher = more text signal (diminishing returns).
FULLTEXT_MAX_FEATURES = int(os.environ.get("ILGA_ML_FULLTEXT_MAX_FEATURES", "400"))
FULLTEXT_MAX_TOKENS = int(os.environ.get("ILGA_ML_FULLTEXT_MAX_TOKENS", "2000"))

# ── Action-based outcome classification ──────────────────────────────────────
#
# All classification now delegates to the action_classifier module which
# uses action_types.json — a comprehensive reference of every action type
# in the IL General Assembly with proper meanings and outcome signals.
#
# The classifier handles ILGA's inconsistent formatting, properly
# distinguishes bill vs. amendment actions, and classifies actions into
# 16 categories with outcome signals (positive_terminal, positive,
# positive_weak, neutral, negative_weak, negative, negative_terminal).

# ── Feature mode ──────────────────────────────────────────────────────────────
# "full"     = Status model — uses ALL features including staleness, slips,
#              action counts, and rule-derived features.
# "forecast" = Forecast model — uses only intrinsic / Day-0 features:
#              sponsor, text (TF-IDF), committee assignment, calendar, and
#              content metadata.  Excludes anything that accumulates over
#              time (staleness, action counts, witness slips, rule events).
FeatureMode = Literal["full", "forecast"]

# Columns excluded in forecast mode.  Defined once so both
# build_feature_matrix and build_panel_feature_matrix stay consistent.
FORECAST_DROP_COLUMNS: frozenset[str] = frozenset(
    {
        # Staleness / activity (accumulate over time)
        "days_since_last_action",
        "days_since_intro",
        "action_count_30d",
        "action_count_60d",
        "action_count_90d",
        "action_velocity_60d",
        "is_stale_90",
        "is_stale_180",
        "has_negative_terminal",
        # Early action counts (depend on post-intro actions)
        "early_action_count",
        "early_cosponsor_actions",
        "early_committee_referrals",
        # Witness slips (accumulate over time)
        "slip_total",
        "slip_proponent",
        "slip_opponent",
        "slip_proponent_ratio",
        "slip_opponent_ratio",
        "slip_written_only",
        "slip_written_ratio",
        "slip_unique_orgs",
        "slip_org_concentration",
        "has_no_slip_data",
        # Rule-derived features (depend on post-intro actions)
        "missed_committee_deadline",
        "missed_floor_deadline",
        "has_favorable_report",
        "was_tabled",
        "on_consent_calendar",
        "has_bipartisan_sponsors",
        # Full-text-derived features (leak "enrolled" / law outcome; see Forecast audit)
        "full_text_word_count",
        "full_text_log_length",
        "full_text_section_count",
        "full_text_citation_count",
        "has_full_text",
        "is_long_bill",
        "is_short_bill",
    }
)

# Snapshot days used by the Forecast model (Day-0 and Day-30 only).
FORECAST_SNAPSHOT_DAYS = [0, 30]

# ── Human-readable feature name mapping (for SHAP explanations) ────────────
FEATURE_NAME_MAPPING: dict[str, str] = {
    # Sponsor Features
    "sponsor_party": "Sponsor's Political Party",
    "sponsor_is_majority": "Sponsored by Majority Party",
    "sponsor_hist_passage_rate": "Sponsor's Historical Success Rate",
    "sponsor_bill_count": "Sponsor's Total Active Bills",
    "sponsor_count": "Number of Co-Sponsors",
    "sponsor_party_democrat": "Sponsor Is Democrat",
    "sponsor_party_republican": "Sponsor Is Republican",
    "has_sponsor_id": "Has Known Sponsor",
    "sponsor_hist_filed": "Sponsor's Previously Filed Bills",
    "has_no_sponsor_history": "Sponsor Has No History",
    # Public Engagement (Witness Slips)
    "slip_proponent": "Public Proponent Slips",
    "slip_opponent": "Public Opponent Slips",
    "slip_ratio": "Ratio of Proponent to Opponent Slips",
    "slip_unique_orgs": "Number of Engaged Organizations",
    "slip_total": "Total Witness Slips",
    "slip_proponent_ratio": "Proponent Slip Ratio",
    "slip_opponent_ratio": "Opponent Slip Ratio",
    "slip_written_only": "Written-Only Slips",
    "slip_written_ratio": "Written Slip Ratio",
    "slip_org_concentration": "Organization Concentration",
    "has_no_slip_data": "No Witness Slip Data",
    # Timing & Momentum
    "month_intro": "Month Introduced",
    "intro_month": "Month Introduced",
    "day_of_year": "Day of the Year",
    "intro_day_of_year": "Day of the Year",
    "is_lame_duck": "Introduced During Lame Duck Session",
    "days_since_intro": "Days Since Introduction",
    "days_since_last_action": "Days Since Last Movement",
    "action_count_30d": "Legislative Actions (Last 30 Days)",
    "action_count_60d": "Legislative Actions (Last 60 Days)",
    "action_count_90d": "Legislative Actions (Last 90 Days)",
    "action_velocity_60d": "Action Velocity (60 Days)",
    "early_action_count": "Initial Momentum (Early Actions)",
    "early_cosponsor_actions": "Early Co-Sponsor Actions",
    "early_committee_referrals": "Early Committee Referrals",
    # Staleness
    "is_stale_90": "Stale (90+ Days Idle)",
    "is_stale_180": "Stale (180+ Days Idle)",
    "has_negative_terminal": "Reached Terminal Negative State",
    # Chamber / Bill Type
    "is_senate_origin": "Originated in Senate",
    "is_house_origin": "Originated in House",
    "is_resolution": "Is a Resolution",
    "is_substantive": "Is Substantive Legislation",
    # Committee
    "committee_advancement_rate": "Committee Advancement Rate",
    "committee_pass_rate": "Committee Pass Rate",
    "committee_bill_volume": "Committee Bill Volume",
    "is_high_throughput_committee": "High-Throughput Committee",
    "has_committee_assignment": "Has Committee Assignment",
    # Rule-derived
    "missed_committee_deadline": "Missed Committee Deadline",
    "missed_floor_deadline": "Missed Floor Deadline",
    "has_favorable_report": "Has Favorable Committee Report",
    "was_tabled": "Was Tabled",
    "on_consent_calendar": "On Consent Calendar",
    "has_bipartisan_sponsors": "Has Bipartisan Sponsors",
    # Content metadata
    "full_text_word_count": "Bill Text Word Count",
    "full_text_log_length": "Bill Text Length (Log)",
    "full_text_section_count": "Number of Sections",
    "full_text_citation_count": "Number of Legal Citations",
    "has_full_text": "Full Text Available",
    "is_long_bill": "Is a Long Bill",
    "is_short_bill": "Is a Short Bill",
}

# Prefixes for one-hot encoded categorical groups that should be
# summed into a single master feature before SHAP ranking.
CATEGORICAL_PREFIXES: list[str] = [
    "sponsor_party_",
]


def humanize_feature_name(raw_name: str) -> str:
    """Map a raw feature name to a human-readable label.

    Checks ``FEATURE_NAME_MAPPING`` first, then falls back to
    title-casing with underscores replaced by spaces.  TF-IDF
    features get a descriptive prefix.
    """
    if raw_name in FEATURE_NAME_MAPPING:
        return FEATURE_NAME_MAPPING[raw_name]
    if raw_name.startswith("tfidf_"):
        term = raw_name.removeprefix("tfidf_").replace("_", " ")
        return f'Synopsis Term: "{term}"'
    if raw_name.startswith("ft_tfidf_"):
        term = raw_name.removeprefix("ft_tfidf_").replace("_", " ")
        return f'Full-Text Term: "{term}"'
    if raw_name.startswith("sponsor_emb_"):
        dim = raw_name.removeprefix("sponsor_emb_")
        return f"Sponsor Network Dimension {dim}"
    return raw_name.replace("_", " ").title()


# ── Stage progress mapping ────────────────────────────────────────────────
# Maps classifier progress_stage values to numeric progress fractions.
_STAGE_PROGRESS: dict[str, float] = {
    "FILED": 0.10,
    "IN_COMMITTEE": 0.25,
    "PASSED_COMMITTEE": 0.40,
    "FLOOR_VOTE": 0.55,
    "CHAMBER_PASSED": 0.60,
    "CROSSED_CHAMBERS": 0.70,
    "PASSED_BOTH": 0.85,
    "GOVERNOR": 0.95,
    "SIGNED": 1.0,
}


def _bill_has_negative_terminal(actions: list[str]) -> bool:
    """Did this bill reach a negative terminal state (governor veto)?

    Delegates to the action classifier — only actual governor vetoes
    (negative_terminal signals) count.  NOT: amendments tabled,
    Rule 19 re-referrals, committee postponements, or sine die refs.
    """
    classified = classify_action_history(actions)
    for ca in classified:
        if ca.is_bill_action and ca.outcome_signal == "negative_terminal":
            return True
    return False


def _bill_advanced(actions: list[str]) -> bool:
    """Did this bill reach a positive outcome?

    A bill is "advanced" if it has any positive bill-level action
    (passed committee, passed floor vote, crossed chambers, signed)
    AND was NOT subsequently vetoed.
    """
    classified = classify_action_history(actions)
    outcome = bill_outcome_from_actions(classified)

    # If vetoed, not advanced
    if outcome["lifecycle_status"] == "VETOED":
        return False

    # Check for any meaningful forward progress
    has_positive = any(
        ca.is_bill_action
        and ca.outcome_signal in ("positive", "positive_terminal")
        and ca.category_id
        in (
            "committee_action",
            "floor_vote",
            "cross_chamber",
            "concurrence",
            "governor",
            "enacted",
        )
        for ca in classified
    )
    return has_positive


def _bill_became_law(actions: list[str]) -> bool:
    """Did this bill become law?"""
    classified = classify_action_history(actions)
    return any(ca.is_bill_action and ca.outcome_signal == "positive_terminal" for ca in classified)


def _bill_lifecycle_status(actions: list[str]) -> str:
    """Classify a bill's lifecycle status for display.

    Delegates to the action classifier's bill_outcome_from_actions().

    Returns one of:
        PASSED   - Became law (signed / public act)
        VETOED   - Vetoed by governor (confirmed terminal)
        OPEN     - Everything else (in committee, idle, tabled, etc.)
    """
    classified = classify_action_history(actions)
    outcome = bill_outcome_from_actions(classified)
    return outcome["lifecycle_status"]


def compute_bill_stage(
    actions: list[str],
) -> tuple[str, float]:
    """Determine the highest legislative stage reached by a bill.

    Uses the action classifier to determine the highest stage from
    the bill's action history.  Only bill-level actions (not amendments)
    contribute to stage progression.

    Returns (stage_name, progress_fraction).
    Progress is 0.0-1.0, with -1.0 for VETOED (terminal).
    """
    classified = classify_action_history(actions)
    outcome = bill_outcome_from_actions(classified)

    # Special case: VETOED
    if outcome["lifecycle_status"] == "VETOED":
        return "VETOED", -1.0

    # Use current_stage (rollbacks applied) so we don't show "Governor" for bills
    # that were re-referred after passing both chambers (e.g. HB3356 Rule 19(b)).
    highest = outcome.get("current_stage") or outcome["highest_stage"]
    # Map classifier stages to our stage names
    stage_map = {
        "FILED": "FILED",
        "IN_COMMITTEE": "IN_COMMITTEE",
        "PASSED_COMMITTEE": "PASSED_COMMITTEE",
        "FLOOR_VOTE": "FLOOR_VOTE",
        "CHAMBER_PASSED": "FLOOR_VOTE",  # map to nearest
        "CROSSED_CHAMBERS": "CROSSED_CHAMBERS",
        "PASSED_BOTH": "PASSED_BOTH",
        "GOVERNOR": "PASSED_BOTH",  # governor action implies passed both
        "SIGNED": "SIGNED",
    }
    stage = stage_map.get(highest, "FILED")
    progress = _STAGE_PROGRESS.get(highest, 0.10)
    return stage, progress


def classify_stuck_status(
    current_stage: str,
    days_since_action: int,
    days_since_intro: int,
    actions: list[str],
) -> tuple[str, str]:
    """Classify a stuck bill into a nuanced sub-status.

    Returns (stuck_status, stuck_reason).

    Sub-statuses:
        DEAD      - Vetoed by governor only (bill "died" by veto)
        STAGNANT  - In committee 180+ days with no action, or tabled/sine die (session ended)
        SLOW      - Some activity, but 60-180 days since last action
        PENDING   - Last action within 60 days
        NEW       - Introduced less than 30 days ago
    """
    # Only an actual governor veto means DEAD.  We use the same precise
    # matching as _bill_has_negative_terminal — skip amendment actions,
    # match only governor-level veto actions.
    if _bill_has_negative_terminal(actions):
        return "DEAD", "Vetoed by the Governor"

    if current_stage == "VETOED":
        return "DEAD", "Vetoed by the Governor"

    # NOTE: We intentionally do NOT classify tabled/sine die/postponed/
    # Rule 19 re-referrals as STAGNANT here.  Those are procedural —
    # the bill may still be alive.  Staleness is determined purely by
    # days_since_action below.

    if days_since_intro < 30:
        return "NEW", "Introduced less than 30 days ago -- too early to assess"

    if days_since_action >= 180:
        months = days_since_action // 30
        return (
            "STAGNANT",
            f"No activity for {months} months (stage: {_stage_label(current_stage)})",
        )

    if days_since_action >= 60:
        months = days_since_action // 30
        return (
            "SLOW",
            f"Last action {months} months ago (stage: {_stage_label(current_stage)})",
        )

    return (
        "PENDING",
        f"Active within last 60 days (stage: {_stage_label(current_stage)})",
    )


def _stage_label(stage: str) -> str:
    """Human-readable stage label."""
    labels = {
        "FILED": "Filed",
        "IN_COMMITTEE": "In Committee",
        "PASSED_COMMITTEE": "Passed Committee",
        "FLOOR_VOTE": "Floor Vote",
        "CROSSED_CHAMBERS": "Crossed Chambers",
        "PASSED_BOTH": "Passed Both Houses",
        "SIGNED": "Signed into Law",
        "VETOED": "Vetoed",
    }
    return labels.get(stage, stage)


# ── Feature building ─────────────────────────────────────────────────────────


def build_bill_labels(
    df_bills: pl.DataFrame,
    df_actions: pl.DataFrame,
) -> pl.DataFrame:
    """Compute binary labels for each bill.

    Returns a DataFrame with bill_id, target_advanced (0/1), target_law (0/1),
    and is_mature (whether the bill is old enough for a reliable label).
    """
    # Group actions by bill
    bill_actions = df_actions.group_by("bill_id").agg(pl.col("action_text").alias("actions"))

    # Join with bills
    df = df_bills.select(["bill_id", "bill_type", "introduction_date"]).join(
        bill_actions, on="bill_id", how="left"
    )

    # Compute labels and maturity
    advanced_labels = []
    law_labels = []
    mature_flags = []

    cutoff_date = datetime.now()

    for row in df.to_dicts():
        actions = row.get("actions") or []
        advanced_labels.append(1 if _bill_advanced(actions) else 0)
        law_labels.append(1 if _bill_became_law(actions) else 0)

        # Is this bill old enough for its label to be meaningful?
        intro = row.get("introduction_date")
        is_mature = False
        if intro:
            try:
                dt = datetime.strptime(intro, "%Y-%m-%d")
                days_old = (cutoff_date - dt).days
                is_mature = days_old >= MATURITY_DAYS
            except ValueError:
                LOGGER.debug(
                    "Unparseable introduction_date in labels: %r for %s",
                    intro,
                    row.get("bill_id", "?"),
                )
        mature_flags.append(is_mature)

    return df.select(["bill_id", "bill_type", "introduction_date"]).with_columns(
        pl.Series("target_advanced", advanced_labels, dtype=pl.Int8),
        pl.Series("target_law", law_labels, dtype=pl.Int8),
        pl.Series("is_mature", mature_flags, dtype=pl.Boolean),
    )


def build_panel_labels(
    df_bills: pl.DataFrame,
    df_actions: pl.DataFrame,
    *,
    snapshot_days: list[int] | None = None,
    observation_days: int | None = None,
) -> pl.DataFrame:
    """Build time-sliced labels for the panel dataset.

    For each bill, creates one row per snapshot day (e.g. 30, 60, 90 days
    after introduction).  For each row the label answers: "did this bill
    advance (or become law) **after** the snapshot date?"

    Only rows where we have observed long enough after the snapshot
    (``observation_days``) are included, so labels are reliable.

    Returns a DataFrame with columns:
        bill_id, snapshot_day, snapshot_date, introduction_date,
        target_advanced_after, target_law_after
    """
    if snapshot_days is None:
        snapshot_days = list(SNAPSHOT_DAYS_AFTER_INTRO)
    if observation_days is None:
        observation_days = OBSERVATION_DAYS_AFTER_SNAPSHOT

    today = datetime.now()

    # Group actions by bill (handle empty actions DataFrame)
    if len(df_actions) > 0:
        bill_actions = df_actions.group_by("bill_id").agg(
            pl.col("action_text").alias("actions"),
            pl.col("date").alias("action_dates"),
        )
    else:
        bill_actions = pl.DataFrame(
            schema={
                "bill_id": pl.Utf8,
                "actions": pl.List(pl.Utf8),
                "action_dates": pl.List(pl.Utf8),
            }
        )

    df = df_bills.select(["bill_id", "bill_type", "introduction_date"]).join(
        bill_actions, on="bill_id", how="left"
    )

    rows: list[dict] = []
    for bill in df.to_dicts():
        bid = bill["bill_id"]
        intro_str = bill.get("introduction_date")
        if not intro_str:
            continue
        try:
            intro_dt = datetime.strptime(intro_str, "%Y-%m-%d")
        except ValueError:
            continue

        all_actions = bill.get("actions") or []
        all_dates = bill.get("action_dates") or []

        # Build (date, action_text) pairs for filtering
        action_pairs: list[tuple[datetime | None, str]] = []
        for d_str, a_text in zip(all_dates, all_actions):
            if d_str:
                try:
                    action_pairs.append((datetime.strptime(d_str, "%Y-%m-%d"), a_text))
                except ValueError:
                    action_pairs.append((None, a_text))
            else:
                action_pairs.append((None, a_text))

        for snap_day in snapshot_days:
            snapshot_dt = intro_dt + _timedelta(days=snap_day)

            # Must have enough observation time after the snapshot
            if (today - snapshot_dt).days < observation_days:
                continue

            # Split actions into "at snapshot" and "after snapshot"
            post_actions = [
                a_text for a_dt, a_text in action_pairs if a_dt is not None and a_dt > snapshot_dt
            ]

            advanced_after = 1 if _bill_advanced(post_actions) else 0
            law_after = 1 if _bill_became_law(post_actions) else 0

            rows.append(
                {
                    "bill_id": bid,
                    "snapshot_day": snap_day,
                    "snapshot_date": snapshot_dt.strftime("%Y-%m-%d"),
                    "introduction_date": intro_str,
                    "target_advanced_after": advanced_after,
                    "target_law_after": law_after,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "bill_id": pl.Utf8,
                "snapshot_day": pl.Int64,
                "snapshot_date": pl.Utf8,
                "introduction_date": pl.Utf8,
                "target_advanced_after": pl.Int8,
                "target_law_after": pl.Int8,
            }
        )

    df_panel = pl.DataFrame(rows).with_columns(
        pl.col("target_advanced_after").cast(pl.Int8),
        pl.col("target_law_after").cast(pl.Int8),
    )

    LOGGER.info(
        "Panel labels: %d rows from %d bills (snapshots at %s days, observation window %d days)",
        len(df_panel),
        df_panel["bill_id"].n_unique(),
        snapshot_days,
        observation_days,
    )

    return df_panel


def build_sponsor_features(
    df_bills: pl.DataFrame,
    df_members: pl.DataFrame,
    df_labels: pl.DataFrame,
    *,
    as_of_date: str | None = None,
) -> pl.DataFrame:
    """Build sponsor-related features for each bill.

    Includes historical passage rate: for each bill, we compute the sponsor's
    passage rate using ONLY bills introduced BEFORE this one (no leakage).

    Parameters
    ----------
    as_of_date:
        If provided (``"YYYY-MM-DD"``), only count historical outcomes from
        bills whose labels are known by this date (i.e. bills introduced at
        least ``MATURITY_DAYS`` before ``as_of_date``).  Used by the panel
        dataset so sponsor history reflects what was knowable at the snapshot.
    """
    member_map = {m["member_id"]: m for m in df_members.to_dicts()}

    # Compute majority party dynamically from member data
    party_counts: dict[str, int] = {}
    for m in member_map.values():
        p = m.get("party", "")
        if p:
            party_counts[p] = party_counts.get(p, 0) + 1
    majority_party = max(party_counts, key=party_counts.get) if party_counts else "Democrat"
    LOGGER.info("Majority party computed from members: %s (%s)", majority_party, party_counts)

    # Sort bills by introduction date for historical rate computation
    bills_sorted = (
        df_bills.join(
            df_labels.select(["bill_id", "target_advanced"]),
            on="bill_id",
            how="left",
        )
        .sort("introduction_date", nulls_last=True)
        .to_dicts()
    )

    # Track running sponsor stats for historical rate
    sponsor_filed: dict[str, int] = {}
    sponsor_advanced: dict[str, int] = {}

    rows = []
    for bill in bills_sorted:
        sponsor_id = bill.get("primary_sponsor_id")
        member = member_map.get(sponsor_id) if sponsor_id else None

        party = member.get("party", "Unknown") if member else "Unknown"
        is_democrat = 1 if party == "Democrat" else 0
        is_republican = 1 if party == "Republican" else 0
        is_majority = 1 if party == majority_party else 0

        # Historical passage rate (BEFORE this bill)
        hist_filed = sponsor_filed.get(sponsor_id, 0) if sponsor_id else 0
        hist_advanced = sponsor_advanced.get(sponsor_id, 0) if sponsor_id else 0
        hist_rate = hist_advanced / hist_filed if hist_filed > 0 else 0.0

        rows.append(
            {
                "bill_id": bill["bill_id"],
                "sponsor_party_democrat": is_democrat,
                "sponsor_party_republican": is_republican,
                "sponsor_is_majority": is_majority,
                "sponsor_count": bill.get("sponsor_count", 0),
                "has_sponsor_id": 1 if sponsor_id else 0,
                "sponsor_hist_filed": hist_filed,
                "sponsor_hist_passage_rate": hist_rate,
                "sponsor_bill_count": (member.get("sponsored_bill_count", 0) if member else 0),
            }
        )

        # Update running stats AFTER computing features (no leakage).
        # Only count bills where we have a known label — otherwise we'd
        # inflate the denominator with unlabeled bills.
        # When as_of_date is set, also require the bill to be old enough
        # that its label would be known by as_of_date.
        label_known = bill.get("target_advanced") is not None
        if as_of_date and label_known:
            bill_intro = bill.get("introduction_date", "")
            if bill_intro:
                try:
                    bill_intro_dt = datetime.strptime(bill_intro, "%Y-%m-%d")
                    cutoff_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
                    label_known = (cutoff_dt - bill_intro_dt).days >= MATURITY_DAYS
                except ValueError:
                    label_known = False

        if sponsor_id and label_known:
            sponsor_filed[sponsor_id] = sponsor_filed.get(sponsor_id, 0) + 1
            if bill["target_advanced"] == 1:
                sponsor_advanced[sponsor_id] = sponsor_advanced.get(sponsor_id, 0) + 1

    return pl.DataFrame(rows)


def build_text_features(
    df_bills: pl.DataFrame,
    max_features: int = 500,
) -> tuple[np.ndarray, TfidfVectorizer, list[str]]:
    """Build TF-IDF features from bill synopsis text.

    Returns (tfidf_matrix, fitted_vectorizer, bill_ids).
    """
    bill_ids = df_bills["bill_id"].to_list()
    texts = df_bills["synopsis_text"].fill_null("").to_list()

    # Clean text
    cleaned = []
    for t in texts:
        t = re.sub(r"\s+", " ", t).strip()
        cleaned.append(t if len(t) > 5 else "")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=3,
        max_df=0.8,
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(cleaned)
    LOGGER.info(
        "TF-IDF: %d bills x %d features",
        tfidf_matrix.shape[0],
        tfidf_matrix.shape[1],
    )

    return tfidf_matrix, vectorizer, bill_ids


# ── Legislative boilerplate patterns (for full-text cleaning) ────────────────

_RE_LEGISLATIVE_BOILERPLATE = re.compile(
    r"(?:"
    r"AN ACT concerning[^.]*\."
    r"|Be it enacted by the People of the State of Illinois[^.]*\."
    r"|ARTICLE \d+"
    r"|WHEREAS[,;]"
    r"|NOW,?\s*THEREFORE"
    r"|Section \d+[\-.]?\d*\."
    r")",
    re.IGNORECASE,
)


def build_full_text_features(
    df_bills: pl.DataFrame,
    max_features: int | None = None,
    max_tokens: int | None = None,
) -> tuple[np.ndarray, TfidfVectorizer, list[str]]:
    """Build TF-IDF features from full bill text.

    Uses module-level FULLTEXT_MAX_FEATURES and FULLTEXT_MAX_TOKENS (env-overridable)
    when max_features/max_tokens are None, so we cap size and runtime by default.

    Handles the extreme length variance (sub-1-page to 500+ pages) with:
    - Truncation to first ``max_tokens`` words
    - ``sublinear_tf=True`` to dampen long-bill term dominance
    - ``norm='l2'`` so long and short bills have equal weight
    - Higher ``min_df`` / lower ``max_df`` than synopsis to filter legal boilerplate

    Returns (tfidf_matrix, fitted_vectorizer, bill_ids).
    When no bills have full_text, returns a zero sparse matrix.
    """
    if max_features is None:
        max_features = FULLTEXT_MAX_FEATURES
    if max_tokens is None:
        max_tokens = FULLTEXT_MAX_TOKENS

    bill_ids = df_bills["bill_id"].to_list()

    # Check if full_text column exists
    if "full_text" not in df_bills.columns:
        LOGGER.info("Full-text TF-IDF: no full_text column — returning zeros.")
        zero_matrix = csr_matrix((len(bill_ids), 0), dtype=np.float32)
        vectorizer = TfidfVectorizer()
        return zero_matrix, vectorizer, bill_ids

    texts = df_bills["full_text"].fill_null("").to_list()

    # Check if any bills have text
    non_empty = sum(1 for t in texts if t and len(t) > 10 and not t.startswith("[SKIPPED:"))
    if non_empty == 0:
        LOGGER.info("Full-text TF-IDF: 0 bills have full_text — returning zeros.")
        zero_matrix = csr_matrix((len(bill_ids), 0), dtype=np.float32)
        vectorizer = TfidfVectorizer()
        return zero_matrix, vectorizer, bill_ids

    LOGGER.info(
        "Full-text TF-IDF: %d of %d bills have full_text (max %d features, %d tokens/bill).",
        non_empty,
        len(bill_ids),
        max_features,
        max_tokens,
    )

    # Clean and truncate
    cleaned = []
    for t in texts:
        if not t or t.startswith("[SKIPPED:"):
            cleaned.append("")
            continue
        # Strip legislative boilerplate
        t = _RE_LEGISLATIVE_BOILERPLATE.sub(" ", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        # Truncate to first max_tokens words
        tokens = t.split()[:max_tokens]
        text = " ".join(tokens) if len(tokens) > 5 else ""
        cleaned.append(text)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2),
        sublinear_tf=True,  # log(tf) to dampen long-bill dominance
        norm="l2",  # normalize so long and short bills have equal weight
    )

    tfidf_matrix = vectorizer.fit_transform(cleaned)
    LOGGER.info(
        "Full-text TF-IDF: %d bills x %d features",
        tfidf_matrix.shape[0],
        tfidf_matrix.shape[1],
    )

    return tfidf_matrix, vectorizer, bill_ids


def build_content_metadata_features(df_bills: pl.DataFrame) -> pl.DataFrame:
    """Build numeric features derived from full bill text content.

    Features capture bill length and complexity signals:
    - ``full_text_word_count``: raw word count (0 if no text)
    - ``full_text_log_length``: log1p(word_count) — prevents 500-page
      bills from having 500x the weight of 1-page bills
    - ``is_long_bill``: 1 if > 10,000 words (~20+ pages)
    - ``is_short_bill``: 1 if < 500 words (~1 page)
    - ``full_text_section_count``: count "Section N." patterns (complexity)
    - ``full_text_citation_count``: count ILCS citations (amends existing law)
    - ``has_full_text``: 1 if full_text is non-empty

    Returns a DataFrame with bill_id + feature columns.
    When ``full_text`` column is missing, returns all-zero features.
    """
    has_column = "full_text" in df_bills.columns

    rows = []
    for bill in df_bills.to_dicts():
        full_text = bill.get("full_text", "") if has_column else ""
        # Treat skip markers as empty
        if full_text and full_text.startswith("[SKIPPED:"):
            full_text = ""

        word_count = len(full_text.split()) if full_text else 0
        section_count = len(re.findall(r"Section\s+\d+", full_text)) if full_text else 0
        citation_count = len(re.findall(r"\d+\s+ILCS\s+\d+", full_text)) if full_text else 0

        rows.append(
            {
                "bill_id": bill["bill_id"],
                "full_text_word_count": word_count,
                "full_text_log_length": float(np.log1p(word_count)),
                "is_long_bill": 1 if word_count > 10_000 else 0,
                "is_short_bill": 1 if 0 < word_count < 500 else 0,
                "full_text_section_count": section_count,
                "full_text_citation_count": citation_count,
                "has_full_text": 1 if word_count > 0 else 0,
            }
        )

    return pl.DataFrame(rows)


def build_slip_features(
    df_bills: pl.DataFrame,
    df_slips: pl.DataFrame,
    *,
    as_of_date: str | None = None,
) -> pl.DataFrame:
    """Build witness slip aggregate features per bill.

    Parameters
    ----------
    as_of_date:
        If provided (``"YYYY-MM-DD"``), only include slips with
        ``hearing_date <= as_of_date``.  Used by the panel dataset.
    """
    df_slips_filtered = df_slips
    if as_of_date and len(df_slips) > 0 and "hearing_date" in df_slips.columns:
        df_slips_filtered = df_slips.filter(
            pl.col("hearing_date").is_not_null() & (pl.col("hearing_date") <= as_of_date)
        )

    if len(df_slips_filtered) == 0:
        return df_bills.select("bill_id").with_columns(
            pl.lit(0).alias("slip_total"),
            pl.lit(0).alias("slip_proponent"),
            pl.lit(0).alias("slip_opponent"),
            pl.lit(0.0).alias("slip_proponent_ratio"),
            pl.lit(0.0).alias("slip_opponent_ratio"),
            pl.lit(0).alias("slip_written_only"),
            pl.lit(0.0).alias("slip_written_ratio"),
            pl.lit(0).alias("slip_unique_orgs"),
            pl.lit(0.0).alias("slip_org_concentration"),
        )

    agg = df_slips_filtered.group_by("bill_id").agg(
        pl.len().alias("slip_total"),
        (pl.col("position") == "Proponent").sum().alias("slip_proponent"),
        (pl.col("position") == "Opponent").sum().alias("slip_opponent"),
        (pl.col("testimony_type").str.contains("(?i)record|written"))
        .sum()
        .alias("slip_written_only"),
        pl.col("organization_clean").n_unique().alias("slip_unique_orgs"),
    )

    agg = agg.with_columns(
        (pl.col("slip_proponent") / pl.col("slip_total").cast(pl.Float64).clip(1, None)).alias(
            "slip_proponent_ratio"
        ),
        (pl.col("slip_opponent") / pl.col("slip_total").cast(pl.Float64).clip(1, None)).alias(
            "slip_opponent_ratio"
        ),
        (pl.col("slip_written_only") / pl.col("slip_total").cast(pl.Float64).clip(1, None)).alias(
            "slip_written_ratio"
        ),
    )

    # Org concentration (HHI)
    org_counts = df_slips_filtered.group_by(["bill_id", "organization_clean"]).agg(
        pl.len().alias("org_count")
    )
    bill_totals = df_slips_filtered.group_by("bill_id").agg(pl.len().alias("bill_total"))
    org_shares = org_counts.join(bill_totals, on="bill_id").with_columns(
        (pl.col("org_count") / pl.col("bill_total").cast(pl.Float64)).alias("share")
    )
    hhi = org_shares.group_by("bill_id").agg(
        (pl.col("share").pow(2).sum()).alias("slip_org_concentration")
    )

    agg = agg.join(hhi, on="bill_id", how="left").fill_null(0.0)
    result = df_bills.select("bill_id").join(agg, on="bill_id", how="left").fill_null(0)

    return result


def build_temporal_features(df_bills: pl.DataFrame) -> pl.DataFrame:
    """Build time-based features for each bill."""
    rows = []
    for bill in df_bills.to_dicts():
        intro_date = bill.get("introduction_date")
        month = None
        day_of_year = None
        is_lame_duck = 0

        if intro_date:
            try:
                dt = datetime.strptime(intro_date, "%Y-%m-%d")
                month = dt.month
                day_of_year = dt.timetuple().tm_yday
                is_lame_duck = 1 if dt.month >= 11 or dt.month == 1 else 0
            except ValueError:
                pass

        bill_type = bill.get("bill_type", "")
        is_senate = 1 if bill_type.startswith("S") else 0
        is_house = 1 if bill_type.startswith("H") else 0
        is_resolution = 1 if bill_type in ("SR", "HR", "SJR", "HJR", "SJRCA", "HJRCA") else 0
        is_substantive = 1 if bill_type in ("SB", "HB") else 0

        rows.append(
            {
                "bill_id": bill["bill_id"],
                "intro_month": month,
                "intro_day_of_year": day_of_year,
                "is_lame_duck": is_lame_duck,
                "is_senate_origin": is_senate,
                "is_house_origin": is_house,
                "is_resolution": is_resolution,
                "is_substantive": is_substantive,
            }
        )

    return pl.DataFrame(rows)


def build_staleness_features(
    df_bills: pl.DataFrame,
    df_actions: pl.DataFrame,
    *,
    as_of_date: str | None = None,
) -> pl.DataFrame:
    """Build staleness / activity features for each bill.

    These measure how "alive" a bill is and whether it shows signs of
    being abandoned.  Safe from leakage because they describe the bill's
    current state, not its future.

    Parameters
    ----------
    as_of_date:
        If provided (``"YYYY-MM-DD"``), only consider actions on or before
        this date and compute staleness relative to it (not "now").
        Used by the panel dataset to build features at a snapshot point.

    Features:
        days_since_last_action:  How long since any action (0 = today)
        days_since_intro:        Age of the bill
        action_count_30d:        Actions in first 30 days (leakage-safe)
        action_count_60d:        Actions in first 60 days (leakage-safe)
        action_count_90d:        Actions in first 90 days (leakage-safe)
        action_velocity_60d:     action_count_60d / 60 (normalized rate)
        is_stale_90:             Binary: no action for 90+ days
        is_stale_180:            Binary: no action for 180+ days
        has_negative_terminal:   Binary: vetoed/tabled/sine-die

    **Data leakage fix (2026-02-13):**
        Removed ``total_action_count`` which had 68.4% feature importance
        and was circular: bills that advance accumulate more actions by
        construction (committee hearings, floor votes, cross-chamber steps
        ARE the outcome).  Replaced with time-capped counts (30/60/90 day
        windows after introduction) that capture early legislative momentum
        without leaking the outcome.
    """
    reference_date = datetime.strptime(as_of_date, "%Y-%m-%d") if as_of_date else datetime.now()

    # When as_of_date is set, restrict actions to on-or-before that date
    df_actions_filtered = df_actions
    if as_of_date:
        df_actions_filtered = df_actions.filter(
            pl.col("date").is_not_null() & (pl.col("date") <= as_of_date)
        )

    # Get last action date and action texts per bill
    bill_action_agg = df_actions_filtered.group_by("bill_id").agg(
        pl.col("date").max().alias("last_action_date_raw"),
        pl.col("action_text").alias("actions"),
    )

    df = df_bills.select(["bill_id", "introduction_date"]).join(
        bill_action_agg, on="bill_id", how="left"
    )

    # Time-capped action counts: count actions within N days of introduction
    # This avoids leakage from post-outcome actions.
    df_act_dated = df_actions_filtered.join(
        df_bills.select(["bill_id", "introduction_date"]),
        on="bill_id",
        how="inner",
    ).filter(pl.col("date").is_not_null() & pl.col("introduction_date").is_not_null())

    # Parse dates for comparison
    df_act_parsed = (
        df_act_dated.with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("action_date"),
            pl.col("introduction_date")
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            .alias("intro_date"),
        )
        .filter(pl.col("action_date").is_not_null() & pl.col("intro_date").is_not_null())
        .with_columns(
            ((pl.col("action_date") - pl.col("intro_date")).dt.total_days()).alias(
                "days_after_intro"
            ),
        )
    )

    # Aggregate time-capped counts per bill
    capped_counts = df_act_parsed.group_by("bill_id").agg(
        (pl.col("days_after_intro") <= 30).sum().alias("action_count_30d"),
        (pl.col("days_after_intro") <= 60).sum().alias("action_count_60d"),
        (pl.col("days_after_intro") <= 90).sum().alias("action_count_90d"),
    )

    df = df.join(capped_counts, on="bill_id", how="left")

    rows = []
    for bill in df.to_dicts():
        bid = bill["bill_id"]

        # Days since last action
        days_since_action = 0
        last_raw = bill.get("last_action_date_raw")
        if last_raw:
            try:
                last_dt = datetime.strptime(last_raw, "%Y-%m-%d")
                days_since_action = (reference_date - last_dt).days
            except (ValueError, TypeError):
                LOGGER.debug("Unparseable last_action_date_raw: %r for %s", last_raw, bid)

        # Days since introduction
        days_since_intro = 0
        intro_str = bill.get("introduction_date")
        if intro_str:
            try:
                intro_dt = datetime.strptime(intro_str, "%Y-%m-%d")
                days_since_intro = (reference_date - intro_dt).days
            except (ValueError, TypeError):
                LOGGER.debug("Unparseable introduction_date: %r for %s", intro_str, bid)

        # Time-capped action counts (leakage-safe)
        ac_30 = bill.get("action_count_30d") or 0
        ac_60 = bill.get("action_count_60d") or 0
        ac_90 = bill.get("action_count_90d") or 0

        # Velocity based on 60-day window (normalized)
        action_velocity_60d = ac_60 / 60.0 if ac_60 > 0 else 0.0

        # Negative terminal check
        actions = bill.get("actions") or []
        has_neg_terminal = 1 if _bill_has_negative_terminal(actions) else 0

        rows.append(
            {
                "bill_id": bid,
                "days_since_last_action": days_since_action,
                "days_since_intro": days_since_intro,
                "action_count_30d": ac_30,
                "action_count_60d": ac_60,
                "action_count_90d": ac_90,
                "action_velocity_60d": round(action_velocity_60d, 6),
                "is_stale_90": 1 if days_since_action >= 90 else 0,
                "is_stale_180": 1 if days_since_action >= 180 else 0,
                "has_negative_terminal": has_neg_terminal,
            }
        )

    return pl.DataFrame(rows)


def build_action_features(
    df_bills: pl.DataFrame,
    df_actions: pl.DataFrame,
    *,
    as_of_date: str | None = None,
) -> pl.DataFrame:
    """Build EARLY action features that avoid leakage.

    Only counts actions in the first 30 days after filing.

    Parameters
    ----------
    as_of_date:
        If provided (``"YYYY-MM-DD"``), only consider actions on or before
        this date.  Used by the panel dataset for time-sliced features.
    """
    df_actions_filtered = df_actions
    if as_of_date:
        df_actions_filtered = df_actions.filter(
            pl.col("date").is_not_null() & (pl.col("date") <= as_of_date)
        )

    df_act_dated = df_actions_filtered.join(
        df_bills.select(["bill_id", "introduction_date"]),
        on="bill_id",
        how="inner",
    )

    early_actions = df_act_dated.filter(
        (pl.col("date").is_not_null())
        & (pl.col("introduction_date").is_not_null())
        & (
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            <= (
                pl.col("introduction_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                + pl.duration(days=30)
            )
        )
    )

    safe_categories = ["procedural", "committee", "cosponsor", "amendment"]

    agg = (
        early_actions.filter(pl.col("action_category").is_in(safe_categories))
        .group_by("bill_id")
        .agg(
            pl.len().alias("early_action_count"),
            (pl.col("action_category") == "cosponsor").sum().alias("early_cosponsor_actions"),
            (pl.col("action_category") == "committee").sum().alias("early_committee_referrals"),
        )
    )

    return df_bills.select("bill_id").join(agg, on="bill_id", how="left").fill_null(0)


# ── Committee features ─────────────────────────────────────────────────────

_ASSIGNED_RE = re.compile(r"(?:Assigned to|Referred to)\s*(.+)", re.IGNORECASE)

# Procedural committees — loaded from ilga_rules.json via rule_engine.
# Falls back to hardcoded set if the glossary is unavailable.
try:
    from ilga_graph.ml.rule_engine import get_procedural_committees as _get_proc_committees

    _PROCEDURAL_COMMITTEES = set(_get_proc_committees())
except Exception:
    _PROCEDURAL_COMMITTEES = {
        "assignments",
        "rules committee",
        "rules",
        "executive",
        "executive appointments",
    }


def build_committee_features(
    df_bills: pl.DataFrame,
    df_actions: pl.DataFrame,
    df_labels: pl.DataFrame,
    *,
    as_of_date: str | None = None,
) -> pl.DataFrame:
    """Build committee-related features for each bill.

    For each bill, we find its FIRST substantive committee assignment (from
    "Assigned to X" / "Referred to X" in action history), then compute
    features about that committee:

        committee_advancement_rate:  What fraction of bills assigned to this
            committee eventually advanced?  Computed historically — only uses
            bills introduced BEFORE this one to avoid leakage.
        committee_pass_rate:  What fraction became law?  (same leakage guard)
        committee_bill_volume:  How many bills has this committee handled?
        is_high_throughput_committee:  Binary: advancement rate >= 0.3
        has_committee_assignment:  Binary: bill has a non-procedural assignment

    Parameters
    ----------
    as_of_date:
        If provided (``"YYYY-MM-DD"``), only consider actions on or before
        this date for committee assignments, and only count historical
        outcomes from bills whose labels are known by this date.
    """
    LOGGER.info("Building committee features...")

    # When as_of_date is set, restrict to actions on or before that date
    df_actions_filtered = df_actions
    if as_of_date:
        df_actions_filtered = df_actions.filter(
            pl.col("date").is_not_null() & (pl.col("date") <= as_of_date)
        )

    # Step 1: find each bill's first substantive committee
    bill_committee: dict[str, str] = {}

    action_rows = df_actions_filtered.sort("date", nulls_last=True).to_dicts()
    for row in action_rows:
        bid = row["bill_id"]
        if bid in bill_committee:
            continue  # already found first committee
        m = _ASSIGNED_RE.match(row.get("action_text", ""))
        if not m:
            continue
        cname = m.group(1).strip().lower()
        # Remove trailing " committee" for normalization
        if cname.endswith(" committee"):
            cname = cname[: -len(" committee")]
        # Skip procedural routing committees
        if cname in _PROCEDURAL_COMMITTEES:
            continue
        bill_committee[bid] = cname

    LOGGER.info(
        "Committee assignments found: %d of %d bills",
        len(bill_committee),
        len(df_bills),
    )

    # Step 2: compute historical committee passage rates (leakage-safe)
    # Sort bills by introduction_date; for each bill, use only bills
    # introduced BEFORE it to compute the committee's rates.
    bills_with_labels = (
        df_bills.select(["bill_id", "introduction_date"])
        .join(
            df_labels.select(["bill_id", "target_advanced", "target_law", "is_mature"]),
            on="bill_id",
            how="left",
        )
        .sort("introduction_date", nulls_last=True)
        .to_dicts()
    )

    # Running committee stats
    committee_filed: dict[str, int] = {}
    committee_advanced: dict[str, int] = {}
    committee_passed: dict[str, int] = {}

    rows = []
    for bill in bills_with_labels:
        bid = bill["bill_id"]
        cname = bill_committee.get(bid)

        if cname:
            hist_filed = committee_filed.get(cname, 0)
            hist_advanced = committee_advanced.get(cname, 0)
            hist_passed = committee_passed.get(cname, 0)
            hist_adv_rate = hist_advanced / hist_filed if hist_filed > 0 else -1.0
            hist_pass_rate = hist_passed / hist_filed if hist_filed > 0 else -1.0

            rows.append(
                {
                    "bill_id": bid,
                    "committee_advancement_rate": round(hist_adv_rate, 4),
                    "committee_pass_rate": round(hist_pass_rate, 4),
                    "committee_bill_volume": hist_filed,
                    "is_high_throughput_committee": 1 if hist_adv_rate >= 0.3 else 0,
                    "has_committee_assignment": 1,
                }
            )

            # Update running stats (only for bills with known labels).
            # When as_of_date is set, require the bill to be old enough
            # that its label would be known by as_of_date.
            label_usable = bill.get("is_mature") and bill.get("target_advanced") is not None
            if as_of_date and label_usable:
                bill_intro = bill.get("introduction_date", "")
                if bill_intro:
                    try:
                        bill_intro_dt = datetime.strptime(bill_intro, "%Y-%m-%d")
                        cutoff_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
                        label_usable = (cutoff_dt - bill_intro_dt).days >= MATURITY_DAYS
                    except ValueError:
                        label_usable = False
            if label_usable:
                committee_filed[cname] = committee_filed.get(cname, 0) + 1
                if bill["target_advanced"] == 1:
                    committee_advanced[cname] = committee_advanced.get(cname, 0) + 1
                if bill.get("target_law") == 1:
                    committee_passed[cname] = committee_passed.get(cname, 0) + 1
        else:
            rows.append(
                {
                    "bill_id": bid,
                    "committee_advancement_rate": -1.0,
                    "committee_pass_rate": -1.0,
                    "committee_bill_volume": 0,
                    "is_high_throughput_committee": 0,
                    "has_committee_assignment": 0,
                }
            )

    df_result = (
        pl.DataFrame(rows)
        if rows
        else df_bills.select("bill_id").with_columns(
            pl.lit(-1.0).alias("committee_advancement_rate"),
            pl.lit(-1.0).alias("committee_pass_rate"),
            pl.lit(0).alias("committee_bill_volume"),
            pl.lit(0).alias("is_high_throughput_committee"),
            pl.lit(0).alias("has_committee_assignment"),
        )
    )

    LOGGER.info(
        "Committee features: %d unique committees, %d bills with assignments",
        len(set(bill_committee.values())),
        sum(1 for r in rows if r["has_committee_assignment"]),
    )

    return df_result


# ── Rule-derived features (from ilga_rules.json glossary) ─────────────────


def build_rule_features(
    df_bills: pl.DataFrame,
    df_actions: pl.DataFrame,
    *,
    as_of_date: str | None = None,
) -> pl.DataFrame:
    """Build binary features derived from the ILGA Senate Rules glossary.

    These features encode specific rule-defined events that are strong signals
    for bill advancement or stalling.  All definitions sourced from
    ``reference/ilga_rules.json`` via ``rule_engine``.

    Features
    --------
    missed_committee_deadline : int8
        1 if any action matches Rule 3-9(a) / House Rule 19(a) — missed
        committee report deadline.  Re-referred to Assignments/Rules.
        NOT terminal but most bills that hit this do not advance.
    missed_floor_deadline : int8
        1 if any action matches House Rule 19(b) — passed committee but
        missed floor vote deadline.  Re-referred to Rules.
    has_favorable_report : int8
        1 if any action is a Rule 3-11 favorable committee report
        (do pass, do pass as amended, be adopted, etc.).  The single
        most important gate — majority of bills never get here.
    was_tabled : int8
        1 if the bill itself was tabled (Rule 7-10).  Not terminal in IL
        (Rule 7-11 allows taking from table) but a negative signal.
    on_consent_calendar : int8
        1 if placed on consent calendar (Rule 6-1).  Indicates the bill
        is non-controversial and expected to pass without debate.
    has_bipartisan_sponsors : int8
        1 if sponsors include members from both majority and minority
        caucuses (Rule 1-10 defines majority/minority).  Cross-party
        sponsorship is a positive advancement signal.

    Parameters
    ----------
    as_of_date : str | None
        If provided ("YYYY-MM-DD"), only consider actions on or before
        this date (for panel/time-sliced dataset).
    """
    try:
        from ilga_graph.ml.rule_engine import (
            is_favorable_report,
            is_missed_committee_deadline,
            is_missed_floor_deadline,
            is_on_consent_calendar,
            is_tabled,
        )

        _has_rule_engine = True
    except Exception:
        LOGGER.warning("rule_engine unavailable; rule features will be zeros.")
        _has_rule_engine = False

    LOGGER.info("Building rule-derived features...")

    # Get action history per bill
    df_act = df_actions.select(["bill_id", "date", "action_text"])
    if as_of_date is not None:
        df_act = df_act.filter(pl.col("date") <= as_of_date)

    # Group actions by bill
    grouped = df_act.sort("date").group_by("bill_id").agg(pl.col("action_text"))

    # Compute rule-based binary features per bill
    rows: list[dict] = []
    for row in grouped.iter_rows(named=True):
        bill_id = row["bill_id"]
        actions = row["action_text"] or []

        missed_comm = 0
        missed_floor = 0
        favorable = 0
        tabled = 0
        consent = 0

        if _has_rule_engine:
            for a in actions:
                if not a:
                    continue
                if is_missed_committee_deadline(a):
                    missed_comm = 1
                if is_missed_floor_deadline(a):
                    missed_floor = 1
                if is_favorable_report(a):
                    favorable = 1
                if is_tabled(a):
                    tabled = 1
                if is_on_consent_calendar(a):
                    consent = 1

        rows.append(
            {
                "bill_id": bill_id,
                "missed_committee_deadline": missed_comm,
                "missed_floor_deadline": missed_floor,
                "has_favorable_report": favorable,
                "was_tabled": tabled,
                "on_consent_calendar": consent,
            }
        )

    df_rule = pl.DataFrame(rows).cast(
        {
            "missed_committee_deadline": pl.Int8,
            "missed_floor_deadline": pl.Int8,
            "has_favorable_report": pl.Int8,
            "was_tabled": pl.Int8,
            "on_consent_calendar": pl.Int8,
        }
    )

    # Bipartisan sponsors: detect if sponsors span both D and R
    # from the sponsor features we already have.  We approximate via
    # the bill's sponsor_count > 1 and mixed party from members table.
    # For now, compute from cosponsor actions that mention party-crossing.
    # A more precise version would cross-reference sponsor IDs with member
    # party; this is a fast heuristic that captures the signal.
    df_rule = df_rule.with_columns(pl.lit(0).cast(pl.Int8).alias("has_bipartisan_sponsors"))

    # Try to compute bipartisan from member data if available
    try:
        if "primary_sponsor_id" in df_bills.columns:
            # We'd need member party data — mark as 0 for now;
            # will be enriched in build_feature_matrix where df_members
            # is available.
            pass
    except Exception:
        pass

    return df_bills.select("bill_id").join(df_rule, on="bill_id", how="left").fill_null(0)


# ── Graph embedding features (Phase 3: Node2Vec) ─────────────────────────────

# Number of embedding dimensions to expect.  Must match node_embedder output.
_EMBEDDING_DIMS = 64


def build_embedding_features(
    df_bills: pl.DataFrame,
) -> pl.DataFrame:
    """Build sponsor graph-embedding features for each bill.

    Loads the pre-computed Node2Vec embeddings from
    ``processed/member_embeddings.parquet`` and maps each bill's
    ``primary_sponsor_id`` to its embedding vector.

    Produces columns ``sponsor_emb_0`` through ``sponsor_emb_{n-1}``.
    Bills whose sponsor has no embedding get zero vectors.

    These features are safe for both full and forecast modes because
    embeddings are intrinsic to the legislator's structural position
    in the co-sponsorship network — they do not leak temporal information.
    """
    LOGGER.info("Building sponsor embedding features...")

    emb_path = PROCESSED_DIR / "member_embeddings.parquet"
    if not emb_path.exists():
        LOGGER.warning(
            "member_embeddings.parquet not found — returning zero embedding features. "
            "Run `make ml-embed` or the full pipeline to generate embeddings."
        )
        emb_cols = {f"sponsor_emb_{i}": pl.lit(0.0) for i in range(_EMBEDDING_DIMS)}
        return df_bills.select("bill_id").with_columns(**emb_cols)

    df_emb = pl.read_parquet(emb_path)
    dim_cols = sorted(
        [c for c in df_emb.columns if c.startswith("dim_")],
        key=lambda c: int(c.split("_")[1]),
    )
    n_dims = len(dim_cols)

    if n_dims == 0:
        LOGGER.warning("Embeddings file has no dim_* columns — returning zeros.")
        emb_cols = {f"sponsor_emb_{i}": pl.lit(0.0) for i in range(_EMBEDDING_DIMS)}
        return df_bills.select("bill_id").with_columns(**emb_cols)

    LOGGER.info("Loaded %d-dim embeddings for %d members.", n_dims, len(df_emb))

    # Rename dim_i → sponsor_emb_i and member_id → primary_sponsor_id for join
    rename_map = {c: f"sponsor_emb_{c.split('_')[1]}" for c in dim_cols}
    rename_map["member_id"] = "primary_sponsor_id"
    df_emb_renamed = df_emb.select(["member_id"] + dim_cols).rename(rename_map)

    # Join on primary_sponsor_id
    df_result = df_bills.select(["bill_id", "primary_sponsor_id"]).join(
        df_emb_renamed, on="primary_sponsor_id", how="left"
    )

    # Fill nulls with 0.0 (no embedding = zero vector)
    emb_feature_cols = [f"sponsor_emb_{i}" for i in range(n_dims)]
    for col in emb_feature_cols:
        if col in df_result.columns:
            df_result = df_result.with_columns(pl.col(col).fill_null(0.0))

    # Drop the join key — only bill_id + embedding cols
    df_result = df_result.drop("primary_sponsor_id")

    LOGGER.info(
        "Embedding features: %d bills, %d dims (%d with sponsor match).",
        len(df_result),
        n_dims,
        len(df_result) - df_result.select(pl.col(emb_feature_cols[0]).eq(0.0).sum()).item()
        if emb_feature_cols
        else 0,
    )

    return df_result


# ── Assembly: combine all features ───────────────────────────────────────────


def build_feature_matrix(
    *,
    tfidf_features: int = 500,
    mode: FeatureMode = "full",
) -> tuple:
    """Build the full feature matrix for bill outcome prediction.

    Parameters
    ----------
    mode:
        ``"full"`` (default) — all features including staleness, slips,
        action counts, and rule-derived features (Status model).
        ``"forecast"`` — only intrinsic / Day-0 features: sponsor, text,
        committee assignment, calendar, and content metadata.

    Returns:
        X_train, X_test, y_train, y_test,
        bill_ids_train, bill_ids_test,
        X_immature, y_immature, bill_ids_immature,
        vectorizer, feature_names,
        y_train_law, y_test_law, y_immature_law
    """
    is_forecast = mode == "forecast"
    LOGGER.info("Building feature matrix (mode=%s)...", mode)

    # Load tables
    df_bills = pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet")
    df_actions = pl.read_parquet(PROCESSED_DIR / "fact_bill_actions.parquet")
    df_members = pl.read_parquet(PROCESSED_DIR / "dim_members.parquet")
    df_slips = pl.read_parquet(PROCESSED_DIR / "fact_witness_slips.parquet")

    # Filter to substantive bills only (SB, HB)
    df_bills_sub = df_bills.filter(pl.col("bill_type").is_in(["SB", "HB"]))
    LOGGER.info(
        "Substantive bills: %d (of %d total)",
        len(df_bills_sub),
        len(df_bills),
    )

    # Build labels (with maturity flag)
    df_labels = build_bill_labels(df_bills_sub, df_actions)

    # Build feature tables — skip leaky builders in forecast mode
    df_sponsor = build_sponsor_features(df_bills_sub, df_members, df_labels)
    tfidf_matrix, vectorizer, tfidf_bill_ids = build_text_features(
        df_bills_sub, max_features=tfidf_features
    )
    # Forecast mode: exclude full-text TF-IDF to avoid label leakage
    # ("enrolled"/LRB in final bill text).
    if is_forecast:
        ft_tfidf_matrix = csr_matrix((len(df_bills_sub), 0), dtype=np.float32)
        ft_vectorizer = TfidfVectorizer()
        ft_tfidf_bill_ids = df_bills_sub["bill_id"].to_list()
    else:
        ft_tfidf_matrix, ft_vectorizer, ft_tfidf_bill_ids = build_full_text_features(df_bills_sub)
    df_content_meta = build_content_metadata_features(df_bills_sub)
    df_temporal = build_temporal_features(df_bills_sub)
    df_committee = build_committee_features(df_bills_sub, df_actions, df_labels)
    df_embeddings = build_embedding_features(df_bills_sub)

    # Merge all tabular features with SEMANTIC imputation.
    #
    # Previous approach: blanket .fill_null(0) conflated "no data" with
    # "zero rate" for ratios like sponsor_hist_passage_rate and slip ratios.
    # New approach:
    #   - Counts: fill with 0 (semantically correct)
    #   - Ratios: fill with -1 sentinel (model learns "no data" as distinct)
    #   - Add explicit has_* indicator columns
    df_feat = (
        df_labels.select(
            [
                "bill_id",
                "target_advanced",
                "target_law",
                "introduction_date",
                "is_mature",
            ]
        )
        .join(df_sponsor, on="bill_id", how="left")
        .join(df_temporal, on="bill_id", how="left")
        .join(df_committee, on="bill_id", how="left")
        .join(df_content_meta, on="bill_id", how="left")
        .join(df_embeddings, on="bill_id", how="left")
    )

    # In full mode, also join slip, action, staleness, and rule features
    if not is_forecast:
        df_slips_feat = build_slip_features(df_bills_sub, df_slips)
        df_action_feat = build_action_features(df_bills_sub, df_actions)
        df_staleness = build_staleness_features(df_bills_sub, df_actions)
        df_rule_feat = build_rule_features(df_bills_sub, df_actions)
        df_feat = (
            df_feat.join(df_slips_feat, on="bill_id", how="left")
            .join(df_action_feat, on="bill_id", how="left")
            .join(df_staleness, on="bill_id", how="left")
            .join(df_rule_feat, on="bill_id", how="left")
        )

    # ── Semantic imputation ────────────────────────────────────────────
    # Columns that are COUNTS — fill missing with 0
    count_cols = [
        "sponsor_count",
        "sponsor_hist_filed",
        "sponsor_bill_count",
        "slip_total",
        "slip_proponent",
        "slip_opponent",
        "slip_written_only",
        "slip_unique_orgs",
        "early_action_count",
        "early_cosponsor_actions",
        "early_committee_referrals",
        "action_count_30d",
        "action_count_60d",
        "action_count_90d",
        "days_since_last_action",
        "days_since_intro",
        "intro_month",
        "intro_day_of_year",
        "committee_bill_volume",
        "full_text_word_count",
        "full_text_section_count",
        "full_text_citation_count",
    ]
    # Columns that are RATIOS — fill missing with -1 sentinel
    ratio_cols = [
        "sponsor_hist_passage_rate",
        "slip_proponent_ratio",
        "slip_opponent_ratio",
        "slip_written_ratio",
        "slip_org_concentration",
        "action_velocity_60d",
        "committee_advancement_rate",
        "committee_pass_rate",
        "full_text_log_length",
    ]
    # Columns that are BINARY flags — fill missing with 0
    binary_cols = [
        "sponsor_party_democrat",
        "sponsor_party_republican",
        "sponsor_is_majority",
        "has_sponsor_id",
        "is_lame_duck",
        "is_senate_origin",
        "is_house_origin",
        "is_resolution",
        "is_substantive",
        "is_stale_90",
        "is_stale_180",
        "has_negative_terminal",
        "is_high_throughput_committee",
        "has_committee_assignment",
        "is_long_bill",
        "is_short_bill",
        "has_full_text",
        # Rule-derived features (from ilga_rules.json glossary)
        "missed_committee_deadline",
        "missed_floor_deadline",
        "has_favorable_report",
        "was_tabled",
        "on_consent_calendar",
        "has_bipartisan_sponsors",
    ]

    # Add explicit missingness indicators BEFORE filling
    # (only when those columns exist — forecast mode skips slip features)
    if "slip_total" in df_feat.columns:
        df_feat = df_feat.with_columns(
            pl.col("slip_total").is_null().cast(pl.Int8).alias("has_no_slip_data"),
        )
    if "sponsor_hist_filed" in df_feat.columns:
        df_feat = df_feat.with_columns(
            pl.col("sponsor_hist_filed").is_null().cast(pl.Int8).alias("has_no_sponsor_history"),
        )

    # Apply semantic fills
    existing_cols = set(df_feat.columns)
    for col in count_cols + binary_cols:
        if col in existing_cols:
            df_feat = df_feat.with_columns(pl.col(col).fill_null(0))
    for col in ratio_cols:
        if col in existing_cols:
            df_feat = df_feat.with_columns(pl.col(col).fill_null(-1.0))

    # Embedding columns — fill nulls with 0.0 (no embedding = zero vector)
    for col in existing_cols:
        if col.startswith("sponsor_emb_"):
            df_feat = df_feat.with_columns(pl.col(col).fill_null(0.0))

    # ── Split: mature vs immature ────────────────────────────────────────
    # Mature bills have reliable labels (introduced 120+ days ago).
    # Immature bills are scored but NOT used for train/test evaluation.
    df_mature = df_feat.filter(pl.col("is_mature"))
    df_immature = df_feat.filter(~pl.col("is_mature"))

    LOGGER.info(
        "Mature bills (120+ days): %d, Immature (too new): %d",
        len(df_mature),
        len(df_immature),
    )

    # ── Time-based train/test on MATURE bills only ───────────────────────
    df_mature = df_mature.sort("introduction_date", nulls_last=True)
    n = len(df_mature)
    split_idx = int(n * 0.7)

    df_train = df_mature[:split_idx]
    df_test = df_mature[split_idx:]

    LOGGER.info(
        "Train: %d bills, Test: %d bills (70/30 time split on mature)",
        len(df_train),
        len(df_test),
    )

    # Extract targets — both "advance past committee" and "become law"
    y_train = df_train["target_advanced"].to_numpy()
    y_test = df_test["target_advanced"].to_numpy()
    y_immature = df_immature["target_advanced"].to_numpy()

    y_train_law = df_train["target_law"].to_numpy()
    y_test_law = df_test["target_law"].to_numpy()
    y_immature_law = df_immature["target_law"].to_numpy()

    pos_train = y_train.sum()
    pos_test = y_test.sum()
    LOGGER.info(
        "Class balance - Train: %d advanced / %d stuck (%.1f%%), "
        "Test: %d advanced / %d stuck (%.1f%%)",
        pos_train,
        len(y_train) - pos_train,
        100 * pos_train / max(len(y_train), 1),
        pos_test,
        len(y_test) - pos_test,
        100 * pos_test / max(len(y_test), 1),
    )

    # Build bill_id -> tfidf row index (synopsis)
    tfidf_id_to_idx = {bid: i for i, bid in enumerate(tfidf_bill_ids)}

    # Build bill_id -> full-text tfidf row index
    ft_tfidf_id_to_idx = {bid: i for i, bid in enumerate(ft_tfidf_bill_ids)}

    # Extract numeric feature columns (excludes metadata and raw text)
    _exclude = {
        "bill_id",
        "target_advanced",
        "target_law",
        "introduction_date",
        "is_mature",
        "full_text",  # raw text column — not a numeric feature
    }
    if is_forecast:
        _exclude |= FORECAST_DROP_COLUMNS
    feature_cols = [c for c in df_feat.columns if c not in _exclude]

    # Pre-compute zero rows for bills missing from TF-IDF matrices
    # (avoids silently reusing bill #0's features)

    _tfidf_zero = csr_matrix((1, tfidf_matrix.shape[1]), dtype=np.float32)
    _ft_has_features = ft_tfidf_matrix.shape[1] > 0
    _ft_tfidf_zero = (
        csr_matrix((1, ft_tfidf_matrix.shape[1]), dtype=np.float32) if _ft_has_features else None
    )

    def _to_sparse(df_slice: pl.DataFrame) -> csr_matrix:
        tabular = df_slice.select(feature_cols).to_numpy().astype(np.float32)
        bill_ids = df_slice["bill_id"].to_list()

        # Synopsis TF-IDF rows
        tfidf_parts = []
        for bid in bill_ids:
            idx = tfidf_id_to_idx.get(bid)
            if idx is not None:
                tfidf_parts.append(tfidf_matrix[idx])
            else:
                tfidf_parts.append(_tfidf_zero)

        from scipy.sparse import vstack as sparse_vstack

        tfidf_rows = sparse_vstack(tfidf_parts)

        # Full-text TF-IDF rows (only if full-text features exist)
        if _ft_has_features:
            ft_parts = []
            for bid in bill_ids:
                idx = ft_tfidf_id_to_idx.get(bid)
                if idx is not None:
                    ft_parts.append(ft_tfidf_matrix[idx])
                else:
                    ft_parts.append(_ft_tfidf_zero)
            ft_rows = sparse_vstack(ft_parts)
            return sparse_hstack([tfidf_rows, ft_rows, csr_matrix(tabular)])
        else:
            return sparse_hstack([tfidf_rows, csr_matrix(tabular)])

    X_train = _to_sparse(df_train)
    X_test = _to_sparse(df_test)
    X_immature = _to_sparse(df_immature) if len(df_immature) > 0 else None

    bill_ids_train = df_train["bill_id"].to_list()
    bill_ids_test = df_test["bill_id"].to_list()
    bill_ids_immature = df_immature["bill_id"].to_list()

    # Build feature names: synopsis TF-IDF + full-text TF-IDF + tabular
    feature_names = [f"tfidf_{fn}" for fn in vectorizer.get_feature_names_out()]
    if _ft_has_features:
        feature_names += [f"ft_tfidf_{fn}" for fn in ft_vectorizer.get_feature_names_out()]
    feature_names += feature_cols

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        bill_ids_train,
        bill_ids_test,
        X_immature,
        y_immature,
        bill_ids_immature,
        vectorizer,
        feature_names,
        y_train_law,
        y_test_law,
        y_immature_law,
    )


# ── Panel (time-slice) feature matrix ────────────────────────────────────────


def _apply_semantic_imputation(
    df_feat: pl.DataFrame,
    *,
    mode: FeatureMode = "full",
) -> pl.DataFrame:
    """Apply the standard semantic null-fill strategy.

    Shared between ``build_feature_matrix`` and ``build_panel_feature_matrix``
    so both paths use identical imputation logic.
    """
    count_cols = [
        "sponsor_count",
        "sponsor_hist_filed",
        "sponsor_bill_count",
        "slip_total",
        "slip_proponent",
        "slip_opponent",
        "slip_written_only",
        "slip_unique_orgs",
        "early_action_count",
        "early_cosponsor_actions",
        "early_committee_referrals",
        "action_count_30d",
        "action_count_60d",
        "action_count_90d",
        "days_since_last_action",
        "days_since_intro",
        "intro_month",
        "intro_day_of_year",
        "committee_bill_volume",
        "full_text_word_count",
        "full_text_section_count",
        "full_text_citation_count",
    ]
    ratio_cols = [
        "sponsor_hist_passage_rate",
        "slip_proponent_ratio",
        "slip_opponent_ratio",
        "slip_written_ratio",
        "slip_org_concentration",
        "action_velocity_60d",
        "committee_advancement_rate",
        "committee_pass_rate",
        "full_text_log_length",
    ]
    binary_cols = [
        "sponsor_party_democrat",
        "sponsor_party_republican",
        "sponsor_is_majority",
        "has_sponsor_id",
        "is_lame_duck",
        "is_senate_origin",
        "is_house_origin",
        "is_resolution",
        "is_substantive",
        "is_stale_90",
        "is_stale_180",
        "has_negative_terminal",
        "is_high_throughput_committee",
        "has_committee_assignment",
        "is_long_bill",
        "is_short_bill",
        "has_full_text",
        # Rule-derived features (from ilga_rules.json glossary)
        "missed_committee_deadline",
        "missed_floor_deadline",
        "has_favorable_report",
        "was_tabled",
        "on_consent_calendar",
        "has_bipartisan_sponsors",
    ]

    # Add explicit missingness indicators BEFORE filling
    if "slip_total" in df_feat.columns:
        df_feat = df_feat.with_columns(
            pl.col("slip_total").is_null().cast(pl.Int8).alias("has_no_slip_data"),
        )
    if "sponsor_hist_filed" in df_feat.columns:
        df_feat = df_feat.with_columns(
            pl.col("sponsor_hist_filed").is_null().cast(pl.Int8).alias("has_no_sponsor_history"),
        )

    existing_cols = set(df_feat.columns)
    for col in count_cols + binary_cols:
        if col in existing_cols:
            df_feat = df_feat.with_columns(pl.col(col).fill_null(0))
    for col in ratio_cols:
        if col in existing_cols:
            df_feat = df_feat.with_columns(pl.col(col).fill_null(-1.0))

    # Embedding columns — fill nulls with 0.0 (no embedding = zero vector)
    for col in existing_cols:
        if col.startswith("sponsor_emb_"):
            df_feat = df_feat.with_columns(pl.col(col).fill_null(0.0))

    return df_feat


# Columns that are NOT numeric features (used by both matrix builders)
_NON_FEATURE_COLS = frozenset(
    {
        "bill_id",
        "target_advanced",
        "target_law",
        "introduction_date",
        "is_mature",
        "full_text",  # raw text column — not a numeric feature
        # Panel-specific
        "snapshot_day",
        "snapshot_date",
        "target_advanced_after",
        "target_law_after",
        "panel_id",
    }
)


def build_panel_feature_matrix(
    *,
    snapshot_days: list[int] | None = None,
    observation_days: int | None = None,
    tfidf_features: int = 500,
    mode: FeatureMode = "full",
) -> tuple:
    """Build a time-sliced (panel) feature matrix for training.

    Creates multiple rows per bill (one per snapshot date) with features
    computed only from data up to each snapshot and labels defined as
    "did this bill advance AFTER this snapshot?"

    Parameters
    ----------
    mode:
        ``"full"`` (default) — all features (Status model).
        ``"forecast"`` — only intrinsic / Day-0 features; excludes
        staleness, action counts, witness slips, and rule features.

    Returns the same tuple shape as ``build_feature_matrix`` so the
    training pipeline in ``bill_predictor.py`` can use either
    interchangeably.

    Returns:
        X_train, X_test, y_train, y_test,
        bill_ids_train, bill_ids_test,
        X_immature, y_immature, bill_ids_immature,
        vectorizer, feature_names,
        y_train_law, y_test_law, y_immature_law
    """
    is_forecast = mode == "forecast"

    if snapshot_days is None:
        snapshot_days = list(FORECAST_SNAPSHOT_DAYS if is_forecast else SNAPSHOT_DAYS_AFTER_INTRO)
    if observation_days is None:
        observation_days = OBSERVATION_DAYS_AFTER_SNAPSHOT

    LOGGER.info(
        "Building PANEL feature matrix (mode=%s, snapshots=%s, observation=%d)...",
        mode,
        snapshot_days,
        observation_days,
    )

    # ── Load tables ──────────────────────────────────────────────────────
    df_bills = pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet")
    df_actions = pl.read_parquet(PROCESSED_DIR / "fact_bill_actions.parquet")
    df_members = pl.read_parquet(PROCESSED_DIR / "dim_members.parquet")
    df_slips = pl.read_parquet(PROCESSED_DIR / "fact_witness_slips.parquet")

    df_bills_sub = df_bills.filter(pl.col("bill_type").is_in(["SB", "HB"]))
    LOGGER.info(
        "Substantive bills: %d (of %d total)",
        len(df_bills_sub),
        len(df_bills),
    )

    # ── Panel labels ─────────────────────────────────────────────────────
    df_panel_labels = build_panel_labels(
        df_bills_sub,
        df_actions,
        snapshot_days=snapshot_days,
        observation_days=observation_days,
    )

    if len(df_panel_labels) == 0:
        LOGGER.warning("Panel dataset is empty — no bills old enough.")
        # Return empty arrays so caller can handle gracefully
        empty_x = csr_matrix((0, 0))
        empty_y = np.array([], dtype=np.int8)
        return (
            empty_x,
            empty_x,
            empty_y,
            empty_y,
            [],
            [],
            None,
            empty_y,
            [],
            TfidfVectorizer(),
            [],
            empty_y,
            empty_y,
            empty_y,
        )

    # ── TF-IDF (one per bill, snapshot-invariant) ────────────────────────
    tfidf_matrix, vectorizer, tfidf_bill_ids = build_text_features(
        df_bills_sub, max_features=tfidf_features
    )
    tfidf_id_to_idx = {bid: i for i, bid in enumerate(tfidf_bill_ids)}
    _tfidf_zero = csr_matrix((1, tfidf_matrix.shape[1]), dtype=np.float32)

    # ── Full-text TF-IDF (snapshot-invariant — skip in forecast to avoid leakage) ──────
    if is_forecast:
        ft_tfidf_matrix = csr_matrix((len(df_bills_sub), 0), dtype=np.float32)
        ft_vectorizer = TfidfVectorizer()
        ft_tfidf_bill_ids = df_bills_sub["bill_id"].to_list()
        _ft_has_features = False
        _ft_tfidf_zero = None
    else:
        ft_tfidf_matrix, ft_vectorizer, ft_tfidf_bill_ids = build_full_text_features(df_bills_sub)
        ft_tfidf_id_to_idx = {bid: i for i, bid in enumerate(ft_tfidf_bill_ids)}
        _ft_has_features = ft_tfidf_matrix.shape[1] > 0
        _ft_tfidf_zero = (
            csr_matrix((1, ft_tfidf_matrix.shape[1]), dtype=np.float32)
            if _ft_has_features
            else None
        )

    # ── Content metadata (snapshot-invariant; skip in forecast — full-text-derived) ────
    df_content_meta = (
        pl.DataFrame() if is_forecast else build_content_metadata_features(df_bills_sub)
    )

    # ── Sponsor embeddings (snapshot-invariant — graph structure is static) ────
    df_embeddings = build_embedding_features(df_bills_sub)

    # ── Build single-row labels for sponsor/committee historical rates ───
    df_labels = build_bill_labels(df_bills_sub, df_actions)

    # ── Per-snapshot feature building ────────────────────────────────────
    # Group panel rows by snapshot_date so we batch-build features once
    # per unique snapshot date (much faster than per-row).
    unique_snapshots = sorted(df_panel_labels["snapshot_date"].unique().to_list())
    LOGGER.info(
        "Building features for %d unique snapshot dates...",
        len(unique_snapshots),
    )

    # We'll accumulate rows here
    panel_feature_rows: list[pl.DataFrame] = []

    for snap_date in unique_snapshots:
        # Bills that have a row at this snapshot
        snap_rows = df_panel_labels.filter(pl.col("snapshot_date") == snap_date)
        snap_bill_ids = snap_rows["bill_id"].to_list()
        df_bills_snap = df_bills_sub.filter(pl.col("bill_id").is_in(snap_bill_ids))

        if len(df_bills_snap) == 0:
            continue

        # Build features with as_of_date = snap_date
        df_sponsor = build_sponsor_features(
            df_bills_snap, df_members, df_labels, as_of_date=snap_date
        )
        df_temporal = build_temporal_features(df_bills_snap)
        df_committee = build_committee_features(
            df_bills_snap, df_actions, df_labels, as_of_date=snap_date
        )

        # Merge intrinsic features (always included; skip content_meta in forecast)
        df_feat_snap = (
            snap_rows.select(
                [
                    "bill_id",
                    "snapshot_day",
                    "snapshot_date",
                    "target_advanced_after",
                    "target_law_after",
                ]
            )
            .join(df_sponsor, on="bill_id", how="left")
            .join(df_temporal, on="bill_id", how="left")
            .join(df_committee, on="bill_id", how="left")
        )
        if not is_forecast and len(df_content_meta) > 0:
            df_feat_snap = df_feat_snap.join(df_content_meta, on="bill_id", how="left")

        # Sponsor embeddings (snapshot-invariant)
        df_feat_snap = df_feat_snap.join(df_embeddings, on="bill_id", how="left")

        # In full mode, also join slip, action, staleness, and rule features
        if not is_forecast:
            df_slips_feat = build_slip_features(df_bills_snap, df_slips, as_of_date=snap_date)
            df_action_feat = build_action_features(df_bills_snap, df_actions, as_of_date=snap_date)
            df_staleness = build_staleness_features(df_bills_snap, df_actions, as_of_date=snap_date)
            df_rule_feat = build_rule_features(df_bills_snap, df_actions, as_of_date=snap_date)
            df_feat_snap = (
                df_feat_snap.join(df_slips_feat, on="bill_id", how="left")
                .join(df_action_feat, on="bill_id", how="left")
                .join(df_staleness, on="bill_id", how="left")
                .join(df_rule_feat, on="bill_id", how="left")
            )

        panel_feature_rows.append(df_feat_snap)

    # Concatenate all snapshot slices (diagonal_relaxed handles minor
    # type mismatches, e.g. Int32 vs Float64, across snapshot slices)
    df_panel = pl.concat(panel_feature_rows, how="diagonal_relaxed")

    # Create composite panel_id for tracking
    df_panel = df_panel.with_columns(
        (pl.col("bill_id") + "_d" + pl.col("snapshot_day").cast(pl.Utf8)).alias("panel_id")
    )

    # ── Semantic imputation ──────────────────────────────────────────────
    df_panel = _apply_semantic_imputation(df_panel, mode=mode)

    LOGGER.info(
        "Panel dataset: %d rows from %d bills",
        len(df_panel),
        df_panel["bill_id"].n_unique(),
    )

    # ── Train/test split by bill_id (no same bill in both splits) ────────
    df_panel = df_panel.sort(["snapshot_date", "bill_id"], nulls_last=True)
    unique_bills = df_panel["bill_id"].unique().sort().to_list()
    n_bills = len(unique_bills)
    split_bill_idx = int(n_bills * 0.7)
    train_bill_ids = set(unique_bills[:split_bill_idx])
    test_bill_ids = set(unique_bills[split_bill_idx:])
    df_train = df_panel.filter(pl.col("bill_id").is_in(train_bill_ids))
    df_test = df_panel.filter(pl.col("bill_id").is_in(test_bill_ids))

    LOGGER.info(
        "Panel split - Train: %d rows (%d bills), Test: %d rows (%d bills) (70/30 by bill_id)",
        len(df_train),
        len(train_bill_ids),
        len(df_test),
        len(test_bill_ids),
    )

    # ── Extract targets ──────────────────────────────────────────────────
    y_train = df_train["target_advanced_after"].to_numpy()
    y_test = df_test["target_advanced_after"].to_numpy()
    y_train_law = df_train["target_law_after"].to_numpy()
    y_test_law = df_test["target_law_after"].to_numpy()

    pos_train = y_train.sum()
    pos_test = y_test.sum()
    LOGGER.info(
        "Panel class balance - Train: %d advanced / %d stuck (%.1f%%), "
        "Test: %d advanced / %d stuck (%.1f%%)",
        pos_train,
        len(y_train) - pos_train,
        100 * pos_train / max(len(y_train), 1),
        pos_test,
        len(y_test) - pos_test,
        100 * pos_test / max(len(y_test), 1),
    )

    # ── Feature columns ──────────────────────────────────────────────────
    _exclude_panel = _NON_FEATURE_COLS
    if is_forecast:
        _exclude_panel = _exclude_panel | FORECAST_DROP_COLUMNS
    feature_cols = [c for c in df_panel.columns if c not in _exclude_panel]

    def _to_sparse(df_slice: pl.DataFrame) -> csr_matrix:
        tabular = df_slice.select(feature_cols).to_numpy().astype(np.float32)
        bill_ids = df_slice["bill_id"].to_list()

        # Synopsis TF-IDF
        tfidf_parts = []
        for bid in bill_ids:
            idx = tfidf_id_to_idx.get(bid)
            if idx is not None:
                tfidf_parts.append(tfidf_matrix[idx])
            else:
                tfidf_parts.append(_tfidf_zero)
        from scipy.sparse import vstack as sparse_vstack

        tfidf_rows = sparse_vstack(tfidf_parts)

        # Full-text TF-IDF (mirrors build_feature_matrix logic)
        if _ft_has_features:
            ft_parts = []
            for bid in bill_ids:
                idx = ft_tfidf_id_to_idx.get(bid)
                if idx is not None:
                    ft_parts.append(ft_tfidf_matrix[idx])
                else:
                    ft_parts.append(_ft_tfidf_zero)
            ft_rows = sparse_vstack(ft_parts)
            return sparse_hstack([tfidf_rows, ft_rows, csr_matrix(tabular)])
        else:
            return sparse_hstack([tfidf_rows, csr_matrix(tabular)])

    X_train = _to_sparse(df_train)
    X_test = _to_sparse(df_test)

    # Use panel_id as the bill identifier (bill_id + snapshot_day)
    ids_train = df_train["panel_id"].to_list()
    ids_test = df_test["panel_id"].to_list()

    feature_names = [f"tfidf_{fn}" for fn in vectorizer.get_feature_names_out()]
    if _ft_has_features:
        feature_names += [f"ft_tfidf_{fn}" for fn in ft_vectorizer.get_feature_names_out()]
    feature_names += feature_cols

    # Immature: not applicable in panel mode — all rows have labels.
    # Return empty arrays for compatibility with run_auto() signature.
    empty_y = np.array([], dtype=np.int8)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        ids_train,
        ids_test,
        None,  # X_immature (not used in panel mode)
        empty_y,  # y_immature
        [],  # bill_ids_immature
        vectorizer,
        feature_names,
        y_train_law,
        y_test_law,
        empty_y,  # y_immature_law
    )
