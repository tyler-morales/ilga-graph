"""Feature engineering for bill outcome prediction.

Builds a feature matrix from the normalized Parquet tables where each row
represents one bill, and features capture:

- **Sponsor features**: party, chamber, historical passage rate, sponsor count
- **Text features**: TF-IDF of synopsis text (top N terms)
- **Slip features**: proponent/opponent counts, ratios, org concentration
- **Temporal features**: month, day of session, is_lame_duck
- **Action features**: action count, speed of early actions

Target: Binary -- 1 if bill passed committee (advanced), 0 if stuck/dead.

**Key design decisions:**
- Only "mature" bills (introduced 90+ days ago) are used for train/test
  to avoid labeling bills as "stuck" when they simply haven't had time yet.
- Time-based train/test split to prevent leakage.
- Stratified k-fold cross-validation within training for model selection.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")

# Bills must be at least this many days old to have a reliable label.
# Bills introduced less than this many days ago are scored but not
# used for training or evaluation.
MATURITY_DAYS = 120

# ── Action-based outcome classification ──────────────────────────────────────

_ADVANCED_TOKENS = (
    "Do Pass",
    "Reported Out Of Committee",
    "Third Reading - Passed",
    "Passed Both Houses",
    "Adopted Both Houses",
    "Resolution Adopted",
    "Public Act",
    "Signed by Governor",
    "Appointment Confirmed",
    "Second Reading",
    "Third Reading",
)

_SIGNED_TOKENS = (
    "Public Act",
    "Signed by Governor",
    "Appointment Confirmed",
)


def _bill_advanced(actions: list[str]) -> bool:
    """Did this bill advance past committee?"""
    for action in actions:
        for token in _ADVANCED_TOKENS:
            if token.lower() in action.lower():
                return True
    return False


def _bill_became_law(actions: list[str]) -> bool:
    """Did this bill become law?"""
    for action in actions:
        for token in _SIGNED_TOKENS:
            if token.lower() in action.lower():
                return True
    return False


# ── Bill pipeline stage classification ────────────────────────────────────────

# Ordered from highest to lowest progress so we can find the "best" stage
_STAGE_DEFINITIONS: list[tuple[str, float, list[str]]] = [
    ("SIGNED", 1.0, ["Public Act", "Signed by Governor", "Appointment Confirmed"]),
    ("VETOED", -1.0, ["Vetoed", "Total Veto", "Amendatory Veto", "Item Veto"]),
    (
        "PASSED_BOTH",
        0.85,
        ["Passed Both Houses", "Adopted Both Houses"],
    ),
    (
        "CROSSED_CHAMBERS",
        0.70,
        ["Passed Both Houses", "Adopted Both Houses", "Arrives in"],
    ),
    (
        "FLOOR_VOTE",
        0.55,
        [
            "Third Reading - Passed",
            "Third Reading",
            "Second Reading",
            "Resolution Adopted",
        ],
    ),
    (
        "PASSED_COMMITTEE",
        0.40,
        ["Do Pass", "Reported Out Of Committee"],
    ),
    ("IN_COMMITTEE", 0.25, ["Assigned to", "Referred to"]),
    (
        "FILED",
        0.10,
        ["First Reading", "Filed with"],
    ),
]

# Death signals that indicate a bill is truly dead
_DEATH_TOKENS = [
    "Vetoed",
    "Total Veto",
    "Amendatory Veto",
    "Item Veto",
    "Tabled",
    "Motion to Table",
    "Postponed",
    "Rule 19",
    "sine die",
    "Session Sine Die",
]


def compute_bill_stage(
    actions: list[str],
) -> tuple[str, float]:
    """Determine the highest legislative stage reached by a bill.

    Returns (stage_name, progress_fraction).
    Progress is 0.0-1.0, with -1.0 for VETOED (terminal).
    """
    best_stage = "FILED"
    best_progress = 0.10

    for action_text in actions:
        al = action_text.lower()
        for stage, progress, tokens in _STAGE_DEFINITIONS:
            for token in tokens:
                if token.lower() in al:
                    if progress > best_progress or (progress < 0 and stage == "VETOED"):
                        best_stage = stage
                        best_progress = progress
                    break

    return best_stage, best_progress


def classify_stuck_status(
    current_stage: str,
    days_since_action: int,
    days_since_intro: int,
    actions: list[str],
) -> tuple[str, str]:
    """Classify a stuck bill into a nuanced sub-status.

    Returns (stuck_status, stuck_reason).

    Sub-statuses:
        DEAD      - Vetoed, tabled, or session-dead
        STAGNANT  - In committee, no action for 180+ days
        SLOW      - Some activity, but 60-180 days since last action
        PENDING   - Last action within 60 days
        NEW       - Introduced less than 30 days ago
    """
    # Check for death signals
    for action_text in actions:
        al = action_text.lower()
        for death_token in _DEATH_TOKENS:
            if death_token.lower() in al:
                reason = f"Bill appears dead: '{action_text.strip()[:60]}'"
                return "DEAD", reason

    if current_stage == "VETOED":
        return "DEAD", "Vetoed by the Governor"

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
                pass
        mature_flags.append(is_mature)

    return df.select(["bill_id", "bill_type", "introduction_date"]).with_columns(
        pl.Series("target_advanced", advanced_labels, dtype=pl.Int8),
        pl.Series("target_law", law_labels, dtype=pl.Int8),
        pl.Series("is_mature", mature_flags, dtype=pl.Boolean),
    )


def build_sponsor_features(
    df_bills: pl.DataFrame,
    df_members: pl.DataFrame,
    df_labels: pl.DataFrame,
) -> pl.DataFrame:
    """Build sponsor-related features for each bill.

    Includes historical passage rate: for each bill, we compute the sponsor's
    passage rate using ONLY bills introduced BEFORE this one (no leakage).
    """
    member_map = {m["member_id"]: m for m in df_members.to_dicts()}

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

        party = member["party"] if member else "Unknown"
        is_democrat = 1 if party == "Democrat" else 0
        is_republican = 1 if party == "Republican" else 0
        is_majority = is_democrat  # Democrats are majority in IL 104th GA

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
                "sponsor_bill_count": (member["sponsored_bill_count"] if member else 0),
            }
        )

        # Update running stats AFTER computing features (no leakage)
        if sponsor_id:
            sponsor_filed[sponsor_id] = sponsor_filed.get(sponsor_id, 0) + 1
            if bill.get("target_advanced") == 1:
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


def build_slip_features(
    df_bills: pl.DataFrame,
    df_slips: pl.DataFrame,
) -> pl.DataFrame:
    """Build witness slip aggregate features per bill."""
    if len(df_slips) == 0:
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

    agg = df_slips.group_by("bill_id").agg(
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
    org_counts = df_slips.group_by(["bill_id", "organization_clean"]).agg(
        pl.len().alias("org_count")
    )
    bill_totals = df_slips.group_by("bill_id").agg(pl.len().alias("bill_total"))
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


def build_action_features(
    df_bills: pl.DataFrame,
    df_actions: pl.DataFrame,
) -> pl.DataFrame:
    """Build EARLY action features that avoid leakage.

    Only counts actions in the first 30 days after filing.
    """
    df_act_dated = df_actions.join(
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


# ── Assembly: combine all features ───────────────────────────────────────────


def build_feature_matrix(
    *,
    tfidf_features: int = 500,
) -> tuple:
    """Build the full feature matrix for bill outcome prediction.

    Returns:
        X_train, X_test, y_train, y_test,
        bill_ids_train, bill_ids_test,
        X_immature, y_immature, bill_ids_immature,
        vectorizer, feature_names
    """
    LOGGER.info("Building feature matrix...")

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

    # Build feature tables
    df_sponsor = build_sponsor_features(df_bills_sub, df_members, df_labels)
    tfidf_matrix, vectorizer, tfidf_bill_ids = build_text_features(
        df_bills_sub, max_features=tfidf_features
    )
    df_slips_feat = build_slip_features(df_bills_sub, df_slips)
    df_temporal = build_temporal_features(df_bills_sub)
    df_action_feat = build_action_features(df_bills_sub, df_actions)

    # Merge all tabular features
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
        .join(df_slips_feat, on="bill_id", how="left")
        .join(df_temporal, on="bill_id", how="left")
        .join(df_action_feat, on="bill_id", how="left")
        .fill_null(0)
    )

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

    # Extract targets
    y_train = df_train["target_advanced"].to_numpy()
    y_test = df_test["target_advanced"].to_numpy()
    y_immature = df_immature["target_advanced"].to_numpy()

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

    # Build bill_id -> tfidf row index
    tfidf_id_to_idx = {bid: i for i, bid in enumerate(tfidf_bill_ids)}

    # Extract numeric feature columns
    feature_cols = [
        c
        for c in df_feat.columns
        if c
        not in (
            "bill_id",
            "target_advanced",
            "target_law",
            "introduction_date",
            "is_mature",
        )
    ]

    def _to_sparse(df_slice: pl.DataFrame) -> csr_matrix:
        tabular = df_slice.select(feature_cols).to_numpy().astype(np.float32)
        tfidf_idx = [tfidf_id_to_idx.get(bid, 0) for bid in df_slice["bill_id"].to_list()]
        tfidf_rows = tfidf_matrix[tfidf_idx]
        return sparse_hstack([tfidf_rows, csr_matrix(tabular)])

    X_train = _to_sparse(df_train)
    X_test = _to_sparse(df_test)
    X_immature = _to_sparse(df_immature) if len(df_immature) > 0 else None

    bill_ids_train = df_train["bill_id"].to_list()
    bill_ids_test = df_test["bill_id"].to_list()
    bill_ids_immature = df_immature["bill_id"].to_list()

    feature_names = [f"tfidf_{fn}" for fn in vectorizer.get_feature_names_out()] + feature_cols

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
    )
