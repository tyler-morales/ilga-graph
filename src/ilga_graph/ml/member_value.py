"""Member Value Model — undervalued/overvalued detection + issue recruiter.

Predicts legislative effectiveness from structural features (network
position, demographics, institutional role, topic profile) and compares
to actual effectiveness to surface undervalued and overvalued members.

Design constraint: N ~ 180 legislators → Ridge Regression with
Leave-One-Out CV (not gradient boosting, which would overfit).

Outputs:
    processed/member_value_scores.parquet  — per-member value profiles
    processed/member_recruitment.json      — per-topic recruitment rankings
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")
CACHE_DIR = Path("cache")

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemberValueProfile:
    """Per-member value assessment from the ML model."""

    member_id: str
    member_name: str
    party: str
    chamber: str

    # Model outputs
    predicted_effectiveness: float = 0.0  # Ridge prediction (0-1)
    actual_effectiveness: float = 0.0  # From Moneyball
    value_residual: float = 0.0  # predicted - actual
    value_percentile: int = 50  # 0-100, higher = more undervalued
    value_label: str = "Fairly Valued"  # Undervalued / Fairly Valued / Overvalued

    # Context
    moneyball_score: float = 0.0
    influence_score: float = 0.0

    # Top 3 topics this member is best recruited for
    top_recruitment_topics: list[str] = field(default_factory=list)


@dataclass
class TopicRecruitmentScore:
    """Per-member-per-topic recruitment assessment."""

    member_id: str
    member_name: str
    party: str
    chamber: str
    topic: str

    # Component scores (all 0-1)
    affinity_score: float = 0.0  # Topic YES rate from coalitions
    effectiveness_score: float = 0.0  # ML-predicted effectiveness
    persuadability_score: float = 0.0  # Higher for Swing tier
    network_reach: float = 0.0  # Betweenness centrality

    # Composite
    recruitment_score: float = 0.0

    # Context
    coalition_tier: str = ""  # Champion / Lean Support / Swing / etc.
    value_label: str = ""  # Undervalued / Fairly Valued / Overvalued
    total_topic_votes: int = 0
    yes_rate: float = 0.0


@dataclass
class MemberValueReport:
    """Full output of the member value pipeline."""

    profiles: dict[str, MemberValueProfile] = field(default_factory=dict)
    topic_rankings: dict[str, list[TopicRecruitmentScore]] = field(default_factory=dict)
    model_r2: float = 0.0
    model_mae: float = 0.0
    n_members: int = 0
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Persuadability mapping (coalition tier → score)
# ═══════════════════════════════════════════════════════════════════════════════

_TIER_PERSUADABILITY: dict[str, float] = {
    "Swing": 1.0,
    "Lean Support": 0.7,
    "Lean Oppose": 0.5,
    "Champion": 0.2,
    "Oppose": 0.1,
}

# Recruitment score weights
_W_AFFINITY = 0.35
_W_EFFECTIVENESS = 0.30
_W_PERSUADABILITY = 0.20
_W_NETWORK = 0.15

# Value label thresholds (percentile-based)
_UNDERVALUED_PCTILE = 70  # top 30% of residuals
_OVERVALUED_PCTILE = 30  # bottom 30% of residuals


# ═══════════════════════════════════════════════════════════════════════════════
# Feature matrix construction
# ═══════════════════════════════════════════════════════════════════════════════


def build_member_feature_matrix(
    moneyball_profiles: dict[str, dict],
    embeddings: dict[str, np.ndarray],
    topic_coalitions: list[dict],
    dim_members: pl.DataFrame,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """Assemble the member feature matrix for the value model.

    Returns
    -------
    X : np.ndarray, shape (n_members, n_features)
    feature_names : list of feature column names
    member_ids : list of member IDs (row order)
    y : np.ndarray, shape (n_members,) — effectiveness_rate target
    """
    console.print("\n[bold]Building member feature matrix...[/]")

    # ── Collect member IDs with Moneyball profiles ──
    member_ids = sorted(moneyball_profiles.keys())
    if not member_ids:
        raise ValueError("No Moneyball profiles available")

    # ── Build dim_members lookup ──
    dm_lookup: dict[str, dict] = {}
    for row in dim_members.to_dicts():
        dm_lookup[row["member_id"]] = row

    # ── Build topic coalition lookup: member_id → {topic: yes_rate} ──
    topic_rates: dict[str, dict[str, float]] = {}
    topic_tiers: dict[str, dict[str, str]] = {}
    topic_names: list[str] = []
    for tc in topic_coalitions:
        topic = tc.get("topic", "")
        if not topic:
            continue
        topic_names.append(topic)
        for mp in tc.get("all_member_profiles", []):
            mid = mp.get("member_id", "")
            if mid:
                topic_rates.setdefault(mid, {})[topic] = mp.get("yes_rate", 0.5)
        # Build tier lookup from the tiers list
        for tier_data in tc.get("tiers", []):
            for tm in tier_data.get("top_members", []):
                # top_members may not have member_id, match by name
                pass  # We'll build tiers differently below

    # Build tier lookup from all_member_profiles + tier thresholds
    for tc in topic_coalitions:
        topic = tc.get("topic", "")
        for mp in tc.get("all_member_profiles", []):
            mid = mp.get("member_id", "")
            yr = mp.get("yes_rate", 0.5)
            if mid and topic:
                tier = _classify_tier(yr)
                topic_tiers.setdefault(mid, {})[topic] = tier

    topic_names = sorted(set(topic_names))

    # ── Assemble feature rows ──
    rows: list[list[float]] = []
    targets: list[float] = []
    valid_member_ids: list[str] = []

    for mid in member_ids:
        mb = moneyball_profiles[mid]
        dm = dm_lookup.get(mid, {})

        # Skip members with no bills (effectiveness undefined)
        laws_filed = mb.get("laws_filed", 0)
        if laws_filed == 0:
            continue

        # ── Demographic features ──
        is_democrat = 1.0 if mb.get("party", "") == "Democrat" else 0.0
        is_republican = 1.0 if mb.get("party", "") == "Republican" else 0.0
        is_senate = 1.0 if mb.get("chamber", "") == "Senate" else 0.0
        career_start = dm.get("career_start")
        career_years = 0.0
        if career_start and isinstance(career_start, (int, float)):
            career_years = max(0, 2025 - int(career_start))

        # ── Network features (non-outcome) ──
        network_centrality = mb.get("network_centrality", 0.0)
        betweenness = mb.get("betweenness", 0.0)
        unique_collaborators = mb.get("unique_collaborators", 0)
        collab_republicans = mb.get("collaborator_republicans", 0)
        collab_democrats = mb.get("collaborator_democrats", 0)

        # ── Institutional features ──
        is_leadership = 1.0 if mb.get("is_leadership", False) else 0.0
        institutional_weight = mb.get("institutional_weight", 0.0)

        # ── Bill volume (input feature, not outcome) ──
        total_primary_bills = mb.get("total_primary_bills", 0)
        pipeline_depth_norm = mb.get("pipeline_depth_normalized", 0.0)

        # ── Cross-party collaboration rate ──
        magnet_score = mb.get("magnet_score", 0.0)
        bridge_score = mb.get("bridge_score", 0.0)

        # ── Topic affinity features (per-topic YES rate) ──
        topic_features = []
        for topic in topic_names:
            rate = topic_rates.get(mid, {}).get(topic, 0.5)
            topic_features.append(rate)

        # ── Assemble row ──
        row = [
            is_democrat,
            is_republican,
            is_senate,
            career_years,
            network_centrality,
            betweenness,
            unique_collaborators,
            collab_republicans,
            collab_democrats,
            is_leadership,
            institutional_weight,
            total_primary_bills,
            pipeline_depth_norm,
            magnet_score,
            bridge_score,
        ] + topic_features

        rows.append(row)
        targets.append(mb.get("effectiveness_rate", 0.0))
        valid_member_ids.append(mid)

    feature_names = [
        "is_democrat",
        "is_republican",
        "is_senate",
        "career_years",
        "network_centrality",
        "betweenness",
        "unique_collaborators",
        "collaborator_republicans",
        "collaborator_democrats",
        "is_leadership",
        "institutional_weight",
        "total_primary_bills",
        "pipeline_depth_norm",
        "magnet_score",
        "bridge_score",
    ] + [f"topic_{t.lower().replace(' ', '_').replace('&', 'and')}" for t in topic_names]

    X_tabular = np.array(rows, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)

    # ── Add PCA-reduced embeddings ──
    n_pca = min(8, len(valid_member_ids) - 1)
    emb_matrix = []
    for mid in valid_member_ids:
        if mid in embeddings:
            emb_matrix.append(embeddings[mid])
        else:
            emb_matrix.append(np.zeros(64))

    emb_arr = np.array(emb_matrix, dtype=np.float64)
    if emb_arr.shape[1] > 0 and n_pca > 0:
        pca = PCA(n_components=n_pca)
        emb_pca = pca.fit_transform(emb_arr)
        X = np.hstack([X_tabular, emb_pca])
        feature_names += [f"emb_pca_{i}" for i in range(n_pca)]
        variance_explained = sum(pca.explained_variance_ratio_)
        console.print(
            f"  PCA: {n_pca} components explain {variance_explained:.1%} of embedding variance"
        )
    else:
        X = X_tabular

    console.print(f"  Feature matrix: {X.shape[0]} members x {X.shape[1]} features")

    return X, feature_names, valid_member_ids, y


# ═══════════════════════════════════════════════════════════════════════════════
# Model training
# ═══════════════════════════════════════════════════════════════════════════════


def train_value_model(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[Ridge, np.ndarray, float, float]:
    """Train Ridge regression with Leave-One-Out CV.

    Returns
    -------
    model : fitted Ridge model
    y_pred_loo : LOO predictions (one per member)
    r2 : LOO R-squared
    mae : LOO mean absolute error
    """
    console.print("\n[bold]Training member value model (Ridge + LOO-CV)...[/]")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge with LOO-CV for predictions
    model = Ridge(alpha=1.0)
    loo = LeaveOneOut()
    y_pred_loo = cross_val_predict(model, X_scaled, y, cv=loo)

    # Clip predictions to valid range
    y_pred_loo = np.clip(y_pred_loo, 0.0, 1.0)

    # Compute LOO metrics
    ss_res = np.sum((y - y_pred_loo) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(y - y_pred_loo))

    console.print(f"  LOO-CV R²: {r2:.3f}")
    console.print(f"  LOO-CV MAE: {mae:.3f}")

    # Fit final model on all data (for feature importance)
    model.fit(X_scaled, y)

    return model, y_pred_loo, r2, mae


# ═══════════════════════════════════════════════════════════════════════════════
# Value score computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_value_scores(
    moneyball_profiles: dict[str, dict],
    embeddings: dict[str, np.ndarray],
    topic_coalitions: list[dict],
    dim_members: pl.DataFrame,
) -> MemberValueReport:
    """Full pipeline: build features → train model → compute residuals.

    Parameters
    ----------
    moneyball_profiles : dict mapping member_id → Moneyball profile dict
    embeddings : dict mapping member_id → numpy embedding vector
    topic_coalitions : list of topic coalition dicts (from topic_coalitions.json)
    dim_members : polars DataFrame from dim_members.parquet

    Returns
    -------
    MemberValueReport with profiles and model metrics
    """
    console.print("\n[bold cyan]Member Value Model — Undervalued / Overvalued Detection[/]")

    # Build feature matrix
    X, feature_names, member_ids, y = build_member_feature_matrix(
        moneyball_profiles,
        embeddings,
        topic_coalitions,
        dim_members,
    )

    if len(member_ids) < 10:
        console.print(
            "[yellow]Too few members with bills filed "
            f"({len(member_ids)}) — skipping value model.[/]"
        )
        return MemberValueReport()

    # Train model
    model, y_pred, r2, mae = train_value_model(X, y)

    # Compute residuals (positive = undervalued)
    residuals = y_pred - y

    # Convert residuals to percentiles
    from scipy.stats import rankdata

    ranks = rankdata(residuals)
    percentiles = ((ranks - 1) / (len(ranks) - 1) * 100).astype(int)

    # Build profiles
    profiles: dict[str, MemberValueProfile] = {}
    for i, mid in enumerate(member_ids):
        mb = moneyball_profiles[mid]
        pct = int(percentiles[i])
        if pct >= _UNDERVALUED_PCTILE:
            label = "Undervalued"
        elif pct <= _OVERVALUED_PCTILE:
            label = "Overvalued"
        else:
            label = "Fairly Valued"

        profiles[mid] = MemberValueProfile(
            member_id=mid,
            member_name=mb.get("member_name", ""),
            party=mb.get("party", ""),
            chamber=mb.get("chamber", ""),
            predicted_effectiveness=float(y_pred[i]),
            actual_effectiveness=float(y[i]),
            value_residual=float(residuals[i]),
            value_percentile=pct,
            value_label=label,
            moneyball_score=mb.get("moneyball_score", 0.0),
        )

    report = MemberValueReport(
        profiles=profiles,
        model_r2=r2,
        model_mae=mae,
        n_members=len(member_ids),
        n_features=X.shape[1],
        feature_names=feature_names,
    )

    # ── Display top undervalued / overvalued ──
    _print_value_table(profiles)

    return report


def compute_recruitment_rankings(
    value_report: MemberValueReport,
    moneyball_profiles: dict[str, dict],
    topic_coalitions: list[dict],
) -> dict[str, list[TopicRecruitmentScore]]:
    """Compute per-topic recruitment rankings.

    For each topic, ranks members by a composite of:
    affinity, predicted effectiveness, persuadability, network reach.
    """
    console.print("\n[bold cyan]Issue-Specific Recruitment Rankings[/]")

    profiles = value_report.profiles
    if not profiles:
        return {}

    # Build topic affinity and tier lookups
    topic_member_data: dict[str, dict[str, dict]] = {}
    for tc in topic_coalitions:
        topic = tc.get("topic", "")
        if not topic:
            continue
        topic_member_data[topic] = {}
        for mp in tc.get("all_member_profiles", []):
            mid = mp.get("member_id", "")
            if mid:
                yr = mp.get("yes_rate", 0.5)
                topic_member_data[topic][mid] = {
                    "yes_rate": yr,
                    "tier": _classify_tier(yr),
                    "total_votes": mp.get("total_votes", 0),
                }

    # Normalize betweenness for network_reach scores
    betweenness_values = [
        moneyball_profiles.get(mid, {}).get("betweenness", 0.0) for mid in profiles
    ]
    max_betweenness = max(betweenness_values) if betweenness_values else 1.0
    if max_betweenness <= 0:
        max_betweenness = 1.0

    # Normalize predicted effectiveness for scoring
    pred_values = [p.predicted_effectiveness for p in profiles.values()]
    max_pred = max(pred_values) if pred_values else 1.0
    if max_pred <= 0:
        max_pred = 1.0

    topic_rankings: dict[str, list[TopicRecruitmentScore]] = {}

    for topic, member_data in topic_member_data.items():
        scores: list[TopicRecruitmentScore] = []

        for mid, vp in profiles.items():
            md = member_data.get(mid)
            if not md:
                continue  # no voting data for this topic

            mb = moneyball_profiles.get(mid, {})
            tier = md["tier"]
            affinity = md["yes_rate"]
            effectiveness = vp.predicted_effectiveness / max_pred
            persuadability = _TIER_PERSUADABILITY.get(tier, 0.5)
            network = mb.get("betweenness", 0.0) / max_betweenness

            composite = (
                _W_AFFINITY * affinity
                + _W_EFFECTIVENESS * effectiveness
                + _W_PERSUADABILITY * persuadability
                + _W_NETWORK * network
            )

            scores.append(
                TopicRecruitmentScore(
                    member_id=mid,
                    member_name=vp.member_name,
                    party=vp.party,
                    chamber=vp.chamber,
                    topic=topic,
                    affinity_score=affinity,
                    effectiveness_score=effectiveness,
                    persuadability_score=persuadability,
                    network_reach=network,
                    recruitment_score=round(composite, 4),
                    coalition_tier=tier,
                    value_label=vp.value_label,
                    total_topic_votes=md.get("total_votes", 0),
                    yes_rate=md["yes_rate"],
                )
            )

        # Sort by recruitment score descending
        scores.sort(key=lambda s: -s.recruitment_score)
        topic_rankings[topic] = scores

    # Populate top_recruitment_topics on value profiles
    for mid, vp in profiles.items():
        topic_scores: list[tuple[str, float]] = []
        for topic, rankings in topic_rankings.items():
            for rs in rankings:
                if rs.member_id == mid:
                    topic_scores.append((topic, rs.recruitment_score))
                    break
        topic_scores.sort(key=lambda x: -x[1])
        vp.top_recruitment_topics = [t for t, _ in topic_scores[:3]]

    # Print summary
    _print_recruitment_summary(topic_rankings)

    return topic_rankings


# ═══════════════════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════════════════


def save_value_outputs(
    report: MemberValueReport,
    topic_rankings: dict[str, list[TopicRecruitmentScore]],
) -> None:
    """Save value scores to parquet and recruitment rankings to JSON."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Value scores → parquet ──
    if report.profiles:
        rows = []
        for p in report.profiles.values():
            rows.append(
                {
                    "member_id": p.member_id,
                    "member_name": p.member_name,
                    "party": p.party,
                    "chamber": p.chamber,
                    "predicted_effectiveness": round(p.predicted_effectiveness, 4),
                    "actual_effectiveness": round(p.actual_effectiveness, 4),
                    "value_residual": round(p.value_residual, 4),
                    "value_percentile": p.value_percentile,
                    "value_label": p.value_label,
                    "moneyball_score": round(p.moneyball_score, 2),
                    "top_recruitment_topics": ",".join(p.top_recruitment_topics),
                }
            )
        df = pl.DataFrame(rows)
        out_path = PROCESSED_DIR / "member_value_scores.parquet"
        df.write_parquet(out_path)
        console.print(f"  Saved {len(rows)} value profiles → {out_path}")

    # ── Recruitment rankings → JSON ──
    if topic_rankings:
        out: dict[str, list[dict]] = {}
        for topic, rankings in topic_rankings.items():
            out[topic] = [asdict(rs) for rs in rankings]

        # Add model metadata
        meta = {
            "model_r2": round(report.model_r2, 4),
            "model_mae": round(report.model_mae, 4),
            "n_members": report.n_members,
            "n_features": report.n_features,
            "topics": out,
        }
        out_path = PROCESSED_DIR / "member_recruitment.json"
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=1)
        console.print(f"  Saved {len(topic_rankings)} topic recruitment rankings → {out_path}")


def load_moneyball_profiles_from_cache() -> dict[str, dict]:
    """Load Moneyball profiles from cache/moneyball.json.

    Returns dict of member_id → profile dict. Returns empty dict
    if cache file doesn't exist.
    """
    mb_path = CACHE_DIR / "moneyball.json"
    if not mb_path.exists():
        LOGGER.warning("Moneyball cache not found at %s", mb_path)
        return {}
    try:
        with open(mb_path) as f:
            raw = json.load(f)
        return raw.get("profiles", {})
    except (json.JSONDecodeError, OSError) as e:
        LOGGER.warning("Failed to load Moneyball cache: %s", e)
        return {}


def load_topic_coalitions() -> list[dict]:
    """Load topic coalitions from processed/topic_coalitions.json."""
    path = PROCESSED_DIR / "topic_coalitions.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def load_embeddings_as_dict() -> dict[str, np.ndarray]:
    """Load member embeddings from parquet into a dict."""
    path = PROCESSED_DIR / "member_embeddings.parquet"
    if not path.exists():
        return {}
    try:
        df = pl.read_parquet(path)
        result: dict[str, np.ndarray] = {}
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        for row in df.to_dicts():
            mid = row["member_id"]
            vec = np.array([row[c] for c in dim_cols], dtype=np.float64)
            result[mid] = vec
        return result
    except Exception as e:
        LOGGER.warning("Failed to load embeddings: %s", e)
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator (called from ml_run.py)
# ═══════════════════════════════════════════════════════════════════════════════


def run_member_value_pipeline() -> tuple[MemberValueReport, dict]:
    """Run the complete member value pipeline.

    Loads all dependencies, trains the model, computes recruitment
    rankings, and saves outputs.

    Returns (report, topic_rankings) for pipeline logging.
    """
    # Load dependencies
    moneyball_profiles = load_moneyball_profiles_from_cache()
    if not moneyball_profiles:
        console.print("[yellow]No Moneyball profiles found — skipping member value model.[/]")
        return MemberValueReport(), {}

    embeddings = load_embeddings_as_dict()
    topic_coalitions = load_topic_coalitions()
    dm_path = PROCESSED_DIR / "dim_members.parquet"
    if not dm_path.exists():
        console.print("[yellow]dim_members.parquet not found — skipping.[/]")
        return MemberValueReport(), {}
    dim_members = pl.read_parquet(dm_path)

    # Train value model
    report = compute_value_scores(
        moneyball_profiles,
        embeddings,
        topic_coalitions,
        dim_members,
    )

    # Compute recruitment rankings
    topic_rankings = compute_recruitment_rankings(
        report,
        moneyball_profiles,
        topic_coalitions,
    )
    report.topic_rankings = topic_rankings

    # Save outputs
    save_value_outputs(report, topic_rankings)

    return report, topic_rankings


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _classify_tier(yes_rate: float) -> str:
    """Classify a member into a coalition tier by YES rate."""
    if yes_rate >= 0.80:
        return "Champion"
    if yes_rate >= 0.65:
        return "Lean Support"
    if yes_rate >= 0.45:
        return "Swing"
    if yes_rate >= 0.25:
        return "Lean Oppose"
    return "Oppose"


def _print_value_table(
    profiles: dict[str, MemberValueProfile],
) -> None:
    """Print top undervalued and overvalued members."""
    sorted_profiles = sorted(
        profiles.values(),
        key=lambda p: -p.value_residual,
    )

    table = Table(
        title="Member Value Assessment",
        show_lines=True,
        title_style="bold green",
    )
    table.add_column("Name", style="bold")
    table.add_column("Party")
    table.add_column("Chamber")
    table.add_column("Predicted Eff.", justify="right")
    table.add_column("Actual Eff.", justify="right")
    table.add_column("Residual", justify="right")
    table.add_column("Label")

    # Top 5 undervalued
    for p in sorted_profiles[:5]:
        label_style = "green" if p.value_label == "Undervalued" else "dim"
        table.add_row(
            p.member_name,
            p.party,
            p.chamber,
            f"{p.predicted_effectiveness:.1%}",
            f"{p.actual_effectiveness:.1%}",
            f"{p.value_residual:+.1%}",
            f"[{label_style}]{p.value_label}[/]",
        )

    table.add_row("", "", "", "", "", "", "", style="dim")

    # Bottom 5 (overvalued / overperformers)
    for p in sorted_profiles[-5:]:
        label_style = "red" if p.value_label == "Overvalued" else "dim"
        table.add_row(
            p.member_name,
            p.party,
            p.chamber,
            f"{p.predicted_effectiveness:.1%}",
            f"{p.actual_effectiveness:.1%}",
            f"{p.value_residual:+.1%}",
            f"[{label_style}]{p.value_label}[/]",
        )

    console.print(table)


def _print_recruitment_summary(
    topic_rankings: dict[str, list[TopicRecruitmentScore]],
) -> None:
    """Print a compact summary of top recruitment targets per topic."""
    table = Table(
        title="Top Recruitment Targets by Topic",
        show_lines=True,
        title_style="bold green",
    )
    table.add_column("Topic", style="bold")
    table.add_column("Top 3 Targets")
    table.add_column("Avg Score", justify="right")

    for topic in sorted(topic_rankings.keys()):
        rankings = topic_rankings[topic]
        top3 = rankings[:3]
        names = ", ".join(f"{r.member_name} ({r.coalition_tier})" for r in top3)
        avg_score = sum(r.recruitment_score for r in top3) / len(top3) if top3 else 0.0
        table.add_row(topic, names, f"{avg_score:.3f}")

    console.print(table)
