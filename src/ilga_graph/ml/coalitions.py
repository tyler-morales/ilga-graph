"""Topic-based voting coalition discovery.

Instead of clustering all members on overall voting agreement (which produces
generic blocs with silhouette ~0.1), this module builds **per-topic coalitions**:

For each policy area (Healthcare, Criminal Justice, Education, etc.):
  1. Identify bills assigned to that topic's committees
  2. Find roll-call votes on those bills
  3. Compute each member's YES rate on that topic
  4. Segment into Champions / Lean Support / Swing / Lean Oppose / Oppose

This produces actionable output like:
  "On Healthcare: 85 Champions (72D/13R), 30 Swing (8D/22R), 45 Oppose (1D/44R)"

Also retains the general (cross-topic) agreement clustering as a secondary
output, but uses k=2 (party-bloc) as the honest baseline since silhouette
is low.

Outputs:
    processed/topic_coalitions.json      -- Per-topic member voting profiles
    processed/coalitions.parquet         -- Member cluster assignments (general)
    processed/coalition_profiles.json    -- General coalition profiles
    processed/member_embeddings.parquet  -- 32-dim vectors per member
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")

console = Console()

# Minimum number of roll-call votes on a topic for a member to be classified
_MIN_MEMBER_VOTES = 3
# Minimum number of bills with votes in a topic for the topic to be reported
_MIN_TOPIC_BILLS = 10


# ── Policy category mapping ───────────────────────────────────────────────────

# Keyword patterns for matching committee NAMES to categories.
_CATEGORY_NAME_KEYWORDS: dict[str, list[str]] = {
    "Transportation": ["transport", "vehicle", "roads"],
    "Agriculture": ["agricultur", "farm", "conservation"],
    "Commerce": [
        "commerce",
        "small business",
        "business",
        "consumer protect",
        "licensed activit",
        "gaming",
        "tourism",
        "trade",
        "economic opportun",
    ],
    "Criminal Justice": [
        "criminal",
        "judiciary",
        "public safety",
        "restorative justice",
        "police",
        "fire committee",
        "gun violence",
        "human rights",
        "immigration",
    ],
    "Education": ["education", "school", "early childhood"],
    "Energy & Environment": [
        "energy",
        "environment",
        "public utilit",
        "utilities",
    ],
    "Healthcare": [
        "health",
        "human services",
        "child welfare",
        "mental health",
        "addiction",
        "medicaid",
        "prescription drug",
        "public health",
        "behavioral",
        "end of life",
    ],
    "Housing": ["housing"],
    "Insurance & Finance": ["insurance", "financial instit"],
    "Labor": ["labor", "workforce", "workers", "wage", "personnel"],
    "Revenue & Pensions": [
        "revenue",
        "pension",
        "appropriat",
        "tax",
        "budget",
    ],
    "State Government": [
        "state government",
        "government admin",
        "election",
        "ethics & election",
        "local government",
        "cities",
        "counties",
        "cybersecur",
        "data analytics",
    ],
}


def _extract_committee_name(action_text: str) -> str | None:
    """Extract committee name from action text like 'Assigned toJudiciary'.

    Handles ILGA format where there is often no space between 'to' and the
    committee name, and committee name may end with ' Committee'.
    """
    import re

    m = re.search(
        r"(?:(?:Re-)?[Aa]ssigned\s+to|(?:Re-)?[Rr]eferred\s+to)\s*"
        r"(.+?)(?:\s+Committee)?(?:\s*;.*|$)",
        action_text,
    )
    if m:
        name = m.group(1).strip().rstrip("-").strip()
        if name.lower() in (
            "rules",
            "assignments",
            "assignments * reports",
            "rules * reports",
            "committee of the whole",
            "resolutions consent calendar",
            "executive",
            "executive appointments",
        ):
            return None
        return name
    return None


def _categorize_committee_name(committee_name: str) -> list[str]:
    """Map a committee name to policy categories using keyword matching."""
    cn_lower = committee_name.lower()
    categories = []
    for cat, keywords in _CATEGORY_NAME_KEYWORDS.items():
        for kw in keywords:
            if kw in cn_lower:
                categories.append(cat)
                break
    return categories


# ── Bill → category mapping ──────────────────────────────────────────────────


def build_bill_categories() -> dict[str, set[str]]:
    """Build bill_id → set of policy categories from actions + witness slips.

    Uses 'assignment' actions (not the old broken 'committee' filter) and
    witness slip hearing_committee fields.
    """
    bill_categories: dict[str, set[str]] = {}

    # From bill actions — assignment category has "Assigned to X" text
    actions_path = PROCESSED_DIR / "fact_bill_actions.parquet"
    if actions_path.exists():
        df_actions = pl.read_parquet(actions_path)
        assignment_actions = df_actions.filter(pl.col("action_category") == "assignment")
        for row in assignment_actions.to_dicts():
            bid = row["bill_id"]
            action = row.get("action_text", "")
            committee_name = _extract_committee_name(action)
            if committee_name:
                cats = _categorize_committee_name(committee_name)
                for cat in cats:
                    bill_categories.setdefault(bid, set()).add(cat)

    # From witness slip hearing_committee
    slips_path = PROCESSED_DIR / "fact_witness_slips.parquet"
    if slips_path.exists():
        df_slips = pl.read_parquet(slips_path)
        if "hearing_committee" in df_slips.columns:
            for row in df_slips.select(["bill_id", "hearing_committee"]).unique().to_dicts():
                hc = row.get("hearing_committee", "") or ""
                if hc:
                    cats = _categorize_committee_name(hc)
                    for cat in cats:
                        bill_categories.setdefault(row["bill_id"], set()).add(cat)

    LOGGER.info("Mapped %d bills to policy categories", len(bill_categories))
    return bill_categories


# ── Topic-based coalitions (NEW — primary output) ────────────────────────────


def discover_topic_coalitions() -> list[dict]:
    """Build per-topic voting coalitions from roll-call vote data.

    For each policy area with enough voted bills:
      1. Find bills in that category
      2. Find vote events on those bills
      3. Compute each member's YES rate on that topic
      4. Segment into tiers: Champion / Lean Support / Swing / Lean Oppose / Oppose
      5. Report with party breakdown, key members

    Returns list of topic coalition dicts and saves to
    processed/topic_coalitions.json.
    """
    console.print("\n[bold]Topic-Based Coalition Discovery[/]")

    # Load data
    casts_path = PROCESSED_DIR / "fact_vote_casts.parquet"
    if not casts_path.exists():
        casts_path = PROCESSED_DIR / "fact_vote_casts_raw.parquet"
    if not casts_path.exists():
        console.print("[dim]No vote cast data.[/]")
        return []

    df_casts = pl.read_parquet(casts_path)
    resolved = df_casts.filter(pl.col("member_id").is_not_null())
    if "vote_cast" not in resolved.columns:
        if "cast" in resolved.columns:
            resolved = resolved.rename({"cast": "vote_cast"})
        else:
            return []

    events_path = PROCESSED_DIR / "fact_vote_events.parquet"
    if not events_path.exists():
        return []
    df_events = pl.read_parquet(events_path)
    df_members = pl.read_parquet(PROCESSED_DIR / "dim_members.parquet")

    member_lookup = {m["member_id"]: m for m in df_members.to_dicts()}

    # Build mappings
    bill_categories = build_bill_categories()
    event_to_bill = {
        r["vote_event_id"]: r["bill_id"]
        for r in df_events.select(["vote_event_id", "bill_id"]).to_dicts()
    }

    # Invert: category → set of bill_ids
    category_bills: dict[str, set[str]] = {}
    for bid, cats in bill_categories.items():
        for cat in cats:
            category_bills.setdefault(cat, set()).add(bid)

    # Find which vote events map to each category
    category_events: dict[str, set[str]] = {}
    for eid, bid in event_to_bill.items():
        for cat in bill_categories.get(bid, set()):
            category_events.setdefault(cat, set()).add(eid)

    # For each topic, compute per-member voting profiles
    topic_results = []

    for cat in sorted(_CATEGORY_NAME_KEYWORDS.keys()):
        cat_event_ids = category_events.get(cat, set())
        cat_bill_ids = category_bills.get(cat, set())
        voted_bill_ids = {event_to_bill[eid] for eid in cat_event_ids if eid in event_to_bill}

        if len(voted_bill_ids) < _MIN_TOPIC_BILLS:
            continue

        # Filter casts to this topic's vote events
        topic_casts = resolved.filter(pl.col("vote_event_id").is_in(list(cat_event_ids)))

        # Per-member: count YES, NO
        member_votes: dict[str, dict[str, int]] = {}
        for row in topic_casts.to_dicts():
            mid = row.get("member_id", "")
            cast = row.get("vote_cast", "")
            if mid and cast in ("Y", "N"):
                stats = member_votes.setdefault(mid, {"Y": 0, "N": 0})
                stats[cast] += 1

        # Segment members into tiers based on YES rate
        tiers: dict[str, list[dict]] = {
            "Champion": [],
            "Lean Support": [],
            "Swing": [],
            "Lean Oppose": [],
            "Oppose": [],
        }

        member_profiles = []
        for mid, stats in member_votes.items():
            total = stats["Y"] + stats["N"]
            if total < _MIN_MEMBER_VOTES:
                continue
            yes_rate = stats["Y"] / total
            m = member_lookup.get(mid, {})

            profile = {
                "member_id": mid,
                "name": m.get("name", ""),
                "party": m.get("party", ""),
                "chamber": m.get("chamber", ""),
                "yes_votes": stats["Y"],
                "no_votes": stats["N"],
                "total_votes": total,
                "yes_rate": round(yes_rate, 3),
            }
            member_profiles.append(profile)

            if yes_rate >= 0.80:
                tiers["Champion"].append(profile)
            elif yes_rate >= 0.65:
                tiers["Lean Support"].append(profile)
            elif yes_rate >= 0.45:
                tiers["Swing"].append(profile)
            elif yes_rate >= 0.25:
                tiers["Lean Oppose"].append(profile)
            else:
                tiers["Oppose"].append(profile)

        # Build tier summaries
        tier_summaries = []
        for tier_name, members in tiers.items():
            if not members:
                continue
            dem = sum(1 for m in members if m["party"] == "Democrat")
            rep = sum(1 for m in members if m["party"] == "Republican")
            avg_yes = sum(m["yes_rate"] for m in members) / len(members) if members else 0
            # Top 5 members by vote count (most active in this topic)
            top_members = sorted(members, key=lambda m: m["total_votes"], reverse=True)[:5]
            tier_summaries.append(
                {
                    "tier": tier_name,
                    "count": len(members),
                    "dem_count": dem,
                    "rep_count": rep,
                    "avg_yes_rate": round(avg_yes, 3),
                    "top_members": [
                        {
                            "name": m["name"],
                            "party": m["party"],
                            "yes_rate": m["yes_rate"],
                            "total_votes": m["total_votes"],
                        }
                        for m in top_members
                    ],
                }
            )

        topic_result = {
            "topic": cat,
            "bills_with_votes": len(voted_bill_ids),
            "total_bills": len(cat_bill_ids),
            "members_with_enough_votes": len(member_profiles),
            "tiers": tier_summaries,
            "all_member_profiles": member_profiles,
        }
        topic_results.append(topic_result)

    # Save
    with open(PROCESSED_DIR / "topic_coalitions.json", "w") as f:
        json.dump(topic_results, f, indent=2)
    LOGGER.info(
        "Saved %d topic coalition profiles to topic_coalitions.json",
        len(topic_results),
    )

    # Display
    _display_topic_coalitions(topic_results)

    return topic_results


def _display_topic_coalitions(topic_results: list[dict]) -> None:
    """Display topic-based coalition results as rich tables."""
    if not topic_results:
        console.print("[dim]No topic coalition data.[/]")
        return

    # Summary table
    console.print()
    summary = Table(
        title="Topic-Based Voting Coalitions",
        show_lines=True,
        title_style="bold cyan",
    )
    summary.add_column("Topic", style="bold")
    summary.add_column("Bills", justify="right")
    summary.add_column("Members", justify="right")
    summary.add_column("Champions", justify="right")
    summary.add_column("Swing", justify="right")
    summary.add_column("Oppose", justify="right")
    summary.add_column("Party Split", no_wrap=True)

    for t in topic_results:
        champ = next((s for s in t["tiers"] if s["tier"] == "Champion"), None)
        swing = next((s for s in t["tiers"] if s["tier"] == "Swing"), None)
        oppose = next((s for s in t["tiers"] if s["tier"] == "Oppose"), None)

        champ_str = ""
        if champ:
            champ_str = (
                f"{champ['count']} ([blue]{champ['dem_count']}D[/]/[red]{champ['rep_count']}R[/])"
            )
        swing_str = ""
        if swing:
            swing_str = (
                f"{swing['count']} ([blue]{swing['dem_count']}D[/]/[red]{swing['rep_count']}R[/])"
            )
        oppose_str = ""
        if oppose:
            oppose_str = (
                f"{oppose['count']} "
                f"([blue]{oppose['dem_count']}D[/]/"
                f"[red]{oppose['rep_count']}R[/])"
            )

        # Overall party split of active voters
        total_dem = sum(s["dem_count"] for s in t["tiers"])
        total_rep = sum(s["rep_count"] for s in t["tiers"])
        party_str = f"[blue]{total_dem}D[/] / [red]{total_rep}R[/]"

        summary.add_row(
            t["topic"],
            str(t["bills_with_votes"]),
            str(t["members_with_enough_votes"]),
            champ_str,
            swing_str,
            oppose_str,
            party_str,
        )
    console.print(summary)

    # Detail tables for topics with interesting swing dynamics
    for t in topic_results:
        swing_tiers = [
            s for s in t["tiers"] if s["tier"] in ("Swing", "Lean Support", "Lean Oppose")
        ]
        swing_count = sum(s["count"] for s in swing_tiers)
        if swing_count < 3:
            continue

        console.print()
        detail = Table(
            title=f"{t['topic']} — Voting Blocs",
            show_lines=True,
        )
        detail.add_column("Tier", style="bold")
        detail.add_column("Members", justify="right")
        detail.add_column("Party Mix")
        detail.add_column("Avg YES Rate", justify="right")
        detail.add_column("Key Members")

        for tier in t["tiers"]:
            party = f"[blue]{tier['dem_count']}D[/] / [red]{tier['rep_count']}R[/]"
            key = ", ".join(f"{m['name']} ({m['yes_rate']:.0%})" for m in tier["top_members"][:3])
            detail.add_row(
                tier["tier"],
                str(tier["count"]),
                party,
                f"{tier['avg_yes_rate']:.0%}",
                key,
            )
        console.print(detail)


# ── Graph construction (for general clustering) ──────────────────────────────


def build_agreement_graph(df_vote_casts: pl.DataFrame) -> nx.Graph:
    """Build a weighted graph: edge weight = agreement rate between members."""
    LOGGER.info("Building agreement-rate graph...")

    resolved = df_vote_casts.filter(pl.col("member_id").is_not_null())
    if len(resolved) == 0:
        LOGGER.warning("No resolved vote casts.")
        return nx.Graph()

    G = nx.Graph()

    events = resolved.group_by("vote_event_id").agg(
        pl.col("member_id"),
        pl.col("vote_cast"),
    )

    pair_stats: dict[tuple[str, str], list[int]] = {}

    for row in events.to_dicts():
        member_ids = row["member_id"]
        casts = row["vote_cast"]

        votes = {}
        for mid, cast in zip(member_ids, casts):
            if cast in ("Y", "N"):
                votes[mid] = cast

        members = list(votes.keys())
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                key = (min(a, b), max(a, b))
                if key not in pair_stats:
                    pair_stats[key] = [0, 0]
                pair_stats[key][1] += 1
                if votes[a] == votes[b]:
                    pair_stats[key][0] += 1

    for (a, b), (agree, total) in pair_stats.items():
        if total >= 10:
            G.add_edge(a, b, weight=agree / total)

    LOGGER.info(
        "  Agreement graph: %d nodes, %d edges (min 10 common votes)",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def add_cosponsor_edges(G: nx.Graph) -> nx.Graph:
    """Add co-sponsorship signal to the agreement graph."""
    LOGGER.info("Adding co-sponsorship edges...")

    bills_path = Path("cache/bills.json")
    if not bills_path.exists():
        return G

    with open(bills_path) as f:
        bills_raw = json.load(f)

    cosponsor_counts: dict[tuple[str, str], int] = {}
    for b in bills_raw.values():
        all_ids = b.get("sponsor_ids", []) + b.get("house_sponsor_ids", [])
        if len(all_ids) < 2:
            continue
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                a, b_id = all_ids[i], all_ids[j]
                key = (min(a, b_id), max(a, b_id))
                cosponsor_counts[key] = cosponsor_counts.get(key, 0) + 1

    if cosponsor_counts:
        max_count = max(cosponsor_counts.values())
        for (a, b), count in cosponsor_counts.items():
            bonus = 0.2 * (count / max_count)
            if G.has_edge(a, b):
                G[a][b]["weight"] = min(1.0, G[a][b]["weight"] + bonus)
            elif count >= 3:
                G.add_edge(a, b, weight=0.5 + bonus)

    LOGGER.info(
        "  After co-sponsorship: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


# ── Graph embeddings ─────────────────────────────────────────────────────────


def compute_embeddings(
    G: nx.Graph,
    n_dims: int = 32,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Compute spectral embeddings from the agreement graph."""
    LOGGER.info("Computing graph embeddings (%d dims)...", n_dims)

    if G.number_of_nodes() == 0:
        return {}, []

    nodes = sorted(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes, weight="weight")

    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    L_norm = diags(np.ones(len(nodes))) - D_inv_sqrt @ A @ D_inv_sqrt

    k = min(n_dims + 1, len(nodes) - 1)
    if k < 2:
        return {}, []

    eigenvalues, eigenvectors = eigsh(L_norm, k=k, which="SM")
    embeddings_matrix = eigenvectors[:, 1 : n_dims + 1]
    embeddings_matrix = normalize(embeddings_matrix)

    embeddings = {node: embeddings_matrix[i] for i, node in enumerate(nodes)}

    LOGGER.info(
        "  Computed %d-dim embeddings for %d members",
        n_dims,
        len(nodes),
    )
    return embeddings, nodes


# ── Clustering (Agglomerative -- general, secondary) ─────────────────────────


def cluster_members(
    embeddings: dict[str, np.ndarray],
    nodes: list[str],
) -> tuple[dict[str, int], float, int]:
    """Cluster members using Agglomerative Clustering.

    Tries k=2 through k=8 and picks the k with best silhouette score.
    Returns ({member_id: cluster_label}, best_silhouette, best_k).
    """
    if not embeddings:
        return {}, 0.0, 0

    X = np.array([embeddings[n] for n in nodes])

    best_labels = None
    best_score = -1.0
    best_k = 2

    for k in range(2, min(9, len(nodes))):
        agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = agg.fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    LOGGER.info(
        "  Clustering: %d blocs, silhouette=%.3f (best of k=2..8)",
        best_k,
        best_score,
    )

    clusters = {nodes[i]: int(best_labels[i]) for i in range(len(nodes))}
    return clusters, best_score, best_k


# ── Validation ───────────────────────────────────────────────────────────────


def validate_coalitions(df: pl.DataFrame) -> dict:
    """Validate discovered coalitions against known party structure."""
    results = {}

    n_blocs = df["coalition_id"].n_unique()
    results["n_blocs"] = n_blocs

    cross_party = 0
    pure_d = 0
    pure_r = 0
    for cid in df["coalition_id"].unique().to_list():
        bloc = df.filter(pl.col("coalition_id") == cid)
        parties = set(bloc["party"].to_list())
        has_d = "Democrat" in parties
        has_r = "Republican" in parties
        if has_d and has_r:
            cross_party += 1
        elif has_d:
            pure_d += 1
        elif has_r:
            pure_r += 1

    results["cross_party_blocs"] = cross_party
    results["pure_democrat_blocs"] = pure_d
    results["pure_republican_blocs"] = pure_r
    results["cross_party_pct"] = round(100 * cross_party / max(n_blocs, 1), 1)

    return results


# ── General coalition characterization ───────────────────────────────────────


def characterize_coalitions(
    df_coalitions: pl.DataFrame,
) -> list[dict]:
    """Analyze what each coalition votes on and generate descriptive names.

    Uses FIXED category mapping (action_category == 'assignment', not 'committee').
    """
    console.print("\n[bold]Characterizing General Coalitions...[/]")

    casts_path = PROCESSED_DIR / "fact_vote_casts.parquet"
    if not casts_path.exists():
        casts_path = PROCESSED_DIR / "fact_vote_casts_raw.parquet"
    if not casts_path.exists():
        return []

    df_casts = pl.read_parquet(casts_path)
    events_path = PROCESSED_DIR / "fact_vote_events.parquet"
    if not events_path.exists():
        return []

    df_events = pl.read_parquet(events_path)
    df_bills = pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet")

    # Build bill categories using the shared helper (FIXED filter)
    bill_categories = build_bill_categories()

    event_to_bill = {
        r["vote_event_id"]: r["bill_id"]
        for r in df_events.select(["vote_event_id", "bill_id"]).to_dicts()
    }

    bill_desc = {
        r["bill_id"]: {
            "bill_number": r.get("bill_number_raw", ""),
            "description": (r.get("description", "") or "")[:80],
        }
        for r in df_bills.to_dicts()
    }

    profiles = []
    unique_cids = sorted(df_coalitions["coalition_id"].unique().to_list())

    resolved = df_casts.filter(pl.col("member_id").is_not_null())
    if "vote_cast" not in resolved.columns:
        if "cast" in resolved.columns:
            resolved = resolved.rename({"cast": "vote_cast"})
        else:
            return []

    for cid in unique_cids:
        bloc = df_coalitions.filter(pl.col("coalition_id") == cid)
        member_ids = set(bloc["member_id"].to_list())
        bloc_casts = resolved.filter(pl.col("member_id").is_in(list(member_ids)))

        # Count categories from YES votes (relative to bloc, not absolute)
        cat_yes: Counter = Counter()
        cat_total: Counter = Counter()
        bill_yes_counts: Counter = Counter()
        total_yes = 0
        total_no = 0

        for row in bloc_casts.to_dicts():
            cast = row.get("vote_cast", "")
            eid = row.get("vote_event_id", "")
            bid = event_to_bill.get(eid)
            if not bid:
                continue

            cats = bill_categories.get(bid, set())
            if cast == "Y":
                total_yes += 1
                bill_yes_counts[bid] += 1
                for cat in cats:
                    cat_yes[cat] += 1
                    cat_total[cat] += 1
            elif cast == "N":
                total_no += 1
                for cat in cats:
                    cat_total[cat] += 1

        # Top categories by YES rate difference (not raw count)
        # This differentiates: a bloc that votes YES 95% on Healthcare
        # from one that votes YES 60% — even if both have lots of votes.
        cat_yes_rates = {}
        for cat in cat_total:
            if cat_total[cat] >= 20:
                cat_yes_rates[cat] = cat_yes[cat] / cat_total[cat]

        # Sort by YES rate descending — the categories they SUPPORT most
        top_cats = sorted(cat_yes_rates, key=lambda c: cat_yes_rates[c], reverse=True)[:3]

        dem = len(bloc.filter(pl.col("party") == "Democrat"))
        rep = len(bloc.filter(pl.col("party") == "Republican"))
        size = len(bloc)

        total_votes = total_yes + total_no
        yes_rate = total_yes / total_votes if total_votes > 0 else 0.5

        cohesion = _compute_cohesion(bloc_casts, member_ids, event_to_bill)

        top_bills = bill_yes_counts.most_common(5)
        sig_bills = []
        for bid, count in top_bills:
            bd = bill_desc.get(bid, {})
            sig_bills.append(
                {
                    "bill_number": bd.get("bill_number", ""),
                    "description": bd.get("description", ""),
                    "yes_votes": count,
                }
            )

        name = _generate_coalition_name(cid, top_cats, dem, rep, size, yes_rate, cohesion)

        profile = {
            "coalition_id": cid,
            "name": name,
            "focus_areas": top_cats,
            "focus_yes_rates": {c: round(cat_yes_rates.get(c, 0), 3) for c in top_cats},
            "size": size,
            "dem_count": dem,
            "rep_count": rep,
            "yes_rate": round(yes_rate, 3),
            "cohesion": round(cohesion, 3),
            "total_votes": total_votes,
            "signature_bills": sig_bills,
        }
        profiles.append(profile)

    # Deduplicate names
    profiles = _deduplicate_coalition_names(profiles)

    with open(PROCESSED_DIR / "coalition_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    LOGGER.info("Saved %d coalition profiles", len(profiles))

    # Update coalitions.parquet with names and focus
    profile_map = {p["coalition_id"]: p for p in profiles}
    names = []
    focuses = []
    for row in df_coalitions.to_dicts():
        cid = row["coalition_id"]
        p = profile_map.get(cid, {})
        names.append(p.get("name", f"Coalition {cid + 1}"))
        focuses.append(", ".join(p.get("focus_areas", [])))

    df_coalitions = df_coalitions.with_columns(
        pl.Series("coalition_name", names, dtype=pl.Utf8),
        pl.Series("coalition_focus", focuses, dtype=pl.Utf8),
    )
    df_coalitions.write_parquet(PROCESSED_DIR / "coalitions.parquet")

    _display_coalition_profiles(profiles)
    return profiles


def _compute_cohesion(
    bloc_casts: pl.DataFrame,
    member_ids: set[str],
    event_to_bill: dict,
) -> float:
    """Compute cohesion: fraction of votes where bloc majority agrees."""
    event_votes: dict[str, Counter] = {}
    for row in bloc_casts.to_dicts():
        eid = row.get("vote_event_id", "")
        cast = row.get("vote_cast", "")
        if cast in ("Y", "N") and eid:
            event_votes.setdefault(eid, Counter())[cast] += 1

    if not event_votes:
        return 0.0

    cohesion_sum = 0.0
    for eid, counts in event_votes.items():
        total = counts["Y"] + counts["N"]
        if total > 0:
            majority = max(counts["Y"], counts["N"])
            cohesion_sum += majority / total

    return cohesion_sum / len(event_votes)


def _generate_coalition_name(
    cid: int,
    top_cats: list[str],
    dem: int,
    rep: int,
    size: int,
    yes_rate: float,
    cohesion: float,
) -> str:
    """Generate a descriptive coalition name from its profile."""
    if dem > 0 and rep > 0:
        ratio = min(dem, rep) / max(dem, rep)
        if ratio > 0.3:
            partisan = "Bipartisan"
        elif dem > rep:
            partisan = "Dem-Leaning"
        else:
            partisan = "GOP-Leaning"
    elif dem > 0:
        partisan = "Democrat"
    elif rep > 0:
        partisan = "Republican"
    else:
        partisan = "Mixed"

    if yes_rate > 0.85:
        style = "Consensus"
    elif yes_rate < 0.55:
        style = "Opposition"
    else:
        style = ""

    if top_cats:
        focus = top_cats[0]
        if len(top_cats) >= 2:
            focus = f"{top_cats[0]} & {top_cats[1]}"
    else:
        focus = "General"

    if cohesion > 0.9:
        cohesion_tag = "Bloc"
    elif cohesion > 0.75:
        cohesion_tag = "Coalition"
    else:
        cohesion_tag = "Caucus"

    parts = []
    if style:
        parts.append(style)
    parts.append(partisan)
    if focus != "General":
        parts.append(focus)
    parts.append(cohesion_tag)

    return " ".join(parts)


def _deduplicate_coalition_names(profiles: list[dict]) -> list[dict]:
    """Ensure every coalition has a unique name."""
    name_counts: Counter = Counter()
    for p in profiles:
        name_counts[p["name"]] += 1

    for p in profiles:
        if name_counts[p["name"]] <= 1:
            continue
        suffix = f" — {p['size']} members ({p['dem_count']}D/{p['rep_count']}R)"
        p["name"] = p["name"] + suffix

    name_counts2: Counter = Counter()
    for p in profiles:
        name_counts2[p["name"]] += 1
    for p in profiles:
        if name_counts2[p["name"]] > 1:
            p["name"] = f"{p['name']} #{p['coalition_id'] + 1}"

    return profiles


def _display_coalition_profiles(profiles: list[dict]) -> None:
    """Show coalition profiles with names and focus areas."""
    console.print()
    table = Table(
        title="General Voting Coalitions (secondary — see Topic Coalitions above)",
        show_lines=True,
    )
    table.add_column("Name", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Party Mix")
    table.add_column("Focus Areas (by YES rate)")
    table.add_column("YES Rate", justify="right")
    table.add_column("Cohesion", justify="right")

    for p in profiles:
        party = f"[blue]{p['dem_count']}D[/] / [red]{p['rep_count']}R[/]"
        focus_parts = []
        for cat in p.get("focus_areas", []):
            yr = p.get("focus_yes_rates", {}).get(cat)
            if yr is not None:
                focus_parts.append(f"{cat} ({yr:.0%})")
            else:
                focus_parts.append(cat)
        focus = ", ".join(focus_parts) or "General"
        table.add_row(
            p["name"],
            str(p["size"]),
            party,
            focus,
            f"{p['yes_rate']:.0%}",
            f"{p['cohesion']:.2f}",
        )
    console.print(table)


def display_coalitions(
    df: pl.DataFrame,
    silhouette: float,
    validation: dict,
) -> None:
    """Show coalition summary with quality metrics."""
    if len(df) == 0:
        console.print("[dim]No coalition data.[/]")
        return

    console.print()
    quality = Table(title="General Coalition Quality", show_lines=True)
    quality.add_column("Metric", style="bold")
    quality.add_column("Value", justify="right")
    quality.add_row("Blocs discovered", str(validation["n_blocs"]))
    quality.add_row("Silhouette score", f"{silhouette:.3f}")

    # Interpret silhouette for the user
    if silhouette < 0.25:
        quality.add_row(
            "Interpretation",
            "[yellow]Weak structure — blocs overlap significantly[/]",
        )
    elif silhouette < 0.50:
        quality.add_row(
            "Interpretation",
            "Moderate structure — some real groupings",
        )
    else:
        quality.add_row(
            "Interpretation",
            "[green]Strong structure — well-separated blocs[/]",
        )

    quality.add_row(
        "Cross-party blocs",
        f"{validation['cross_party_blocs']} ({validation['cross_party_pct']}%)",
    )
    quality.add_row(
        "Members classified",
        f"{len(df)} / {len(df)} (100%)",
    )
    console.print(quality)

    console.print()
    table = Table(
        title="General Voting Clusters",
        show_lines=True,
    )
    table.add_column("Bloc", style="bold", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Party Mix")
    table.add_column("Chamber")
    table.add_column("Sample Members")

    for cid in sorted(df["coalition_id"].unique().to_list()):
        bloc = df.filter(pl.col("coalition_id") == cid)
        dem = len(bloc.filter(pl.col("party") == "Democrat"))
        rep = len(bloc.filter(pl.col("party") == "Republican"))
        senate = len(bloc.filter(pl.col("chamber") == "Senate"))
        house = len(bloc.filter(pl.col("chamber") == "House"))
        names = bloc["name"].to_list()

        party_str = f"[blue]{dem}D[/] / [red]{rep}R[/]"
        chamber_str = f"{senate}S / {house}H"
        sample = ", ".join(names[:3])
        if len(names) > 3:
            sample += f" +{len(names) - 3}"

        table.add_row(str(cid), str(len(bloc)), party_str, chamber_str, sample)

    console.print(table)


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_coalition_discovery() -> pl.DataFrame:
    """Full automated coalition discovery pipeline.

    1. Topic-based coalitions (PRIMARY) — per-topic voting blocs
    2. General agreement clustering (SECONDARY) — overall voting similarity,
       now using Node2Vec embeddings (Phase 3) instead of spectral embeddings.
    """
    console.print("\n[bold]Coalition Discovery[/]")

    casts_path = PROCESSED_DIR / "fact_vote_casts.parquet"
    if not casts_path.exists():
        casts_path = PROCESSED_DIR / "fact_vote_casts_raw.parquet"

    df_casts = pl.read_parquet(casts_path)
    df_members = pl.read_parquet(PROCESSED_DIR / "dim_members.parquet")

    # ── Step 1: Topic-based coalitions (PRIMARY) ──
    discover_topic_coalitions()

    # ── Step 2: General clustering (SECONDARY) — using Node2Vec embeddings ──
    console.print("\n[bold]General Clustering[/] [dim](Node2Vec embeddings)[/]")

    # Load pre-computed Node2Vec embeddings (generated by node_embedder.py).
    # Falls back to spectral embeddings from the agreement graph if the
    # Node2Vec embeddings file doesn't exist yet.
    from ilga_graph.ml.node_embedder import load_embeddings as _load_n2v

    embeddings = _load_n2v()

    if embeddings:
        nodes = sorted(embeddings.keys())
        n_dims = len(next(iter(embeddings.values())))
        LOGGER.info(
            "Using Node2Vec embeddings: %d members, %d dims",
            len(nodes),
            n_dims,
        )
        console.print(f"  Using [cyan]Node2Vec[/] embeddings: {len(nodes)} members, {n_dims}d")
    else:
        # Fallback: compute spectral embeddings from agreement graph
        LOGGER.info("Node2Vec embeddings not available — falling back to spectral.")
        console.print("  [yellow]Fallback:[/] computing spectral embeddings")
        G = build_agreement_graph(df_casts)
        G = add_cosponsor_edges(G)
        embeddings, nodes = compute_embeddings(G, n_dims=32)
        n_dims = 32

    clusters, silhouette, best_k = cluster_members(embeddings, nodes)

    member_lookup = {m["member_id"]: m for m in df_members.to_dicts()}

    rows = []
    for mid, cluster_id in clusters.items():
        m = member_lookup.get(mid, {})
        emb = embeddings.get(mid, np.zeros(n_dims))
        rows.append(
            {
                "member_id": mid,
                "name": m.get("name", ""),
                "party": m.get("party", ""),
                "chamber": m.get("chamber", ""),
                "district": m.get("district"),
                "coalition_id": cluster_id,
                **{f"emb_{i}": float(emb[i]) for i in range(len(emb))},
            }
        )

    df_coalitions = pl.DataFrame(rows)
    df_coalitions.write_parquet(PROCESSED_DIR / "coalitions.parquet")
    LOGGER.info("Saved coalitions to processed/coalitions.parquet")

    # Save embeddings (Node2Vec embeddings are already saved by
    # node_embedder.py; this re-save ensures the file exists even
    # in the spectral fallback path)
    emb_rows = []
    for mid in nodes:
        emb = embeddings.get(mid)
        if emb is not None:
            emb_rows.append(
                {
                    "member_id": mid,
                    **{f"dim_{i}": float(emb[i]) for i in range(len(emb))},
                }
            )
    if emb_rows:
        pl.DataFrame(emb_rows).write_parquet(PROCESSED_DIR / "member_embeddings.parquet")

    validation = validate_coalitions(df_coalitions)
    display_coalitions(df_coalitions, silhouette, validation)

    # Characterize general coalitions with FIXED category mapping
    characterize_coalitions(df_coalitions)

    df_coalitions = pl.read_parquet(PROCESSED_DIR / "coalitions.parquet")
    return df_coalitions
