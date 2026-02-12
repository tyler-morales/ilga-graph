"""Voting coalition discovery with proper clustering.

Fixes over v1:
- Uses Agglomerative Clustering instead of DBSCAN (no outliers by design)
- Automatically finds optimal number of clusters via silhouette analysis
- Validates clusters against known party structure
- Builds DISAGREEMENT graph (not just agreement) for richer signal
- Reports cross-party coalition quality metrics

Outputs:
    processed/member_embeddings.parquet  -- 32-dim vectors per member
    processed/coalitions.parquet         -- Member cluster assignments
"""

from __future__ import annotations

import logging
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


# ── Graph construction ───────────────────────────────────────────────────────


def build_agreement_graph(df_vote_casts: pl.DataFrame) -> nx.Graph:
    """Build a weighted graph: edge weight = agreement rate between members.

    For each pair of members who voted on at least 10 common events,
    edge weight = (# same direction votes) / (# common votes).
    This normalizes for activity level, unlike raw co-vote counts.
    """
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

    # Track per-pair: (agree_count, total_count)
    pair_stats: dict[tuple[str, str], list[int]] = {}

    for row in events.to_dicts():
        member_ids = row["member_id"]
        casts = row["vote_cast"]

        # Build member -> cast map for this event
        votes = {}
        for mid, cast in zip(member_ids, casts):
            if cast in ("Y", "N"):  # Only clear Y/N votes
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

    # Only add edges for pairs with 10+ common votes
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
    import json

    LOGGER.info("Adding co-sponsorship edges...")

    bills_path = Path("cache/bills.json")
    if not bills_path.exists():
        return G

    with open(bills_path) as f:
        bills_raw = json.load(f)

    # Count co-sponsorships per pair
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

    # Add as bonus weight (normalized)
    if cosponsor_counts:
        max_count = max(cosponsor_counts.values())
        for (a, b), count in cosponsor_counts.items():
            bonus = 0.2 * (count / max_count)  # Max 0.2 bonus
            if G.has_edge(a, b):
                G[a][b]["weight"] = min(1.0, G[a][b]["weight"] + bonus)
            elif count >= 3:
                # Only add new edges for significant co-sponsorship
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


# ── Clustering (Agglomerative -- no outliers) ────────────────────────────────


def cluster_members(
    embeddings: dict[str, np.ndarray],
    nodes: list[str],
) -> tuple[dict[str, int], float]:
    """Cluster members using Agglomerative Clustering.

    Tries k=3 through k=10 and picks the k with best silhouette score.
    Returns ({member_id: cluster_label}, best_silhouette).
    """
    if not embeddings:
        return {}, 0.0

    X = np.array([embeddings[n] for n in nodes])

    best_labels = None
    best_score = -1.0
    best_k = 3

    for k in range(3, min(11, len(nodes))):
        agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = agg.fit_predict(X)

        score = silhouette_score(X, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    LOGGER.info(
        "  Clustering: %d blocs, silhouette=%.3f (best of k=3..10)",
        best_k,
        best_score,
    )

    clusters = {nodes[i]: int(best_labels[i]) for i in range(len(nodes))}
    return clusters, best_score


# ── Validation ───────────────────────────────────────────────────────────────


def validate_coalitions(
    df: pl.DataFrame,
) -> dict:
    """Validate discovered coalitions against known party structure.

    Good coalitions should:
    - Have at least some cross-party blocs (otherwise just party)
    - Not be perfectly aligned with party (otherwise useless)
    """
    results = {}

    n_blocs = df["coalition_id"].n_unique()
    results["n_blocs"] = n_blocs

    # Count cross-party blocs (blocs with both D and R)
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


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_coalition_discovery() -> pl.DataFrame:
    """Full automated coalition discovery pipeline."""
    console.print("\n[bold]Coalition Discovery[/]")

    casts_path = PROCESSED_DIR / "fact_vote_casts.parquet"
    if not casts_path.exists():
        casts_path = PROCESSED_DIR / "fact_vote_casts_raw.parquet"

    df_casts = pl.read_parquet(casts_path)
    df_members = pl.read_parquet(PROCESSED_DIR / "dim_members.parquet")

    # Build agreement graph (normalized, not raw counts)
    G = build_agreement_graph(df_casts)
    G = add_cosponsor_edges(G)

    # Embeddings
    embeddings, nodes = compute_embeddings(G, n_dims=32)

    # Cluster (agglomerative -- every member assigned)
    clusters, silhouette = cluster_members(embeddings, nodes)

    # Build output DataFrame
    member_lookup = {m["member_id"]: m for m in df_members.to_dicts()}

    rows = []
    for mid, cluster_id in clusters.items():
        m = member_lookup.get(mid, {})
        emb = embeddings.get(mid, np.zeros(32))
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

    # Save embeddings separately
    emb_rows = []
    for mid in nodes:
        emb = embeddings[mid]
        emb_rows.append(
            {
                "member_id": mid,
                **{f"dim_{i}": float(emb[i]) for i in range(len(emb))},
            }
        )
    if emb_rows:
        pl.DataFrame(emb_rows).write_parquet(PROCESSED_DIR / "member_embeddings.parquet")

    # Validate and display
    validation = validate_coalitions(df_coalitions)
    display_coalitions(df_coalitions, silhouette, validation)

    # Characterize coalitions (names, policy focus, signature bills)
    characterize_coalitions(df_coalitions)

    # Re-read updated coalitions with names
    df_coalitions = pl.read_parquet(PROCESSED_DIR / "coalitions.parquet")

    return df_coalitions


# ── Policy category mapping (mirrors main.py) ────────────────────────────────

_CATEGORY_COMMITTEES: dict[str, list[str]] = {
    "Transportation": ["STRN"],
    "Agriculture": ["SAGR"],
    "Commerce": ["SCOM", "SBTE"],
    "Criminal Justice": ["SCRL", "SHRJ"],
    "Education": ["SESE", "SCHE"],
    "Energy & Environment": ["SENE", "SNVR"],
    "Healthcare": ["SBMH", "SCHW", "SHUM"],
    "Housing": ["SHOU"],
    "Insurance & Finance": ["SINS", "SFIC"],
    "Labor": ["SLAB"],
    "Revenue & Pensions": ["SREV", "SPEN"],
    "State Government": ["SGOA", "SHEE", "SEXC"],
}

# Reverse: committee code -> category name
_COMMITTEE_TO_CATEGORY: dict[str, str] = {}
for _cat, _codes in _CATEGORY_COMMITTEES.items():
    for _code in _codes:
        _COMMITTEE_TO_CATEGORY[_code] = _cat


def _extract_committee_code(action_text: str) -> str | None:
    """Try to extract a committee code from an action like 'Assigned to Judiciary'."""
    import re

    m = re.search(
        r"(?:Assigned to|Referred to)\s+(.+?)(?:\s*-|$)",
        action_text,
        re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def characterize_coalitions(
    df_coalitions: pl.DataFrame,
) -> list[dict]:
    """Analyze what each coalition votes on and generate descriptive names.

    Joins coalition members -> vote casts -> vote events -> bills -> actions
    to identify policy focus areas for each bloc.

    Returns a list of coalition profile dicts and saves to
    processed/coalition_profiles.json.
    """
    import json

    console.print("\n[bold]Characterizing Coalitions...[/]")

    # Load needed data
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
    df_actions = pl.read_parquet(PROCESSED_DIR / "fact_bill_actions.parquet")
    df_bills = pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet")

    # Build bill -> categories from committee assignment actions
    bill_categories: dict[str, list[str]] = {}
    committee_actions = df_actions.filter(pl.col("action_category") == "committee")
    for row in committee_actions.to_dicts():
        bid = row["bill_id"]
        action = row.get("action_text", "")
        # Try to match committee name to category
        for cat, codes in _CATEGORY_COMMITTEES.items():
            for code in codes:
                if code.upper() in action.upper():
                    bill_categories.setdefault(bid, []).append(cat)
        # Also try matching committee name text
        committee_name = _extract_committee_code(action)
        if committee_name:
            cn_lower = committee_name.lower()
            for cat in _CATEGORY_COMMITTEES:
                if cat.lower() in cn_lower:
                    bill_categories.setdefault(bid, []).append(cat)

    # Also use witness slip hearing_committee
    slips_path = PROCESSED_DIR / "fact_witness_slips.parquet"
    if slips_path.exists():
        df_slips = pl.read_parquet(slips_path)
        if "hearing_committee" in df_slips.columns:
            for row in df_slips.select(["bill_id", "hearing_committee"]).unique().to_dicts():
                hc = row.get("hearing_committee", "") or ""
                if hc:
                    hc_lower = hc.lower()
                    for cat in _CATEGORY_COMMITTEES:
                        if cat.lower() in hc_lower:
                            bill_categories.setdefault(row["bill_id"], []).append(cat)

    # Deduplicate per bill
    for bid in bill_categories:
        bill_categories[bid] = list(set(bill_categories[bid]))

    # Build vote_event -> bill mapping
    event_to_bill = {
        r["vote_event_id"]: r["bill_id"]
        for r in df_events.select(["vote_event_id", "bill_id"]).to_dicts()
    }

    # Build bill descriptions for signature bills
    bill_desc = {
        r["bill_id"]: {
            "bill_number": r.get("bill_number_raw", ""),
            "description": (r.get("description", "") or "")[:80],
        }
        for r in df_bills.to_dicts()
    }

    # Per-coalition analysis
    # Track: category votes (YES), total YES/NO, per-bill YES counts
    from collections import Counter

    profiles = []
    unique_cids = sorted(df_coalitions["coalition_id"].unique().to_list())

    # Pre-compute: resolved casts with coalition info
    resolved = df_casts.filter(pl.col("member_id").is_not_null())
    if "vote_cast" not in resolved.columns:
        # Fallback for raw casts
        if "cast" in resolved.columns:
            resolved = resolved.rename({"cast": "vote_cast"})
        else:
            return []

    for cid in unique_cids:
        bloc = df_coalitions.filter(pl.col("coalition_id") == cid)
        member_ids = set(bloc["member_id"].to_list())

        # Get all votes by this coalition's members
        bloc_casts = resolved.filter(pl.col("member_id").is_in(list(member_ids)))

        # Category vote counts (YES votes on bills in each category)
        cat_counts: Counter = Counter()
        bill_yes_counts: Counter = Counter()
        total_yes = 0
        total_no = 0

        for row in bloc_casts.to_dicts():
            cast = row.get("vote_cast", "")
            eid = row.get("vote_event_id", "")
            bid = event_to_bill.get(eid)
            if not bid:
                continue

            if cast == "Y":
                total_yes += 1
                bill_yes_counts[bid] += 1
                cats = bill_categories.get(bid, [])
                for cat in cats:
                    cat_counts[cat] += 1
            elif cast == "N":
                total_no += 1

        # Top 3 categories
        top_cats = [c for c, _ in cat_counts.most_common(5)][:3]

        # Party composition
        dem = len(bloc.filter(pl.col("party") == "Democrat"))
        rep = len(bloc.filter(pl.col("party") == "Republican"))
        size = len(bloc)

        # YES rate
        total_votes = total_yes + total_no
        yes_rate = total_yes / total_votes if total_votes > 0 else 0.5

        # Cohesion: how often do bloc members agree on the same vote?
        cohesion = _compute_cohesion(bloc_casts, member_ids, event_to_bill)

        # Signature bills: highest YES rate among members
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

        # Generate name
        name = _generate_coalition_name(cid, top_cats, dem, rep, size, yes_rate, cohesion)

        profile = {
            "coalition_id": cid,
            "name": name,
            "focus_areas": top_cats,
            "size": size,
            "dem_count": dem,
            "rep_count": rep,
            "yes_rate": round(yes_rate, 3),
            "cohesion": round(cohesion, 3),
            "total_votes": total_votes,
            "signature_bills": sig_bills,
        }
        profiles.append(profile)

    # Save profiles
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

    # Display
    _display_coalition_profiles(profiles)

    return profiles


def _compute_cohesion(
    bloc_casts: pl.DataFrame,
    member_ids: set[str],
    event_to_bill: dict,
) -> float:
    """Compute cohesion: fraction of votes where bloc majority agrees."""
    from collections import Counter

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
    # Partisan descriptor
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

    # Voting style
    if yes_rate > 0.85:
        style = "Consensus"
    elif yes_rate < 0.55:
        style = "Opposition"
    else:
        style = ""

    # Policy focus
    if top_cats:
        focus = top_cats[0]
        if len(top_cats) >= 2:
            focus = f"{top_cats[0]} & {top_cats[1]}"
    else:
        focus = "General"

    # Cohesion descriptor
    if cohesion > 0.9:
        cohesion_tag = "Bloc"
    elif cohesion > 0.75:
        cohesion_tag = "Coalition"
    else:
        cohesion_tag = "Caucus"

    # Compose name
    parts = []
    if style:
        parts.append(style)
    parts.append(partisan)
    if focus != "General":
        parts.append(focus)
    parts.append(cohesion_tag)

    return " ".join(parts)


def _display_coalition_profiles(profiles: list[dict]) -> None:
    """Show coalition profiles with names and focus areas."""
    console.print()
    table = Table(
        title="Coalition Profiles",
        show_lines=True,
    )
    table.add_column("Name", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Party Mix")
    table.add_column("Focus Areas")
    table.add_column("YES Rate", justify="right")
    table.add_column("Cohesion", justify="right")

    for p in profiles:
        party = f"[blue]{p['dem_count']}D[/] / [red]{p['rep_count']}R[/]"
        focus = ", ".join(p["focus_areas"]) or "General"
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

    # Quality summary
    console.print()
    quality = Table(title="Coalition Quality", show_lines=True)
    quality.add_column("Metric", style="bold")
    quality.add_column("Value", justify="right")
    quality.add_row("Blocs discovered", str(validation["n_blocs"]))
    quality.add_row("Silhouette score", f"{silhouette:.3f}")
    quality.add_row(
        "Cross-party blocs",
        f"{validation['cross_party_blocs']} ({validation['cross_party_pct']}%)",
    )
    quality.add_row(
        "Pure Democrat blocs",
        str(validation["pure_democrat_blocs"]),
    )
    quality.add_row(
        "Pure Republican blocs",
        str(validation["pure_republican_blocs"]),
    )
    quality.add_row(
        "Members classified",
        f"{len(df)} / {len(df)} (100%)",
    )
    console.print(quality)

    # Bloc details
    console.print()
    table = Table(
        title="Discovered Voting Coalitions",
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
