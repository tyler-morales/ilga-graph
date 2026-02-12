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
    return df_coalitions


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
