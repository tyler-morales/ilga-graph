"""Automated voting coalition discovery.

Builds a legislator graph from co-voting and co-sponsorship patterns, computes
graph embeddings, and clusters members into voting blocs. Fully unsupervised.

Outputs:
    processed/member_embeddings.parquet  -- 64-dim vectors per member
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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")

console = Console()


# ── Graph construction ───────────────────────────────────────────────────────


def build_covote_graph(df_vote_casts: pl.DataFrame) -> nx.Graph:
    """Build a weighted graph where edges = co-voting frequency.

    Two members get an edge if they voted the same way (both Y or both N)
    on the same vote event. Weight = number of shared same-direction votes.
    """
    LOGGER.info("Building co-vote graph...")

    # Only use resolved casts with member_id
    resolved = df_vote_casts.filter(pl.col("member_id").is_not_null())
    if len(resolved) == 0:
        LOGGER.warning("No resolved vote casts. Returning empty graph.")
        return nx.Graph()

    # For each vote event, group members by their vote direction
    # Then every pair who voted the same way gets an edge increment
    G = nx.Graph()

    events = resolved.group_by("vote_event_id").agg(
        pl.col("member_id"),
        pl.col("vote_cast"),
    )

    edge_weights: dict[tuple[str, str], int] = {}

    for row in events.to_dicts():
        member_ids = row["member_id"]
        casts = row["vote_cast"]

        # Group by vote direction
        by_cast: dict[str, list[str]] = {}
        for mid, cast in zip(member_ids, casts):
            by_cast.setdefault(cast, []).append(mid)

        # For each direction group, add edges between all pairs
        for cast_group in by_cast.values():
            for i in range(len(cast_group)):
                for j in range(i + 1, len(cast_group)):
                    a, b = cast_group[i], cast_group[j]
                    key = (min(a, b), max(a, b))
                    edge_weights[key] = edge_weights.get(key, 0) + 1

    # Add edges to graph
    for (a, b), weight in edge_weights.items():
        G.add_edge(a, b, weight=weight)

    LOGGER.info(
        "  Co-vote graph: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def build_cosponsor_graph(df_bills: pl.DataFrame) -> nx.Graph:
    """Build a co-sponsorship graph from bill sponsor IDs.

    Reads from the raw bills JSON since dim_bills only has the primary sponsor.
    Two members share an edge if they co-sponsor the same bill.
    """
    import json

    LOGGER.info("Building co-sponsorship graph...")

    bills_path = Path("cache/bills.json")
    if not bills_path.exists():
        LOGGER.warning("cache/bills.json not found.")
        return nx.Graph()

    with open(bills_path) as f:
        bills_raw = json.load(f)

    G = nx.Graph()
    edge_weights: dict[tuple[str, str], int] = {}

    for b in bills_raw.values():
        all_ids = b.get("sponsor_ids", []) + b.get("house_sponsor_ids", [])
        if len(all_ids) < 2:
            continue

        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                a, b_id = all_ids[i], all_ids[j]
                key = (min(a, b_id), max(a, b_id))
                edge_weights[key] = edge_weights.get(key, 0) + 1

    for (a, b_id), weight in edge_weights.items():
        G.add_edge(a, b_id, weight=weight)

    LOGGER.info(
        "  Co-sponsor graph: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def merge_graphs(g1: nx.Graph, g2: nx.Graph) -> nx.Graph:
    """Merge two weighted graphs, summing edge weights."""
    G = nx.Graph()
    for u, v, d in g1.edges(data=True):
        w = d.get("weight", 1)
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    for u, v, d in g2.edges(data=True):
        w = d.get("weight", 1)
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G


# ── Graph embeddings (spectral) ─────────────────────────────────────────────


def compute_embeddings(
    G: nx.Graph,
    n_dims: int = 32,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Compute spectral embeddings for each node in the graph.

    Uses the normalized Laplacian eigenvectors as node embeddings.
    This is a lightweight alternative to Node2Vec that requires no
    extra dependencies and works well for moderate-sized graphs.
    """
    LOGGER.info("Computing graph embeddings (%d dims)...", n_dims)

    if G.number_of_nodes() == 0:
        return {}, []

    # Get adjacency matrix
    nodes = sorted(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes, weight="weight")

    # Compute normalized Laplacian eigenvectors
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # avoid division by zero
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    L_norm = diags(np.ones(len(nodes))) - D_inv_sqrt @ A @ D_inv_sqrt

    # Use min of available dims and requested dims
    k = min(n_dims + 1, len(nodes) - 1)
    if k < 2:
        LOGGER.warning("Graph too small for embeddings.")
        return {}, []

    eigenvalues, eigenvectors = eigsh(L_norm, k=k, which="SM")

    # Skip the first eigenvector (trivial constant vector)
    embeddings_matrix = eigenvectors[:, 1 : n_dims + 1]

    # Normalize embeddings
    embeddings_matrix = normalize(embeddings_matrix)

    embeddings = {node: embeddings_matrix[i] for i, node in enumerate(nodes)}

    LOGGER.info("  Computed %d-dim embeddings for %d members", n_dims, len(nodes))
    return embeddings, nodes


# ── Clustering ───────────────────────────────────────────────────────────────


def cluster_members(
    embeddings: dict[str, np.ndarray],
    nodes: list[str],
    *,
    eps: float = 0.5,
    min_samples: int = 3,
) -> dict[str, int]:
    """Cluster members using DBSCAN on their embeddings.

    Returns {member_id: cluster_label}. Label -1 = outlier/unassigned.
    """
    if not embeddings:
        return {}

    X = np.array([embeddings[n] for n in nodes])

    # Try multiple eps values and pick the one with best silhouette
    from sklearn.metrics import silhouette_score

    best_labels = None
    best_score = -1
    best_eps = eps

    for trial_eps in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        db = DBSCAN(eps=trial_eps, min_samples=min_samples, metric="cosine")
        labels = db.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue

        # Only compute silhouette on assigned (non-outlier) points
        mask = labels != -1
        if mask.sum() < 2:
            continue

        score = silhouette_score(X[mask], labels[mask], metric="cosine")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_eps = trial_eps

    if best_labels is None:
        # Fallback: use PCA + K-means instead
        LOGGER.info("  DBSCAN found no clusters. Falling back to K-means.")
        from sklearn.cluster import KMeans

        # Use PCA to reduce first
        pca = PCA(n_components=min(8, X.shape[1]))
        X_pca = pca.fit_transform(X)

        # Try 3-6 clusters
        best_labels_km = None
        best_score_km = -1
        for k in range(3, min(7, len(nodes))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_km = km.fit_predict(X_pca)
            score_km = silhouette_score(X_pca, labels_km)
            if score_km > best_score_km:
                best_score_km = score_km
                best_labels_km = labels_km

        best_labels = (
            best_labels_km if best_labels_km is not None else np.zeros(len(nodes), dtype=int)
        )
        best_score = best_score_km if best_score_km > 0 else 0

    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    outliers = int((best_labels == -1).sum())
    LOGGER.info(
        "  Clustering: %d clusters, %d outliers (silhouette=%.3f, eps=%.2f)",
        n_clusters,
        outliers,
        best_score,
        best_eps,
    )

    return {nodes[i]: int(best_labels[i]) for i in range(len(nodes))}


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_coalition_discovery() -> pl.DataFrame:
    """Full automated coalition discovery pipeline.

    Returns DataFrame of members with cluster assignments.
    """
    console.print("\n[bold]Coalition Discovery[/]")

    # Load resolved vote casts
    casts_path = PROCESSED_DIR / "fact_vote_casts.parquet"
    if not casts_path.exists():
        # Fall back to raw casts
        casts_path = PROCESSED_DIR / "fact_vote_casts_raw.parquet"

    df_casts = pl.read_parquet(casts_path)
    df_members = pl.read_parquet(PROCESSED_DIR / "dim_members.parquet")

    # Build graphs
    G_covote = build_covote_graph(df_casts)
    G_cosponsor = build_cosponsor_graph(pl.read_parquet(PROCESSED_DIR / "dim_bills.parquet"))

    # Merge: co-voting + co-sponsorship
    G = merge_graphs(G_covote, G_cosponsor)
    LOGGER.info(
        "Merged graph: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    # Embeddings
    embeddings, nodes = compute_embeddings(G, n_dims=32)

    # Cluster
    clusters = cluster_members(embeddings, nodes)

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

    # Save
    df_coalitions.write_parquet(PROCESSED_DIR / "coalitions.parquet")
    LOGGER.info("Saved coalitions to processed/coalitions.parquet")

    # Also save embeddings separately
    emb_rows = []
    for mid in nodes:
        emb = embeddings[mid]
        emb_rows.append({"member_id": mid, **{f"dim_{i}": float(emb[i]) for i in range(len(emb))}})
    if emb_rows:
        pl.DataFrame(emb_rows).write_parquet(PROCESSED_DIR / "member_embeddings.parquet")

    display_coalitions(df_coalitions)
    return df_coalitions


def display_coalitions(df: pl.DataFrame) -> None:
    """Show coalition summary."""
    if len(df) == 0:
        console.print("[dim]No coalition data.[/]")
        return

    # Group by coalition
    groups = df.group_by("coalition_id").agg(
        pl.len().alias("members"),
        pl.col("party").value_counts().alias("party_breakdown"),
        pl.col("name").alias("member_names"),
    )

    console.print()
    table = Table(title="Discovered Voting Coalitions", show_lines=True)
    table.add_column("Coalition", style="bold", justify="center")
    table.add_column("Members", justify="right")
    table.add_column("Party Mix")
    table.add_column("Sample Members")

    for row in groups.sort("coalition_id").to_dicts():
        cid = row["coalition_id"]
        count = row["members"]
        names = row["member_names"]

        # Get party breakdown from the coalition df
        coalition_members = df.filter(pl.col("coalition_id") == cid)
        dem = len(coalition_members.filter(pl.col("party") == "Democrat"))
        rep = len(coalition_members.filter(pl.col("party") == "Republican"))

        label = "Outliers" if cid == -1 else f"Bloc {cid}"
        party_str = f"[blue]{dem}D[/] / [red]{rep}R[/]"
        sample = ", ".join(names[:4])
        if len(names) > 4:
            sample += f" +{len(names) - 4} more"

        table.add_row(label, str(count), party_str, sample)

    console.print(table)
