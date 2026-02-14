"""Generate Node2Vec embeddings for legislators from the co-sponsorship graph.

Uses the Node2Vec algorithm (second-order random walks → Word2Vec skip-gram)
to produce dense vector representations for each legislator node.  These
embeddings capture the structural position of each legislator within the
co-sponsorship network — legislators who frequently collaborate will have
nearby vectors.

Tuning guidance:
  - ``p`` (return parameter): Lower ``p`` encourages revisiting recent nodes
    (local / BFS-like exploration → homophily).
  - ``q`` (in-out parameter): Lower ``q`` encourages exploring outward
    (global / DFS-like exploration → structural equivalence).
  - ``p=1, q=1`` is the balanced default (equivalent to DeepWalk).

Outputs:
    processed/member_embeddings.parquet  — one row per legislator with
                                           dim_0 … dim_{n-1} columns.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
from rich.console import Console

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")

console = Console()

# ── Default hyperparameters (overridable via env vars) ────────────────────────
_DIMENSIONS = int(os.environ.get("ILGA_N2V_DIMS", "64"))
_WALK_LENGTH = int(os.environ.get("ILGA_N2V_WALK_LENGTH", "30"))
_NUM_WALKS = int(os.environ.get("ILGA_N2V_NUM_WALKS", "200"))
_P = float(os.environ.get("ILGA_N2V_P", "1.0"))
_Q = float(os.environ.get("ILGA_N2V_Q", "1.0"))
_WINDOW = int(os.environ.get("ILGA_N2V_WINDOW", "10"))
_WORKERS = int(os.environ.get("ILGA_N2V_WORKERS", "4"))


def generate_embeddings(
    G: nx.Graph,
    *,
    dimensions: int = _DIMENSIONS,
    walk_length: int = _WALK_LENGTH,
    num_walks: int = _NUM_WALKS,
    p: float = _P,
    q: float = _Q,
    window: int = _WINDOW,
    workers: int = _WORKERS,
) -> dict[str, np.ndarray]:
    """Run Node2Vec on a NetworkX graph and return per-node embeddings.

    Parameters
    ----------
    G : nx.Graph
        Weighted undirected graph (from ``graph_builder.build_cosponsor_graph``).
    dimensions : int
        Embedding vector size (default 64).
    walk_length : int
        Length of each random walk (default 30).
    num_walks : int
        Number of random walks per node (default 200).
    p : float
        Return parameter — controls likelihood of revisiting a node.
    q : float
        In-out parameter — controls search to differentiate inward vs outward nodes.
    window : int
        Window size for the skip-gram model (default 10).
    workers : int
        Number of parallel workers (default 4).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from node ID (member_id) to embedding vector.
    """
    from node2vec import Node2Vec

    if G.number_of_nodes() == 0:
        LOGGER.warning("Empty graph — no embeddings to generate.")
        return {}

    # Isolated nodes (degree 0) cause Node2Vec to produce degenerate walks.
    # We generate embeddings only for the connected component and assign
    # zero vectors to isolated nodes afterward.
    connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
    isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]

    if not connected_nodes:
        LOGGER.warning("All nodes are isolated — returning zero embeddings.")
        return {n: np.zeros(dimensions) for n in G.nodes()}

    G_connected = G.subgraph(connected_nodes).copy()

    LOGGER.info(
        "Running Node2Vec: %d connected nodes (%d isolated), "
        "dims=%d, walks=%d, length=%d, p=%.1f, q=%.1f",
        len(connected_nodes),
        len(isolated_nodes),
        dimensions,
        num_walks,
        walk_length,
        p,
        q,
    )

    console.print(
        f"[bold]Node2Vec[/] — {len(connected_nodes)} nodes, "
        f"{G_connected.number_of_edges()} edges, "
        f"{dimensions}d embeddings"
    )

    # Initialize and fit Node2Vec
    node2vec_model = Node2Vec(
        G_connected,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        quiet=False,
    )

    model = node2vec_model.fit(
        window=window,
        min_count=1,
        batch_words=4,
    )

    # Extract embeddings
    embeddings: dict[str, np.ndarray] = {}
    for node in connected_nodes:
        embeddings[str(node)] = model.wv.get_vector(str(node))

    # Assign zero vectors for isolated nodes
    for node in isolated_nodes:
        embeddings[str(node)] = np.zeros(dimensions)

    LOGGER.info(
        "Generated %d-dim embeddings for %d legislators (%d connected, %d isolated).",
        dimensions,
        len(embeddings),
        len(connected_nodes),
        len(isolated_nodes),
    )

    return embeddings


def save_embeddings(
    embeddings: dict[str, np.ndarray],
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Save embeddings to a Parquet file.

    Parameters
    ----------
    embeddings : dict[str, np.ndarray]
        Mapping from member_id to embedding vector.
    output_path : Path | None
        Output file path. Defaults to ``processed/member_embeddings.parquet``.

    Returns
    -------
    pl.DataFrame
        The saved DataFrame with columns ``member_id, dim_0, ..., dim_{n-1}``.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "member_embeddings.parquet"

    if not embeddings:
        LOGGER.warning("No embeddings to save.")
        return pl.DataFrame()

    # Detect dimensionality from first vector
    sample = next(iter(embeddings.values()))
    n_dims = len(sample)

    rows = []
    for member_id, vec in sorted(embeddings.items()):
        row: dict = {"member_id": member_id}
        for i in range(n_dims):
            row[f"dim_{i}"] = float(vec[i])
        rows.append(row)

    df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    LOGGER.info("Saved %d embeddings (%d dims) to %s", len(rows), n_dims, output_path)
    console.print(f"  Saved [cyan]{len(rows)}[/] embeddings ({n_dims}d) → {output_path}")

    return df


def load_embeddings(
    path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load embeddings from a Parquet file.

    Parameters
    ----------
    path : Path | None
        Path to the embeddings Parquet. Defaults to
        ``processed/member_embeddings.parquet``.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from member_id to embedding vector.
    """
    if path is None:
        path = PROCESSED_DIR / "member_embeddings.parquet"

    if not path.exists():
        LOGGER.warning("Embeddings file not found: %s", path)
        return {}

    df = pl.read_parquet(path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]

    embeddings: dict[str, np.ndarray] = {}
    for row in df.to_dicts():
        vec = np.array([row[c] for c in dim_cols], dtype=np.float32)
        embeddings[row["member_id"]] = vec

    LOGGER.info("Loaded %d embeddings (%d dims) from %s", len(embeddings), len(dim_cols), path)
    return embeddings


def run_embedding_pipeline(
    *,
    dimensions: int = _DIMENSIONS,
    walk_length: int = _WALK_LENGTH,
    num_walks: int = _NUM_WALKS,
    p: float = _P,
    q: float = _Q,
) -> pl.DataFrame:
    """End-to-end: build graph → generate embeddings → save to Parquet.

    This is the top-level function called by ``ml_run.py`` and the
    ``make ml-embed`` Makefile target.

    Returns the embeddings DataFrame.
    """
    from ilga_graph.ml.graph_builder import build_cosponsor_graph

    console.print("\n[bold]Graph Embedding Generation (Node2Vec)[/]")

    # Step 1: Build co-sponsorship graph
    G = build_cosponsor_graph()

    # Step 2: Generate Node2Vec embeddings
    embeddings = generate_embeddings(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
    )

    # Step 3: Save to Parquet
    df = save_embeddings(embeddings)

    return df
