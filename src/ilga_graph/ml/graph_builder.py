"""Build the weighted co-sponsorship graph for the Illinois General Assembly.

Constructs a NetworkX undirected graph where:
  - **Nodes** = legislators (member IDs from dim_members.parquet)
  - **Edges** = co-sponsorship relationships
  - **Weights** = number of bills two legislators co-sponsored together

The graph is the foundation for Node2Vec embedding generation
(``node_embedder.py``) and can also feed coalition clustering.

Design decisions:
  - Only substantive bills (SB/HB) are counted — resolutions are excluded.
  - Bills with more than ``max_sponsors_per_bill`` co-sponsors are dropped
    as likely ceremonial / non-controversial "token" co-sponsorships
    (risk mitigation from the Phase 3 design doc).
  - Every legislator in ``dim_members.parquet`` is added as a node even
    if they have zero co-sponsorship edges, so downstream embeddings
    cover the full legislature.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
import polars as pl

LOGGER = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
PROCESSED_DIR = Path("processed")

# Bill type prefixes considered substantive (not ceremonial resolutions)
_SUBSTANTIVE_PREFIXES = ("SB", "HB")


def build_cosponsor_graph(
    *,
    min_shared_bills: int = 1,
    max_sponsors_per_bill: int | None = 40,
) -> nx.Graph:
    """Build a weighted, undirected co-sponsorship graph.

    Parameters
    ----------
    min_shared_bills:
        Minimum number of co-sponsored bills for an edge to exist.
        Default ``1`` keeps all edges; increase to prune weak ties.
    max_sponsors_per_bill:
        Bills with more than this many sponsors are excluded from
        edge counting (likely ceremonial / non-controversial).
        Set to ``None`` to disable.

    Returns
    -------
    nx.Graph
        Undirected graph with ``weight`` attribute on each edge
        equal to the number of shared co-sponsorships.
    """
    G = nx.Graph()

    # ── Add all legislators as nodes ──────────────────────────────────────
    members_path = PROCESSED_DIR / "dim_members.parquet"
    if members_path.exists():
        df_members = pl.read_parquet(members_path)
        for mid in df_members["member_id"].to_list():
            G.add_node(mid)
        LOGGER.info("Added %d legislator nodes from dim_members.", G.number_of_nodes())
    else:
        LOGGER.warning("dim_members.parquet not found — nodes will come from bills only.")

    # ── Build edges from co-sponsorship data ──────────────────────────────
    bills_path = CACHE_DIR / "bills.json"
    if not bills_path.exists():
        LOGGER.warning("cache/bills.json not found — returning empty graph.")
        return G

    with open(bills_path) as f:
        bills_raw = json.load(f)

    cosponsor_counts: dict[tuple[str, str], int] = {}
    bills_used = 0
    bills_skipped_type = 0
    bills_skipped_sponsors = 0

    for bill in bills_raw.values():
        # Filter: only substantive bills (SB, HB)
        bill_number = bill.get("bill_number", "")
        if not any(bill_number.startswith(p) for p in _SUBSTANTIVE_PREFIXES):
            bills_skipped_type += 1
            continue

        all_ids = bill.get("sponsor_ids", []) + bill.get("house_sponsor_ids", [])
        if len(all_ids) < 2:
            continue

        # Filter: skip bills with too many sponsors (ceremonial fluff)
        if max_sponsors_per_bill is not None and len(all_ids) > max_sponsors_per_bill:
            bills_skipped_sponsors += 1
            continue

        bills_used += 1
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                a, b = all_ids[i], all_ids[j]
                key = (min(a, b), max(a, b))
                cosponsor_counts[key] = cosponsor_counts.get(key, 0) + 1

    # ── Add weighted edges ────────────────────────────────────────────────
    edges_added = 0
    for (a, b), count in cosponsor_counts.items():
        if count >= min_shared_bills:
            G.add_edge(a, b, weight=count)
            edges_added += 1

    LOGGER.info(
        "Co-sponsorship graph: %d nodes, %d edges "
        "(from %d bills; skipped %d non-substantive, %d high-sponsor)",
        G.number_of_nodes(),
        edges_added,
        bills_used,
        bills_skipped_type,
        bills_skipped_sponsors,
    )

    return G
