"""Application startup: load/scrape data, analytics, export, ML, influence, init_db."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import config as cfg
from .analytics import (
    build_member_committee_roles,
    compute_committee_stats,
)
from .analytics_cache import load_analytics_cache, save_analytics_cache
from .app_state import state
from .constants import CATEGORY_COMMITTEES
from .data_source import get_data_dir, is_using_mocks
from .etl import (
    ScrapedData,
    compute_analytics,
    export_vault,
    load_from_cache,
    load_or_scrape_data,
    load_stale_cache_fallback,
)
from .models import Bill
from .moneyball import build_cosponsor_edges
from .run_log import append_startup_run
from .seating import process_seating
from .startup_banner import _Colors, format_startup_table, log_startup_timing
from .vote_name_normalizer import normalize_vote_events
from .voting_record import (
    build_all_category_bill_sets,
    build_member_vote_index,
)
from .zip_crosswalk import load_zip_crosswalk

LOGGER = logging.getLogger(__name__)

DEV_MODE = cfg.DEV_MODE
INCREMENTAL = cfg.INCREMENTAL
LOAD_ONLY = cfg.LOAD_ONLY

if DEV_MODE:
    _SCRAPE_MEMBER_LIMIT = cfg.MEMBER_LIMIT or 20
    _EXPORT_MEMBER_LIMIT: int | None = None
    _EXPORT_COMMITTEE_LIMIT: int | None = None
    _EXPORT_BILL_LIMIT: int | None = 100
else:
    _SCRAPE_MEMBER_LIMIT = cfg.MEMBER_LIMIT
    _EXPORT_MEMBER_LIMIT = None
    _EXPORT_COMMITTEE_LIMIT = None
    _EXPORT_BILL_LIMIT = None


def _collect_unique_bills_by_number(bills_lookup: dict[str, Bill]) -> dict[str, Bill]:
    """Build a bill_number -> Bill lookup from the leg_id -> Bill dict."""
    unique: dict[str, Bill] = {}
    for b in bills_lookup.values():
        if b.bill_number not in unique:
            unique[b.bill_number] = b
    return unique


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    import time as _time

    # In-memory job store for campaign send progress (single-worker; keyed by job_id).
    _app.state.send_jobs = {}

    t_startup_begin = _time.perf_counter()
    elapsed_load = 0.0
    elapsed_analytics = 0.0
    elapsed_seating = 0.0
    elapsed_export = 0.0
    elapsed_committee = 0.0
    elapsed_votes = 0.0
    elapsed_voting_records = 0.0
    elapsed_slips = 0.0
    elapsed_zip = 0.0
    elapsed_graph = 0.0
    elapsed_ml = 0.0
    elapsed_influence = 0.0
    data: ScrapedData | None = None

    if DEV_MODE:
        if LOAD_ONLY:
            LOGGER.warning(
                "\u26a0\ufe0f DEV MODE (cache startup): scrape limit (%d/chamber) is inactive "
                "under ILGA_LOAD_ONLY=1; vault export bill cap=%s%s.",
                _SCRAPE_MEMBER_LIMIT,
                _EXPORT_BILL_LIMIT or "all",
                " (using mocks)" if is_using_mocks() else "",
            )
        else:
            LOGGER.warning(
                "\u26a0\ufe0f DEV MODE (scrape startup): %d members/chamber, "
                "vault export bill cap=%s%s.",
                _SCRAPE_MEMBER_LIMIT,
                _EXPORT_BILL_LIMIT or "all",
                " (using mocks)" if is_using_mocks() else "",
            )
    elif LOAD_ONLY:
        LOGGER.info(
            "LOAD-ONLY startup: serving from cache (no scrape); vault export bill cap=%s.",
            _EXPORT_BILL_LIMIT or "all",
        )

    # ── Step 1: Load or scrape data (resilient) ──────────────────────────
    t_load = _time.perf_counter()
    if LOAD_ONLY:
        data = load_from_cache()
        if data is None:
            LOGGER.warning("ILGA_LOAD_ONLY=1 but no cache found. Trying stale-cache fallback...")
            try:
                data = load_stale_cache_fallback()
                state.members = data.members
                LOGGER.warning(
                    "Loaded stale cache: %d members, %d bills.",
                    len(data.members),
                    len(data.bills_lookup),
                )
            except Exception:
                LOGGER.exception("Stale-cache fallback failed. App will start with EMPTY state.")
                data = ScrapedData(
                    members=[],
                    bills_lookup={},
                    committees=[],
                    committee_rosters={},
                    committee_bills={},
                )
                state.members = []
        else:
            state.members = data.members
        elapsed_load = _time.perf_counter() - t_load
    else:
        try:
            data = load_or_scrape_data(
                limit=_SCRAPE_MEMBER_LIMIT,
                dev_mode=DEV_MODE,
                incremental=INCREMENTAL,
                sb_limit=100,
                hb_limit=100,
            )
            state.members = data.members
            elapsed_load = _time.perf_counter() - t_load
        except Exception:
            LOGGER.exception("ETL load/scrape failed. Attempting stale-cache fallback...")
            try:
                data = load_stale_cache_fallback()
                state.members = data.members
                elapsed_load = _time.perf_counter() - t_startup_begin
                LOGGER.warning(
                    "Loaded stale cache: %d members, %d bills.",
                    len(data.members),
                    len(data.bills_lookup),
                )
            except Exception:
                LOGGER.exception(
                    "Stale-cache fallback also failed. "
                    "App will start with EMPTY state (health.ready=false)."
                )
                data = ScrapedData(
                    members=[],
                    bills_lookup={},
                    committees=[],
                    committee_rosters={},
                    committee_bills={},
                )
                state.members = []

    # ── Step 2: Compute analytics (or load from cache when fresh) ───────────
    try:
        t_analytics = _time.perf_counter()
        cached = load_analytics_cache(cfg.CACHE_DIR)
        if cached is not None:
            state.scorecards, state.moneyball = cached
        else:
            state.scorecards, state.moneyball = compute_analytics(
                state.members,
                data.committee_rosters,
            )
            save_analytics_cache(
                state.scorecards,
                state.moneyball,
                cfg.CACHE_DIR,
            )
        elapsed_analytics = _time.perf_counter() - t_analytics
    except Exception:
        LOGGER.exception("Analytics computation failed; scorecards will be empty.")

    # ── Step 2a: Build co-sponsorship adjacency for graph visualization ──
    try:
        t_graph = _time.perf_counter()
        state.cosponsor_adjacency = build_cosponsor_edges(state.members)
        elapsed_graph = _time.perf_counter() - t_graph
        LOGGER.info(
            "Co-sponsorship graph: %d nodes, %d total edges.",
            len(state.cosponsor_adjacency),
            sum(len(peers) for peers in state.cosponsor_adjacency.values()) // 2,
        )
    except Exception:
        LOGGER.exception("Co-sponsorship graph build failed; /explore will have no edges.")

    # ── Step 2b: Seating chart analytics ─────────────────────────────────
    try:
        t_seating = _time.perf_counter()
        seating_path = get_data_dir() / "senate_seats.json"
        process_seating(state.members, seating_path)
        elapsed_seating = _time.perf_counter() - t_seating
    except Exception:
        LOGGER.exception("Seating chart processing failed; seating fields will be empty.")

    # ── Step 3: Export vault ─────────────────────────────────────────────
    try:
        t_export = _time.perf_counter()
        export_vault(
            data,
            state.scorecards,
            state.moneyball,
            member_export_limit=_EXPORT_MEMBER_LIMIT,
            committee_export_limit=_EXPORT_COMMITTEE_LIMIT,
            bill_export_limit=_EXPORT_BILL_LIMIT,
        )
        elapsed_export = _time.perf_counter() - t_export
    except Exception:
        LOGGER.exception("Vault export failed; Obsidian vault may be stale.")

    state.member_lookup = {m.name: m for m in state.members}
    state.member_lookup_by_id = {m.id: m for m in state.members}
    state.bill_lookup = _collect_unique_bills_by_number(data.bills_lookup)
    state.bills_lookup = data.bills_lookup
    state.bills = list(state.bill_lookup.values())
    state.committees = data.committees
    state.committee_lookup = {c.code: c for c in data.committees}
    state.committee_rosters = data.committee_rosters
    state.committee_bills = data.committee_bills

    # ── Step 3b: Compute committee-level stats & member reverse index ────
    try:
        t_committee = _time.perf_counter()
        state.committee_stats = compute_committee_stats(
            state.committees,
            state.committee_bills,
            data.bills_lookup,
        )
        state.member_committee_roles = build_member_committee_roles(
            state.committees,
            state.committee_rosters,
            state.committee_stats,
        )
        elapsed_committee = _time.perf_counter() - t_committee
        LOGGER.info(
            "Committee stats: %d committees, %d members with roles.",
            len(state.committee_stats),
            len(state.member_committee_roles),
        )
    except Exception:
        LOGGER.exception("Committee stats computation failed; power dashboard will be empty.")

    # ── Step 4: Build vote events from per-bill data ─────────────────────
    try:
        t_votes = _time.perf_counter()
        for bill in state.bills:
            for ve in bill.vote_events:
                state.vote_events.append(ve)
                state.vote_lookup.setdefault(ve.bill_number, []).append(ve)

        if state.vote_events:
            normalize_vote_events(state.vote_events, state.member_lookup)
        elapsed_votes = _time.perf_counter() - t_votes
        LOGGER.info("Built %d vote events from bill data.", len(state.vote_events))
    except Exception:
        LOGGER.exception("Vote event loading failed; vote data will be empty.")

    # ── Step 4b: Build per-member voting records & category bill sets ────
    try:
        t_vr = _time.perf_counter()
        bn_lookup = _collect_unique_bills_by_number(data.bills_lookup)
        state.member_vote_records = build_member_vote_index(
            state.vote_events,
            state.member_lookup,
            bn_lookup,
        )
        state.category_bill_sets = build_all_category_bill_sets(
            CATEGORY_COMMITTEES,
            state.committee_bills,
        )
        elapsed_voting_records = _time.perf_counter() - t_vr
        LOGGER.info(
            "Voting records: %d members indexed, %d category bill sets (%0.2fs).",
            len(state.member_vote_records),
            len(state.category_bill_sets),
            elapsed_voting_records,
        )
    except Exception:
        LOGGER.exception("Voting record computation failed; voting records will be empty.")

    # ── Step 5: Build witness slips from per-bill data ───────────────────
    try:
        t_slips = _time.perf_counter()
        for bill in state.bills:
            for ws in bill.witness_slips:
                state.witness_slips.append(ws)
                state.witness_slips_lookup.setdefault(ws.bill_number, []).append(ws)
        elapsed_slips = _time.perf_counter() - t_slips
        LOGGER.info("Built %d witness slips from bill data.", len(state.witness_slips))
    except Exception:
        LOGGER.exception("Witness slip loading failed; slip data will be empty.")

    # ── Step 6: Load ZIP-to-district crosswalk ───────────────────────────
    try:
        t_zip = _time.perf_counter()
        state.zip_to_district = load_zip_crosswalk()
        elapsed_zip = _time.perf_counter() - t_zip
        LOGGER.info("ZIP crosswalk loaded: %d ZCTAs.", len(state.zip_to_district))
    except Exception:
        LOGGER.exception("ZIP crosswalk loading failed; advocacy search will be limited.")

    # ── Step 7: Load ML intelligence data (optional) ────────────────────
    try:
        t_ml = _time.perf_counter()
        from .ml_loader import load_ml_data

        state.ml = load_ml_data()
        elapsed_ml = _time.perf_counter() - t_ml
        if state.ml and state.ml.available:
            LOGGER.info(
                "ML intelligence loaded: %d predictions, %d coalitions, "
                "%d anomalies, %d backtest runs.",
                len(state.ml.bill_scores),
                len(state.ml.coalitions),
                len(state.ml.anomalies),
                len(state.ml.accuracy_history),
            )
        else:
            LOGGER.info("ML data not available (run 'make ml-run' to generate).")
    except Exception:
        LOGGER.exception("ML data loading failed (non-critical).")

    # ── Step 8: Compute influence engine ─────────────────────────────────
    try:
        t_inf = _time.perf_counter()
        from .influence import (
            compute_influence_scores,
            compute_sponsor_pull,
            compute_vote_pivotality,
        )

        if state.vote_events:
            state.pivotality = compute_vote_pivotality(state.vote_events, state.member_lookup)

        bill_scores_map: dict[str, float] = {}
        ml_data = state.ml
        if ml_data and hasattr(ml_data, "available") and ml_data.available:
            bill_scores_map = {s.bill_id: s.prob_advance for s in ml_data.bill_scores if s.bill_id}
        if bill_scores_map:
            state.sponsor_pull = compute_sponsor_pull(state.members, bill_scores_map)

        if state.moneyball:
            state.influence = compute_influence_scores(
                state.moneyball.profiles,
                state.pivotality,
                state.sponsor_pull,
                state.member_lookup_by_id,
            )

        from .influence import enrich_coalitions_with_influence

        if state.influence and ml_data and hasattr(ml_data, "coalitions"):
            coalition_dicts = [
                {
                    "member_id": c.member_id,
                    "name": c.name,
                    "coalition_id": c.coalition_id,
                    "coalition_name": c.coalition_name,
                }
                for c in ml_data.coalitions
            ]
            mb_profiles = state.moneyball.profiles if state.moneyball else None
            state.coalition_influence = enrich_coalitions_with_influence(
                coalition_dicts, state.influence, mb_profiles
            )
        else:
            state.coalition_influence = []

        elapsed_influence = _time.perf_counter() - t_inf
        LOGGER.info(
            "Influence engine: %d pivotality, %d pull, %d influence profiles (%.2fs).",
            len(state.pivotality),
            len(state.sponsor_pull),
            len(state.influence),
            elapsed_influence,
        )
    except Exception:
        LOGGER.exception("Influence engine failed (non-critical).")

    # ── Print startup summary table ──────────────────────────────────────
    elapsed_total = _time.perf_counter() - t_startup_begin
    exported_bill_count = (
        len(state.bills)
        if _EXPORT_BILL_LIMIT is None
        else min(len(state.bills), _EXPORT_BILL_LIMIT)
    )
    graph_edge_count = (
        sum(len(peers) for peers in state.cosponsor_adjacency.values()) // 2
        if state.cosponsor_adjacency
        else 0
    )
    ml_data = state.ml
    summary = format_startup_table(
        elapsed_total,
        elapsed_load,
        elapsed_analytics,
        elapsed_seating,
        elapsed_export,
        elapsed_committee,
        elapsed_votes,
        elapsed_voting_records,
        elapsed_slips,
        elapsed_zip,
        elapsed_graph,
        elapsed_ml,
        elapsed_influence,
        len(state.members),
        len(data.committees),
        len(state.bills),
        exported_bill_count,
        len(state.member_committee_roles),
        len(state.member_vote_records),
        len(state.category_bill_sets),
        len(state.vote_events),
        len(state.witness_slips),
        len(state.vote_lookup),
        len(state.witness_slips_lookup),
        len(state.zip_to_district),
        len(state.cosponsor_adjacency),
        graph_edge_count,
        len(ml_data.bill_scores) if ml_data and ml_data.available else 0,
        len(ml_data.coalitions) if ml_data and ml_data.available else 0,
        len(ml_data.anomalies) if ml_data and ml_data.available else 0,
        len(state.pivotality),
        len(state.sponsor_pull),
        len(state.influence),
        LOAD_ONLY,
        DEV_MODE,
        is_using_mocks(),
    )
    print(summary, flush=True)

    if state.moneyball and state.moneyball.mvp_house_non_leadership:
        mvp = state.moneyball.profiles[state.moneyball.mvp_house_non_leadership]
        print(
            f"  🏆 MVP (House, non-leadership): {mvp.member_name} (Score: {mvp.moneyball_score})",
            flush=True,
        )

    c = _Colors
    app_url = cfg.APP_BASE_URL
    graphql_url = f"{app_url}/graphql"
    if cfg.DOCS_BASE_URL:
        docs_url = cfg.DOCS_BASE_URL
        docs_note = ""
    elif "127.0.0.1" in app_url:
        docs_url = "http://127.0.0.1:8001"
        docs_note = f"  {c.DIM}(make docs-serve){c.RESET}"
    else:
        docs_url = app_url
        docs_note = f"  {c.DIM}(docs){c.RESET}"
    print(
        f"\n  {c.BOLD}Services:{c.RESET}\n"
        f"    {c.WHITE}Website    {c.BRIGHT_CYAN}{app_url}{c.RESET}\n"
        f"    {c.WHITE}GraphQL    {c.BRIGHT_CYAN}{graphql_url}{c.RESET}\n"
        f"    {c.WHITE}Docs       {c.BRIGHT_CYAN}{docs_url}{c.RESET}{docs_note}\n",
        flush=True,
    )

    log_startup_timing(
        elapsed_total,
        elapsed_load,
        elapsed_analytics,
        elapsed_seating,
        elapsed_export,
        elapsed_votes,
        elapsed_slips,
        elapsed_zip,
        elapsed_graph,
        elapsed_ml,
        elapsed_influence,
        len(state.members),
        len(state.bills),
        len(state.vote_events),
        len(state.witness_slips),
        len(state.zip_to_district),
        DEV_MODE,
        is_using_mocks(),
    )
    append_startup_run(
        elapsed_total,
        elapsed_load,
        elapsed_analytics,
        elapsed_seating,
        elapsed_export,
        elapsed_votes,
        elapsed_slips,
        elapsed_zip,
        len(state.members),
        len(state.bills),
        len(state.vote_events),
        len(state.witness_slips),
        len(state.zip_to_district),
        DEV_MODE,
        is_using_mocks(),
    )

    from .db import init_db

    await init_db()

    yield
