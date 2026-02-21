from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import strawberry
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from strawberry.fastapi import GraphQLRouter

from . import config as cfg
from .analytics import (
    build_member_committee_roles,
    compute_advancement_analytics,
    compute_committee_stats,
    controversial_score,
    lobbyist_alignment,
)
from .analytics_cache import load_analytics_cache, save_analytics_cache
from .app_state import state
from .constants import CATEGORY_COMMITTEES
from .date_parse import parse_bill_date, safe_parse_date
from .etl import (
    ScrapedData,
    _link_members_to_bills,
    compute_analytics,
    export_vault,
    load_from_cache,
    load_or_scrape_data,
    load_stale_cache_fallback,
)
from .models import Bill, Member, WitnessSlip
from .moneyball import build_cosponsor_edges
from .routers.advocacy import router as _advocacy_router
from .routers.auth import router as _auth_router
from .routers.bills import router as _bills_router
from .routers.explore import router as _explore_router
from .routers.feedback import router as _feedback_router
from .routers.intelligence import router as _intelligence_router
from .routers.outreach import router as _outreach_router
from .run_log import append_startup_run, get_log_path, load_recent_runs
from .schema import (
    BillAdvancementAnalyticsType,
    BillConnection,
    BillSlipAnalyticsType,
    BillSortField,
    BillType,
    BillVoteTimelineType,
    Chamber,
    CommitteeConnection,
    CommitteeType,
    LeaderboardSortField,
    LobbyistAlignmentEntryType,
    MemberConnection,
    MemberSortField,
    MemberType,
    PageInfo,
    SearchConnection,
    SearchEntityType,
    SearchResultType,
    SortOrder,
    VoteEventConnection,
    VoteEventType,
    WitnessSlipConnection,
    WitnessSlipSummaryConnection,
    WitnessSlipSummaryType,
    WitnessSlipType,
    paginate,
)
from .scraper import ILGAScraper
from .scrapers.bills import load_bill_cache
from .search import EntityType as SearchEntityTypeEnum
from .search import search_all
from .seating import process_seating
from .security import (
    CSRF_COOKIE_NAME,
    CSRF_MAX_AGE_SECONDS,
    generate_csrf_token,
)
from .startup_banner import _Colors, format_startup_table, log_startup_timing
from .vote_name_normalizer import normalize_vote_events
from .vote_timeline import compute_bill_vote_timeline
from .voting_record import (
    build_all_category_bill_sets,
    build_member_vote_index,
)
from .zip_crosswalk import load_zip_crosswalk

# Backward compat for tests
_parse_bill_date = parse_bill_date
_safe_parse_date = safe_parse_date

# ── Configure logging ────────────────────────────────────────────────────────
# Ensure our application logs show up in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
    stream=sys.stderr,
    force=True,
)
LOGGER = logging.getLogger(__name__)

# Re-export for backward compatibility (scripts/scrape.py imports from here)
get_bill_status_urls = cfg.get_bill_status_urls


# ── Startup timing log & summary table ──────────────────────────────────────

from pathlib import Path  # noqa: E402

# ── Mode flags (from config) ──────────────────────────────────────────────────
DEV_MODE = cfg.DEV_MODE
SEED_MODE = cfg.SEED_MODE
INCREMENTAL = cfg.INCREMENTAL
LOAD_ONLY = cfg.LOAD_ONLY

# When DEV_MODE is on, override scrape + export limits:
#   - Scrape 20 members per chamber (40 total)
#   - Export all members, all committees, latest 100 bills
if DEV_MODE:
    _SCRAPE_MEMBER_LIMIT = cfg.MEMBER_LIMIT or 20
    _EXPORT_MEMBER_LIMIT: int | None = None  # export all scraped members
    _EXPORT_COMMITTEE_LIMIT: int | None = None  # only ~142, export all
    _EXPORT_BILL_LIMIT: int | None = 100  # latest 100 by most-recent action
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


def _load_stale_cache_fallback() -> ScrapedData:
    """Best-effort fallback: load whatever JSON caches exist on disk.

    Used when the primary ETL scrape fails so the app can serve stale data
    instead of starting completely empty.  Raises if no usable cache is found.
    """
    scraper = ILGAScraper(request_delay=0, seed_fallback=SEED_MODE)

    # Members + bills (normalized cache)
    from .scraper import load_normalized_cache  # local to avoid circular at top-level

    normalized = load_normalized_cache(seed_fallback=SEED_MODE)
    if normalized is not None:
        members, bills_lookup = normalized
    else:
        members = []
        bills_lookup = {}

    # Bills cache (independent of member cache)
    if not bills_lookup:
        bills_lookup = load_bill_cache(seed_fallback=SEED_MODE) or {}

    # Committees (best-effort)
    try:
        committees, committee_rosters, committee_bills = scraper.fetch_all_committees()
    except Exception:
        LOGGER.warning("Committee cache also unavailable.")
        committees, committee_rosters, committee_bills = [], {}, {}

    if not members and not bills_lookup:
        raise RuntimeError("No usable cache data found for stale-cache fallback.")

    # Re-link members to bills
    _link_members_to_bills(members, bills_lookup)

    return ScrapedData(
        members=members,
        bills_lookup=bills_lookup,
        committees=committees,
        committee_rosters=committee_rosters,
        committee_bills=committee_bills,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    import time as _time

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
                " (seed fallback ON)" if SEED_MODE else "",
            )
        else:
            LOGGER.warning(
                "\u26a0\ufe0f DEV MODE (scrape startup): %d members/chamber, "
                "vault export bill cap=%s%s.",
                _SCRAPE_MEMBER_LIMIT,
                _EXPORT_BILL_LIMIT or "all",
                " (seed fallback ON)" if SEED_MODE else "",
            )
    elif LOAD_ONLY:
        LOGGER.info(
            "LOAD-ONLY startup: serving from cache (no scrape); vault export bill cap=%s.",
            _EXPORT_BILL_LIMIT or "all",
        )

    # ── Step 1: Load or scrape data (resilient) ──────────────────────────
    t_load = _time.perf_counter()
    if LOAD_ONLY:
        data = load_from_cache(seed_fallback=SEED_MODE)
        if data is None:
            LOGGER.warning("ILGA_LOAD_ONLY=1 but no cache found. Trying stale-cache fallback...")
            try:
                data = load_stale_cache_fallback(seed_fallback=SEED_MODE)
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
                seed_mode=SEED_MODE,
                incremental=INCREMENTAL,
                sb_limit=100,
                hb_limit=100,
            )
            state.members = data.members
            elapsed_load = _time.perf_counter() - t_load
        except Exception:
            LOGGER.exception("ETL load/scrape failed. Attempting stale-cache fallback...")
            try:
                data = load_stale_cache_fallback(seed_fallback=SEED_MODE)
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
        cached = load_analytics_cache(
            cfg.CACHE_DIR,
            cfg.MOCK_DEV_DIR,
            SEED_MODE,
        )
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
        seating_path = cfg.MOCK_DEV_DIR / "senate_seats.json"
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
    state.bills_lookup = data.bills_lookup  # leg_id -> Bill (for bill detail action_history)
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

        # ── Normalize vote names to canonical member names ──
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

    # ── Step 8: Compute influence engine (pivotality + sponsor pull + score)
    try:
        t_inf = _time.perf_counter()
        from .influence import (
            compute_influence_scores,
            compute_sponsor_pull,
            compute_vote_pivotality,
        )

        # 8a. Vote pivotality (from scraped vote events)
        if state.vote_events:
            state.pivotality = compute_vote_pivotality(state.vote_events, state.member_lookup)

        # 8b. Sponsor pull (from ML bill scores, if available)
        bill_scores_map: dict[str, float] = {}
        ml_data = state.ml
        if ml_data and hasattr(ml_data, "available") and ml_data.available:
            bill_scores_map = {s.bill_id: s.prob_advance for s in ml_data.bill_scores if s.bill_id}
        if bill_scores_map:
            state.sponsor_pull = compute_sponsor_pull(state.members, bill_scores_map)

        # 8c. Unified influence score (needs id-keyed lookup, not name-keyed)
        if state.moneyball:
            state.influence = compute_influence_scores(
                state.moneyball.profiles,
                state.pivotality,
                state.sponsor_pull,
                state.member_lookup_by_id,
            )

        # 8d. Enrich coalitions with influence data
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
        SEED_MODE,
    )
    print(summary, flush=True)

    # Show MVP
    if state.moneyball and state.moneyball.mvp_house_non_leadership:
        mvp = state.moneyball.profiles[state.moneyball.mvp_house_non_leadership]
        print(
            f"  🏆 MVP (House, non-leadership): {mvp.member_name} (Score: {mvp.moneyball_score})",
            flush=True,
        )

    # Show service URLs (use ILGA_APP_BASE_URL in production so logs show your domain)
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

    # ── Log to timing file for historical tracking ──
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
        SEED_MODE,
    )
    # ── Unified run log (for /logs dashboard and make logs) ──
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
        SEED_MODE,
    )

    # ── Step N: Initialize auth + outreach DB ────────────────────────────
    from .db import init_db

    await init_db()

    yield


# ── GraphQL schema ───────────────────────────────────────────────────────────


def _member_career_start(member: Member) -> int:
    """Return the earliest career start year, or a large value if unknown."""
    if member.career_ranges:
        return min(cr.start_year for cr in member.career_ranges)
    return 9999


def _mb_profile(member_id: str):
    """Safely get moneyball profile for a member."""
    if state.moneyball is None:
        return None
    return state.moneyball.profiles.get(member_id)


def _resolve_chamber(chamber: Chamber | None) -> str | None:
    """Convert a ``Chamber`` enum value to the string used in data models."""
    if chamber is None:
        return None
    return chamber.value


@strawberry.type
class Query:
    @strawberry.field(description="Look up a single member by exact name.")
    def member(self, name: str) -> MemberType | None:
        model = state.member_lookup.get(name)
        if model is None:
            return None
        return MemberType.from_model(
            model,
            state.scorecards.get(model.id),
            _mb_profile(model.id),
        )

    @strawberry.field(
        description="Paginated list of members with optional sorting and chamber filter.",
    )
    def members(
        self,
        sort_by: MemberSortField | None = None,
        sort_order: SortOrder | None = None,
        chamber: Chamber | None = None,
        offset: int = 0,
        limit: int = 0,
    ) -> MemberConnection:
        result = list(state.members)
        chamber_str = _resolve_chamber(chamber)

        # ── Filtering ──
        if chamber_str is not None:
            result = [m for m in result if m.chamber.lower() == chamber_str.lower()]

        if sort_by is not None:
            reverse = sort_order == SortOrder.DESC
            if sort_by == MemberSortField.CAREER_START:
                result.sort(key=_member_career_start, reverse=reverse)
            elif sort_by == MemberSortField.NAME:
                result.sort(key=lambda m: m.name, reverse=reverse)

        page, page_info = paginate(result, offset, limit)
        return MemberConnection(
            items=[
                MemberType.from_model(m, state.scorecards.get(m.id), _mb_profile(m.id))
                for m in page
            ],
            page_info=page_info,
        )

    @strawberry.field(description="Ranked leaderboard by Moneyball Score or any analytics metric.")
    def moneyball_leaderboard(
        self,
        chamber: Chamber | None = None,
        exclude_leadership: bool = False,
        limit: int = 0,
        offset: int = 0,
        sort_by: LeaderboardSortField | None = None,
        sort_order: SortOrder | None = None,
    ) -> MemberConnection:
        """Returns all members by default (limit=0 means no cap).

        Use ``chamber=HOUSE, excludeLeadership=true, limit=1`` to get the MVP.
        """
        if state.moneyball is None:
            return MemberConnection(
                items=[],
                page_info=PageInfo(total_count=0, has_next_page=False, has_previous_page=False),
            )

        chamber_str = _resolve_chamber(chamber)

        # ── Base ranking (by moneyball_score) ──
        if chamber_str and chamber_str.lower() == "house":
            ids = (
                state.moneyball.rankings_house_non_leadership
                if exclude_leadership
                else state.moneyball.rankings_house
            )
        elif chamber_str and chamber_str.lower() == "senate":
            ids = (
                state.moneyball.rankings_senate_non_leadership
                if exclude_leadership
                else state.moneyball.rankings_senate
            )
        else:
            ids = state.moneyball.rankings_overall

        # Resolve to Member models
        id_set = set(ids)
        members = [m for m in state.members if m.id in id_set]

        # ── Optional re-sort by analytics field ──
        if sort_by is not None:
            scorecards = state.scorecards
            profiles = state.moneyball.profiles
            reverse = sort_order == SortOrder.DESC

            def _sort_key(m: Member) -> float:
                if sort_by == LeaderboardSortField.MONEYBALL_SCORE:
                    return profiles[m.id].moneyball_score if m.id in profiles else 0.0
                if sort_by == LeaderboardSortField.EFFECTIVENESS_SCORE:
                    return scorecards[m.id].effectiveness_score if m.id in scorecards else 0.0
                if sort_by == LeaderboardSortField.PIPELINE_DEPTH:
                    return profiles[m.id].pipeline_depth_avg if m.id in profiles else 0.0
                if sort_by == LeaderboardSortField.NETWORK_CENTRALITY:
                    return profiles[m.id].network_centrality if m.id in profiles else 0.0
                if sort_by == LeaderboardSortField.HEAT_SCORE:
                    return float(scorecards[m.id].heat_score) if m.id in scorecards else 0.0
                if sort_by == LeaderboardSortField.SUCCESS_RATE:
                    return scorecards[m.id].success_rate if m.id in scorecards else 0.0
                if sort_by == LeaderboardSortField.MAGNET_SCORE:
                    return scorecards[m.id].magnet_score if m.id in scorecards else 0.0
                if sort_by == LeaderboardSortField.BRIDGE_SCORE:
                    return scorecards[m.id].bridge_score if m.id in scorecards else 0.0
                return 0.0

            members.sort(key=_sort_key, reverse=reverse)
        else:
            # Preserve the pre-computed ranking order
            rank = {mid: i for i, mid in enumerate(ids)}
            members.sort(key=lambda m: rank.get(m.id, len(ids)))

        page, page_info = paginate(members, offset, limit)
        return MemberConnection(
            items=[
                MemberType.from_model(m, state.scorecards.get(m.id), _mb_profile(m.id))
                for m in page
            ],
            page_info=page_info,
        )

    @strawberry.field(description="All vote events for a specific bill (floor + committee).")
    def votes(self, bill_number: str) -> list[VoteEventType]:
        events = state.vote_lookup.get(bill_number, [])
        return [VoteEventType.from_model(v) for v in events]

    @strawberry.field(
        description=(
            "Full vote timeline for a bill in one chamber,"
            " tracking every member's journey across committee and floor events."
        ),
    )
    def bill_vote_timeline(self, bill_number: str, chamber: Chamber) -> BillVoteTimelineType | None:
        return compute_bill_vote_timeline(state.vote_lookup, bill_number, chamber.value)

    @strawberry.field(
        description="All scraped vote events, optionally filtered by type and chamber.",
    )
    def all_vote_events(
        self,
        vote_type: str | None = None,
        chamber: Chamber | None = None,
        offset: int = 0,
        limit: int = 0,
    ) -> VoteEventConnection:
        result = list(state.vote_events)
        chamber_str = _resolve_chamber(chamber)
        if vote_type is not None:
            result = [v for v in result if v.vote_type == vote_type]
        if chamber_str is not None:
            result = [v for v in result if v.chamber.lower() == chamber_str.lower()]
        page, page_info = paginate(result, offset, limit)
        return VoteEventConnection(
            items=[VoteEventType.from_model(v) for v in page],
            page_info=page_info,
        )

    @strawberry.field(description="Look up a single bill by bill number (e.g. 'SB1527').")
    def bill(self, number: str) -> BillType | None:
        model = state.bill_lookup.get(number)
        return BillType.from_model(model) if model else None

    @strawberry.field(
        description="Paginated list of bills with optional sorting and date-range filtering.",
    )
    def bills(
        self,
        sort_by: BillSortField | None = None,
        sort_order: SortOrder | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        offset: int = 0,
        limit: int = 0,
    ) -> BillConnection:
        result = list(state.bills)

        # ── date filtering (with safe parsing) ──
        if date_from is not None:
            from_dt = _safe_parse_date(date_from, "dateFrom")
            if from_dt is not None:
                result = [b for b in result if parse_bill_date(b.last_action_date) >= from_dt]
        if date_to is not None:
            to_dt = _safe_parse_date(date_to, "dateTo")
            if to_dt is not None:
                result = [b for b in result if parse_bill_date(b.last_action_date) <= to_dt]

        # ── sorting ──
        if sort_by is not None:
            reverse = sort_order == SortOrder.DESC
            if sort_by == BillSortField.LAST_ACTION_DATE:
                result.sort(
                    key=lambda b: parse_bill_date(b.last_action_date),
                    reverse=reverse,
                )
            elif sort_by == BillSortField.BILL_NUMBER:
                result.sort(key=lambda b: b.bill_number, reverse=reverse)

        page, page_info = paginate(result, offset, limit)
        return BillConnection(
            items=[BillType.from_model(b) for b in page],
            page_info=page_info,
        )

    # ── Committee queries ─────────────────────────────────────────────────

    @strawberry.field(description="Look up a single committee by its code (e.g. 'SAGR').")
    def committee(self, code: str) -> CommitteeType | None:
        model = state.committee_lookup.get(code)
        if model is None:
            return None
        return CommitteeType.from_model(
            model,
            roster=state.committee_rosters.get(code),
            bill_numbers=state.committee_bills.get(code),
        )

    @strawberry.field(description="Paginated list of committees.")
    def committees(
        self,
        offset: int = 0,
        limit: int = 0,
    ) -> CommitteeConnection:
        page, page_info = paginate(state.committees, offset, limit)
        return CommitteeConnection(
            items=[
                CommitteeType.from_model(
                    c,
                    roster=state.committee_rosters.get(c.code),
                    bill_numbers=state.committee_bills.get(c.code),
                )
                for c in page
            ],
            page_info=page_info,
        )

    # ── Witness slip queries ──────────────────────────────────────────────

    @strawberry.field(description="Witness slips for a specific bill.")
    def witness_slips(
        self,
        bill_number: str,
        offset: int = 0,
        limit: int = 0,
    ) -> WitnessSlipConnection:
        slips = state.witness_slips_lookup.get(bill_number, [])
        page, page_info = paginate(slips, offset, limit)
        return WitnessSlipConnection(
            items=[WitnessSlipType.from_model(ws) for ws in page],
            page_info=page_info,
        )

    def _witness_slip_summary_for_slips(
        self, bill_number: str, slips: list[WitnessSlip]
    ) -> WitnessSlipSummaryType:
        pro = sum(1 for s in slips if s.position == "Proponent")
        opp = sum(1 for s in slips if s.position == "Opponent")
        no_pos = sum(1 for s in slips if s.position and "no position" in s.position.lower())
        return WitnessSlipSummaryType(
            bill_number=bill_number,
            total_count=len(slips),
            proponent_count=pro,
            opponent_count=opp,
            no_position_count=no_pos,
        )

    @strawberry.field(
        description="Per-bill witness slip counts by position (no paging).",
    )
    def witness_slip_summary(self, bill_number: str) -> WitnessSlipSummaryType | None:
        slips = state.witness_slips_lookup.get(bill_number, [])
        if not slips:
            return None
        return self._witness_slip_summary_for_slips(bill_number, slips)

    @strawberry.field(
        description="All bills with witness slips, summarized (sorted by slip volume descending).",
    )
    def witness_slip_summaries(
        self,
        offset: int = 0,
        limit: int = 0,
    ) -> WitnessSlipSummaryConnection:
        all_summaries = [
            self._witness_slip_summary_for_slips(bill_number, slips)
            for bill_number, slips in state.witness_slips_lookup.items()
        ]
        all_summaries.sort(key=lambda s: s.total_count, reverse=True)
        page, page_info = paginate(all_summaries, offset, limit)
        return WitnessSlipSummaryConnection(items=page, page_info=page_info)

    @strawberry.field(
        description="Witness-slip analytics for a bill (controversy score 0–1).",
    )
    def bill_slip_analytics(self, bill_number: str) -> BillSlipAnalyticsType | None:
        if not state.witness_slips_lookup.get(bill_number):
            return None
        score = controversial_score(state.witness_slips, bill_number)
        return BillSlipAnalyticsType(
            bill_number=bill_number,
            controversy_score=score,
        )

    @strawberry.field(
        description="Orgs filing as proponents on member's sponsored bills (by count desc).",
    )
    def member_slip_alignment(self, member_name: str) -> list[LobbyistAlignmentEntryType]:
        member = state.member_lookup.get(member_name)
        if member is None:
            return []
        alignment = lobbyist_alignment(state.witness_slips, member)
        return [
            LobbyistAlignmentEntryType(
                organization=org,
                proponent_count=count,
            )
            for org, count in alignment.items()
        ]

    # ----- New Query Field for Advancement Analytics -----
    @strawberry.field(
        description="Analytics categorizing bills by witness slip volume and advancement status.",
    )
    def bill_advancement_analytics_summary(
        self,
        volume_percentile_threshold: float = 0.9,
    ) -> BillAdvancementAnalyticsType:
        analytics_results = compute_advancement_analytics(
            state.bills,
            state.witness_slips,
            volume_percentile_threshold=volume_percentile_threshold,
        )
        return BillAdvancementAnalyticsType(
            high_volume_stalled=analytics_results.get("high_volume_stalled", []),
            high_volume_passed=analytics_results.get("high_volume_passed", []),
        )

    # ----- End New Query Field -----

    # ── Unified search ─────────────────────────────────────────────────────

    @strawberry.field(
        description=(
            "Unified free-text search across members, bills, and committees. "
            "Returns results ranked by relevance. Use entityTypes to restrict "
            "which kinds of entities are searched."
        ),
    )
    def search(
        self,
        query: str,
        entity_types: list[SearchEntityType] | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> SearchConnection:
        # Map GraphQL enum values to the internal EntityType enum.
        filter_set: set[SearchEntityTypeEnum] | None = None
        if entity_types:
            _map = {
                SearchEntityType.MEMBER: SearchEntityTypeEnum.MEMBER,
                SearchEntityType.BILL: SearchEntityTypeEnum.BILL,
                SearchEntityType.COMMITTEE: SearchEntityTypeEnum.COMMITTEE,
            }
            filter_set = {_map[et] for et in entity_types}

        all_hits = search_all(
            query=query,
            members=state.members,
            bills=state.bills,
            committees=state.committees,
            entity_types=filter_set,
        )

        page, page_info = paginate(all_hits, offset, limit)

        items: list[SearchResultType] = []
        for hit in page:
            member_type = None
            bill_type = None
            committee_type = None

            if hit.member is not None:
                sc = state.scorecards.get(hit.member.id)
                mb = _mb_profile(hit.member.id)
                member_type = MemberType.from_model(hit.member, sc, mb)
            elif hit.bill is not None:
                bill_type = BillType.from_model(hit.bill)
            elif hit.committee is not None:
                committee_type = CommitteeType.from_model(hit.committee)

            items.append(
                SearchResultType(
                    entity_type=hit.entity_type.value,
                    match_field=hit.match_field,
                    match_snippet=hit.match_snippet,
                    relevance_score=round(hit.relevance_score, 4),
                    member=member_type,
                    bill=bill_type,
                    committee=committee_type,
                )
            )

        return SearchConnection(items=items, page_info=page_info)

    # ── ML Intelligence queries ─────────────────────────────────────────

    @strawberry.field(
        description="Bill predictions from the ML pipeline.",
    )
    def bill_predictions(
        self,
        outcome: str | None = None,
        min_confidence: float | None = None,
        reliable_only: bool = False,
        forecasts_only: bool = False,
        stuck_status: str | None = None,
        stage: str | None = None,
        sort_by: str = "prob_advance",
        offset: int = 0,
        limit: int = 50,
    ) -> BillPredictionConnection:
        ml = state.ml
        if not ml or not ml.available:
            empty_page = PageInfo(
                total_count=0,
                has_next_page=False,
                has_previous_page=False,
            )
            return BillPredictionConnection(items=[], page_info=empty_page)

        results = list(ml.bill_scores)
        if outcome:
            results = [s for s in results if s.predicted_outcome == outcome.upper()]
        if min_confidence is not None:
            results = [s for s in results if s.confidence >= min_confidence]
        if reliable_only:
            results = [s for s in results if s.label_reliable]
        if forecasts_only:
            results = [s for s in results if not s.label_reliable]
        if stuck_status:
            results = [s for s in results if s.stuck_status == stuck_status.upper()]
        if stage:
            results = [s for s in results if s.current_stage == stage.upper()]

        if sort_by == "confidence":
            results.sort(key=lambda s: -s.confidence)
        else:
            results.sort(key=lambda s: -s.prob_advance)

        page, page_info = paginate(results, offset, limit)
        return BillPredictionConnection(
            items=[_bill_score_to_type(s) for s in page],
            page_info=page_info,
        )

    @strawberry.field(description="Prediction for a single bill by number.")
    def bill_prediction(self, bill_number: str) -> BillPredictionType | None:
        ml = state.ml
        if not ml or not ml.available:
            return None
        for s in ml.bill_scores:
            if s.bill_number == bill_number:
                return _bill_score_to_type(s)
        return None

    @strawberry.field(
        description="SHAP-based explanation for why a bill received its prediction score.",
    )
    def prediction_explanation(self, bill_id: str) -> PredictionExplanation | None:
        ml = state.ml
        if not ml or not ml.available or ml.explainer is None:
            return None
        row_idx = ml._bill_id_to_row.get(bill_id)
        if row_idx is None:
            return None
        try:
            row = ml.feature_matrix[row_idx]
            result = ml.explainer.explain_prediction(row, ml.feature_names)
            return PredictionExplanation(
                base_value=result["base_value"],
                top_positive_factors=[
                    PredictionFactor(**f) for f in result["top_positive_factors"]
                ],
                top_negative_factors=[
                    PredictionFactor(**f) for f in result["top_negative_factors"]
                ],
            )
        except Exception:
            import logging

            logging.getLogger(__name__).exception("SHAP explanation failed for %s", bill_id)
            return None

    @strawberry.field(description="Discovered voting coalitions from the ML pipeline.")
    def voting_coalitions(self) -> list[CoalitionGroupType]:
        ml = state.ml
        if not ml or not ml.available:
            return []
        groups: dict[int, list] = {}
        for m in ml.coalitions:
            groups.setdefault(m.coalition_id, []).append(m)
        # Build profile lookup
        prof_map = {p.coalition_id: p for p in ml.coalition_profiles}
        result = []
        for cid, members in sorted(groups.items()):
            dem = sum(1 for m in members if m.party == "Democrat")
            rep = sum(1 for m in members if m.party == "Republican")
            prof = prof_map.get(cid)
            result.append(
                CoalitionGroupType(
                    coalition_id=cid,
                    name=prof.name if prof else f"Coalition {cid + 1}",
                    size=len(members),
                    dem_count=dem,
                    rep_count=rep,
                    focus_areas=prof.focus_areas if prof else [],
                    yes_rate=round(prof.yes_rate, 3) if prof else 0.0,
                    cohesion=round(prof.cohesion, 3) if prof else 0.0,
                    signature_bills=[
                        SignatureBillType(
                            bill_number=b.get("bill_number", ""),
                            description=b.get("description", ""),
                            yes_votes=b.get("yes_votes", 0),
                        )
                        for b in (prof.signature_bills if prof else [])[:5]
                    ],
                    members=[
                        CoalitionMemberType(
                            name=m.name,
                            party=m.party,
                            chamber=m.chamber,
                            district=m.district,
                        )
                        for m in members
                    ],
                )
            )
        return result

    @strawberry.field(description="Slip anomalies (astroturfing detection) from the ML pipeline.")
    def slip_anomalies(
        self,
        flagged_only: bool = False,
        min_score: float | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> SlipAnomalyConnection:
        ml = state.ml
        if not ml or not ml.available:
            empty_page = PageInfo(total_count=0, has_next_page=False, has_previous_page=False)
            return SlipAnomalyConnection(items=[], page_info=empty_page)
        results = list(ml.anomalies)
        if flagged_only:
            results = [a for a in results if a.is_anomaly]
        if min_score is not None:
            results = [a for a in results if a.anomaly_score >= min_score]
        results.sort(key=lambda a: -a.anomaly_score)
        page, page_info = paginate(results, offset, limit)
        return SlipAnomalyConnection(
            items=[
                SlipAnomalyType(
                    bill_number=a.bill_number,
                    description=a.description,
                    total_slips=a.total_slips,
                    anomaly_score=round(a.anomaly_score, 4),
                    is_anomaly=a.is_anomaly,
                    anomaly_reason=a.anomaly_reason,
                    top_org_share=round(a.top_org_share, 4),
                    org_hhi=round(a.org_hhi, 4),
                    position_unanimity=round(a.position_unanimity, 4),
                    n_proponent=a.n_proponent,
                    n_opponent=a.n_opponent,
                    unique_orgs=a.unique_orgs,
                )
                for a in page
            ],
            page_info=page_info,
        )

    @strawberry.field(description="Model quality assessment from the ML pipeline.")
    def model_quality(self) -> ModelQualityType | None:
        ml = state.ml
        if not ml or not ml.available or not ml.quality:
            return None
        q = ml.quality
        trust = q.get("trust_assessment", {})
        ts = q.get("test_set_metrics", {})
        return ModelQualityType(
            model_selected=q.get("model_selected", ""),
            trust_overall=trust.get("overall", "UNKNOWN"),
            strengths=trust.get("strengths", []),
            issues=trust.get("issues", []),
            test_roc_auc=ts.get("roc_auc"),
            test_accuracy=ts.get("accuracy"),
            test_precision_pos=ts.get("precision_pos"),
            test_recall_pos=ts.get("recall_pos"),
            test_f1_pos=ts.get("f1_pos"),
            top_features=[
                FeatureImportanceType(name=f["name"], importance=round(f["importance"], 4))
                for f in q.get("top_features", [])[:15]
            ],
            last_run_date=ml.last_run_date,
        )

    @strawberry.field(description="Prediction accuracy history across pipeline runs.")
    def prediction_accuracy(self, limit_runs: int = 20) -> list[AccuracySnapshotType]:
        ml = state.ml
        if not ml or not ml.available:
            return []
        runs = ml.accuracy_history[-limit_runs:]
        return [
            AccuracySnapshotType(
                run_date=r.run_date,
                snapshot_date=r.snapshot_date,
                days_elapsed=r.days_elapsed,
                total_testable=r.total_testable,
                correct=r.correct,
                accuracy=round(r.accuracy, 4),
                precision_advance=round(r.precision_advance, 4),
                recall_advance=round(r.recall_advance, 4),
                f1_advance=round(r.f1_advance, 4),
                model_version=r.model_version,
                biggest_misses=[
                    PredictionMissType(
                        bill_number=m.get("bill_number", ""),
                        description=m.get("description", ""),
                        predicted=m.get("predicted", ""),
                        actual=m.get("actual", ""),
                        confidence=m.get("confidence", 0),
                    )
                    for m in r.biggest_misses[:10]
                ],
            )
            for r in runs
        ]


# ── ML Intelligence GraphQL types ────────────────────────────────────────


@strawberry.type
class BillPredictionType:
    bill_number: str
    description: str
    sponsor: str
    prob_advance: float
    prob_law: float
    predicted_outcome: str
    predicted_destination: str
    confidence: float
    label_reliable: bool
    chamber_origin: str
    introduction_date: str
    # Pipeline stage (v4)
    current_stage: str = ""
    stage_progress: float = 0.0
    stage_label: str = ""
    days_since_action: int = 0
    last_action_text: str = ""
    last_action_date: str = ""
    # Stuck analysis (v4)
    stuck_status: str = ""
    stuck_reason: str = ""
    # Forecast model (v8): intrinsic-only P(law) — no staleness/slips
    forecast_score: float = 0.0
    forecast_confidence: str = ""


@strawberry.type
class BillPredictionConnection:
    items: list[BillPredictionType]
    page_info: PageInfo


# ── SHAP explanation types (v9) ──────────────────────────────────────────


@strawberry.type
class PredictionFactor:
    """A single feature's contribution to a bill's prediction."""

    feature: str  # Human-readable name
    impact: str  # e.g. "+12.4%"
    raw_impact: float  # Numeric for sorting / styling


@strawberry.type
class PredictionExplanation:
    """SHAP-based explanation for a single bill prediction."""

    base_value: float
    top_positive_factors: list[PredictionFactor]
    top_negative_factors: list[PredictionFactor]


@strawberry.type
class CoalitionMemberType:
    name: str
    party: str
    chamber: str
    district: str


@strawberry.type
class SignatureBillType:
    bill_number: str
    description: str
    yes_votes: int


@strawberry.type
class CoalitionGroupType:
    coalition_id: int
    name: str
    size: int
    dem_count: int
    rep_count: int
    focus_areas: list[str]
    yes_rate: float
    cohesion: float
    signature_bills: list[SignatureBillType]
    members: list[CoalitionMemberType]


@strawberry.type
class SlipAnomalyType:
    bill_number: str
    description: str
    total_slips: int
    anomaly_score: float
    is_anomaly: bool
    anomaly_reason: str
    top_org_share: float
    org_hhi: float
    position_unanimity: float
    n_proponent: int
    n_opponent: int
    unique_orgs: int


@strawberry.type
class SlipAnomalyConnection:
    items: list[SlipAnomalyType]
    page_info: PageInfo


@strawberry.type
class FeatureImportanceType:
    name: str
    importance: float


@strawberry.type
class ModelQualityType:
    model_selected: str
    trust_overall: str
    strengths: list[str]
    issues: list[str]
    test_roc_auc: float | None
    test_accuracy: float | None
    test_precision_pos: float | None
    test_recall_pos: float | None
    test_f1_pos: float | None
    top_features: list[FeatureImportanceType]
    last_run_date: str


@strawberry.type
class PredictionMissType:
    bill_number: str
    description: str
    predicted: str
    actual: str
    confidence: float


@strawberry.type
class AccuracySnapshotType:
    run_date: str
    snapshot_date: str
    days_elapsed: int
    total_testable: int
    correct: int
    accuracy: float
    precision_advance: float
    recall_advance: float
    f1_advance: float
    model_version: str
    biggest_misses: list[PredictionMissType]


# ── ML Intelligence helpers ──────────────────────────────────────────────


def _bill_score_to_type(s) -> BillPredictionType:
    """Convert a BillScore dataclass to a GraphQL BillPredictionType."""
    return BillPredictionType(
        bill_number=s.bill_number,
        description=s.description,
        sponsor=s.sponsor,
        prob_advance=round(s.prob_advance, 4),
        prob_law=round(getattr(s, "prob_law", 0.0), 4),
        predicted_outcome=s.predicted_outcome,
        predicted_destination=getattr(s, "predicted_destination", "Stuck"),
        confidence=round(s.confidence, 4),
        label_reliable=s.label_reliable,
        chamber_origin=s.chamber_origin,
        introduction_date=s.introduction_date,
        current_stage=s.current_stage,
        stage_progress=round(s.stage_progress, 2),
        stage_label=s.stage_label,
        days_since_action=s.days_since_action,
        last_action_text=s.last_action_text,
        last_action_date=s.last_action_date,
        stuck_status=s.stuck_status,
        stuck_reason=s.stuck_reason,
        forecast_score=round(getattr(s, "forecast_score", 0.0), 4),
        forecast_confidence=getattr(s, "forecast_confidence", ""),
    )


from strawberry.extensions import QueryDepthLimiter  # noqa: E402

from .loaders import create_loaders  # noqa: E402


async def get_graphql_context() -> dict:
    """Request-scoped context with state and batch loaders for GraphQL."""
    return create_loaders(state)


schema = strawberry.Schema(
    query=Query,
    extensions=[QueryDepthLimiter(max_depth=10)],
)
graphql_app = GraphQLRouter(schema, context_getter=get_graphql_context)

app = FastAPI(title="ILGA Graph", lifespan=lifespan)

# ── Static files & Jinja2 templates ──────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATE_DIR = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

# Dev bar is available when running in dev profile (never rendered in prod)
templates.env.globals["dev_available"] = DEV_MODE
# SEO, share cards, and analytics (base template uses these)
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["site_name"] = cfg.SITE_NAME
templates.env.globals["meta_description"] = cfg.META_DESCRIPTION
templates.env.globals["og_image_url"] = cfg.OG_IMAGE_URL
# Umami script only in prod (and only when website ID is set)
templates.env.globals["umami_enabled"] = cfg.PROFILE == "prod" and bool(cfg.UMAMI_WEBSITE_ID)
templates.env.globals["umami_website_id"] = cfg.UMAMI_WEBSITE_ID
templates.env.globals["umami_script_url"] = cfg.UMAMI_SCRIPT_URL
templates.env.globals["show_beta_banner"] = cfg.BETA_BANNER
templates.env.globals["beta_banner_feedback_url"] = cfg.BETA_BANNER_REPORT_URL


@app.get("/", include_in_schema=False)
def _root() -> RedirectResponse:
    """Redirect root to the advocacy page."""
    return RedirectResponse(url="/advocacy", status_code=302)


@app.get("/favicon.ico", include_in_schema=False)
def _favicon() -> FileResponse:
    """Serve theme-matching favicon (Kei truck SVG) at /favicon.ico."""
    path = _STATIC_DIR / "favicon.svg"
    return FileResponse(path, media_type="image/svg+xml")


# ── CORS middleware ──────────────────────────────────────────────────────────
_cors_origins = [o.strip() for o in cfg.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API key authentication middleware ────────────────────────────────────────
@app.middleware("http")
async def _api_key_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
    """Require ``X-API-Key`` header when ``ILGA_API_KEY`` env var is set.

    Skips auth for the health endpoint and for OPTIONS (CORS preflight).
    """
    if cfg.API_KEY:
        exempt = {"/", "/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}
        path = request.url.path
        if (
            path not in exempt
            and not path.startswith("/advocacy")
            and not path.startswith("/auth")
            and not path.startswith("/outreach")
            and not path.startswith("/report-bug")
            and not path.startswith("/explore")
            and not path.startswith("/intelligence")
            and not path.startswith("/api/graph")
            and not path.startswith("/api/dev")
            and not path.startswith("/static")
            and request.method != "OPTIONS"
        ):
            provided = request.headers.get("X-API-Key", "")
            if provided != cfg.API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
    return await call_next(request)


# ── CSRF cookie middleware ───────────────────────────────────────────────────
@app.middleware("http")
async def _csrf_cookie_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
    """Set XSRF-TOKEN cookie and request.state.csrf_token for form/fetch POST protection."""
    token = generate_csrf_token()
    request.state.csrf_token = token  # type: ignore[attr-defined]
    response: Response = await call_next(request)
    if hasattr(response, "set_cookie"):
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=token,
            max_age=CSRF_MAX_AGE_SECONDS,
            path="/",
            httponly=False,  # So JS can read and send in body/header for fetch()
            samesite="strict",
            secure=cfg.PROFILE == "prod",
        )
    return response


# ── Security headers middleware ──────────────────────────────────────────────
@app.middleware("http")
async def _security_headers_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
    """Add security headers to every response."""
    response: Response = await call_next(request)
    if hasattr(response, "headers"):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ── Request logging middleware ───────────────────────────────────────────────
@app.middleware("http")
async def _request_logging_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
    """Log every request with method, path, and response time."""
    import time as _t

    t0 = _t.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (_t.perf_counter() - t0) * 1000
    LOGGER.info(
        "%s %s %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ── Health endpoint ──────────────────────────────────────────────────────────
@app.get("/logs")
async def logs_dashboard(request: Request):
    """Unified run log dashboard — scrape, ML, startup. Minimal 2000s-hacker UI."""
    runs = load_recent_runs(n=100)
    # Bottleneck summary: avg phase duration per task
    task_phases: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        if r.task not in task_phases:
            task_phases[r.task] = {}
        for p in r.phases:
            name = p.get("name", "?")
            if name not in task_phases[r.task]:
                task_phases[r.task][name] = []
            task_phases[r.task][name].append(p.get("duration_s") or 0)
    bottleneck: list[tuple[str, list[tuple[str, float]]]] = []
    for task, phases in task_phases.items():
        by_name = [(name, sum(durs) / len(durs) if durs else 0) for name, durs in phases.items()]
        by_name.sort(key=lambda x: x[1], reverse=True)
        bottleneck.append((task, by_name[:5]))
    return templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            "runs": runs,
            "bottleneck": bottleneck,
            "log_path": str(get_log_path()),
        },
    )


@app.get("/health")
async def health() -> dict:
    """Service health check with data counts."""
    return {
        "status": "ok",
        "ready": len(state.members) > 0,
        "members": len(state.members),
        "bills": len(state.bills),
        "committees": len(state.committees),
        "vote_events": len(state.vote_events),
    }


# ── Dev bar API ──────────────────────────────────────────────────────────────


@app.get("/api/dev/members")
async def dev_members():
    """Return first 20 members as JSON for the dev bar member dropdown. Only active in DEV_MODE."""
    if not DEV_MODE:
        return JSONResponse(status_code=404, content={"detail": "Not available"})
    return [{"id": m.id, "name": m.name} for m in state.members[:20]]


app.include_router(graphql_app, prefix="/graphql")

app.include_router(_advocacy_router, prefix="/advocacy")
app.include_router(_auth_router)
app.include_router(_feedback_router)
app.include_router(_bills_router, prefix="/api")
app.include_router(_explore_router)
app.include_router(_intelligence_router, prefix="/intelligence")
app.include_router(_outreach_router)
