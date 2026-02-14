from __future__ import annotations

import functools
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

import strawberry
from fastapi import FastAPI, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from strawberry.fastapi import GraphQLRouter

from . import config as cfg
from .analytics import (
    CommitteeStats,
    MemberScorecard,
    build_member_committee_roles,
    compute_advancement_analytics,
    compute_committee_stats,
    controversial_score,
    lobbyist_alignment,
)
from .analytics_cache import load_analytics_cache, save_analytics_cache
from .etl import (
    ScrapedData,
    _link_members_to_bills,
    compute_analytics,
    export_vault,
    load_from_cache,
    load_or_scrape_data,
    load_stale_cache_fallback,
)
from .metrics_definitions import MONEYBALL_ONE_LINER
from .models import Bill, Committee, CommitteeMemberRole, Member, VoteEvent, WitnessSlip
from .moneyball import MoneyballReport, build_cosponsor_edges, compute_power_badges
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
from .vote_name_normalizer import normalize_vote_events
from .vote_timeline import compute_bill_vote_timeline
from .voting_record import (
    VotingSummary,
    build_all_category_bill_sets,
    build_member_vote_index,
)
from .zip_crosswalk import ZipDistrictInfo, load_zip_crosswalk

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


class _Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    # Bright colors
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


def _format_startup_table(
    elapsed_total: float,
    elapsed_load: float,
    elapsed_analytics: float,
    elapsed_seating: float,
    elapsed_export: float,
    elapsed_committee: float,
    elapsed_votes: float,
    elapsed_voting_records: float,
    elapsed_slips: float,
    elapsed_zip: float,
    member_count: int,
    committee_count: int,
    bill_count: int,
    exported_bill_count: int,
    member_committee_role_count: int,
    member_vote_record_count: int,
    category_bill_set_count: int,
    vote_event_count: int,
    slip_count: int,
    bills_with_votes: int,
    bills_with_slips: int,
    zcta_count: int,
    load_only: bool,
    dev_mode: bool,
    seed_mode: bool,
) -> str:
    """Format a chronological ETL startup table with phase/time/detail."""
    c = _Colors

    def row(phase: str, label: str, sec: float, detail: str) -> str:
        return (
            f"{c.CYAN}{phase:<10}{c.RESET} "
            f"{c.WHITE}{label:<32}{c.RESET}"
            f"{c.BRIGHT_GREEN}{sec:>8.2f}s{c.RESET}  "
            f"{c.WHITE}{detail}{c.RESET}"
        )

    mode_bits = [
        f"load_only={load_only}",
        f"dev_mode={dev_mode}",
        f"seed_mode={seed_mode}",
    ]
    mode_line = ", ".join(mode_bits)

    lines = [
        "",
        f"{c.BOLD}{c.CYAN}{'=' * 100}{c.RESET}",
        f"{c.BOLD}{c.BRIGHT_CYAN}🚀 Application Startup Complete (chronological ETL view){c.RESET}",
        f"{c.DIM}Mode: {mode_line}{c.RESET}",
        f"{c.BOLD}{c.CYAN}{'=' * 100}{c.RESET}",
        "",
        f"{c.BOLD}{'Phase':<10} {'Step':<32} {'Time':>8}  {'Details'}{c.RESET}",
        f"{c.GRAY}{'-' * 100}{c.RESET}",
    ]

    # 1) Extract / Load core data.
    load_detail = f"{member_count} members, {committee_count} committees, {bill_count} bills"
    if load_only:
        load_detail += f" {c.DIM}(cache-only startup){c.RESET}"
    elif seed_mode and elapsed_load < 0.5:
        load_detail += f" {c.DIM}(seed fallback){c.RESET}"
    else:
        load_detail += f" {c.DIM}(cache/scrape){c.RESET}"
    lines.append(row("Extract", "1) Load core data", elapsed_load, load_detail))

    # 2) Transform analytics.
    lines.append(
        row(
            "Transform",
            "2) Compute analytics",
            elapsed_analytics,
            f"{member_count} scorecards + Moneyball profiles",
        )
    )

    # 3) Transform seating enrichment.
    lines.append(
        row(
            "Transform",
            "3) Seating enrichment",
            elapsed_seating,
            "Senate seat blocks + seatmate affinity",
        )
    )

    # 4) Load/export Obsidian artifacts.
    export_detail = f"{exported_bill_count} bills exported ({bill_count} in memory)"
    lines.append(
        row(
            "Load",
            "4) Export vault artifacts",
            elapsed_export,
            export_detail,
        )
    )

    # 5) Transform committee-level indexes.
    committee_detail = (
        f"{committee_count} committee stats, {member_committee_role_count} members with roles"
    )
    lines.append(
        row(
            "Transform",
            "5) Committee indexes",
            elapsed_committee,
            committee_detail,
        )
    )

    # 6) Transform vote events and bill-level vote lookup.
    vote_detail = f"{vote_event_count} vote events"
    if bills_with_votes > 0:
        vote_detail += f" ({bills_with_votes} bills)"
    if elapsed_votes < 0.1 and vote_event_count > 0:
        vote_detail += f" {c.DIM}(cached){c.RESET}"
    lines.append(
        row(
            "Transform",
            "6) Vote event index + normalize",
            elapsed_votes,
            vote_detail,
        )
    )

    # 7) Transform member voting records and policy category lookups.
    voting_records_detail = (
        f"{member_vote_record_count} members, {category_bill_set_count} category bill sets"
    )
    lines.append(
        row(
            "Transform",
            "7) Member voting records",
            elapsed_voting_records,
            voting_records_detail,
        )
    )

    # 8) Transform witness slips.
    slip_detail = f"{slip_count} slips"
    if bills_with_slips > 0:
        slip_detail += f" ({bills_with_slips} bills)"
    if elapsed_slips < 0.1 and slip_count > 0:
        slip_detail += f" {c.DIM}(cached){c.RESET}"
    lines.append(row("Transform", "8) Witness slip index", elapsed_slips, slip_detail))

    # 9) Load reference crosswalk.
    lines.append(
        row(
            "Reference",
            "9) ZIP district crosswalk",
            elapsed_zip,
            f"{zcta_count} ZCTAs → IL Senate/House districts",
        )
    )

    lines.extend(
        [
            f"{c.GRAY}{'-' * 100}{c.RESET}",
            f"{c.BOLD}{'Total':<43}{c.BRIGHT_CYAN}{elapsed_total:>8.2f}s{c.RESET}  "
            f"{c.DIM}{mode_line}{c.RESET}",
            f"{c.BOLD}{c.CYAN}{'=' * 100}{c.RESET}",
            "",
        ]
    )

    return "\n".join(lines)


def _log_startup_timing(
    total_s: float,
    load_s: float,
    analytics_s: float,
    seating_s: float,
    export_s: float,
    votes_s: float,
    slips_s: float,
    zip_s: float,
    member_count: int,
    bill_count: int,
    vote_count: int,
    slip_count: int,
    zcta_count: int,
    dev_mode: bool,
    seed_mode: bool,
) -> None:
    """Append startup timing to .startup_timings.csv for historical tracking."""
    log_file = Path(".startup_timings.csv")
    is_new = not log_file.exists()

    with open(log_file, "a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "timestamp,total_s,load_s,analytics_s,seating_s,export_s,votes_s,slips_s,zip_s,"
                "members,bills,votes,slips,zctas,dev_mode,seed_mode\n"
            )
        f.write(
            f"{datetime.now().isoformat()},{total_s:.2f},{load_s:.2f},{analytics_s:.2f},"
            f"{seating_s:.2f},{export_s:.2f},{votes_s:.2f},{slips_s:.2f},{zip_s:.2f},"
            f"{member_count},{bill_count},{vote_count},{slip_count},{zcta_count},"
            f"{dev_mode},{seed_mode}\n"
        )
    LOGGER.debug("Startup timing logged to %s", log_file)


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


# ── App state container ──────────────────────────────────────────────────────


class AppState:
    def __init__(self) -> None:
        self.members: list[Member] = []
        self.member_lookup: dict[str, Member] = {}  # name-keyed (for vote normalization, schema)
        self.member_lookup_by_id: dict[
            str, Member
        ] = {}  # id-keyed (for influence, graph, deep-dives)
        self.bills: list[Bill] = []
        self.bill_lookup: dict[str, Bill] = {}
        self.committees: list[Committee] = []
        self.committee_lookup: dict[str, Committee] = {}
        self.committee_rosters: dict[str, list[CommitteeMemberRole]] = {}
        self.committee_bills: dict[str, list[str]] = {}
        self.committee_stats: dict[str, CommitteeStats] = {}
        self.member_committee_roles: dict[str, list[dict]] = {}
        self.scorecards: dict[str, MemberScorecard] = {}
        self.moneyball: MoneyballReport | None = None
        self.vote_events: list[VoteEvent] = []
        self.vote_lookup: dict[str, list[VoteEvent]] = {}  # bill_number -> votes
        self.member_vote_records: dict[str, VotingSummary] = {}  # member_name -> voting summary
        self.category_bill_sets: dict[str, set[str]] = {}  # category -> bill_numbers
        self.witness_slips: list[WitnessSlip] = []
        self.witness_slips_lookup: dict[str, list[WitnessSlip]] = {}  # bill_number -> slips
        self.zip_to_district: dict[str, ZipDistrictInfo] = {}  # ZCTA -> district info
        self.ml: object | None = None  # MLData from ml_loader (optional)
        # ── Co-sponsorship graph (for /explore visualization) ──
        self.cosponsor_adjacency: dict[str, set[str]] = {}  # member_id -> {peer_ids}
        # ── Influence engine (computed after vote data + ML) ──
        self.pivotality: dict = {}  # member_name -> MemberPivotality
        self.sponsor_pull: dict = {}  # member_id -> SponsorPull
        self.influence: dict = {}  # member_id -> InfluenceProfile
        self.coalition_influence: list = []  # CoalitionInfluence per bloc


state = AppState()


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
        state.cosponsor_adjacency = build_cosponsor_edges(state.members)
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
            _CATEGORY_COMMITTEES,
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
        from .ml_loader import load_ml_data

        state.ml = load_ml_data()
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
    elapsed_influence = 0.0
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
    summary = _format_startup_table(
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
        LOAD_ONLY,
        DEV_MODE,
        SEED_MODE,
    )
    print(summary, flush=True)

    # Show MVP
    if state.moneyball and state.moneyball.mvp_house_non_leadership:
        mvp = state.moneyball.profiles[state.moneyball.mvp_house_non_leadership]
        print(
            f"  🏆 MVP (House, non-leadership): {mvp.member_name} (Score: {mvp.moneyball_score})\n",
            flush=True,
        )

    # ── Log to timing file for historical tracking ──
    _log_startup_timing(
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

    yield


# ── GraphQL schema ───────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=16384)
def _parse_bill_date(date_str: str) -> datetime:
    """Parse 'M/D/YYYY' into a datetime for sorting.

    Unparseable dates return ``datetime.max`` so they sort *after* valid dates
    in ascending order rather than polluting the front of the list.
    """
    try:
        return datetime.strptime(date_str, "%m/%d/%Y")
    except (ValueError, TypeError):
        return datetime.max


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


def _safe_parse_date(date_str: str, param_name: str) -> datetime | None:
    """Parse an ISO date string, returning None and logging on failure."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        LOGGER.warning("Invalid date for %s: %r", param_name, date_str)
        return None


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
                result = [b for b in result if _parse_bill_date(b.last_action_date) >= from_dt]
        if date_to is not None:
            to_dt = _safe_parse_date(date_to, "dateTo")
            if to_dt is not None:
                result = [b for b in result if _parse_bill_date(b.last_action_date) <= to_dt]

        # ── sorting ──
        if sort_by is not None:
            reverse = sort_order == SortOrder.DESC
            if sort_by == BillSortField.LAST_ACTION_DATE:
                result.sort(
                    key=lambda b: _parse_bill_date(b.last_action_date),
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
        exempt = {"/health", "/docs", "/openapi.json", "/redoc"}
        path = request.url.path
        if (
            path not in exempt
            and not path.startswith("/advocacy")
            and not path.startswith("/explore")
            and not path.startswith("/api/graph")
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


# ── SSR Advocacy routes ──────────────────────────────────────────────────────

# Policy categories mapped to Senate committee codes.  When the user picks a
# category the Power Broker / Ally search is restricted to members who sit on
# at least one of the listed committees.  "All" (empty string) = no filter.
_CATEGORY_COMMITTEES: dict[str, list[str]] = {
    "": [],  # No filter — default
    "Transportation": ["STRN"],
    "Agriculture": ["SAGR"],
    "Commerce & Small Business": ["SCOM", "SBTE"],
    "Criminal Justice": ["SCRL", "SHRJ"],
    "Education": ["SESE", "SCHE"],
    "Energy & Environment": ["SENE", "SNVR"],
    "Healthcare & Human Services": ["SBMH", "SCHW", "SHUM"],
    "Housing": ["SHOU"],
    "Insurance & Finance": ["SINS", "SFIC"],
    "Labor": ["SLAB"],
    "Revenue & Pensions": ["SREV", "SPEN"],
    "State Government": ["SGOA", "SHEE", "SEXC"],
}

# Labels in display order for the dropdown.
CATEGORY_CHOICES: list[tuple[str, str]] = [
    ("", "All categories"),
    ("Transportation", "Transportation"),
    ("Agriculture", "Agriculture"),
    ("Commerce & Small Business", "Commerce & Small Business"),
    ("Criminal Justice", "Criminal Justice"),
    ("Education", "Education"),
    ("Energy & Environment", "Energy & Environment"),
    ("Healthcare & Human Services", "Healthcare & Human Services"),
    ("Housing", "Housing"),
    ("Insurance & Finance", "Insurance & Finance"),
    ("Labor", "Labor"),
    ("Revenue & Pensions", "Revenue & Pensions"),
    ("State Government", "State Government"),
]


def _committee_member_ids(committee_codes: list[str]) -> set[str]:
    """Return a set of member IDs that sit on any of the given committees."""
    ids: set[str] = set()
    for code in committee_codes:
        for role in state.committee_rosters.get(code, []):
            ids.add(role.member_id)
    return ids


def _build_influence_dict(member: Member) -> dict | None:
    """Build influence data for a card from the influence engine state."""
    ip = state.influence.get(member.id)
    if not ip:
        return None

    piv = state.pivotality.get(member.name)
    sp = state.sponsor_pull.get(member.id)

    d: dict = {
        "score": ip.influence_score,
        "label": ip.influence_label,
        "rank_overall": ip.rank_overall,
        "rank_chamber": ip.rank_chamber,
        "signals": ip.influence_signals,
        # Component breakdowns
        "moneyball_pct": round(ip.moneyball_normalized * 100, 1),
        "betweenness_pct": round(ip.betweenness_normalized * 100, 1),
        "pivotality_pct": round(ip.pivotality_normalized * 100, 1),
        "pull_pct": round(ip.pull_normalized * 100, 1),
    }

    if piv:
        d["close_votes"] = piv.close_votes_total
        d["pivotal_winning"] = piv.pivotal_winning
        d["swing_votes"] = piv.swing_votes

    if sp:
        d["sponsor_lift"] = round(sp.sponsor_lift, 3)
        d["cosponsor_lift"] = round(sp.cosponsor_lift, 3)

    return d


def _member_to_card(member: Member, *, why: str = "", badges: list[str] | None = None) -> dict:
    """Convert a Member to a template-friendly dict for card rendering.

    Includes empirical stats first (laws passed, passage rate, cross-party %),
    then Moneyball composite with a short explanation so derived metrics are clear.
    A ``scorecard`` sub-dict is attached when scorecard data exists so the
    template can render the full Legislative Scorecard (Lawmaking / Resolutions
    / Overall).
    ``script_hint`` is set to ``""`` here; callers populate it with an
    evidence-based hint via the ``_build_script_hint_*`` helpers.
    """
    phone = None
    for office in member.offices:
        if office.phone:
            phone = office.phone
            break

    mb = None
    if state.moneyball:
        mb = state.moneyball.profiles.get(member.id)

    # Empirical (raw) stats — directly from data
    laws_filed = mb.laws_filed if mb else None
    laws_passed = mb.laws_passed if mb else None
    passage_rate_pct = round((mb.effectiveness_rate * 100), 1) if mb and mb.laws_filed else None
    bridge_pct = round((mb.bridge_score * 100), 1) if mb else None

    # ── Full scorecard (Lawmaking / Resolutions / Overall) ──
    sc = state.scorecards.get(member.id)
    scorecard_dict = None
    if sc is not None and (sc.primary_bill_count > 0 or sc.law_heat_score > 0):
        scorecard_dict = {
            # Lawmaking (HB/SB)
            "laws_filed": sc.law_heat_score,
            "laws_passed": sc.law_passed_count,
            "law_pass_rate_pct": round(sc.law_success_rate * 100, 1),
            "magnet_score": round(sc.magnet_score, 1),
            "bridge_pct": round(sc.bridge_score * 100, 1),
            # Resolutions (HR/SR/HJR/SJR)
            "resolutions_filed": sc.resolutions_count,
            "resolutions_passed": sc.resolutions_passed_count,
            "resolution_pass_rate_pct": round(sc.resolution_pass_rate * 100, 1),
            # Overall
            "total_bills": sc.primary_bill_count,
            "total_passed": sc.passed_count,
            "overall_pass_rate_pct": round(sc.success_rate * 100, 1),
            "vetoed_count": sc.vetoed_count,
            "stuck_count": sc.stuck_count,
            "in_progress_count": sc.in_progress_count,
        }

    # ── Moneyball component breakdown for template ──
    moneyball_dict = None
    if mb:
        moneyball_dict = {
            "effectiveness_rate_pct": round(mb.effectiveness_rate * 100, 1),
            "pipeline_depth_avg": round(mb.pipeline_depth_avg, 2),
            "magnet_score": round(mb.magnet_score, 1),
            "bridge_pct": round(mb.bridge_score * 100, 1),
            "network_centrality": round(mb.network_centrality, 3),
            "institutional_weight": round(mb.institutional_weight, 2),
        }

    # ── Influence Network (human-readable power picture) ──
    influence_dict = None
    if mb and mb.unique_collaborators > 0:
        # Determine bipartisan label based on party balance of collaborators
        total_collab = (
            mb.collaborator_republicans + mb.collaborator_democrats + mb.collaborator_other
        )
        minority_share = (
            min(mb.collaborator_republicans, mb.collaborator_democrats) / total_collab
            if total_collab > 0
            else 0.0
        )
        if minority_share >= 0.3:
            bipartisan_label = "high bipartisan reach"
        elif minority_share >= 0.15:
            bipartisan_label = "moderate bipartisan reach"
        else:
            bipartisan_label = ""

        influence_dict = {
            "unique_collaborators": mb.unique_collaborators,
            "collaborator_republicans": mb.collaborator_republicans,
            "collaborator_democrats": mb.collaborator_democrats,
            "collaborator_other": mb.collaborator_other,
            "bipartisan_label": bipartisan_label,
            "magnet_score": round(mb.magnet_score, 1),
            "magnet_vs_chamber": mb.magnet_vs_chamber,
            "cosponsor_passage_rate_pct": round(mb.cosponsor_passage_rate * 100, 1),
            "cosponsor_passage_multiplier": mb.cosponsor_passage_multiplier,
            "chamber_median_cosponsor_rate_pct": round(
                mb.chamber_median_cosponsor_rate * 100,
                1,
            ),
            "passage_rate_vs_caucus": mb.passage_rate_vs_caucus,
            "caucus_avg_passage_rate_pct": round(mb.caucus_avg_passage_rate * 100, 1),
        }

    # ── Committee assignments with roles and stats ──
    committee_roles = state.member_committee_roles.get(member.id, [])

    # ── Voting Record ──
    # Always show the full voting record — category filtering would reduce it
    # to only bills currently pending in those committees (a tiny subset) and
    # make the section vanish for most members.
    voting_record_dict: dict | None = None
    vr = state.member_vote_records.get(member.name)
    if vr and vr.total_votes > 0:
        voting_record_dict = {
            "total_votes": vr.total_votes,
            "total_floor_votes": vr.total_floor_votes,
            "total_committee_votes": vr.total_committee_votes,
            "yes_count": vr.yes_count,
            "no_count": vr.no_count,
            "present_count": vr.present_count,
            "nv_count": vr.nv_count,
            "yes_rate_pct": vr.yes_rate_pct,
            "party_alignment_pct": vr.party_alignment_pct,
            "party_defection_count": vr.party_defection_count,
            "is_persuadable": vr.party_defection_count > 0,
            "records": [
                {
                    "bill_number": r.bill_number,
                    "bill_description": r.bill_description,
                    "date": r.date,
                    "vote": r.vote,
                    "bill_status": r.bill_status,
                    "vote_type": r.vote_type,
                    "bill_status_url": r.bill_status_url,
                }
                for r in vr.records
            ],
        }

    # ── Institutional Power Badges ──
    power_badges_list: list[dict] = []
    chamber_size = 0
    if mb:
        chamber_size = (
            (
                len(state.moneyball.rankings_house)
                if member.chamber == "House"
                else len(state.moneyball.rankings_senate)
            )
            if state.moneyball
            else 0
        )
        raw_badges = compute_power_badges(mb, committee_roles, chamber_size)
        power_badges_list = [
            {
                "label": pb.label,
                "icon": pb.icon,
                "explanation": pb.explanation,
                "css_class": pb.css_class,
            }
            for pb in raw_badges
        ]

    # ── Ranking context (human-readable power position) ──
    rank_chamber = mb.rank_chamber if mb else None
    rank_percentile = None
    if mb and chamber_size > 0:
        rank_percentile = round((1 - (mb.rank_chamber - 1) / chamber_size) * 100)

    # ── Party abbreviation for compact display ──
    if "republican" in (member.party or "").lower():
        party_abbr = "R"
    elif "democrat" in (member.party or "").lower():
        party_abbr = "D"
    elif member.party:
        party_abbr = member.party[:1]
    else:
        party_abbr = ""

    # ── Active bill count ──
    active_count = 0
    for bid in member.sponsored_bill_ids or []:
        b = state.bill_lookup.get(bid)
        if b and b.last_action:
            action_lower = b.last_action.lower()
            if not any(
                kw in action_lower
                for kw in (
                    "public act",
                    "effective date",
                    "vetoed",
                    "tabled",
                    "postponed indefinitely",
                    "session sine die",
                )
            ):
                active_count += 1

    return {
        "name": member.name,
        "id": member.id,
        "district": member.district,
        "party": member.party,
        "party_abbr": party_abbr,
        "chamber": member.chamber,
        "role": member.role,
        "phone": phone,
        "email": member.email,
        "laws_filed": laws_filed,
        "laws_passed": laws_passed,
        "passage_rate_pct": passage_rate_pct,
        "bridge_score": round(mb.bridge_score, 4) if mb else None,
        "bridge_pct": bridge_pct,
        "moneyball_score": round(mb.moneyball_score, 2) if mb else None,
        "moneyball_explanation": MONEYBALL_ONE_LINER,
        "moneyball": moneyball_dict,
        "influence_network": influence_dict,
        "member_url": member.member_url,
        "why": why,
        "badges": badges or [],
        "power_badges": power_badges_list,
        "script_hint": "",
        "scorecard": scorecard_dict,
        "committee_roles": committee_roles,
        "voting_record": voting_record_dict,
        "rank_chamber": rank_chamber,
        "chamber_size": chamber_size,
        "rank_percentile": rank_percentile,
        "active_bills": active_count,
        # ── Influence engine ──
        "influence": _build_influence_dict(member),
    }


# ── Evidence-based script hint builders ──────────────────────────────────────


def _stats_sentence(card: dict) -> str:
    """Build a short stats clause from the card's empirical fields.

    Example: "They've passed 4 of 12 laws (33.3% passage rate) and 25.0% of
    their bills have cross-party co-sponsors."
    """
    parts: list[str] = []
    if card.get("laws_filed") and card.get("laws_passed") is not None:
        parts.append(
            f"they've passed {card['laws_passed']} of {card['laws_filed']} laws "
            f"({card['passage_rate_pct'] or 0}% passage rate)"
        )
    if card.get("bridge_pct") is not None and card["bridge_pct"] > 0:
        parts.append(f"{card['bridge_pct']}% of their bills have cross-party co-sponsors")
    if parts:
        return (
            parts[0][0].upper()
            + parts[0][1:]
            + (" and " + parts[1] if len(parts) > 1 else "")
            + "."
        )
    return ""


def _build_script_hint_senator(card: dict, zip_code: str, district: str) -> str:
    """Evidence-based script hint for Your Senator."""
    stats = _stats_sentence(card)
    stats_line = f" {stats}" if stats else ""
    return (
        f"This is YOUR state senator. Call their office, say you live in "
        f"ZIP {zip_code} (District {district}), and tell them you support "
        f"kei truck legalization in Illinois. Constituent calls are tracked "
        f"\u2014 yours counts.{stats_line}"
    )


def _build_script_hint_rep(card: dict, zip_code: str, district: str) -> str:
    """Evidence-based script hint for Your Representative."""
    stats = _stats_sentence(card)
    stats_line = f" {stats}" if stats else ""
    return (
        f"This is YOUR state representative. They vote on bills in the House "
        f"before they reach the Senate. Call their office, reference ZIP "
        f"{zip_code} (District {district}), and ask them to sponsor or support "
        f"kei truck legislation.{stats_line}"
    )


def _build_script_hint_broker(card: dict, broker_why: str) -> str:
    """Evidence-based script hint for Power Broker."""
    # Determine if chosen as Chair or by Moneyball
    is_chair = "Chair of the" in broker_why
    stats = _stats_sentence(card)

    if is_chair:
        lead = (
            "This legislator chairs the committee that controls whether your bill "
            "gets a hearing \u2014 they are the institutional gatekeeper."
        )
    else:
        lead = (
            "This legislator has the highest overall influence score (Moneyball) "
            "in the Senate outside your district."
        )

    evidence = f" {stats}" if stats else ""
    action = (
        " When you call, reference the bill, mention broad constituent support, "
        "and ask for their co-sponsorship."
    )
    return lead + evidence + action


def _build_script_hint_ally(card: dict) -> str:
    """Evidence-based script hint for Potential Ally."""
    bridge = card.get("bridge_pct")
    if bridge and bridge > 0:
        evidence = (
            f" They have a {bridge}% cross-party co-sponsorship rate \u2014 "
            "meaning they regularly work across the aisle."
        )
    else:
        evidence = ""
    return (
        "This senator sits physically next to yours in the chamber."
        + evidence
        + " Ask your senator to partner with them on kei truck legislation "
        "\u2014 proximity and bipartisan track record make them a natural ally."
    )


def _build_script_hint_super_ally(card: dict) -> str:
    """Evidence-based script hint for Super Ally (merged Broker + Ally)."""
    stats = _stats_sentence(card)
    evidence = f" {stats}" if stats else ""
    return (
        "This legislator is both the most influential senator in the chamber "
        "AND a physical neighbor of your senator \u2014 a uniquely powerful "
        "advocacy target." + evidence + " Ask your senator to partner with "
        "them directly. When you call their office, reference the bill and "
        "the fact that they sit next to your senator. This is your "
        "highest-value strategic target."
    )


def _find_member_by_district(chamber: str, district: str) -> Member | None:
    """Find a member by chamber (case-insensitive) and district number."""
    chamber_lower = chamber.lower()
    for m in state.members:
        if m.chamber.lower() == chamber_lower and m.district == district:
            return m
    return None


def _find_power_broker(
    exclude_district: str,
    *,
    committee_ids: set[str] | None = None,
    committee_codes: list[str] | None = None,
    category_name: str = "",
) -> tuple[Member | None, str]:
    """Find the top Senate Power Broker.

    Selection priority:
    1.  When a policy *committee_codes* filter is active, first look for the
        **Committee Chair** of the relevant committee(s).  A Chair is the
        strongest institutional voice on a topic — they control what bills
        get heard.
    2.  If no Chair is found (or no committee filter is active), fall back to
        the Senate member with the **highest Moneyball score**.

    Returns ``(Member | None, why_text)``.
    """
    member_lookup = {m.id: m for m in state.members}

    # ── Priority 1: Committee Chair (when a policy category is selected) ──
    if committee_codes:
        for code in committee_codes:
            for cmr in state.committee_rosters.get(code, []):
                role_lower = cmr.role.lower()
                # Match "Chairperson", "Chair" but NOT "Vice-Chair*"
                if "chair" in role_lower and "vice" not in role_lower:
                    chair_member = member_lookup.get(cmr.member_id)
                    if (
                        chair_member
                        and chair_member.chamber == "Senate"
                        and chair_member.district != exclude_district
                    ):
                        committee_name = ""
                        cmt = state.committee_lookup.get(code)
                        if cmt:
                            committee_name = cmt.name
                        parts = [f"Chair of the {committee_name or code} committee"]
                        if category_name:
                            parts.append(
                                f"the institutional gatekeeper for {category_name} legislation"
                            )
                        mb = (
                            state.moneyball.profiles.get(cmr.member_id) if state.moneyball else None
                        )
                        if mb:
                            parts.append(
                                f"Moneyball score: {mb.moneyball_score}, "
                                f"effectiveness: {mb.effectiveness_rate:.0%}"
                            )
                        why = ". ".join(parts) + "."
                        return chair_member, why

    # ── Priority 2: Highest Moneyball score (fallback) ──
    if not state.moneyball:
        return None, ""

    best_profile = None
    for profile in state.moneyball.profiles.values():
        if profile.chamber != "Senate":
            continue
        if profile.district == exclude_district:
            continue
        if committee_ids and profile.member_id not in committee_ids:
            continue
        if best_profile is None or profile.moneyball_score > best_profile.moneyball_score:
            best_profile = profile

    if best_profile is None:
        return None, ""

    member = member_lookup.get(best_profile.member_id)
    if member is None:
        return None, ""

    parts = [
        f"Highest Moneyball score ({best_profile.moneyball_score}) "
        f"in the Senate outside your district",
    ]
    if category_name:
        parts.append(f"sits on a {category_name} committee")
    parts.append(
        f"effectiveness: {best_profile.effectiveness_rate:.0%}, "
        f"{best_profile.unique_collaborators} collaborators"
    )
    why = ". ".join(parts) + "."
    return member, why


def _find_ally(
    senator: Member,
    *,
    committee_ids: set[str] | None = None,
    category_name: str = "",
) -> tuple[Member | None, str]:
    """Find the best Ally from the senator's seatmates.

    Returns ``(Member | None, why_text)``.
    """
    if not senator.seatmate_names:
        return None, ""

    best_member = None
    best_bridge = -1.0

    for seatmate_name in senator.seatmate_names:
        member = state.member_lookup.get(seatmate_name)
        if member is None:
            continue
        if committee_ids and member.id not in committee_ids:
            continue

        bridge = 0.0
        if state.moneyball:
            mb = state.moneyball.profiles.get(member.id)
            if mb:
                bridge = mb.bridge_score

        if bridge > best_bridge:
            best_bridge = bridge
            best_member = member

    # Fallback: if committee filter excluded everyone, try without it.
    if best_member is None and committee_ids:
        return _find_ally(senator, committee_ids=None, category_name="")

    # Fallback: if no one has a bridge score, pick first resolved seatmate.
    if best_member is None:
        for seatmate_name in senator.seatmate_names:
            member = state.member_lookup.get(seatmate_name)
            if member is not None:
                best_member = member
                break

    if best_member is None:
        return None, ""

    why_parts = ["Sits next to your senator in the chamber"]
    if category_name and committee_ids and best_member.id in committee_ids:
        why_parts.append(f"also on a {category_name} committee")
    if state.moneyball:
        mb = state.moneyball.profiles.get(best_member.id)
        if mb and mb.bridge_score > 0:
            why_parts.append(
                f"bridge score of {mb.bridge_score:.0%} (cross-party co-sponsorship rate)"
            )
    if senator.seatmate_affinity > 0:
        why_parts.append(f"{senator.seatmate_affinity:.0%} bill overlap with seatmates")
    why = ". ".join(why_parts) + "."
    return best_member, why


@app.get("/advocacy")
async def advocacy_index(request: Request):
    """Render the advocacy search page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Kei Truck Freedom",
            "categories": CATEGORY_CHOICES,
        },
    )


@app.post("/advocacy/search")
async def advocacy_search(
    request: Request,
    zip_code: str = Form(...),
    category: str = Form(""),
):
    """Look up advocacy targets for a given ZIP code and optional policy category.

    Returns up to four cards (or three if Power Broker and Ally are the same
    person, merged into a single "Super Ally" card):

    1. **Your Senator** — IL Senate member for this ZIP's district.
    2. **Your Representative** — IL House member for this ZIP's district.
    3. **Power Broker** — highest Moneyball score in the Senate (different district).
    4. **Potential Ally** — senator's physical seatmate with highest bridge score.

    When *category* is provided, Power Broker and Ally are filtered to members
    who sit on a committee in that policy area.

    When the request comes from htmx (``HX-Request`` header), only the
    results partial is returned.
    """
    zip_code = zip_code.strip()
    category = category.strip()
    is_htmx = request.headers.get("HX-Request") == "true"

    # ── Lookup ZIP in crosswalk ──
    district_info = state.zip_to_district.get(zip_code)
    if district_info is None:
        error = (
            f"ZIP code {zip_code!r} not found in Illinois district data. "
            "Please enter a valid 5-digit Illinois ZIP code."
        )
        tpl = "_results_partial.html" if is_htmx else "index.html"
        return templates.TemplateResponse(
            tpl,
            {
                "request": request,
                "title": "Kei Truck Freedom",
                "categories": CATEGORY_CHOICES,
                "error": error,
            },
        )

    senate_district = district_info.il_senate
    house_district = district_info.il_house
    warnings: list[str] = []

    # ── Committee filter ──
    committee_codes = _CATEGORY_COMMITTEES.get(category, [])
    committee_ids = _committee_member_ids(committee_codes) if committee_codes else None
    category_label = category if category else ""

    # ── Find Your Senator ──
    senator_member = (
        _find_member_by_district("senate", senate_district) if senate_district else None
    )
    senator_card = None
    if senator_member:
        senator_card = _member_to_card(
            senator_member,
            why=f"Represents IL Senate District {senate_district}, which contains ZIP {zip_code}.",
        )
        senator_card["script_hint"] = _build_script_hint_senator(
            senator_card,
            zip_code,
            senate_district,
        )
    elif senate_district:
        warnings.append(
            f"Senate District {senate_district} (for ZIP {zip_code}) — "
            "senator not in current data (dev/seed mode has limited members)."
        )

    # ── Find Your Representative ──
    rep_member = _find_member_by_district("house", house_district) if house_district else None
    rep_card = None
    if rep_member:
        rep_card = _member_to_card(
            rep_member,
            why=f"Represents IL House District {house_district}, which contains ZIP {zip_code}.",
        )
        rep_card["script_hint"] = _build_script_hint_rep(
            rep_card,
            zip_code,
            house_district,
        )
    elif house_district:
        warnings.append(
            f"House District {house_district} (for ZIP {zip_code}) — "
            "representative not in current data (dev/seed mode has limited members)."
        )

    # ── Find Power Broker ──
    exclude_dist = senate_district or ""
    broker_member, broker_why = _find_power_broker(
        exclude_dist,
        committee_ids=committee_ids,
        committee_codes=committee_codes or None,
        category_name=category_label,
    )

    # ── Find Potential Ally ──
    ally_member, ally_why = (
        _find_ally(
            senator_member,
            committee_ids=committee_ids,
            category_name=category_label,
        )
        if senator_member
        else (None, "")
    )

    # ── Merge: if broker and ally are the same person → "Super Ally" ──
    broker_card = None
    ally_card = None
    super_ally_card = None

    if broker_member and ally_member and broker_member.id == ally_member.id:
        # Same person — merge into a Super Ally with both badges.
        merged_why = (
            f"This legislator is both the most influential senator in the chamber "
            f"AND a physical neighbor of your senator — a uniquely powerful advocacy target. "
            f"{broker_why} {ally_why}"
        )
        super_ally_card = _member_to_card(
            broker_member,
            why=merged_why,
            badges=["Power Broker", "Potential Ally"],
        )
        super_ally_card["script_hint"] = _build_script_hint_super_ally(super_ally_card)
    else:
        if broker_member:
            broker_card = _member_to_card(broker_member, why=broker_why)
            broker_card["script_hint"] = _build_script_hint_broker(
                broker_card,
                broker_why,
            )
        if ally_member:
            ally_card = _member_to_card(ally_member, why=ally_why)
            ally_card["script_hint"] = _build_script_hint_ally(ally_card)

    error = "; ".join(warnings) if warnings else None

    member_count = len(state.members)
    zip_count = len(state.zip_to_district)
    tpl = "_results_partial.html" if is_htmx else "results.html"
    return templates.TemplateResponse(
        tpl,
        {
            "request": request,
            "title": "Kei Truck Freedom",
            "categories": CATEGORY_CHOICES,
            "seed_mode": SEED_MODE,
            "member_count": member_count,
            "zip_count": zip_count,
            "zip": zip_code,
            "category": category,
            "senate_district": senate_district,
            "house_district": house_district,
            "senator": senator_card,
            "representative": rep_card,
            "broker": broker_card,
            "ally": ally_card,
            "super_ally": super_ally_card,
            "error": error,
        },
    )


# ── ML Intelligence Dashboard routes ─────────────────────────────────────


@app.get("/intelligence")
async def intelligence_summary(request: Request):
    """Executive summary: narrative-driven intelligence overview."""
    ml = state.ml
    available = ml and ml.available

    if not available:
        return templates.TemplateResponse(
            "intelligence_summary.html",
            {"request": request, "title": "Intelligence", "available": False},
        )

    # ── Model confidence ──
    trust_level = ml.quality.get("trust_assessment", {}).get("overall", "")
    roc_auc = ml.quality.get("test_set_metrics", {}).get("roc_auc")
    accuracy_pct = None
    if ml.accuracy_history:
        latest = ml.accuracy_history[-1]
        accuracy_pct = latest.accuracy * 100

    total_bills_scored = len(ml.bill_scores)
    n_coalitions = len(set(m.coalition_id for m in ml.coalitions))
    flagged_anomalies = sum(1 for a in ml.anomalies if a.is_anomaly)

    # ── Bills to Watch: OPEN bills with interesting signals ──
    open_bills = [s for s in ml.bill_scores if s.lifecycle_status == "OPEN"]
    bills_to_watch = []

    # Category 1: High confidence ADVANCE predictions on open bills
    advance_preds = sorted(
        [s for s in open_bills if s.predicted_outcome == "ADVANCE" and s.confidence >= 0.75],
        key=lambda s: -s.prob_advance,
    )
    for s in advance_preds[:3]:
        why = f"High confidence advance prediction ({s.confidence:.0%})"
        if s.days_since_action > 60:
            why += f" despite {s.days_since_action} days idle"
        bills_to_watch.append(
            {
                "bill_id": s.bill_id,
                "bill_number": s.bill_number,
                "description": s.description,
                "sponsor": s.sponsor,
                "prob_advance": s.prob_advance,
                "prob_law": getattr(s, "prob_law", 0.0),
                "predicted_outcome": s.predicted_outcome,
                "predicted_destination": getattr(s, "predicted_destination", "Stuck"),
                "confidence": s.confidence,
                "stage_label": s.stage_label,
                "forecast_score": getattr(s, "forecast_score", 0.0),
                "forecast_confidence": getattr(s, "forecast_confidence", ""),
                "why": why,
            }
        )

    # Category 2: Surprise — model says ADVANCE but bill is in early stage
    surprises = sorted(
        [
            s
            for s in open_bills
            if s.predicted_outcome == "ADVANCE"
            and s.prob_advance >= 0.65
            and s.current_stage in ("IN_COMMITTEE", "FILED")
        ],
        key=lambda s: -s.prob_advance,
    )
    for s in surprises[:2]:
        if not any(b["bill_id"] == s.bill_id for b in bills_to_watch):
            bills_to_watch.append(
                {
                    "bill_id": s.bill_id,
                    "bill_number": s.bill_number,
                    "description": s.description,
                    "sponsor": s.sponsor,
                    "prob_advance": s.prob_advance,
                    "prob_law": getattr(s, "prob_law", 0.0),
                    "predicted_outcome": s.predicted_outcome,
                    "predicted_destination": getattr(s, "predicted_destination", "Stuck"),
                    "confidence": s.confidence,
                    "stage_label": s.stage_label,
                    "forecast_score": getattr(s, "forecast_score", 0.0),
                    "forecast_confidence": getattr(s, "forecast_confidence", ""),
                    "why": (
                        f"Surprise: still in {s.stage_label} but "
                        f"{s.prob_advance:.0%} chance of advancing"
                    ),
                }
            )

    # Category 3: High confidence STUCK on bills people might expect to move
    stuck_surprises = sorted(
        [
            s
            for s in open_bills
            if s.predicted_outcome == "STUCK"
            and s.confidence >= 0.80
            and s.current_stage in ("PASSED_COMMITTEE", "FLOOR_VOTE")
        ],
        key=lambda s: s.prob_advance,
    )
    for s in stuck_surprises[:2]:
        if not any(b["bill_id"] == s.bill_id for b in bills_to_watch):
            bills_to_watch.append(
                {
                    "bill_id": s.bill_id,
                    "bill_number": s.bill_number,
                    "description": s.description,
                    "sponsor": s.sponsor,
                    "prob_advance": s.prob_advance,
                    "prob_law": getattr(s, "prob_law", 0.0),
                    "predicted_outcome": s.predicted_outcome,
                    "predicted_destination": getattr(s, "predicted_destination", "Stuck"),
                    "confidence": s.confidence,
                    "stage_label": s.stage_label,
                    "forecast_score": getattr(s, "forecast_score", 0.0),
                    "forecast_confidence": getattr(s, "forecast_confidence", ""),
                    "why": (
                        f"Warning: reached {s.stage_label} but model "
                        f"predicts stall ({s.confidence:.0%} confidence)"
                    ),
                }
            )

    # ── Power Movers: top influencers ──
    power_movers = []
    influence_profiles = sorted(
        state.influence.values(),
        key=lambda p: p.influence_score,
        reverse=True,
    )
    for p in influence_profiles[:8]:
        member = state.member_lookup_by_id.get(p.member_id)
        if member:
            power_movers.append(
                {
                    "member_id": p.member_id,
                    "name": p.member_name,
                    "party": p.party,
                    "chamber": p.chamber,
                    "district": member.district,
                    "score": p.influence_score,
                    "label": p.influence_label,
                    "rank": p.rank_overall,
                    "signals": p.influence_signals,
                }
            )

    # ── Coalition Landscape ──
    coalitions_summary = []
    prof_map = {cp.coalition_id: cp for cp in ml.coalition_profiles}
    ci_map = {ci.coalition_id: ci for ci in state.coalition_influence}
    coalition_groups: dict[int, list] = {}
    for m in ml.coalitions:
        coalition_groups.setdefault(m.coalition_id, []).append(m)

    for cid, members in sorted(coalition_groups.items()):
        prof = prof_map.get(cid)
        ci = ci_map.get(cid)
        dem = sum(1 for m in members if m.party == "Democrat")
        rep = sum(1 for m in members if m.party == "Republican")
        coalitions_summary.append(
            {
                "name": prof.name if prof else f"Coalition {cid + 1}",
                "size": len(members),
                "dem": dem,
                "rep": rep,
                "cohesion": prof.cohesion if prof else 0,
                "focus_areas": prof.focus_areas if prof else [],
                "top_influencer": ci.top_influencer_name if ci else None,
            }
        )

    # ── Top anomalies ──
    top_anomalies = []
    for a in sorted(ml.anomalies, key=lambda a: -a.anomaly_score):
        if a.is_anomaly:
            top_anomalies.append(
                {
                    "bill_number": a.bill_number,
                    "description": a.description,
                    "reason": a.anomaly_reason,
                    "total_slips": a.total_slips,
                }
            )
            if len(top_anomalies) >= 5:
                break

    return templates.TemplateResponse(
        "intelligence_summary.html",
        {
            "request": request,
            "title": "Intelligence",
            "available": True,
            "trust_level": trust_level,
            "roc_auc": roc_auc,
            "accuracy_pct": accuracy_pct,
            "total_bills_scored": total_bills_scored,
            "n_coalitions": n_coalitions,
            "flagged_anomalies": flagged_anomalies,
            "bills_to_watch": bills_to_watch,
            "power_movers": power_movers,
            "coalitions_summary": coalitions_summary,
            "top_anomalies": top_anomalies,
            "last_run": ml.last_run_date,
        },
    )


@app.get("/intelligence/raw")
async def intelligence_raw(request: Request):
    """Raw data tables — the original tabbed ML dashboard for power users."""
    ml = state.ml
    available = ml and ml.available

    # Summary stats for the overview
    summary = {}
    if available:
        scores = ml.bill_scores
        advance_count = sum(1 for s in scores if s.predicted_outcome == "ADVANCE")
        stuck_count = sum(1 for s in scores if s.predicted_outcome == "STUCK")
        forecast_count = sum(1 for s in scores if not s.label_reliable)
        flagged = sum(1 for a in ml.anomalies if a.is_anomaly)
        n_coalitions = len(set(m.coalition_id for m in ml.coalitions))
        # Destination-based counts
        dest_law = sum(
            1
            for s in scores
            if getattr(s, "predicted_destination", "").startswith("→ Law")
            or getattr(s, "predicted_destination", "") == "Became Law"
        )
        dest_floor = sum(
            1
            for s in scores
            if getattr(s, "predicted_destination", "") in ("→ Floor", "→ Passed", "→ Governor")
        )
        dest_stuck = sum(1 for s in scores if getattr(s, "predicted_destination", "") == "Stuck")
        summary = {
            "total_predictions": len(scores),
            "advance_count": advance_count,
            "stuck_count": stuck_count,
            "dest_law": dest_law,
            "dest_floor": dest_floor,
            "dest_stuck": dest_stuck,
            "forecast_count": forecast_count,
            "flagged_anomalies": flagged,
            "total_anomalies": len(ml.anomalies),
            "n_coalitions": n_coalitions,
            "n_coalition_members": len(ml.coalitions),
            "last_run": ml.last_run_date,
            "model": ml.quality.get("model_selected", ""),
            "trust": ml.quality.get("trust_assessment", {}).get("overall", ""),
            "roc_auc": ml.quality.get("test_set_metrics", {}).get("roc_auc"),
            "accuracy_runs": len(ml.accuracy_history),
            "n_committees": len(state.committees),
            "active_committees": sum(
                1 for cs in state.committee_stats.values() if cs.total_bills >= 10
            ),
        }

    return templates.TemplateResponse(
        "intelligence.html",
        {
            "request": request,
            "title": "ML Intelligence",
            "available": available,
            "summary": summary,
            "ml": ml,
        },
    )


@app.get("/intelligence/predictions")
async def intelligence_predictions(request: Request):
    """Tab: bill predictions."""
    ml = state.ml
    if not ml or not ml.available:
        return templates.TemplateResponse(
            "_intelligence_predictions.html",
            {"request": request, "predictions": [], "ml": None},
        )

    predictions = sorted(ml.bill_scores, key=lambda s: -s.prob_advance)
    return templates.TemplateResponse(
        "_intelligence_predictions.html",
        {"request": request, "predictions": predictions, "ml": ml},
    )


@app.get("/intelligence/coalitions")
async def intelligence_coalitions(request: Request):
    """Tab: voting coalitions."""
    ml = state.ml
    if not ml or not ml.available:
        return templates.TemplateResponse(
            "_intelligence_coalitions.html",
            {"request": request, "groups": [], "ml": None},
        )

    groups: dict[int, list] = {}
    for m in ml.coalitions:
        groups.setdefault(m.coalition_id, []).append(m)

    # Build profile lookup
    prof_map = {p.coalition_id: p for p in ml.coalition_profiles}

    coalition_list = []
    for cid, members in sorted(groups.items()):
        dem = sum(1 for m in members if m.party == "Democrat")
        rep = sum(1 for m in members if m.party == "Republican")
        members_sorted = sorted(members, key=lambda m: (m.party, m.name))
        prof = prof_map.get(cid)
        coalition_list.append(
            {
                "id": cid,
                "name": prof.name if prof else f"Coalition {cid + 1}",
                "size": len(members),
                "dem": dem,
                "rep": rep,
                "cross_party": dem > 0 and rep > 0,
                "focus_areas": prof.focus_areas if prof else [],
                "yes_rate": prof.yes_rate if prof else 0.0,
                "cohesion": prof.cohesion if prof else 0.0,
                "signature_bills": (prof.signature_bills[:5] if prof else []),
                "members": members_sorted,
            }
        )

    return templates.TemplateResponse(
        "_intelligence_coalitions.html",
        {"request": request, "groups": coalition_list, "ml": ml},
    )


@app.get("/intelligence/anomalies")
async def intelligence_anomalies(request: Request):
    """Tab: anomaly detection."""
    ml = state.ml
    if not ml or not ml.available:
        return templates.TemplateResponse(
            "_intelligence_anomalies.html",
            {"request": request, "anomalies": [], "ml": None},
        )

    anomalies = sorted(ml.anomalies, key=lambda a: -a.anomaly_score)
    return templates.TemplateResponse(
        "_intelligence_anomalies.html",
        {"request": request, "anomalies": anomalies, "ml": ml},
    )


@app.get("/intelligence/influence")
async def intelligence_influence(request: Request):
    """Tab: influence leaderboard."""
    profiles = list(state.influence.values())
    if not profiles:
        return templates.TemplateResponse(
            "_intelligence_influence.html",
            {"request": request, "profiles": [], "coalition_influence": []},
        )

    profiles.sort(key=lambda p: p.influence_score, reverse=True)

    # Build template-friendly dicts
    profile_dicts = [
        {
            "rank_overall": p.rank_overall,
            "name": p.member_name,
            "chamber": p.chamber,
            "party": p.party,
            "score": p.influence_score,
            "label": p.influence_label,
            "moneyball_pct": round(p.moneyball_normalized * 100, 1),
            "betweenness_pct": round(p.betweenness_normalized * 100, 1),
            "pivotality_pct": round(p.pivotality_normalized * 100, 1),
            "pull_pct": round(p.pull_normalized * 100, 1),
            "signals": p.influence_signals,
        }
        for p in profiles
    ]

    # Coalition influence
    ci_dicts = [
        {
            "coalition_id": ci.coalition_id,
            "coalition_name": ci.coalition_name,
            "total_members": ci.total_members,
            "avg_influence": ci.avg_influence,
            "high_influence_count": ci.high_influence_count,
            "top_influencer_name": ci.top_influencer_name,
            "top_influencer_score": ci.top_influencer_score,
            "top_influencer_label": ci.top_influencer_label,
            "bridge_member_name": ci.bridge_member_name,
            "bridge_member_betweenness": ci.bridge_member_betweenness,
        }
        for ci in state.coalition_influence
    ]

    return templates.TemplateResponse(
        "_intelligence_influence.html",
        {
            "request": request,
            "profiles": profile_dicts,
            "coalition_influence": ci_dicts,
        },
    )


# Procedural/routing committees: bills are assigned here after passing substantive
# committees (e.g. "Referred to Rules * Reports"). "Advanced" in our pipeline
# means last_action = Do Pass/Reported Out, so these show 0% and are misleading.
_PROCEDURAL_COMMITTEE_NAMES = frozenset(
    {
        "rules * reports",
        "assignments * reports",
        "committee of the whole",
        "assignments",
        "rules committee",
    }
)


@app.get("/intelligence/committees")
async def intelligence_committees(request: Request):
    """Tab: committee power dashboard."""
    if not state.committees:
        return templates.TemplateResponse(
            "_intelligence_committees.html",
            {
                "request": request,
                "committees": [],
                "top_by_volume": [],
                "top_by_passage": [],
                "top_law_factories": [],
            },
        )

    # Build template-friendly committee dicts
    committee_dicts = []
    for c in state.committees:
        cstats = state.committee_stats.get(c.code)
        roster = state.committee_rosters.get(c.code, [])

        # Determine chamber from code prefix
        chamber = "Senate" if c.code.startswith("S") else "House"

        # Procedural committees (Rules, Assignments) route bills after passage;
        # our "advanced" count is 0 there, which is misleading.
        is_procedural = c.name.strip().lower() in _PROCEDURAL_COMMITTEE_NAMES

        # Find the chair
        chair_name = None
        chair_id = None
        for cmr in roster:
            if cmr.role.lower() == "chair":
                chair_name = cmr.member_name
                chair_id = cmr.member_id
                break

        committee_dicts.append(
            {
                "code": c.code,
                "name": c.name,
                "chamber": chamber,
                "total_bills": cstats.total_bills if cstats else 0,
                "advanced_count": cstats.advanced_count if cstats else 0,
                "passed_count": cstats.passed_count if cstats else 0,
                "advancement_rate": cstats.advancement_rate if cstats else 0.0,
                "pass_rate": cstats.pass_rate if cstats else 0.0,
                "chair": chair_name,
                "chair_id": chair_id,
                "member_count": len(roster),
                "is_procedural": is_procedural,
            }
        )

    # Sort by total bills (busiest first)
    committee_dicts.sort(key=lambda x: -x["total_bills"])

    # Insight cards: exclude procedural from passage/law so they don't dominate
    substantive = [c for c in committee_dicts if not c["is_procedural"]]
    active = [c for c in substantive if c["total_bills"] >= 10]
    top_by_volume = sorted(committee_dicts, key=lambda x: -x["total_bills"])[:10]
    top_by_passage = sorted(active, key=lambda x: -x["advancement_rate"])[:10]
    top_law_factories = sorted(
        [c for c in substantive if c["passed_count"] > 0],
        key=lambda x: -x["passed_count"],
    )[:10]

    return templates.TemplateResponse(
        "_intelligence_committees.html",
        {
            "request": request,
            "committees": committee_dicts,
            "top_by_volume": top_by_volume,
            "top_by_passage": top_by_passage,
            "top_law_factories": top_law_factories,
        },
    )


@app.get("/intelligence/accuracy")
async def intelligence_accuracy(request: Request):
    """Tab: accuracy history / feedback loop."""
    ml = state.ml
    if not ml or not ml.available:
        return templates.TemplateResponse(
            "_intelligence_accuracy.html",
            {"request": request, "history": [], "quality": {}, "ml": None},
        )

    return templates.TemplateResponse(
        "_intelligence_accuracy.html",
        {
            "request": request,
            "history": ml.accuracy_history,
            "quality": ml.quality,
            "ml": ml,
        },
    )


# Canonical labels for witness-slip org names (avoids duplicate rows for Self/self/NA/None etc.)
_CANONICAL_NO_ORG = "No organization"
_CANONICAL_INDIVIDUAL = "Individual"
_ORG_NORMALIZE_MAP = None


def _get_org_normalize_map() -> dict[str, str]:
    """Lazy-build map from normalized raw org string -> canonical display name."""
    global _ORG_NORMALIZE_MAP
    if _ORG_NORMALIZE_MAP is not None:
        return _ORG_NORMALIZE_MAP
    # No-organization variants (case-insensitive match keys)
    no_org = (
        "na",
        "n/a",
        "none",
        "not applicable",
        "not specified",
        "no organization",
        "(no organization)",
        "—",
        "-",
        "",
    )
    # Individual/self variants
    individual = (
        "self",
        "myself",
        "on behalf of self",
        "individual",
        "citizen",
        "family",
        "personal",
        "retired",
        "private citizen",
        "self-employed",
        "me",
    )
    m = {}
    for v in no_org:
        m[v.strip().lower()] = _CANONICAL_NO_ORG
    for v in individual:
        m[v.strip().lower()] = _CANONICAL_INDIVIDUAL
    _ORG_NORMALIZE_MAP = m
    return _ORG_NORMALIZE_MAP


def _canonical_organization_name(raw: str) -> str:
    """Map raw witness-slip organization string to a canonical name for grouping."""
    s = (raw or "").strip()
    if not s:
        return _CANONICAL_NO_ORG
    key = s.lower()
    canonical = _get_org_normalize_map().get(key)
    if canonical is not None:
        return canonical
    return s  # keep original for real org names


def _bill_description_for_slip_bill_number(bill_number: str) -> str:
    """Resolve bill description for a witness-slip bill number (may lack leading zeros)."""
    import re

    bill = getattr(state, "bill_lookup", {}).get(bill_number)
    if bill:
        return bill.description or ""
    # Normalize and match (e.g. HB100 vs HB0100)
    m = re.match(r"([A-Za-z]+)0*(\d+)", (bill_number or "").strip(), re.IGNORECASE)
    if m:
        norm = f"{m.group(1).upper()}{m.group(2)}"
        for b in getattr(state, "bills", []):
            m2 = re.match(r"([A-Za-z]+)0*(\d+)", (b.bill_number or "").strip(), re.IGNORECASE)
            if m2 and f"{m2.group(1).upper()}{m2.group(2)}" == norm:
                return b.description or ""
    return ""


@app.get("/intelligence/witness-slips")
async def intelligence_witness_slips(request: Request):
    """Tab: witness slips and organization/lobbying influence on bills."""
    lookup = getattr(state, "witness_slips_lookup", {})
    if not lookup:
        return templates.TemplateResponse(
            "_intelligence_witness_slips.html",
            {
                "request": request,
                "bill_slips": [],
                "top_organizations": [],
                "anomaly_by_bill": {},
            },
        )

    # Build bill_number -> anomaly for flagged/suspicious bills
    anomaly_by_bill = {}
    ml = getattr(state, "ml", None)
    if ml and getattr(ml, "anomalies", None):
        for a in ml.anomalies:
            if getattr(a, "is_anomaly", False) and getattr(a, "bill_number", None):
                anomaly_by_bill[a.bill_number] = a
        # Also by normalized bill_id (e.g. leg_id) for cross-reference
        for a in ml.anomalies:
            if getattr(a, "is_anomaly", False) and getattr(a, "bill_id", None):
                # bill_id may be HB0100-style; match to slip keys
                bn = getattr(a, "bill_number", None) or a.bill_id
                if bn and bn not in anomaly_by_bill:
                    anomaly_by_bill[bn] = a

    # Per-bill: total, pro/opp/no_pos, controversy, top orgs
    bill_slips = []
    for bill_number, slips in lookup.items():
        pro = sum(1 for s in slips if s.position == "Proponent")
        opp = sum(1 for s in slips if s.position == "Opponent")
        no_pos = sum(1 for s in slips if s.position and "no position" in s.position.lower())
        total = len(slips)
        controversy = (opp / (pro + opp)) if (pro + opp) > 0 else 0.0
        desc = _bill_description_for_slip_bill_number(bill_number)
        # Top organizations: (name, count, pro, opp) — use canonical names to merge Self/NA/etc.
        org_counts = {}
        for s in slips:
            org = _canonical_organization_name(s.organization or "")
            if org not in org_counts:
                org_counts[org] = {"total": 0, "pro": 0, "opp": 0}
            org_counts[org]["total"] += 1
            if s.position == "Proponent":
                org_counts[org]["pro"] += 1
            elif s.position == "Opponent":
                org_counts[org]["opp"] += 1
        top_orgs = sorted(
            [
                {"name": org, "total": d["total"], "pro": d["pro"], "opp": d["opp"]}
                for org, d in org_counts.items()
            ],
            key=lambda x: -x["total"],
        )[:10]
        anomaly = anomaly_by_bill.get(bill_number)
        bill_slips.append(
            {
                "bill_number": bill_number,
                "description": desc,
                "total_count": total,
                "proponent_count": pro,
                "opponent_count": opp,
                "no_position_count": no_pos,
                "controversy": controversy,
                "top_organizations": top_orgs,
                "is_flagged": bool(anomaly),
                "anomaly_reason": getattr(anomaly, "anomaly_reason", "") if anomaly else "",
            }
        )
    bill_slips.sort(key=lambda x: -x["total_count"])

    # Global top organizations (across all bills) — canonical names to merge duplicates
    org_global = {}
    for slips in lookup.values():
        for s in slips:
            org = _canonical_organization_name(s.organization or "")
            org_global[org] = org_global.get(org, 0) + 1
    top_organizations = sorted(org_global.items(), key=lambda x: -x[1])[:50]

    return templates.TemplateResponse(
        "_intelligence_witness_slips.html",
        {
            "request": request,
            "bill_slips": bill_slips,
            "top_organizations": top_organizations,
            "anomaly_by_bill": anomaly_by_bill,
        },
    )


# ── Intelligence deep-dive routes ─────────────────────────────────────────


@app.get("/intelligence/member/{member_id}")
async def intelligence_member_detail(request: Request, member_id: str):
    """Deep-dive on a single legislator's influence profile."""
    member = state.member_lookup_by_id.get(member_id)
    if not member:
        # Fallback: try name-keyed lookup (member_id could be a name)
        member = state.member_lookup.get(member_id)
    if not member:
        return templates.TemplateResponse(
            "intelligence_member.html",
            {
                "request": request,
                "member": None,
                "influence": None,
                "moneyball": None,
                "narrative": None,
                "top_bills": [],
                "coalition": None,
            },
        )

    # ── Influence profile ──
    ip = state.influence.get(member_id)
    influence_dict = None
    if ip:
        piv = state.pivotality.get(member.name)
        sp = state.sponsor_pull.get(member_id)
        influence_dict = {
            "score": ip.influence_score,
            "label": ip.influence_label,
            "rank_overall": ip.rank_overall,
            "rank_chamber": ip.rank_chamber,
            "moneyball_pct": round(ip.moneyball_normalized * 100, 1),
            "betweenness_pct": round(ip.betweenness_normalized * 100, 1),
            "pivotality_pct": round(ip.pivotality_normalized * 100, 1),
            "pull_pct": round(ip.pull_normalized * 100, 1),
            "signals": ip.influence_signals,
            "pivotal_winning": piv.pivotal_winning if piv else 0,
            "swing_votes": piv.swing_votes if piv else 0,
            "close_votes_total": piv.close_votes_total if piv else 0,
            "sponsor_lift": sp.sponsor_lift if sp else 0,
            "cosponsor_lift": sp.cosponsor_lift if sp else 0,
        }

    # ── Moneyball profile ──
    mb = state.moneyball.profiles.get(member_id) if state.moneyball else None
    moneyball_dict = None
    if mb:
        moneyball_dict = {
            "laws_passed": mb.laws_passed,
            "effectiveness_rate": mb.effectiveness_rate,
            "magnet_score": mb.magnet_score,
            "bridge_score": mb.bridge_score,
            "unique_collaborators": mb.unique_collaborators,
            "moneyball_score": mb.moneyball_score,
            "badges": mb.badges,
        }

    # ── Build narrative ──
    narrative_parts = []
    if ip:
        narrative_parts.append(
            f"{member.name} ranks #{ip.rank_overall} overall in the Illinois General Assembly"
        )
        if ip.influence_label == "High":
            narrative_parts.append("with high legislative influence")
        elif ip.influence_label == "Moderate":
            narrative_parts.append("with moderate influence")

    if mb:
        if mb.laws_passed > 0:
            narrative_parts.append(
                f"They have passed {mb.laws_passed} law{'s' if mb.laws_passed != 1 else ''} "
                f"with a {mb.effectiveness_rate:.0%} effectiveness rate"
            )
        if mb.unique_collaborators > 20:
            narrative_parts.append(
                f"and collaborate with {mb.unique_collaborators} different legislators"
            )
        if mb.bridge_score > 0.3:
            narrative_parts.append(
                f"({mb.bridge_score:.0%} of their laws have cross-party co-sponsors)"
            )

    if ip and ip.influence_signals:
        narrative_parts.append(". " + ip.influence_signals[0])

    narrative = ", ".join(narrative_parts) + "." if narrative_parts else None

    # ── Top bills ──
    top_bills = []
    ml = state.ml
    if ml and ml.available:
        member_bills = [
            s for s in ml.bill_scores if s.sponsor and member.name and member.name in s.sponsor
        ]
        member_bills.sort(key=lambda s: -s.prob_advance)
        for s in member_bills[:10]:
            top_bills.append(
                {
                    "bill_id": s.bill_id,
                    "bill_number": s.bill_number,
                    "description": s.description,
                    "prob_advance": s.prob_advance,
                    "prob_law": getattr(s, "prob_law", 0.0),
                    "predicted_destination": getattr(s, "predicted_destination", "Stuck"),
                    "stage_label": s.stage_label,
                    "lifecycle_status": s.lifecycle_status,
                    "forecast_score": getattr(s, "forecast_score", 0.0),
                    "forecast_confidence": getattr(s, "forecast_confidence", ""),
                }
            )

    # ── Coalition membership ──
    coalition = None
    if ml and ml.coalitions:
        for cm in ml.coalitions:
            if cm.member_id == member_id:
                prof_map = {p.coalition_id: p for p in ml.coalition_profiles}
                prof = prof_map.get(cm.coalition_id)
                coalition_members = [m for m in ml.coalitions if m.coalition_id == cm.coalition_id]
                coalition = {
                    "name": prof.name if prof else f"Coalition {cm.coalition_id + 1}",
                    "size": len(coalition_members),
                    "cohesion": prof.cohesion if prof else 0,
                    "focus_areas": prof.focus_areas if prof else [],
                }
                break

    return templates.TemplateResponse(
        "intelligence_member.html",
        {
            "request": request,
            "member": member,
            "influence": influence_dict,
            "moneyball": moneyball_dict,
            "narrative": narrative,
            "top_bills": top_bills,
            "coalition": coalition,
        },
    )


@app.get("/intelligence/bill/{bill_id}")
async def intelligence_bill_detail(request: Request, bill_id: str):
    """Deep-dive on a single bill's prediction and context."""
    ml = state.ml
    bill = None
    if ml and ml.available:
        for s in ml.bill_scores:
            if s.bill_id == bill_id:
                bill = s
                break

    if not bill:
        return templates.TemplateResponse(
            "intelligence_bill.html",
            {"request": request, "bill": None, "sponsor_influence": None, "anomaly": None},
        )

    # ── Sponsor influence ──
    sponsor_influence = None
    # Find sponsor member_id
    sponsor_member = None
    for m in state.members:
        if m.name and bill.sponsor and m.name in bill.sponsor:
            sponsor_member = m
            break

    if sponsor_member:
        ip = state.influence.get(sponsor_member.id)
        sp = state.sponsor_pull.get(sponsor_member.id)
        if ip:
            sponsor_influence = {
                "member_id": sponsor_member.id,
                "label": ip.influence_label,
                "rank": ip.rank_overall,
                "signals": ip.influence_signals,
                "sponsor_lift": sp.sponsor_lift if sp else 0,
            }
        # Add sponsor_id to bill for linking
        bill_dict_extra = {"sponsor_id": sponsor_member.id}
    else:
        bill_dict_extra = {"sponsor_id": None}

    # ── Anomaly data ──
    anomaly = None
    if ml and ml.anomalies:
        for a in ml.anomalies:
            if a.bill_id == bill_id:
                anomaly = {
                    "total_slips": a.total_slips,
                    "n_proponent": a.n_proponent,
                    "n_opponent": a.n_opponent,
                    "unique_orgs": a.unique_orgs,
                    "anomaly_score": a.anomaly_score,
                    "is_anomaly": a.is_anomaly,
                    "anomaly_reason": a.anomaly_reason,
                }
                break

    # Build a dict-like object with all bill fields + extras
    class _BillCtx:
        """Template-friendly bill context."""

        def __init__(self, score, extras):
            self.bill_id = score.bill_id
            self.bill_number = score.bill_number
            self.description = score.description
            self.sponsor = score.sponsor
            self.prob_advance = score.prob_advance
            self.prob_law = getattr(score, "prob_law", 0.0)
            self.predicted_outcome = score.predicted_outcome
            self.predicted_destination = getattr(score, "predicted_destination", "Stuck")
            self.confidence = score.confidence
            self.label_reliable = score.label_reliable
            self.chamber_origin = score.chamber_origin
            self.introduction_date = score.introduction_date
            self.current_stage = score.current_stage
            self.stage_progress = score.stage_progress
            self.stage_label = score.stage_label
            self.days_since_action = score.days_since_action
            self.last_action_text = score.last_action_text
            self.last_action_date = score.last_action_date
            self.stuck_status = score.stuck_status
            self.stuck_reason = score.stuck_reason
            self.lifecycle_status = score.lifecycle_status
            self.rule_context = getattr(score, "rule_context", "")
            self.forecast_score = getattr(score, "forecast_score", 0.0)
            self.forecast_confidence = getattr(score, "forecast_confidence", "")
            self.sponsor_id = extras.get("sponsor_id")

    bill_ctx = _BillCtx(bill, bill_dict_extra)

    # ── Classified action history ──
    action_history = []
    # Find the actual bill object from state.bills to get action_history
    bill_obj = state.bills_lookup.get(bill_id) if hasattr(state, "bills_lookup") else None
    if bill_obj and bill_obj.action_history:
        for ae in bill_obj.action_history:
            action_history.append(
                {
                    "date": ae.date,
                    "action": ae.action,
                    "chamber": ae.chamber,
                    "action_category": ae.action_category or "other",
                    "action_category_label": ae.action_category_label or "Other",
                    "outcome_signal": ae.outcome_signal or "neutral",
                    "meaning": ae.meaning or "",
                    "rule_reference": getattr(ae, "rule_reference", "") or "",
                }
            )

    return templates.TemplateResponse(
        "intelligence_bill.html",
        {
            "request": request,
            "bill": bill_ctx,
            "sponsor_influence": sponsor_influence,
            "anomaly": anomaly,
            "action_history": action_history,
        },
    )


# ── Legislative Power Map routes ──────────────────────────────────────────


@app.get("/explore")
async def explore_page(request: Request):
    """Render the interactive Legislative Power Map."""
    return templates.TemplateResponse(
        "explore.html",
        {
            "request": request,
            "title": "Legislative Power Map",
            "categories": CATEGORY_CHOICES,
        },
    )


@app.get("/api/graph")
async def graph_data(
    topic: str = "",
    zip: str = "",
    focus: str = "relevant",
):
    """Return graph data (nodes + edges) for the Legislative Power Map.

    Query params:
    - topic: policy category name (e.g. "Transportation") — highlights
      members on relevant committees.
    - zip: Illinois ZIP code — identifies the user's senator and
      representative.
    - focus: "relevant" (default) — only top influencers + topic + your
      legislators; "all" — all 180 members.

    Returns JSON with nodes, edges, your_legislators, topic_committees, meta.
    """
    # ── Resolve topic to committee codes and member IDs ──
    topic = topic.strip()
    committee_codes = _CATEGORY_COMMITTEES.get(topic, [])
    topic_member_ids: set[str] = set()
    topic_committees_list: list[dict] = []
    if committee_codes:
        for code in committee_codes:
            cmembers: list[str] = []
            for role in state.committee_rosters.get(code, []):
                if role.member_id:
                    topic_member_ids.add(role.member_id)
                    cmembers.append(role.member_id)
            cmt = state.committee_lookup.get(code)
            topic_committees_list.append(
                {
                    "code": code,
                    "name": cmt.name if cmt else code,
                    "member_ids": cmembers,
                }
            )

    # ── Resolve ZIP to user's legislators ──
    zip_code = zip.strip()
    your_senator_id: str | None = None
    your_rep_id: str | None = None
    if zip_code:
        district_info = state.zip_to_district.get(zip_code)
        if district_info:
            if district_info.il_senate:
                sen = _find_member_by_district("senate", district_info.il_senate)
                if sen:
                    your_senator_id = sen.id
            if district_info.il_house:
                rep = _find_member_by_district("house", district_info.il_house)
                if rep:
                    your_rep_id = rep.id

    your_legislator_ids = set()
    if your_senator_id:
        your_legislator_ids.add(your_senator_id)
    if your_rep_id:
        your_legislator_ids.add(your_rep_id)

    # ── Build nodes ──
    nodes: list[dict] = []
    for member in state.members:
        mb = state.moneyball.profiles.get(member.id) if state.moneyball else None
        ip = state.influence.get(member.id)

        influence_score = ip.influence_score if ip else (mb.moneyball_score if mb else 0.0)
        influence_label = ip.influence_label if ip else ""

        # Committee roles for this member
        member_committees: list[dict] = []
        for cr in state.member_committee_roles.get(member.id, []):
            member_committees.append(
                {
                    "code": cr.get("code", ""),
                    "name": cr.get("name", ""),
                    "role": cr.get("role", ""),
                    "is_leadership": cr.get("is_leadership", False),
                }
            )

        # Party abbreviation
        party_lower = (member.party or "").lower()
        if "republican" in party_lower:
            party_abbr = "R"
        elif "democrat" in party_lower:
            party_abbr = "D"
        else:
            party_abbr = member.party[:1] if member.party else ""

        is_topic_relevant = member.id in topic_member_ids if topic_member_ids else False
        is_your_legislator = member.id in your_legislator_ids

        nodes.append(
            {
                "id": member.id,
                "name": member.name,
                "party": party_abbr,
                "chamber": member.chamber,
                "district": member.district,
                "influence_score": round(influence_score, 2),
                "influence_label": influence_label,
                "moneyball_score": round(mb.moneyball_score, 2) if mb else 0.0,
                "moneyball_rank": mb.rank_chamber if mb else 0,
                "is_leadership": mb.is_leadership if mb else False,
                "role": member.role or "",
                "committees": member_committees,
                "laws_passed": mb.laws_passed if mb else 0,
                "laws_filed": mb.laws_filed if mb else 0,
                "bridge_score": round(mb.bridge_score, 4) if mb else 0.0,
                "effectiveness_rate": round(mb.effectiveness_rate, 4) if mb else 0.0,
                "is_topic_relevant": is_topic_relevant,
                "is_your_legislator": is_your_legislator,
                "influence_signals": ip.influence_signals if ip else [],
            }
        )

    # ── Optional: restrict to relevant members only ──
    RELEVANT_TOP_N = 50
    if focus.strip().lower() == "relevant":
        if topic_member_ids:
            # Topic selected: only members on that topic's committees + your legislators
            relevant_ids = topic_member_ids | your_legislator_ids
        else:
            # No topic: top influencers + your legislators
            by_influence = sorted(nodes, key=lambda n: n["influence_score"], reverse=True)
            relevant_ids = your_legislator_ids | {n["id"] for n in by_influence[:RELEVANT_TOP_N]}
        nodes = [n for n in nodes if n["id"] in relevant_ids]

    # ── Build edges (pruned for performance) ──
    # Full adjacency can have 15k+ edges which slows SVG rendering.
    # Strategy: keep edges where at least one endpoint is "important"
    # (high influence, topic-relevant, or user's legislator).
    # For the remaining, cap at top N connections per member.
    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    adjacency = state.cosponsor_adjacency

    # Build node influence lookup for edge prioritization
    node_influence: dict[str, float] = {}
    for n in nodes:
        node_influence[n["id"]] = n["influence_score"]

    # Important member IDs: always keep all their edges
    important_ids = topic_member_ids | your_legislator_ids
    # Also include top 20 by influence
    top_by_influence = sorted(nodes, key=lambda n: n["influence_score"], reverse=True)[:20]
    important_ids |= {n["id"] for n in top_by_influence}

    MAX_EDGES_PER_NODE = 8  # for non-important members

    for member_id, peers in adjacency.items():
        is_important = member_id in important_ids

        if is_important:
            # Keep all edges for important nodes
            target_peers = peers
        else:
            # For regular members, keep top N by peer influence
            sorted_peers = sorted(
                peers,
                key=lambda pid: node_influence.get(pid, 0),
                reverse=True,
            )
            target_peers = sorted_peers[:MAX_EDGES_PER_NODE]

        for peer_id in target_peers:
            edge_key = tuple(sorted((member_id, peer_id)))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "source": member_id,
                        "target": peer_id,
                    }
                )

    # When focus=relevant, drop edges whose endpoints are not both in nodes
    node_ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

    return {
        "nodes": nodes,
        "edges": edges,
        "your_legislators": {
            "senator": your_senator_id,
            "representative": your_rep_id,
        },
        "topic_committees": topic_committees_list,
        "meta": {
            "total_members": len(nodes),
            "total_edges": len(edges),
            "topic": topic,
            "zip": zip_code,
            "focus": focus.strip().lower() or "all",
        },
    }


app.include_router(graphql_app, prefix="/graphql")
