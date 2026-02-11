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
from .moneyball import MoneyballReport, compute_power_badges
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
        self.member_lookup: dict[str, Member] = {}
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
            cfg.CACHE_DIR, cfg.MOCK_DEV_DIR, SEED_MODE,
        )
        if cached is not None:
            state.scorecards, state.moneyball = cached
        else:
            state.scorecards, state.moneyball = compute_analytics(
                state.members, data.committee_rosters,
            )
            save_analytics_cache(
                state.scorecards, state.moneyball, cfg.CACHE_DIR,
            )
        elapsed_analytics = _time.perf_counter() - t_analytics
    except Exception:
        LOGGER.exception("Analytics computation failed; scorecards will be empty.")

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
            state.committees, state.committee_bills, data.bills_lookup,
        )
        state.member_committee_roles = build_member_committee_roles(
            state.committees, state.committee_rosters, state.committee_stats,
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
            state.vote_events, state.member_lookup, bn_lookup,
        )
        state.category_bill_sets = build_all_category_bill_sets(
            _CATEGORY_COMMITTEES, state.committee_bills,
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

            items.append(SearchResultType(
                entity_type=hit.entity_type.value,
                match_field=hit.match_field,
                match_snippet=hit.match_snippet,
                relevance_score=round(hit.relevance_score, 4),
                member=member_type,
                bill=bill_type,
                committee=committee_type,
            ))

        return SearchConnection(items=items, page_info=page_info)


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
                mb.chamber_median_cosponsor_rate * 100, 1,
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
            len(state.moneyball.rankings_house)
            if member.chamber == "House"
            else len(state.moneyball.rankings_senate)
        ) if state.moneyball else 0
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
    for bid in (member.sponsored_bill_ids or []):
        b = state.bill_lookup.get(bid)
        if b and b.last_action:
            action_lower = b.last_action.lower()
            if not any(kw in action_lower for kw in (
                "public act", "effective date", "vetoed", "tabled",
                "postponed indefinitely", "session sine die",
            )):
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
        parts.append(
            f"{card['bridge_pct']}% of their bills have cross-party co-sponsors"
        )
    if parts:
        return parts[0][0].upper() + parts[0][1:] + (
            " and " + parts[1] if len(parts) > 1 else ""
        ) + "."
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
                        parts = [
                            f"Chair of the {committee_name or code} committee"
                        ]
                        if category_name:
                            parts.append(
                                f"the institutional gatekeeper for {category_name} legislation"
                            )
                        mb = (
                            state.moneyball.profiles.get(cmr.member_id)
                            if state.moneyball
                            else None
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
        why_parts.append(
            f"{senator.seatmate_affinity:.0%} bill overlap with seatmates"
        )
    why = ". ".join(why_parts) + "."
    return best_member, why


@app.get("/advocacy")
async def advocacy_index(request: Request):
    """Render the advocacy search page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Kei Truck Freedom",
        "categories": CATEGORY_CHOICES,
    })


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
        return templates.TemplateResponse(tpl, {
            "request": request,
            "title": "Kei Truck Freedom",
            "categories": CATEGORY_CHOICES,
            "error": error,
        })

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
            senator_card, zip_code, senate_district,
        )
    elif senate_district:
        warnings.append(
            f"Senate District {senate_district} (for ZIP {zip_code}) — "
            "senator not in current data (dev/seed mode has limited members)."
        )

    # ── Find Your Representative ──
    rep_member = (
        _find_member_by_district("house", house_district) if house_district else None
    )
    rep_card = None
    if rep_member:
        rep_card = _member_to_card(
            rep_member,
            why=f"Represents IL House District {house_district}, which contains ZIP {zip_code}.",
        )
        rep_card["script_hint"] = _build_script_hint_rep(
            rep_card, zip_code, house_district,
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
            senator_member, committee_ids=committee_ids, category_name=category_label,
        )
        if senator_member
        else (None, "")
    )

    # ── Merge: if broker and ally are the same person → "Super Ally" ──
    broker_card = None
    ally_card = None
    super_ally_card = None

    if (
        broker_member
        and ally_member
        and broker_member.id == ally_member.id
    ):
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
                broker_card, broker_why,
            )
        if ally_member:
            ally_card = _member_to_card(ally_member, why=ally_why)
            ally_card["script_hint"] = _build_script_hint_ally(ally_card)

    error = "; ".join(warnings) if warnings else None

    member_count = len(state.members)
    zip_count = len(state.zip_to_district)
    tpl = "_results_partial.html" if is_htmx else "results.html"
    return templates.TemplateResponse(tpl, {
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
    })


app.include_router(graphql_app, prefix="/graphql")
