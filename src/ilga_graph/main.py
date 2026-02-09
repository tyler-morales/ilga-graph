from __future__ import annotations

import functools
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

import strawberry
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from strawberry.fastapi import GraphQLRouter

from .analytics import (
    MemberScorecard,
    compute_advancement_analytics,  # Import the new function
    compute_all_scorecards,
    controversial_score,
    lobbyist_alignment,
)
from .exporter import ObsidianExporter
from .models import Bill, Committee, CommitteeMemberRole, Member, VoteEvent, WitnessSlip
from .moneyball import MoneyballReport, compute_moneyball
from .schema import (
    BillAdvancementAnalyticsType,  # Import the new GraphQL type
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
    SortOrder,
    VoteEventConnection,
    VoteEventType,
    WitnessSlipConnection,
    WitnessSlipSummaryConnection,
    WitnessSlipSummaryType,
    WitnessSlipType,
    paginate,
)
from .scraper import ILGAScraper, extract_and_normalize, save_normalized_cache
from .scrapers.votes import scrape_specific_bills
from .scrapers.witness_slips import scrape_all_witness_slips
from .vote_name_normalizer import normalize_vote_events
from .vote_timeline import compute_bill_vote_timeline

# ── Configure logging ────────────────────────────────────────────────────────
# Ensure our application logs show up in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
    stream=sys.stderr,
    force=True,
)
LOGGER = logging.getLogger(__name__)

TEST_MEMBER_URL = os.getenv("ILGA_TEST_MEMBER_URL", "").strip()
TEST_MEMBER_CHAMBER = os.getenv("ILGA_TEST_MEMBER_CHAMBER", "Senate").strip() or "Senate"
MEMBER_LIMIT = int(os.getenv("ILGA_MEMBER_LIMIT", "0"))
CORS_ORIGINS = os.getenv("ILGA_CORS_ORIGINS", "*").strip()
API_KEY = os.getenv("ILGA_API_KEY", "").strip()

# ── Bill status URLs: single source of truth for votes + witness slips ───────

DEFAULT_BILL_STATUS_URLS = [
    # Senate bills (votes + witness slips)
    "https://www.ilga.gov/Legislation/BillStatus?DocNum=852&GAID=18&DocTypeID=SB&LegId=158575&SessionID=114",
    "https://www.ilga.gov/Legislation/BillStatus?DocNum=8&GAID=18&DocTypeID=SB&LegId=157098&SessionID=114",
    "https://www.ilga.gov/Legislation/BillStatus?DocNum=9&GAID=18&DocTypeID=SB&LegId=157099&SessionID=114",
    # House bills (votes + witness slips — HB0034 has high-volume slips)
    "https://www.ilga.gov/Legislation/BillStatus?DocNum=576&GAID=18&DocTypeID=HB&LegId=156254&SessionID=114",
    "https://www.ilga.gov/Legislation/BillStatus?DocNum=34&GAID=18&DocTypeID=HB&LegId=155692&SessionID=114",
]


def get_bill_status_urls() -> list[str]:
    """Return bill status URLs from env or defaults.

    Used by both the FastAPI lifespan and ``scripts/scrape.py`` so the same
    bills are scraped for vote events **and** witness slips.
    """
    custom = os.getenv("ILGA_VOTE_BILL_URLS", "").strip()
    if custom:
        return [u.strip() for u in custom.split(",") if u.strip()]
    return list(DEFAULT_BILL_STATUS_URLS)


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
    elapsed_export: float,
    elapsed_votes: float,
    member_count: int,
    committee_count: int,
    bill_count: int,
    vote_event_count: int,
    dev_mode: bool,
    seed_mode: bool,
) -> str:
    """Format a nice table showing startup breakdown with colors."""
    c = _Colors

    lines = [
        "",
        f"{c.BOLD}{c.CYAN}{'=' * 80}{c.RESET}",
        f"{c.BOLD}{c.BRIGHT_CYAN}🚀 Application Startup Complete{c.RESET}",
        f"{c.BOLD}{c.CYAN}{'=' * 80}{c.RESET}",
        "",
        f"{c.BOLD}{'Step':<20} {'Time':>8}  {'Details':<45}{c.RESET}",
        f"{c.GRAY}{'-' * 80}{c.RESET}",
    ]

    # Load step
    load_icon = "📦"
    load_detail = f"{member_count} members, {committee_count} committees"
    if seed_mode and elapsed_load < 0.5:
        load_detail += f" {c.DIM}(from seed){c.RESET}"
    lines.append(
        f"{c.GREEN}✓{c.RESET} {load_icon} {c.WHITE}Load Data{c.RESET}"
        f"{c.BRIGHT_GREEN}{elapsed_load:>11.2f}s{c.RESET}  "
        f"{c.WHITE}{load_detail}{c.RESET}"
    )

    # Analytics step
    lines.append(
        f"{c.GREEN}✓{c.RESET} 📊 {c.WHITE}Analytics{c.RESET}"
        f"{c.BRIGHT_GREEN}{elapsed_analytics:>14.2f}s{c.RESET}  "
        f"{c.WHITE}{member_count} scorecards, {member_count} profiles{c.RESET}"
    )

    # Export step
    lines.append(
        f"{c.GREEN}✓{c.RESET} 📝 {c.WHITE}Export Vault{c.RESET}"
        f"{c.BRIGHT_GREEN}{elapsed_export:>11.2f}s{c.RESET}  "
        f"{c.WHITE}{bill_count} bills exported{c.RESET}"
    )

    # Votes step
    vote_detail = f"{vote_event_count} events"
    if elapsed_votes < 0.1:
        vote_detail += f" {c.DIM}(cached){c.RESET}"
    lines.append(
        f"{c.GREEN}✓{c.RESET} 🗳️  {c.WHITE}Roll-Call Votes{c.RESET}"
        f"{c.BRIGHT_GREEN}{elapsed_votes:>9.2f}s{c.RESET}  "
        f"{c.WHITE}{vote_detail}{c.RESET}"
    )

    lines.extend(
        [
            f"{c.GRAY}{'-' * 80}{c.RESET}",
            f"{c.BOLD}{'Total':<20} {c.BRIGHT_CYAN}{elapsed_total:>8.2f}s{c.RESET}  "
            f"{c.DIM}Dev: {dev_mode}, Seed: {seed_mode}{c.RESET}",
            f"{c.BOLD}{c.CYAN}{'=' * 80}{c.RESET}",
            "",
        ]
    )

    return "\n".join(lines)


def _log_startup_timing(
    total_s: float,
    load_s: float,
    analytics_s: float,
    export_s: float,
    votes_s: float,
    member_count: int,
    bill_count: int,
    dev_mode: bool,
    seed_mode: bool,
) -> None:
    """Append startup timing to .startup_timings.csv for historical tracking."""
    log_file = Path(".startup_timings.csv")
    is_new = not log_file.exists()

    with open(log_file, "a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "timestamp,total_s,load_s,analytics_s,export_s,votes_s,members,bills,dev_mode,seed_mode\n"
            )
        f.write(
            f"{datetime.now().isoformat()},{total_s:.2f},{load_s:.2f},{analytics_s:.2f},"
            f"{export_s:.2f},{votes_s:.2f},{member_count},{bill_count},{dev_mode},{seed_mode}\n"
        )
    LOGGER.debug("Startup timing logged to %s", log_file)


# ── Mode flags ────────────────────────────────────────────────────────────────
# DEV_MODE: lighter scrape limits, faster request delays.
# SEED_MODE: load from mocks/dev/ when cache/ is missing (instant startup).
DEV_MODE = os.getenv("ILGA_DEV_MODE", "1") == "1"
SEED_MODE = os.getenv("ILGA_SEED_MODE", "1") == "1"

# When DEV_MODE is on, override scrape + export limits:
#   - Scrape 20 members per chamber (40 total)
#   - Export all members, all committees, latest 100 bills
if DEV_MODE:
    _SCRAPE_MEMBER_LIMIT = MEMBER_LIMIT or 20
    _EXPORT_MEMBER_LIMIT: int | None = None  # export all scraped members
    _EXPORT_COMMITTEE_LIMIT: int | None = None  # only ~142, export all
    _EXPORT_BILL_LIMIT: int | None = 100  # latest 100 by most-recent action
else:
    _SCRAPE_MEMBER_LIMIT = MEMBER_LIMIT
    _EXPORT_MEMBER_LIMIT = None
    _EXPORT_COMMITTEE_LIMIT = None
    _EXPORT_BILL_LIMIT = None


# ── Composable ETL steps ─────────────────────────────────────────────────────


@dataclass
class ScrapedData:
    """Container for raw scraped/loaded data before analytics."""

    members: list[Member]
    committees: list[Committee]
    committee_rosters: dict[str, list[CommitteeMemberRole]]
    committee_bills: dict[str, list[str]]


def load_or_scrape_data(
    *,
    limit: int = 0,
    dev_mode: bool = False,
    seed_mode: bool = False,
) -> ScrapedData:
    """Load data from cache/seed or scrape from ilga.gov.

    Parameters
    ----------
    limit:
        Max members to scrape per chamber (0 = all).
    dev_mode:
        Use faster request delays when scraping.
    seed_mode:
        Fall back to ``mocks/dev/`` when ``cache/`` is missing.
    """
    scraper = ILGAScraper(
        request_delay=0.25 if dev_mode else 1.0,
        seed_fallback=seed_mode,
    )

    committees, committee_rosters, committee_bills = scraper.fetch_all_committees()

    senate_members = scraper.fetch_members("Senate", limit=limit)
    house_members = scraper.fetch_members("House", limit=limit)
    members = senate_members + house_members

    if TEST_MEMBER_URL:
        test_member = scraper.fetch_member_by_url(TEST_MEMBER_URL, TEST_MEMBER_CHAMBER)
        if test_member is not None:
            members.append(test_member)

    # ── Save normalized cache if we scraped fresh data ──
    # (If loaded from cache, sponsored/co_sponsor bills are already populated.)
    # Detect fresh scrape: if any member has bills but no sponsored_bill_ids yet.
    needs_normalize = any(m.sponsored_bills and not m.sponsored_bill_ids for m in members)
    if needs_normalize and members:
        all_bills = extract_and_normalize(members)
        save_normalized_cache(members, all_bills)
        LOGGER.info(
            "Normalized %d members → %d unique bills.",
            len(members),
            len(all_bills),
        )

    return ScrapedData(
        members=members,
        committees=committees,
        committee_rosters=committee_rosters,
        committee_bills=committee_bills,
    )


def compute_analytics(
    members: list[Member],
) -> tuple[dict[str, MemberScorecard], MoneyballReport]:
    """Compute scorecards and moneyball analytics for all members."""
    scorecards = compute_all_scorecards(members)
    moneyball = compute_moneyball(members, scorecards=scorecards)
    return scorecards, moneyball


def export_vault(
    data: ScrapedData,
    scorecards: dict[str, MemberScorecard],
    moneyball: MoneyballReport,
    *,
    member_export_limit: int | None = None,
    committee_export_limit: int | None = None,
    bill_export_limit: int | None = None,
) -> None:
    """Export the Obsidian vault from processed data."""
    ObsidianExporter(
        committees=data.committees,
        committee_rosters=data.committee_rosters,
        committee_bills=data.committee_bills,
        member_export_limit=member_export_limit,
        committee_export_limit=committee_export_limit,
        bill_export_limit=bill_export_limit,
    ).export(data.members, scorecards=scorecards, moneyball=moneyball)


def run_etl(
    limit: int = 0,
    *,
    member_export_limit: int | None = None,
    committee_export_limit: int | None = None,
    bill_export_limit: int | None = None,
) -> list[Member]:
    """Full ETL pipeline: scrape/load -> analytics -> vault export."""
    data = load_or_scrape_data(
        limit=limit,
        dev_mode=DEV_MODE,
        seed_mode=SEED_MODE,
    )
    scorecards, moneyball = compute_analytics(data.members)
    export_vault(
        data,
        scorecards,
        moneyball,
        member_export_limit=member_export_limit,
        committee_export_limit=committee_export_limit,
        bill_export_limit=bill_export_limit,
    )
    return data.members


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
        self.scorecards: dict[str, MemberScorecard] = {}
        self.moneyball: MoneyballReport | None = None
        self.vote_events: list[VoteEvent] = []
        self.vote_lookup: dict[str, list[VoteEvent]] = {}  # bill_number -> votes
        self.witness_slips: list[WitnessSlip] = []
        self.witness_slips_lookup: dict[str, list[WitnessSlip]] = {}  # bill_number -> slips


state = AppState()


def _collect_unique_bills(members: list[Member]) -> dict[str, Bill]:
    unique: dict[str, Bill] = {}
    for m in members:
        for b in m.bills:
            if b.bill_number not in unique:
                unique[b.bill_number] = b
    return unique


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    import time as _time

    t_startup_begin = _time.perf_counter()

    if DEV_MODE:
        LOGGER.warning(
            "\u26a0\ufe0f DEV MODE: %d members/chamber, top %s bills%s",
            _SCRAPE_MEMBER_LIMIT,
            _EXPORT_BILL_LIMIT or "all",
            " (seed ON)" if SEED_MODE else "",
        )

    # ── Step 1: Load or scrape data ──
    t_load = _time.perf_counter()
    data = load_or_scrape_data(
        limit=_SCRAPE_MEMBER_LIMIT,
        dev_mode=DEV_MODE,
        seed_mode=SEED_MODE,
    )
    state.members = data.members
    elapsed_load = _time.perf_counter() - t_load

    # ── Step 2: Compute analytics ──
    t_analytics = _time.perf_counter()
    state.scorecards, state.moneyball = compute_analytics(state.members)
    elapsed_analytics = _time.perf_counter() - t_analytics

    # ── Step 3: Export vault ──
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

    state.member_lookup = {m.name: m for m in state.members}
    state.bill_lookup = _collect_unique_bills(state.members)
    state.bills = list(state.bill_lookup.values())
    state.committees = data.committees
    state.committee_lookup = {c.code: c for c in data.committees}
    state.committee_rosters = data.committee_rosters
    state.committee_bills = data.committee_bills

    # ── Level 4: Scrape roll-call votes ──
    # Uses the same bill status URLs as witness slips (single source of truth).
    _vote_bill_urls = get_bill_status_urls()

    t_votes = _time.perf_counter()
    state.vote_events = scrape_specific_bills(
        _vote_bill_urls,
        request_delay=0.3,
        use_cache=True,
        seed_fallback=SEED_MODE,
    )
    for ve in state.vote_events:
        state.vote_lookup.setdefault(ve.bill_number, []).append(ve)

    # ── Normalize vote names to canonical member names ──
    normalize_vote_events(state.vote_events, state.member_lookup)

    elapsed_votes = _time.perf_counter() - t_votes

    # ── Level 5: Scrape / load witness slips (same bills as votes) ──
    state.witness_slips = scrape_all_witness_slips(
        _vote_bill_urls,
        request_delay=0.3,
        use_cache=True,
        seed_fallback=SEED_MODE,
    )
    for ws in state.witness_slips:
        state.witness_slips_lookup.setdefault(ws.bill_number, []).append(ws)

    # ── Print startup summary table ──
    elapsed_total = _time.perf_counter() - t_startup_begin
    summary = _format_startup_table(
        elapsed_total,
        elapsed_load,
        elapsed_analytics,
        elapsed_export,
        elapsed_votes,
        len(state.members),
        len(data.committees),
        len(state.bills),
        len(state.vote_events),
        DEV_MODE,
        SEED_MODE,
    )
    print(summary, flush=True)

    # Show MVP
    if state.moneyball.mvp_house_non_leadership:
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
        elapsed_export,
        elapsed_votes,
        len(state.members),
        len(state.bills),
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


from strawberry.extensions import QueryDepthLimiter  # noqa: E402

schema = strawberry.Schema(
    query=Query,
    extensions=[QueryDepthLimiter(max_depth=10)],
)
graphql_app = GraphQLRouter(schema)

app = FastAPI(title="ILGA Graph", lifespan=lifespan)

# ── CORS middleware ──────────────────────────────────────────────────────────
_cors_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
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
    if API_KEY:
        exempt = {"/health", "/docs", "/openapi.json", "/redoc"}
        if request.url.path not in exempt and request.method != "OPTIONS":
            provided = request.headers.get("X-API-Key", "")
            if provided != API_KEY:
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


app.include_router(graphql_app, prefix="/graphql")
