"""ETL orchestration: load from cache, scrape, compute analytics, export.

The API can start without running scrapers by using ``load_from_cache()``
when ``ILGA_LOAD_ONLY`` is set. Scraping is done out-of-process via
``scripts/scrape.py`` or a scheduler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import config as cfg
from .analytics import MemberScorecard, compute_all_scorecards
from .exporter import ObsidianExporter
from .models import Bill, Committee, CommitteeMemberRole, Member
from .moneyball import MoneyballReport, compute_moneyball, populate_member_roles
from .scraper import ILGAScraper, load_normalized_cache, save_normalized_cache
from .scrapers.bills import (
    incremental_bill_scrape,
    load_bill_cache,
    scrape_all_bill_indexes,
    scrape_all_bills,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ScrapedData:
    """Container for raw scraped/loaded data before analytics."""

    members: list[Member]
    bills_lookup: dict[str, Bill]  # leg_id -> Bill (from bill scraper)
    committees: list[Committee]
    committee_rosters: dict[str, list[CommitteeMemberRole]]
    committee_bills: dict[str, list[str]]


def _link_members_to_bills(
    members: list[Member],
    bills_lookup: dict[str, Bill],
) -> None:
    """Build member-bill relationships from Bill.sponsor_ids.

    For each bill, its ``sponsor_ids`` and ``house_sponsor_ids`` contain
    member IDs extracted from the BillStatus page.  We build a reverse
    index so each member knows which bills they sponsor / co-sponsor.
    """
    member_by_id: dict[str, Member] = {m.id: m for m in members}

    for m in members:
        m.sponsored_bills = []
        m.co_sponsor_bills = []
        m.sponsored_bill_ids = []
        m.co_sponsor_bill_ids = []

    for bill in bills_lookup.values():
        all_sponsor_ids = bill.sponsor_ids + bill.house_sponsor_ids
        if not all_sponsor_ids:
            continue

        chief_id = all_sponsor_ids[0]
        co_ids = all_sponsor_ids[1:]

        if chief_id in member_by_id:
            m = member_by_id[chief_id]
            m.sponsored_bills.append(bill)
            m.sponsored_bill_ids.append(bill.leg_id)

        for co_id in co_ids:
            if co_id in member_by_id:
                m = member_by_id[co_id]
                m.co_sponsor_bills.append(bill)
                m.co_sponsor_bill_ids.append(bill.leg_id)


def load_from_cache(
    *,
    seed_fallback: bool = False,
) -> ScrapedData | None:
    """Load members, bills, and committees from cache only (no scraping).

    Returns ``ScrapedData`` if normalized cache exists, otherwise ``None``.
    Use this when ``ILGA_LOAD_ONLY`` is set so the API can start without
    running scrapers.
    """
    normalized = load_normalized_cache(seed_fallback=seed_fallback)
    if normalized is None:
        return None

    members, bills_lookup = normalized
    if not bills_lookup:
        bills_lookup = load_bill_cache(seed_fallback=seed_fallback) or {}

    scraper = ILGAScraper(request_delay=0, seed_fallback=seed_fallback)
    try:
        committees, committee_rosters, committee_bills = scraper.fetch_all_committees()
    except Exception:
        LOGGER.warning("Committee cache unavailable during load_from_cache.")
        committees, committee_rosters, committee_bills = [], {}, {}

    _link_members_to_bills(members, bills_lookup)
    LOGGER.info(
        "Loaded from cache: %d members, %d bills, %d committees.",
        len(members),
        len(bills_lookup),
        len(committees),
    )
    return ScrapedData(
        members=members,
        bills_lookup=bills_lookup,
        committees=committees,
        committee_rosters=committee_rosters,
        committee_bills=committee_bills,
    )


def load_stale_cache_fallback(
    *,
    seed_fallback: bool = False,
) -> ScrapedData:
    """Best-effort load from whatever JSON caches exist (no scrape).

    Raises if no usable cache is found. Use when primary load/scrape fails
    so the app can serve stale data instead of starting empty.
    """
    normalized = load_normalized_cache(seed_fallback=seed_fallback)
    if normalized is not None:
        members, bills_lookup = normalized
    else:
        members = []
        bills_lookup = {}

    if not bills_lookup:
        bills_lookup = load_bill_cache(seed_fallback=seed_fallback) or {}

    scraper = ILGAScraper(request_delay=0, seed_fallback=seed_fallback)
    try:
        committees, committee_rosters, committee_bills = scraper.fetch_all_committees()
    except Exception:
        LOGGER.warning("Committee cache also unavailable.")
        committees, committee_rosters, committee_bills = [], {}, {}

    if not members and not bills_lookup:
        raise RuntimeError("No usable cache data found for stale-cache fallback.")

    _link_members_to_bills(members, bills_lookup)
    return ScrapedData(
        members=members,
        bills_lookup=bills_lookup,
        committees=committees,
        committee_rosters=committee_rosters,
        committee_bills=committee_bills,
    )


def load_or_scrape_data(
    *,
    limit: int = 0,
    dev_mode: bool = False,
    seed_mode: bool = False,
    incremental: bool = False,
    sb_limit: int = 100,
    hb_limit: int = 100,
    save_cache: bool = True,
) -> ScrapedData:
    """Load data from cache/seed or scrape from ilga.gov.

    Parameters
    ----------
    save_cache:
        If True (default), saves normalized cache (members.json, bills.json)
        at the end.  Set to False when the caller will do additional
        transformations (e.g. merging vote events / witness slips) before
        saving — avoids writing an incomplete intermediate cache.
    """
    request_delay = 0.25 if dev_mode else 0.5

    scraper = ILGAScraper(
        request_delay=request_delay,
        seed_fallback=seed_mode,
    )

    # ── EXTRACT: committees ──────────────────────────────────────────────
    committees, committee_rosters, committee_bills = scraper.fetch_all_committees()

    # ── EXTRACT: members ─────────────────────────────────────────────────
    senate_members = scraper.fetch_members("Senate", limit=limit)
    house_members = scraper.fetch_members("House", limit=limit)
    members = senate_members + house_members

    if cfg.TEST_MEMBER_URL:
        test_member = scraper.fetch_member_by_url(
            cfg.TEST_MEMBER_URL, cfg.TEST_MEMBER_CHAMBER
        )
        if test_member is not None:
            members.append(test_member)

    # ── EXTRACT: bills ───────────────────────────────────────────────────
    if incremental:
        LOGGER.info("Incremental bill scrape (SB limit=%d, HB limit=%d)...", sb_limit, hb_limit)
        bills_lookup = incremental_bill_scrape(
            sb_limit=sb_limit,
            hb_limit=hb_limit,
            request_delay=request_delay,
            rescrape_recent_days=30,
        )
    else:
        bills_lookup = load_bill_cache(seed_fallback=seed_mode)
        if bills_lookup is None:
            LOGGER.info("Full bill scrape (SB limit=%d, HB limit=%d)...", sb_limit, hb_limit)
            index = scrape_all_bill_indexes(
                sb_limit=sb_limit,
                hb_limit=hb_limit,
                request_delay=request_delay,
            )
            bills_lookup = scrape_all_bills(
                index,
                request_delay=request_delay,
                use_cache=False,
                seed_fallback=seed_mode,
            )
        else:
            LOGGER.info("Loaded %d bills from cache.", len(bills_lookup))

    # ── TRANSFORM: link members ↔ bills ──────────────────────────────────
    _link_members_to_bills(members, bills_lookup)
    LOGGER.info(
        "Linked %d members to %d bills via sponsor IDs.",
        len(members),
        len(bills_lookup),
    )

    # ── PERSIST (optional) ───────────────────────────────────────────────
    if save_cache:
        save_normalized_cache(members, bills_lookup)

    return ScrapedData(
        members=members,
        bills_lookup=bills_lookup,
        committees=committees,
        committee_rosters=committee_rosters,
        committee_bills=committee_bills,
    )


def compute_analytics(
    members: list[Member],
    committee_rosters: dict[str, list[CommitteeMemberRole]] | None = None,
) -> tuple[dict[str, MemberScorecard], MoneyballReport]:
    """Compute scorecards and moneyball analytics for all members."""
    if committee_rosters is not None:
        populate_member_roles(members, committee_rosters)
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
    ).export(
        data.members,
        scorecards=scorecards,
        moneyball=moneyball,
        all_bills=data.bills_lookup.values(),
    )


def run_etl(
    limit: int = 0,
    *,
    dev_mode: bool = False,
    seed_mode: bool = False,
    member_export_limit: int | None = None,
    committee_export_limit: int | None = None,
    bill_export_limit: int | None = None,
    incremental: bool = False,
    sb_limit: int = 100,
    hb_limit: int = 100,
) -> list[Member]:
    """Full ETL pipeline: scrape/load -> analytics -> vault export."""
    data = load_or_scrape_data(
        limit=limit,
        dev_mode=dev_mode,
        seed_mode=seed_mode,
        incremental=incremental,
        sb_limit=sb_limit,
        hb_limit=hb_limit,
    )
    scorecards, moneyball = compute_analytics(data.members, data.committee_rosters)
    from .seating import process_seating

    process_seating(data.members, cfg.MOCK_DEV_DIR / "senate_seats.json")
    export_vault(
        data,
        scorecards,
        moneyball,
        member_export_limit=member_export_limit,
        committee_export_limit=committee_export_limit,
        bill_export_limit=bill_export_limit,
    )
    return data.members
