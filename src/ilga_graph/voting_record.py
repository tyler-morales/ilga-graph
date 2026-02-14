"""Voting record analytics for legislators.

Builds a per-member reverse index of vote events, computes party-alignment
statistics, and supports category-based filtering via committee-bill mappings.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from .analytics import BillStatus, classify_bill_status
from .models import Bill, Member, VoteEvent

LOGGER = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class MemberVoteRecord:
    """A single vote cast by a legislator on one bill/event."""

    bill_number: str
    bill_description: str  # from Bill.description
    date: str  # from VoteEvent.date
    vote: str  # "YES" | "NO" | "PRESENT" | "NV"
    bill_status: str  # "PASSED" | "FAILED" | "STUCK" | "IN_PROGRESS"
    vote_type: str  # "floor" | "committee"
    bill_status_url: str = ""  # ILGA BillStatus page URL


@dataclass
class VotingSummary:
    """Aggregated voting statistics for a single legislator."""

    total_votes: int = 0  # all votes (floor + committee)
    total_floor_votes: int = 0
    total_committee_votes: int = 0
    yes_count: int = 0
    no_count: int = 0
    present_count: int = 0
    nv_count: int = 0
    yes_rate_pct: float = 0.0
    party_alignment_pct: float = 0.0  # % of floor votes matching party majority
    party_defection_count: int = 0  # times voted against party majority
    records: list[MemberVoteRecord] = field(default_factory=list)  # sorted date desc


# ── Helpers ──────────────────────────────────────────────────────────────────

_STATUS_DISPLAY = {
    BillStatus.PASSED: "Passed",
    BillStatus.VETOED: "Vetoed",
    BillStatus.STUCK: "Stuck",
    BillStatus.IN_PROGRESS: "In Progress",
}


def _bill_status_label(bill: Bill | None) -> str:
    """Human-readable bill outcome label."""
    if bill is None:
        return "Unknown"
    status = classify_bill_status(bill.last_action)
    return _STATUS_DISPLAY.get(status, "Unknown")


def _parse_vote_date_sort_key(date_str: str) -> tuple[int, int, int]:
    """Parse 'May 31, 2025' into (year, month, day) for sorting.

    Falls back to (0, 0, 0) for unparseable dates so they sort to the end.
    """
    import calendar

    parts = date_str.replace(",", "").split()
    if len(parts) != 3:
        return (0, 0, 0)
    try:
        month_name, day_str, year_str = parts
        month_abbrevs = {m: i for i, m in enumerate(calendar.month_name) if m}
        month_num = month_abbrevs.get(month_name, 0)
        return (int(year_str), month_num, int(day_str))
    except (ValueError, KeyError, IndexError):
        return (0, 0, 0)


# ── Core: build member vote index ────────────────────────────────────────────


def build_member_vote_index(
    vote_events: list[VoteEvent],
    member_lookup: dict[str, Member],
    bills_lookup: dict[str, Bill],
) -> dict[str, VotingSummary]:
    """Build a voting record summary for every legislator.

    Iterates all vote events, finds each member in the yea/nay/present/nv
    lists, and creates ``MemberVoteRecord`` entries. Then computes party
    alignment for floor votes.

    Returns ``{member_name: VotingSummary}``.
    """
    # Step 1: collect raw records per member
    records_by_member: dict[str, list[MemberVoteRecord]] = defaultdict(list)

    for event in vote_events:
        bill = bills_lookup.get(event.bill_number)
        bill_desc = bill.description if bill else event.bill_number
        bill_status = _bill_status_label(bill)
        bill_url = bill.status_url if bill else ""

        for name in event.yea_votes:
            records_by_member[name].append(
                MemberVoteRecord(
                    bill_number=event.bill_number,
                    bill_description=bill_desc,
                    date=event.date,
                    vote="YES",
                    bill_status=bill_status,
                    vote_type=event.vote_type,
                    bill_status_url=bill_url,
                )
            )
        for name in event.nay_votes:
            records_by_member[name].append(
                MemberVoteRecord(
                    bill_number=event.bill_number,
                    bill_description=bill_desc,
                    date=event.date,
                    vote="NO",
                    bill_status=bill_status,
                    vote_type=event.vote_type,
                    bill_status_url=bill_url,
                )
            )
        for name in event.present_votes:
            records_by_member[name].append(
                MemberVoteRecord(
                    bill_number=event.bill_number,
                    bill_description=bill_desc,
                    date=event.date,
                    vote="PRESENT",
                    bill_status=bill_status,
                    vote_type=event.vote_type,
                    bill_status_url=bill_url,
                )
            )
        for name in event.nv_votes:
            records_by_member[name].append(
                MemberVoteRecord(
                    bill_number=event.bill_number,
                    bill_description=bill_desc,
                    date=event.date,
                    vote="NV",
                    bill_status=bill_status,
                    vote_type=event.vote_type,
                    bill_status_url=bill_url,
                )
            )

    # Step 2: compute party alignment per member
    party_stats = _compute_all_party_alignment(vote_events, member_lookup)

    # Step 3: assemble VotingSummary for each member
    result: dict[str, VotingSummary] = {}

    for member_name, records in records_by_member.items():
        # Sort by date descending (most recent first)
        records.sort(key=lambda r: _parse_vote_date_sort_key(r.date), reverse=True)

        # Count ALL votes for the summary stats shown to users
        yes_count = sum(1 for r in records if r.vote == "YES")
        no_count = sum(1 for r in records if r.vote == "NO")
        present_count = sum(1 for r in records if r.vote == "PRESENT")
        nv_count = sum(1 for r in records if r.vote == "NV")
        total = len(records)
        total_floor = sum(1 for r in records if r.vote_type == "floor")
        total_committee = total - total_floor

        yes_rate = round((yes_count / total * 100), 1) if total > 0 else 0.0

        alignment_pct, defection_count = party_stats.get(member_name, (0.0, 0))

        result[member_name] = VotingSummary(
            total_votes=total,
            total_floor_votes=total_floor,
            total_committee_votes=total_committee,
            yes_count=yes_count,
            no_count=no_count,
            present_count=present_count,
            nv_count=nv_count,
            yes_rate_pct=yes_rate,
            party_alignment_pct=alignment_pct,
            party_defection_count=defection_count,
            records=records,
        )

    LOGGER.info(
        "Built voting records for %d members (%d total vote events).",
        len(result),
        len(vote_events),
    )
    return result


# ── Party alignment ──────────────────────────────────────────────────────────


def _compute_all_party_alignment(
    vote_events: list[VoteEvent],
    member_lookup: dict[str, Member],
) -> dict[str, tuple[float, int]]:
    """Compute party-alignment stats for all members in a single pass.

    For each *floor* vote event:
      1. Classify voters by party.
      2. Determine each party's majority direction (YES or NO).
      3. For each voter, check if they voted with their party's majority.

    Returns ``{member_name: (alignment_pct, defection_count)}``.
    Skips votes where the party is split 50/50 (no clear party line).
    """
    # member_name -> [aligned_count, total_applicable, defection_count]
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    # Build a quick name -> party lookup from member_lookup
    name_to_party: dict[str, str] = {}
    for member in member_lookup.values():
        name_to_party[member.name] = member.party

    for event in vote_events:
        if event.vote_type != "floor":
            continue

        # Count yea/nay per party for this event
        party_yea: dict[str, int] = defaultdict(int)
        party_nay: dict[str, int] = defaultdict(int)

        for name in event.yea_votes:
            party = name_to_party.get(name, "")
            if party:
                party_yea[party] += 1
        for name in event.nay_votes:
            party = name_to_party.get(name, "")
            if party:
                party_nay[party] += 1

        # Determine majority direction per party
        party_majority: dict[str, str] = {}  # party -> "YES" or "NO"
        for party in set(list(party_yea.keys()) + list(party_nay.keys())):
            yea = party_yea.get(party, 0)
            nay = party_nay.get(party, 0)
            if yea > nay:
                party_majority[party] = "YES"
            elif nay > yea:
                party_majority[party] = "NO"
            # Tie: skip this party for this vote (no clear party line)

        # Score each voter
        for name in event.yea_votes:
            party = name_to_party.get(name, "")
            if party and party in party_majority:
                stats[name][1] += 1  # total applicable
                if party_majority[party] == "YES":
                    stats[name][0] += 1  # aligned
                else:
                    stats[name][2] += 1  # defection
        for name in event.nay_votes:
            party = name_to_party.get(name, "")
            if party and party in party_majority:
                stats[name][1] += 1
                if party_majority[party] == "NO":
                    stats[name][0] += 1
                else:
                    stats[name][2] += 1
        # PRESENT and NV voters: count as defection if party had a clear majority
        for name in event.present_votes + event.nv_votes:
            party = name_to_party.get(name, "")
            if party and party in party_majority:
                stats[name][1] += 1
                stats[name][2] += 1  # not voting with party = defection

    # Convert to (alignment_pct, defection_count)
    result: dict[str, tuple[float, int]] = {}
    for name, (aligned, total, defections) in stats.items():
        pct = round((aligned / total * 100), 1) if total > 0 else 0.0
        result[name] = (pct, defections)

    return result


# ── Category filtering ───────────────────────────────────────────────────────


def build_category_bill_set(
    category: str,
    category_committees: dict[str, list[str]],
    committee_bills: dict[str, list[str]],
) -> set[str]:
    """Return the set of bill numbers associated with a policy category.

    Unions the bill lists from all committee codes mapped to the given category.
    """
    codes = category_committees.get(category, [])
    bill_set: set[str] = set()
    for code in codes:
        bill_set.update(committee_bills.get(code, []))
    return bill_set


def build_all_category_bill_sets(
    category_committees: dict[str, list[str]],
    committee_bills: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Precompute bill sets for every category in one pass."""
    result: dict[str, set[str]] = {}
    for category in category_committees:
        if category:  # skip the empty "" (All) category
            result[category] = build_category_bill_set(
                category,
                category_committees,
                committee_bills,
            )
    return result


def filter_summary_by_category(
    summary: VotingSummary,
    category_bill_numbers: set[str],
) -> VotingSummary:
    """Return a new VotingSummary filtered to only include category-relevant bills.

    Recalculates all counts and rates from the filtered record set.
    """
    filtered_records = [r for r in summary.records if r.bill_number in category_bill_numbers]

    if not filtered_records:
        return VotingSummary(records=[])

    yes_count = sum(1 for r in filtered_records if r.vote == "YES")
    no_count = sum(1 for r in filtered_records if r.vote == "NO")
    present_count = sum(1 for r in filtered_records if r.vote == "PRESENT")
    nv_count = sum(1 for r in filtered_records if r.vote == "NV")
    total = len(filtered_records)
    total_floor = sum(1 for r in filtered_records if r.vote_type == "floor")

    yes_rate = round((yes_count / total * 100), 1) if total > 0 else 0.0

    # Party alignment is not recalculated for filtered subset — it's a global
    # stat. We preserve the original values from the unfiltered summary.
    return VotingSummary(
        total_votes=total,
        total_floor_votes=total_floor,
        total_committee_votes=total - total_floor,
        yes_count=yes_count,
        no_count=no_count,
        present_count=present_count,
        nv_count=nv_count,
        yes_rate_pct=yes_rate,
        party_alignment_pct=summary.party_alignment_pct,
        party_defection_count=summary.party_defection_count,
        records=filtered_records,
    )
