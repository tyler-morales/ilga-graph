"""Query helpers: member money trail and bill money context."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..models import Bill, Member, VoteEvent


@dataclass
class CampaignCommitteeView:
    sbe_committee_id: str
    name: str
    type_of_committee: str
    status: str
    party: str
    match_method: str
    match_confidence: float
    member_id: str | None


@dataclass
class DonorAggregate:
    name: str
    total_amount: float
    receipt_count: int
    occupation: str = ""
    employer: str = ""
    contributor_member_id: str | None = None


@dataclass
class CampaignReceiptView:
    sbe_receipt_id: str
    received_date: str
    amount: float
    contributor_name: str
    occupation: str
    employer: str
    city: str
    state: str
    committee_id: str
    committee_name: str
    contributor_member_id: str | None = None


@dataclass
class MemberMoneyTrail:
    member_id: str
    member_name: str
    window_start: str
    window_end: str
    committee_count: int
    receipt_count: int
    total_received: float
    committees: list[CampaignCommitteeView] = field(default_factory=list)
    top_donors: list[DonorAggregate] = field(default_factory=list)
    recent_receipts: list[CampaignReceiptView] = field(default_factory=list)
    match_notes: str = ""


@dataclass
class MemberMoneyTrailSummary:
    member_id: str
    member_name: str
    role: str
    total_received: float
    receipt_count: int
    top_donors: list[DonorAggregate] = field(default_factory=list)


@dataclass
class SharedDonor:
    name: str
    total_amount: float
    member_ids: list[str]


@dataclass
class BillMoneyContext:
    bill_number: str
    description: str
    window_start: str
    window_end: str
    total_received_across_sponsors: float
    sponsor_trails: list[MemberMoneyTrailSummary] = field(default_factory=list)
    overlapping_donors: list[SharedDonor] = field(default_factory=list)
    top_donors_across_sponsors: list[DonorAggregate] = field(default_factory=list)
    match_notes: str = ""


def _contributor_name(row: dict[str, Any]) -> str:
    last = (row.get("last_only_name") or "").strip()
    first = (row.get("first_name") or "").strip()
    return f"{first} {last}".strip() if first else last


def _committee_view(match: dict[str, Any], committee: dict[str, Any]) -> CampaignCommitteeView:
    return CampaignCommitteeView(
        sbe_committee_id=committee.get("committee_id", ""),
        name=committee.get("name", ""),
        type_of_committee=committee.get("type_of_committee", ""),
        status=committee.get("status", ""),
        party=committee.get("party", ""),
        match_method=match.get("method", ""),
        match_confidence=float(match.get("confidence") or 0.0),
        member_id=match.get("member_id"),
    )


def _receipt_view(row: dict[str, Any], committee_name: str) -> CampaignReceiptView:
    received = row.get("received_date") or ""
    if hasattr(received, "isoformat"):
        received = received.isoformat()
    return CampaignReceiptView(
        sbe_receipt_id=str(row.get("receipt_id") or ""),
        received_date=str(received),
        amount=float(row.get("amount") or 0.0),
        contributor_name=_contributor_name(row),
        occupation=row.get("occupation") or "",
        employer=row.get("employer") or "",
        city=row.get("city") or "",
        state=row.get("state") or "",
        committee_id=row.get("committee_id") or "",
        committee_name=committee_name,
        contributor_member_id=row.get("contributor_member_id"),
    )


def _aggregate_donors(receipts: list[dict[str, Any]], limit: int) -> list[DonorAggregate]:
    buckets: dict[str, DonorAggregate] = {}
    for row in receipts:
        name = _contributor_name(row)
        if not name:
            continue
        bucket = buckets.get(name)
        if bucket is None:
            bucket = DonorAggregate(
                name=name,
                total_amount=0.0,
                receipt_count=0,
                occupation=row.get("occupation") or "",
                employer=row.get("employer") or "",
                contributor_member_id=row.get("contributor_member_id"),
            )
            buckets[name] = bucket
        bucket.total_amount += float(row.get("amount") or 0.0)
        bucket.receipt_count += 1
    ranked = sorted(buckets.values(), key=lambda d: d.total_amount, reverse=True)
    return ranked[:limit]


def member_money_trail(
    index: Any,
    member: Member,
    *,
    limit: int = 25,
) -> MemberMoneyTrail | None:
    if index is None:
        return None
    matches = list(index.matches_by_member.get(member.id, []))
    receipts: list[dict[str, Any]] = []
    committees: list[CampaignCommitteeView] = []
    for match in matches:
        cid = match.get("committee_id")
        committee = index.committees_by_id.get(cid, {})
        committees.append(_committee_view(match, committee))
        receipts.extend(index.receipts_by_committee.get(cid, []))
    receipts.sort(key=lambda r: str(r.get("received_date") or ""), reverse=True)
    total = sum(float(r.get("amount") or 0.0) for r in receipts)
    names = {c.name for c in committees}
    return MemberMoneyTrail(
        member_id=member.id,
        member_name=member.name,
        window_start=index.window_start,
        window_end=index.window_end,
        committee_count=len(committees),
        receipt_count=len(receipts),
        total_received=round(total, 2),
        committees=committees,
        top_donors=_aggregate_donors(receipts, limit),
        recent_receipts=[
            _receipt_view(r, index.committees_by_id.get(r.get("committee_id"), {}).get("name", ""))
            for r in receipts[:limit]
        ],
        match_notes=(
            f"Linked via {', '.join(sorted(names))}" if names else "No SBE committee matched"
        ),
    )


def _bill_member_roles(
    bill: Bill,
    members_by_id: dict[str, Member],
    vote_events: list[VoteEvent],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    sponsor_ids = list(bill.sponsor_ids) + list(bill.house_sponsor_ids)
    if sponsor_ids:
        roles[sponsor_ids[0]] = "chief_sponsor"
        for mid in sponsor_ids[1:]:
            roles.setdefault(mid, "sponsor")
    else:
        for member in members_by_id.values():
            if member.name == bill.primary_sponsor:
                roles[member.id] = "chief_sponsor"
                break
    for event in vote_events:
        for raw in event.yea_votes + event.nay_votes:
            member = members_by_id.get(raw)
            if member:
                roles.setdefault(member.id, "voter")
    return roles


def bill_money_context(
    index: Any,
    bill: Bill,
    members_by_id: dict[str, Member],
    *,
    vote_events: list[VoteEvent] | None = None,
    limit: int = 25,
) -> BillMoneyContext | None:
    if index is None:
        return None
    roles = _bill_member_roles(bill, members_by_id, vote_events or [])
    summaries: list[MemberMoneyTrailSummary] = []
    all_receipts: list[dict[str, Any]] = []
    donors_by_name: dict[str, set[str]] = defaultdict(set)
    donor_totals: dict[str, float] = defaultdict(float)

    for member_id, role in roles.items():
        member = members_by_id.get(member_id)
        if member is None:
            continue
        trail = member_money_trail(index, member, limit=limit)
        if trail is None:
            continue
        summaries.append(
            MemberMoneyTrailSummary(
                member_id=member.id,
                member_name=member.name,
                role=role,
                total_received=trail.total_received,
                receipt_count=trail.receipt_count,
                top_donors=trail.top_donors[: min(5, limit)],
            )
        )
        for match in index.matches_by_member.get(member.id, []):
            for row in index.receipts_by_committee.get(match.get("committee_id"), []):
                all_receipts.append(row)
                name = _contributor_name(row)
                if name:
                    donors_by_name[name].add(member.id)
                    donor_totals[name] += float(row.get("amount") or 0.0)

    overlapping = [
        SharedDonor(name=name, total_amount=round(donor_totals[name], 2), member_ids=sorted(mids))
        for name, mids in donors_by_name.items()
        if len(mids) >= 2
    ]
    overlapping.sort(key=lambda d: d.total_amount, reverse=True)
    total = sum(s.total_received for s in summaries)
    return BillMoneyContext(
        bill_number=bill.bill_number,
        description=bill.description,
        window_start=index.window_start,
        window_end=index.window_end,
        total_received_across_sponsors=round(total, 2),
        sponsor_trails=summaries,
        overlapping_donors=overlapping[:limit],
        top_donors_across_sponsors=_aggregate_donors(all_receipts, limit),
        match_notes=(
            "Money is campaign receipts to matched candidate committees, "
            "not earmarked to this bill."
        ),
    )
