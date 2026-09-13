"""Helpers for intelligence routes: witness-slip org normalization, bill lookup, money views."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .app_state import state
from .campaign_finance.service import bill_money_context, member_money_trail
from .models import Bill, Member, VoteEvent

_CANONICAL_NO_ORG = "No organization"
_CANONICAL_INDIVIDUAL = "Individual"
_ORG_NORMALIZE_MAP: dict[str, str] | None = None


def get_org_normalize_map() -> dict[str, str]:
    """Lazy-build map from normalized raw org string -> canonical display name."""
    global _ORG_NORMALIZE_MAP
    if _ORG_NORMALIZE_MAP is not None:
        return _ORG_NORMALIZE_MAP
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
    m: dict[str, str] = {}
    for v in no_org:
        m[v.strip().lower()] = _CANONICAL_NO_ORG
    for v in individual:
        m[v.strip().lower()] = _CANONICAL_INDIVIDUAL
    _ORG_NORMALIZE_MAP = m
    return _ORG_NORMALIZE_MAP


def canonical_organization_name(raw: str) -> str:
    """Map raw witness-slip organization string to a canonical name for grouping."""
    s = (raw or "").strip()
    if not s:
        return _CANONICAL_NO_ORG
    key = s.lower()
    canonical = get_org_normalize_map().get(key)
    if canonical is not None:
        return canonical
    return s


def bill_description_for_slip_bill_number(bill_number: str) -> str:
    """Resolve bill description for a witness-slip bill number (may lack leading zeros)."""
    bill = getattr(state, "bill_lookup", {}).get(bill_number)
    if bill:
        return bill.description or ""
    m = re.match(r"([A-Za-z]+)0*(\d+)", (bill_number or "").strip(), re.IGNORECASE)
    if m:
        norm = f"{m.group(1).upper()}{m.group(2)}"
        for b in getattr(state, "bills", []):
            m2 = re.match(r"([A-Za-z]+)0*(\d+)", (b.bill_number or "").strip(), re.IGNORECASE)
            if m2 and f"{m2.group(1).upper()}{m2.group(2)}" == norm:
                return b.description or ""
    return ""


_MATCH_METHOD_LABELS = {
    "gold": "Gold override",
    "office_district_name": "Office + district + name",
    "name_chamber": "Name + chamber",
    "unresolved": "Unresolved",
}

_ROLE_LABELS = {
    "chief_sponsor": "Chief sponsor",
    "sponsor": "Sponsor",
    "voter": "Voter",
}


def format_usd(amount: float) -> str:
    """Format a dollar amount for intelligence money tables."""
    return f"${amount:,.2f}"


def campaign_finance_summary_view(index: Any) -> dict[str, Any] | None:
    """Template dict for ingest window / match-rate KPIs. None when index is missing."""
    if index is None:
        return None
    stats = getattr(index, "match_stats", {}) or {}
    match_rate = float(stats.get("match_rate") or 0.0)
    return {
        "source": getattr(index, "source", ""),
        "window_start": getattr(index, "window_start", ""),
        "window_end": getattr(index, "window_end", ""),
        "generated_at": getattr(index, "generated_at", ""),
        "legislative_committees": int(stats.get("legislative_committees") or 0),
        "accepted": int(stats.get("accepted") or 0),
        "review": int(stats.get("review") or 0),
        "match_rate": match_rate,
        "match_rate_pct": round(match_rate * 100, 1),
        "members_matched": int(stats.get("members_matched") or 0),
        "receipts_indexed": int(stats.get("receipts_indexed") or 0),
    }


# Bundled mocks/dev is 15 members / 140 receipts. Statewide ingest is on the
# order of a full General Assembly and tens of thousands of receipts. Hide
# fixture copy only when both floors clear so a larger sample cannot pass as live.
_FULL_SCALE_MIN_MEMBERS = 50
_FULL_SCALE_MIN_RECEIPTS = 1000


def is_sample_scale_finance(summary: Mapping[str, Any] | None) -> bool:
    """True when the loaded finance index is fixture/dev-scale or missing."""
    if not summary:
        return True
    members = int(summary.get("members_matched") or 0)
    receipts = int(summary.get("receipts_indexed") or 0)
    return members < _FULL_SCALE_MIN_MEMBERS or receipts < _FULL_SCALE_MIN_RECEIPTS


def _committee_dict(committee: Any) -> dict[str, Any]:
    method = committee.match_method or ""
    status = committee.status or ""
    return {
        "sbe_committee_id": committee.sbe_committee_id,
        "name": committee.name,
        "type_of_committee": committee.type_of_committee,
        "status": status,
        "status_label": "Active" if status == "A" else (status or "—"),
        "party": committee.party,
        "match_method": method,
        "match_method_label": _MATCH_METHOD_LABELS.get(method, method or "—"),
        "match_confidence": committee.match_confidence,
        "match_confidence_pct": round(float(committee.match_confidence or 0.0) * 100),
        "member_id": committee.member_id,
    }


def _donor_dict(donor: Any) -> dict[str, Any]:
    return {
        "name": donor.name,
        "total_amount": donor.total_amount,
        "total_amount_display": format_usd(donor.total_amount),
        "receipt_count": donor.receipt_count,
        "occupation": donor.occupation,
        "employer": donor.employer,
        "contributor_member_id": donor.contributor_member_id,
    }


def _receipt_dict(receipt: Any) -> dict[str, Any]:
    location = ", ".join(part for part in (receipt.city, receipt.state) if part)
    return {
        "sbe_receipt_id": receipt.sbe_receipt_id,
        "received_date": receipt.received_date,
        "amount": receipt.amount,
        "amount_display": format_usd(receipt.amount),
        "contributor_name": receipt.contributor_name,
        "occupation": receipt.occupation,
        "employer": receipt.employer,
        "city": receipt.city,
        "state": receipt.state,
        "location": location,
        "committee_id": receipt.committee_id,
        "committee_name": receipt.committee_name,
        "contributor_member_id": receipt.contributor_member_id,
    }


def member_money_trail_view(
    index: Any,
    member: Member,
    *,
    limit: int = 25,
) -> dict[str, Any] | None:
    """Template dict for a member money trail. None when the finance index is missing."""
    trail = member_money_trail(index, member, limit=limit)
    if trail is None:
        return None
    return {
        "member_id": trail.member_id,
        "member_name": trail.member_name,
        "window_start": trail.window_start,
        "window_end": trail.window_end,
        "committee_count": trail.committee_count,
        "receipt_count": trail.receipt_count,
        "total_received": trail.total_received,
        "total_received_display": format_usd(trail.total_received),
        "committees": [_committee_dict(c) for c in trail.committees],
        "top_donors": [_donor_dict(d) for d in trail.top_donors],
        "recent_receipts": [_receipt_dict(r) for r in trail.recent_receipts],
        "match_notes": trail.match_notes,
    }


def bill_money_context_view(
    index: Any,
    bill: Bill,
    members_by_id: dict[str, Member],
    *,
    vote_events: list[VoteEvent] | None = None,
    limit: int = 25,
) -> dict[str, Any] | None:
    """Template dict for bill money context. None when the finance index is missing."""
    ctx = bill_money_context(
        index,
        bill,
        members_by_id,
        vote_events=vote_events,
        limit=limit,
    )
    if ctx is None:
        return None
    sponsor_trails = []
    for summary in ctx.sponsor_trails:
        sponsor_trails.append(
            {
                "member_id": summary.member_id,
                "member_name": summary.member_name,
                "role": summary.role,
                "role_label": _ROLE_LABELS.get(summary.role, summary.role or "—"),
                "total_received": summary.total_received,
                "total_received_display": format_usd(summary.total_received),
                "receipt_count": summary.receipt_count,
                "top_donors": [_donor_dict(d) for d in summary.top_donors],
            }
        )
    overlapping = []
    for donor in ctx.overlapping_donors:
        names = [members_by_id[mid].name for mid in donor.member_ids if mid in members_by_id]
        overlapping.append(
            {
                "name": donor.name,
                "total_amount": donor.total_amount,
                "total_amount_display": format_usd(donor.total_amount),
                "member_ids": donor.member_ids,
                "member_names": names,
            }
        )
    return {
        "bill_number": ctx.bill_number,
        "description": ctx.description,
        "window_start": ctx.window_start,
        "window_end": ctx.window_end,
        "total_received_across_sponsors": ctx.total_received_across_sponsors,
        "total_received_across_sponsors_display": format_usd(ctx.total_received_across_sponsors),
        "sponsor_trails": sponsor_trails,
        "overlapping_donors": overlapping,
        "top_donors_across_sponsors": [_donor_dict(d) for d in ctx.top_donors_across_sponsors],
        "match_notes": ctx.match_notes,
    }


def top_funded_member_rows(
    index: Any,
    members_by_id: dict[str, Member],
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Members with matched SBE committees, ranked by total receipts in the window."""
    if index is None:
        return []
    rows: list[dict[str, Any]] = []
    for member_id in getattr(index, "matches_by_member", {}):
        member = members_by_id.get(member_id)
        if member is None:
            continue
        trail = member_money_trail(index, member, limit=5)
        if trail is None:
            continue
        top_donor = trail.top_donors[0].name if trail.top_donors else ""
        rows.append(
            {
                "member_id": member.id,
                "member_name": member.name,
                "party": member.party,
                "chamber": member.chamber,
                "district": member.district,
                "total_received": trail.total_received,
                "total_received_display": format_usd(trail.total_received),
                "receipt_count": trail.receipt_count,
                "committee_count": trail.committee_count,
                "top_donor": top_donor,
            }
        )
    rows.sort(key=lambda r: r["total_received"], reverse=True)
    return rows[:limit]
