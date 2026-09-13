"""Helpers for intelligence routes: witness-slip org normalization, bill lookup, money views."""

from __future__ import annotations

import re
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

# Sitting ILGA is 177 members. Below these floors the loaded index is
# fixture/sample-scale, not a statewide disclosure graph.
_SAMPLE_SCALE_MEMBER_MAX = 99
_SAMPLE_SCALE_RECEIPT_MAX = 4999

_HONORIFICS = frozenset({"jr", "sr", "ii", "iii", "iv"})
_ACTION_BY_SEN = re.compile(r"\bby(?=Sen\.?\b)")
_ACTION_BY_REP = re.compile(r"\bby(?=Rep\.?\b)")
_ACTION_TO_CAPS = re.compile(r"\bto(?=[A-Z])")
_ACTION_DO_PASS = re.compile(r"Do Pass(?=[A-Z])")


def is_sample_scale_finance(members_matched: int, receipts_indexed: int) -> bool:
    """True when the loaded index is fixture/sample-scale, not statewide."""
    return (
        members_matched <= _SAMPLE_SCALE_MEMBER_MAX
        or receipts_indexed <= _SAMPLE_SCALE_RECEIPT_MAX
    )


def format_ilga_action_text(text: str) -> str:
    """Insert missing spaces in ILGA actions jammed by ``get_text(strip=True)``."""
    if not text:
        return ""
    spaced = _ACTION_BY_SEN.sub("by ", text)
    spaced = _ACTION_BY_REP.sub("by ", spaced)
    spaced = _ACTION_TO_CAPS.sub("to ", spaced)
    spaced = _ACTION_DO_PASS.sub("Do Pass ", spaced)
    return re.sub(r" {2,}", " ", spaced).strip()


def _person_name_tokens(name: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return [token for token in cleaned.split() if token and token not in _HONORIFICS]


def contributor_matches_member(contributor_name: str, member_name: str) -> bool:
    """Display-only check: contributor looks like the same person as the member."""
    contrib = _person_name_tokens(contributor_name)
    member_tokens = _person_name_tokens(member_name)
    if len(contrib) < 2 or len(member_tokens) < 2:
        return False
    last = member_tokens[-1]
    first = member_tokens[0]
    if last not in contrib:
        return False
    return any(
        token == first or token.startswith(first) or first.startswith(token)
        for token in contrib
        if token != last
    )


def member_glance_narrative(
    name: str,
    *,
    rank_overall: int | None = None,
    influence_label: str = "",
    laws_passed: int = 0,
    effectiveness_rate: float = 0.0,
    unique_collaborators: int = 0,
    bridge_score: float = 0.0,
    influence_signal: str = "",
) -> str | None:
    """Member-page 'At a Glance' copy. Uses the member's name — never 'they'."""
    sentences: list[str] = []
    if rank_overall is not None:
        rank_bit = f"{name} ranks #{rank_overall} overall in the Illinois General Assembly"
        label = (influence_label or "").strip().lower()
        if label == "high":
            rank_bit += " with high legislative influence"
        elif label == "moderate":
            rank_bit += " with moderate legislative influence"
        elif label == "low":
            rank_bit += " with lower legislative influence"
        sentences.append(rank_bit + ".")
    if laws_passed > 0:
        noun = "law" if laws_passed == 1 else "laws"
        sentences.append(
            f"{name} has passed {laws_passed} {noun} "
            f"with a {effectiveness_rate:.0%} effectiveness rate."
        )
    if unique_collaborators > 20:
        collab = f"{name} collaborates with {unique_collaborators} different legislators"
        if bridge_score > 0.3:
            collab += f" ({bridge_score:.0%} of those laws have cross-party co-sponsors)"
        sentences.append(collab + ".")
    elif laws_passed > 0 and bridge_score > 0.3:
        sentences.append(f"{bridge_score:.0%} of {name}'s laws have cross-party co-sponsors.")
    signal = (influence_signal or "").strip()
    if signal:
        if not signal.endswith("."):
            signal += "."
        sentences.append(signal)
    return " ".join(sentences) if sentences else None


def _is_self_contributor(
    *,
    contributor_name: str,
    contributor_member_id: str | None,
    member_id: str,
    member_name: str,
) -> bool:
    if contributor_member_id and contributor_member_id == member_id:
        return True
    return contributor_matches_member(contributor_name, member_name)


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
        "sample_scale": is_sample_scale_finance(
            int(stats.get("members_matched") or 0),
            int(stats.get("receipts_indexed") or 0),
        ),
    }


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


def _donor_dict(
    donor: Any,
    *,
    member_id: str = "",
    member_name: str = "",
) -> dict[str, Any]:
    contributor_member_id = donor.contributor_member_id
    return {
        "name": donor.name,
        "total_amount": donor.total_amount,
        "total_amount_display": format_usd(donor.total_amount),
        "receipt_count": donor.receipt_count,
        "occupation": donor.occupation,
        "employer": donor.employer,
        "contributor_member_id": contributor_member_id,
        "is_self": _is_self_contributor(
            contributor_name=donor.name,
            contributor_member_id=contributor_member_id,
            member_id=member_id,
            member_name=member_name,
        ),
    }


def _receipt_dict(
    receipt: Any,
    *,
    member_id: str = "",
    member_name: str = "",
) -> dict[str, Any]:
    location = ", ".join(part for part in (receipt.city, receipt.state) if part)
    contributor_member_id = receipt.contributor_member_id
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
        "contributor_member_id": contributor_member_id,
        "is_self": _is_self_contributor(
            contributor_name=receipt.contributor_name,
            contributor_member_id=contributor_member_id,
            member_id=member_id,
            member_name=member_name,
        ),
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
        "top_donors": [
            _donor_dict(d, member_id=trail.member_id, member_name=trail.member_name)
            for d in trail.top_donors
        ],
        "recent_receipts": [
            _receipt_dict(r, member_id=trail.member_id, member_name=trail.member_name)
            for r in trail.recent_receipts
        ],
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
                "top_donors": [
                    _donor_dict(
                        d,
                        member_id=summary.member_id,
                        member_name=summary.member_name,
                    )
                    for d in summary.top_donors
                ],
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
        "top_donors_across_sponsors": [
            _donor_dict(d) for d in ctx.top_donors_across_sponsors
        ],
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
        if trail.receipt_count <= 0 or trail.total_received <= 0:
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
