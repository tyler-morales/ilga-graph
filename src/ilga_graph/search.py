"""Unified in-memory search across Members, Bills, and Committees.

The search engine performs case-insensitive matching with tiered relevance
scoring so that exact ID / name matches surface above substring hits in
descriptions or bios.  Results are returned as ``SearchHit`` dataclasses
sorted by ``relevance_score`` descending.

No external dependencies — the current data volume (~177 members, ~15 000
bills, ~50 committees) is well within linear-scan territory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import Bill, Committee, Member

# ── Public types ──────────────────────────────────────────────────────────────


class EntityType(str, Enum):
    MEMBER = "member"
    BILL = "bill"
    COMMITTEE = "committee"


@dataclass
class SearchHit:
    """One search result with relevance metadata."""

    entity_type: EntityType
    entity_id: str  # member.id, bill.bill_number, committee.code
    match_field: str  # e.g. "name", "description", "synopsis"
    match_snippet: str  # contextual excerpt around the match
    relevance_score: float  # 0.0 – 1.0; higher is better
    # The underlying model object — exactly one is set.
    member: Member | None = None
    bill: Bill | None = None
    committee: Committee | None = None


# ── Relevance tiers ──────────────────────────────────────────────────────────

_SCORE_EXACT_ID = 1.0
_SCORE_EXACT_NAME = 0.95
_SCORE_PREFIX_NAME = 0.80
_SCORE_CONTAINS_NAME = 0.60
_SCORE_CONTAINS_DESC = 0.40
_SCORE_CONTAINS_SECONDARY = 0.20


# ── Snippet helper ────────────────────────────────────────────────────────────


def _snippet(text: str, query_lower: str, context_chars: int = 80) -> str:
    """Extract a short excerpt of *text* centred on the first occurrence of *query_lower*.

    Returns at most ``2 * context_chars`` characters with an ellipsis on each
    side when the text is trimmed.
    """
    if not text:
        return ""
    idx = text.lower().find(query_lower)
    if idx == -1:
        # Shouldn't happen if the caller verified the match, but be safe.
        return text[:context_chars * 2] + ("..." if len(text) > context_chars * 2 else "")

    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(query_lower) + context_chars)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


# ── Per-entity search functions ───────────────────────────────────────────────


def _best_hit(candidates: list[tuple[float, str, str]]) -> tuple[float, str, str] | None:
    """Return the candidate with the highest relevance score, or *None*."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])


def _check_field(
    value: str,
    query_lower: str,
    *,
    exact_score: float,
    prefix_score: float,
    contains_score: float,
    field_name: str,
) -> tuple[float, str, str] | None:
    """Check a single text field against the query.

    Returns (score, field_name, snippet) or None.
    """
    val_lower = value.lower()
    if val_lower == query_lower:
        return (exact_score, field_name, _snippet(value, query_lower))
    if val_lower.startswith(query_lower):
        return (prefix_score, field_name, _snippet(value, query_lower))
    if query_lower in val_lower:
        return (contains_score, field_name, _snippet(value, query_lower))
    return None


def _search_members(
    members: list[Member],
    query_lower: str,
) -> list[SearchHit]:
    """Search members across name, party, district, chamber, role, committees, bio_text."""
    hits: list[SearchHit] = []

    for m in members:
        candidates: list[tuple[float, str, str]] = []

        # Primary: name
        r = _check_field(
            m.name, query_lower,
            exact_score=_SCORE_EXACT_NAME,
            prefix_score=_SCORE_PREFIX_NAME,
            contains_score=_SCORE_CONTAINS_NAME,
            field_name="name",
        )
        if r:
            candidates.append(r)

        # Primary: id
        r = _check_field(
            m.id, query_lower,
            exact_score=_SCORE_EXACT_ID,
            prefix_score=_SCORE_PREFIX_NAME,
            contains_score=_SCORE_CONTAINS_NAME,
            field_name="id",
        )
        if r:
            candidates.append(r)

        # Secondary fields
        for field_name, value in [
            ("role", m.role),
            ("party", m.party),
            ("district", m.district),
            ("chamber", m.chamber),
            ("email", m.email or ""),
        ]:
            r = _check_field(
                value, query_lower,
                exact_score=_SCORE_CONTAINS_SECONDARY + 0.05,
                prefix_score=_SCORE_CONTAINS_SECONDARY + 0.02,
                contains_score=_SCORE_CONTAINS_SECONDARY,
                field_name=field_name,
            )
            if r:
                candidates.append(r)

        # Committees (list of committee code strings)
        for comm in m.committees:
            r = _check_field(
                comm, query_lower,
                exact_score=_SCORE_CONTAINS_DESC,
                prefix_score=_SCORE_CONTAINS_SECONDARY + 0.05,
                contains_score=_SCORE_CONTAINS_SECONDARY,
                field_name="committees",
            )
            if r:
                candidates.append(r)
                break  # one committee match is enough

        # Description-tier: bio_text
        if m.bio_text:
            r = _check_field(
                m.bio_text, query_lower,
                exact_score=_SCORE_CONTAINS_DESC,
                prefix_score=_SCORE_CONTAINS_DESC,
                contains_score=_SCORE_CONTAINS_DESC,
                field_name="bio_text",
            )
            if r:
                candidates.append(r)

        best = _best_hit(candidates)
        if best:
            score, field_name, snippet = best
            hits.append(SearchHit(
                entity_type=EntityType.MEMBER,
                entity_id=m.id,
                match_field=field_name,
                match_snippet=snippet,
                relevance_score=score,
                member=m,
            ))

    return hits


def _search_bills(
    bills: list[Bill],
    query_lower: str,
) -> list[SearchHit]:
    """Search bills across bill_number, description, synopsis, primary_sponsor, last_action."""
    hits: list[SearchHit] = []

    for b in bills:
        candidates: list[tuple[float, str, str]] = []

        # Primary: bill_number (exact or prefix is very high value)
        r = _check_field(
            b.bill_number, query_lower,
            exact_score=_SCORE_EXACT_ID,
            prefix_score=_SCORE_PREFIX_NAME,
            contains_score=_SCORE_CONTAINS_NAME,
            field_name="bill_number",
        )
        if r:
            candidates.append(r)

        # Description / synopsis
        r = _check_field(
            b.description, query_lower,
            exact_score=_SCORE_CONTAINS_DESC + 0.05,
            prefix_score=_SCORE_CONTAINS_DESC + 0.02,
            contains_score=_SCORE_CONTAINS_DESC,
            field_name="description",
        )
        if r:
            candidates.append(r)

        if b.synopsis:
            r = _check_field(
                b.synopsis, query_lower,
                exact_score=_SCORE_CONTAINS_DESC + 0.03,
                prefix_score=_SCORE_CONTAINS_DESC + 0.01,
                contains_score=_SCORE_CONTAINS_DESC,
                field_name="synopsis",
            )
            if r:
                candidates.append(r)

        # Secondary
        for field_name, value in [
            ("primary_sponsor", b.primary_sponsor),
            ("last_action", b.last_action),
            ("chamber", b.chamber),
        ]:
            r = _check_field(
                value, query_lower,
                exact_score=_SCORE_CONTAINS_SECONDARY + 0.05,
                prefix_score=_SCORE_CONTAINS_SECONDARY + 0.02,
                contains_score=_SCORE_CONTAINS_SECONDARY,
                field_name=field_name,
            )
            if r:
                candidates.append(r)

        best = _best_hit(candidates)
        if best:
            score, field_name, snippet = best
            hits.append(SearchHit(
                entity_type=EntityType.BILL,
                entity_id=b.bill_number,
                match_field=field_name,
                match_snippet=snippet,
                relevance_score=score,
                bill=b,
            ))

    return hits


def _search_committees(
    committees: list[Committee],
    query_lower: str,
) -> list[SearchHit]:
    """Search committees across code and name."""
    hits: list[SearchHit] = []

    for c in committees:
        candidates: list[tuple[float, str, str]] = []

        # Code (exact match is very valuable)
        r = _check_field(
            c.code, query_lower,
            exact_score=_SCORE_EXACT_ID,
            prefix_score=_SCORE_PREFIX_NAME,
            contains_score=_SCORE_CONTAINS_NAME,
            field_name="code",
        )
        if r:
            candidates.append(r)

        # Name
        r = _check_field(
            c.name, query_lower,
            exact_score=_SCORE_EXACT_NAME,
            prefix_score=_SCORE_PREFIX_NAME,
            contains_score=_SCORE_CONTAINS_NAME,
            field_name="name",
        )
        if r:
            candidates.append(r)

        best = _best_hit(candidates)
        if best:
            score, field_name, snippet = best
            hits.append(SearchHit(
                entity_type=EntityType.COMMITTEE,
                entity_id=c.code,
                match_field=field_name,
                match_snippet=snippet,
                relevance_score=score,
                committee=c,
            ))

    return hits


# ── Public API ────────────────────────────────────────────────────────────────


def search_all(
    query: str,
    members: list[Member],
    bills: list[Bill],
    committees: list[Committee],
    entity_types: set[EntityType] | None = None,
) -> list[SearchHit]:
    """Run a unified search across all entity types.

    Parameters
    ----------
    query:
        Free-text search string.  Matching is case-insensitive.
    members, bills, committees:
        The full in-memory datasets to search over.
    entity_types:
        Optional filter — when provided, only entities of the given types are
        searched.  ``None`` means search everything.

    Returns
    -------
    list[SearchHit]
        Results sorted by ``relevance_score`` descending.  No pagination is
        applied here — the caller is responsible for offset/limit slicing.
    """
    if not query or not query.strip():
        return []

    query_lower = query.strip().lower()
    hits: list[SearchHit] = []

    if entity_types is None or EntityType.MEMBER in entity_types:
        hits.extend(_search_members(members, query_lower))

    if entity_types is None or EntityType.BILL in entity_types:
        hits.extend(_search_bills(bills, query_lower))

    if entity_types is None or EntityType.COMMITTEE in entity_types:
        hits.extend(_search_committees(committees, query_lower))

    # Sort by relevance descending, then entity_type name for stable ordering
    hits.sort(key=lambda h: (-h.relevance_score, h.entity_type.value, h.entity_id))

    return hits
