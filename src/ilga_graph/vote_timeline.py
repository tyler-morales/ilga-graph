"""Compute a bill's vote timeline — tracks every member's journey across
committee and floor events for a single chamber.

Extracted from ``main.py`` so the logic is independently testable and
the GraphQL resolver stays thin.
"""

from __future__ import annotations

import re

from .models import VoteEvent
from .schema import BillVoteTimelineType, MemberVoteJourneyType

_SUFFIX_RE = re.compile(r"\s+(?:jr\.?|sr\.?|iii|ii|iv)\s*$", re.IGNORECASE)


def _norm(name: str) -> str:
    """Normalize a voter name to bare lower-case last name for matching.

    Handles three formats:

    - ``"Murphy, Laura M"`` → ``"murphy"``  (comma-delimited: text before comma)
    - ``"Murphy"``          → ``"murphy"``  (bare last name)
    - ``"Laura M. Murphy"`` → ``"murphy"``  (canonical full name: last word)
    """
    if not name:
        return ""
    if "," in name:
        # Comma-delimited: last name is everything before the first comma
        last = name.split(",")[0].strip().lower()
    else:
        # No comma: last name is the last whitespace-delimited token
        # (handles both bare "Murphy" and canonical "Laura M. Murphy")
        last = name.split()[-1].strip().lower() if name.strip() else ""
    last = _SUFFIX_RE.sub("", last)
    return last


def _vote_code(norm_name: str, event: VoteEvent) -> str:
    """Return the vote code for *norm_name* in *event*, or ``'--'`` if absent."""
    for n in event.yea_votes:
        if _norm(n) == norm_name:
            return "Y"
    for n in event.nay_votes:
        if _norm(n) == norm_name:
            return "N"
    for n in event.present_votes:
        if _norm(n) == norm_name:
            return "P"
    for n in event.nv_votes:
        if _norm(n) == norm_name:
            return "NV"
    return "--"


def compute_bill_vote_timeline(
    vote_lookup: dict[str, list[VoteEvent]],
    bill_number: str,
    chamber: str,
) -> BillVoteTimelineType | None:
    """Build a :class:`BillVoteTimelineType` for *bill_number* in *chamber*.

    Returns ``None`` when no vote events match.
    """
    events = vote_lookup.get(bill_number, [])
    chamber_events = [e for e in events if e.chamber.lower() == chamber.lower()]
    if not chamber_events:
        return None

    # ── Sort events chronologically ──────────────────────────────────────
    def _event_sort_key(e: VoteEvent) -> tuple[str, int]:
        return (e.date, 0 if e.vote_type == "committee" else 1)

    chamber_events.sort(key=_event_sort_key)

    # ── Build event labels ───────────────────────────────────────────────
    event_labels: list[str] = []
    for e in chamber_events:
        prefix = "Committee" if e.vote_type == "committee" else "Floor"
        event_labels.append(f"{prefix}: {e.description} ({e.date})")

    # ── Canonical display-name registry ──────────────────────────────────
    # Committee PDFs give "Last, First"; floor PDFs give "Last" or "Last, I."
    norm_to_display: dict[str, str] = {}
    for e in chamber_events:
        for n in e.yea_votes + e.nay_votes + e.present_votes + e.nv_votes:
            key = _norm(n)
            existing = norm_to_display.get(key)
            if existing is None or ("," in n and "," not in existing):
                norm_to_display[key] = n

    # ── Committee vs floor event indices ─────────────────────────────────
    committee_event_indices: set[int] = set()
    floor_event_indices: set[int] = set()
    for idx, e in enumerate(chamber_events):
        if e.vote_type == "committee":
            committee_event_indices.add(idx)
        else:
            floor_event_indices.add(idx)

    # ── All unique normalized member names ───────────────────────────────
    all_norm_names: set[str] = set()
    for e in chamber_events:
        for n in e.yea_votes + e.nay_votes + e.present_votes + e.nv_votes:
            all_norm_names.add(_norm(n))

    # ── Build per-member journeys ────────────────────────────────────────
    journeys: list[MemberVoteJourneyType] = []
    for nk in sorted(all_norm_names):
        display = norm_to_display.get(nk, nk)
        votes: list[str] = []
        first_appearance = ""
        last_vote = "--"
        is_committee = False
        changed = False
        prior_yn: str | None = None

        for idx, e in enumerate(chamber_events):
            code = _vote_code(nk, e)
            votes.append(code)
            if code != "--":
                if not first_appearance:
                    first_appearance = event_labels[idx]
                last_vote = code
                if idx in committee_event_indices:
                    is_committee = True
                if code in ("Y", "N"):
                    if prior_yn is not None and prior_yn != code:
                        changed = True
                    prior_yn = code

        journeys.append(
            MemberVoteJourneyType(
                member_name=display,
                chamber=chamber,
                votes=votes,
                first_appearance=first_appearance,
                last_vote=last_vote,
                changed=changed,
                is_committee_member=is_committee,
            )
        )

    # ── Derive analytics ─────────────────────────────────────────────────
    committee_yea_norms: set[str] = set()
    committee_nay_norms: set[str] = set()
    committee_all_norms: set[str] = set()
    for idx in committee_event_indices:
        e = chamber_events[idx]
        for n in e.yea_votes:
            committee_yea_norms.add(_norm(n))
        for n in e.nay_votes:
            committee_nay_norms.add(_norm(n))
        for n in e.yea_votes + e.nay_votes + e.present_votes + e.nv_votes:
            committee_all_norms.add(_norm(n))

    floor_yea_norms: set[str] = set()
    floor_nay_norms: set[str] = set()
    floor_nv_norms: set[str] = set()
    floor_all_norms: set[str] = set()
    for idx in floor_event_indices:
        e = chamber_events[idx]
        for n in e.yea_votes:
            floor_yea_norms.add(_norm(n))
        for n in e.nay_votes:
            floor_nay_norms.add(_norm(n))
        for n in e.nv_votes + e.present_votes:
            floor_nv_norms.add(_norm(n))
        for n in e.yea_votes + e.nay_votes + e.present_votes + e.nv_votes:
            floor_all_norms.add(_norm(n))

    _dn = lambda nk: norm_to_display.get(nk, nk)  # noqa: E731

    committee_to_floor_flips = sorted(
        _dn(nk)
        for nk in committee_all_norms
        if (
            (nk in committee_yea_norms and nk in floor_nay_norms)
            or (nk in committee_nay_norms and nk in floor_yea_norms)
        )
    )

    committee_voted_norms = committee_yea_norms | committee_nay_norms
    committee_to_floor_dropoffs = sorted(
        _dn(nk)
        for nk in committee_voted_norms
        if nk in floor_nv_norms and nk not in floor_yea_norms and nk not in floor_nay_norms
    )

    floor_newcomers = sorted(_dn(nk) for nk in floor_all_norms if nk not in committee_all_norms)

    consistent_yea = sorted(
        _dn(nk)
        for nk in all_norm_names
        if all(
            _vote_code(nk, chamber_events[idx]) in ("Y", "--") for idx in range(len(chamber_events))
        )
        and any(_vote_code(nk, chamber_events[idx]) == "Y" for idx in range(len(chamber_events)))
    )

    consistent_nay = sorted(
        _dn(nk)
        for nk in all_norm_names
        if all(
            _vote_code(nk, chamber_events[idx]) in ("N", "--") for idx in range(len(chamber_events))
        )
        and any(_vote_code(nk, chamber_events[idx]) == "N" for idx in range(len(chamber_events)))
    )

    return BillVoteTimelineType(
        bill_number=bill_number,
        chamber=chamber,
        event_labels=event_labels,
        journeys=journeys,
        committee_to_floor_flips=committee_to_floor_flips,
        committee_to_floor_dropoffs=committee_to_floor_dropoffs,
        floor_newcomers=floor_newcomers,
        consistent_yea=consistent_yea,
        consistent_nay=consistent_nay,
    )
