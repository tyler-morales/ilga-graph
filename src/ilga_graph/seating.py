"""Senate seating-chart analytics (Whisper Network).

Loads the hierarchical senate seating JSON, fuzzy-matches seat names to
:class:`Member` objects, identifies physical neighbors (the Aisle Rule),
and computes a co-sponsorship affinity score for each member's seatmates.

Public entry point: :func:`process_seating`.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from .models import Member

LOGGER = logging.getLogger(__name__)

# Sentinels that represent empty/unoccupied seats.
_SKIP_NAMES = frozenset({"vacant", "null", ""})

# Regex to strip suffixes like ", Jr." / ", III" from canonical member names.
_SUFFIX_RE = re.compile(r",?\s+(?:Jr\.?|Sr\.?|III|II|IV)\s*$", re.IGNORECASE)


# ── Name-parsing helpers ─────────────────────────────────────────────────────


def _norm(name: str) -> str:
    """Lowercase, strip periods, collapse whitespace."""
    return re.sub(r"\s+", " ", name.lower().replace(".", "").replace("-", " ")).strip()


def _parse_member_name(full_name: str) -> tuple[str, str, str]:
    """Parse ``Member.name`` into ``(first_middle, last, suffix)``.

    Mirrors the logic in ``vote_name_normalizer._parse_member_name`` so
    we stay consistent across the codebase.

    Examples::

        "Laura M. Murphy"       -> ("Laura M.", "Murphy", "")
        "Napoleon Harris, III"  -> ("Napoleon", "Harris", "III")
        "Neil Anderson"         -> ("Neil", "Anderson", "")
    """
    suffix = ""
    m = _SUFFIX_RE.search(full_name)
    if m:
        suffix = m.group().strip().lstrip(",").strip()
        full_name = full_name[: m.start()].strip()

    parts = full_name.split()
    if len(parts) <= 1:
        return ("", full_name, suffix)

    last = parts[-1]
    first_middle = " ".join(parts[:-1])
    return (first_middle, last, suffix)


def _parse_seat_name(raw: str) -> tuple[str, str]:
    """Parse a seating-chart name into ``(initial_or_empty, last_name_parts)``.

    Returns
    -------
    (initial, last) where *initial* is a single uppercase letter or ``""``,
    and *last* is the normalized remainder.

    Examples::

        "N. HARRIS"        -> ("N", "harris")
        "Glowiak Hilton"   -> ("",  "glowiak hilton")
        "SIMS"             -> ("",  "sims")
        "D. Turner"        -> ("D", "turner")
        "E. Harss"         -> ("E", "harss")
    """
    normed = raw.strip()
    # Detect "X. Rest" pattern (single letter followed by period/space)
    m = re.match(r"^([A-Za-z])\.\s+(.+)$", normed)
    if m:
        return (m.group(1).upper(), _norm(m.group(2)))
    return ("", _norm(normed))


# ── Index builder ─────────────────────────────────────────────────────────────


class _MemberIndex:
    """Efficient lookup structure for fuzzy seat-name matching.

    Builds three maps from Senate members:
    1. ``by_last`` -- normalized last name -> list[Member]
    2. ``by_initial_last`` -- (initial, norm_last) -> Member
    3. ``by_compound`` -- normalized "last1 last2" -> Member (for hyphenated /
       multi-word last names like "Glowiak Hilton")
    """

    def __init__(self, members: list[Member]) -> None:
        self.by_last: dict[str, list[Member]] = {}
        self.by_initial_last: dict[tuple[str, str], Member] = {}
        self.by_compound: dict[str, Member] = {}

        # Count last-name occurrences so we can flag ambiguities.
        last_counts: Counter[str] = Counter()
        parsed: list[tuple[Member, str, str]] = []  # (member, first_middle, last)

        for m in members:
            if m.chamber.lower() != "senate":
                continue
            first_middle, last, _suffix = _parse_member_name(m.name)
            norm_last = _norm(last)
            parsed.append((m, first_middle, last))
            last_counts[norm_last] += 1

        for m, first_middle, last in parsed:
            norm_last = _norm(last)

            # 1. Last-name list (always populated, may be ambiguous)
            self.by_last.setdefault(norm_last, []).append(m)

            # 2. Initial + last (for disambiguation like "D. Turner" / "S. Turner")
            if first_middle:
                initial = first_middle[0].upper()
                key = (initial, norm_last)
                # First writer wins (should be unique for Senate)
                if key not in self.by_initial_last:
                    self.by_initial_last[key] = m

            # 3. Compound / multi-word last names
            #    e.g. "Kimberly A. Glowiak Hilton" -> check if the last N words
            #    form a compound last that appears in the seating chart.
            #    We store "first_middle_parts[-1:] + last" as compound keys.
            name_parts = m.name.split()
            if len(name_parts) >= 3:
                # Try the last 2 words as compound last name
                compound_2 = _norm(" ".join(name_parts[-2:]))
                if compound_2 not in self.by_compound:
                    self.by_compound[compound_2] = m
                # Try the last 3 words for triple-barrel names
                if len(name_parts) >= 4:
                    compound_3 = _norm(" ".join(name_parts[-3:]))
                    if compound_3 not in self.by_compound:
                        self.by_compound[compound_3] = m

    def match(self, seat_name: str) -> Member | None:
        """Resolve a seating-chart name to a :class:`Member`, or ``None``."""
        if seat_name.strip().lower() in _SKIP_NAMES:
            return None

        initial, norm_last = _parse_seat_name(seat_name)

        # Strategy 1: initial + last (most specific, handles "D. Turner")
        if initial:
            member = self.by_initial_last.get((initial, norm_last))
            if member is not None:
                return member

        # Strategy 2: compound last name ("Glowiak Hilton", "Loughran Cappel")
        if " " in norm_last:
            member = self.by_compound.get(norm_last)
            if member is not None:
                return member

        # Strategy 3: unique bare last name
        candidates = self.by_last.get(norm_last, [])
        if len(candidates) == 1:
            return candidates[0]

        # Strategy 4: hyphenated names -- the JSON may use hyphen or space
        #   e.g. "EDLY-ALLEN" -> try "edly allen" in compound map
        if "-" in seat_name:
            dehyphenated = _norm(seat_name.replace("-", " "))
            member = self.by_compound.get(dehyphenated)
            if member is not None:
                return member
            # Also try as bare last name with the hyphen removed
            candidates = self.by_last.get(dehyphenated, [])
            if len(candidates) == 1:
                return candidates[0]

        if candidates:
            LOGGER.warning(
                "Seating: ambiguous seat name %r matches %d members: %s",
                seat_name,
                len(candidates),
                [m.name for m in candidates],
            )
        else:
            LOGGER.debug(
                "Seating: no member match for seat %r (expected in dev/seed mode).",
                seat_name,
            )
        return None


# ── Neighbor discovery (Aisle Rule) ──────────────────────────────────────────


def _neighbors_in_section(seats: list[str], idx: int) -> list[str]:
    """Return the raw seat names at ``idx-1`` and ``idx+1`` within *seats*.

    Skips VACANT / null sentinels.  Does **not** cross section boundaries
    (the Aisle Rule).
    """
    result: list[str] = []
    for offset in (-1, 1):
        ni = idx + offset
        if 0 <= ni < len(seats):
            name = seats[ni].strip()
            if name.lower() not in _SKIP_NAMES:
                result.append(name)
    return result


# ── Affinity calculator ──────────────────────────────────────────────────────


def _compute_seatmate_affinity(member: Member, seatmates: list[Member]) -> float:
    """Fraction of *member*'s bills where at least one seatmate is a co-sponsor.

    Returns 0.0 when the member has no bills or no matched seatmates.
    """
    if not seatmates:
        return 0.0

    all_bills = member.sponsored_bills + member.co_sponsor_bills
    if not all_bills:
        return 0.0

    # Build a set of bill leg_ids that *any* seatmate sponsors or co-sponsors.
    seatmate_bill_ids: set[str] = set()
    for sm in seatmates:
        for b in sm.sponsored_bills:
            seatmate_bill_ids.add(b.leg_id)
        for b in sm.co_sponsor_bills:
            seatmate_bill_ids.add(b.leg_id)

    overlap = sum(1 for b in all_bills if b.leg_id in seatmate_bill_ids)
    return round(overlap / len(all_bills), 4)


# ── Public entry point ───────────────────────────────────────────────────────


def process_seating(
    members: list[Member],
    seating_json_path: Path | str,
) -> None:
    """Load the senate seating chart and populate seating fields on *members*.

    Mutates each matched :class:`Member` in place, setting:
    - ``seat_block_id``  (e.g. ``"ring1-FarLeft"``)
    - ``seat_ring``  (1-4)
    - ``seatmate_names``  (canonical ``Member.name`` of physical neighbors)
    - ``seatmate_affinity``  (co-sponsorship overlap with neighbors)

    Parameters
    ----------
    members:
        Full member list (House members are ignored; Senate members are
        mutated in place).
    seating_json_path:
        Path to the senate seating JSON file.
    """
    path = Path(seating_json_path)
    if not path.exists():
        LOGGER.warning("Seating chart not found at %s; skipping.", path)
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    index = _MemberIndex(members)

    # First pass: assign seat metadata and collect neighbor raw names.
    # seat_key -> (Member, [neighbor_raw_names])
    seat_assignments: dict[str, tuple[Member, list[str]]] = {}

    matched_count = 0
    total_seats = 0

    for ring in data.get("rings", []):
        ring_num = ring.get("ring_number", 0)
        for section in ring.get("sections", []):
            label = section.get("label", "unknown")
            # Normalize label for block_id (strip spaces, camelCase-ish)
            block_id = f"ring{ring_num}-{label.replace(' ', '')}"
            seats = section.get("seats", [])

            for seat_idx, seat_name in enumerate(seats):
                if seat_name.strip().lower() in _SKIP_NAMES:
                    continue
                total_seats += 1

                member = index.match(seat_name)
                if member is None:
                    continue

                matched_count += 1
                member.seat_block_id = block_id
                member.seat_ring = ring_num

                neighbor_names = _neighbors_in_section(seats, seat_idx)
                seat_assignments[member.id] = (member, neighbor_names)

    LOGGER.info(
        "Seating chart: matched %d/%d occupied seats to members.",
        matched_count,
        total_seats,
    )

    # Second pass: resolve neighbor names to Members and compute affinity.
    for member, neighbor_raw_names in seat_assignments.values():
        seatmate_members: list[Member] = []
        seatmate_canonical: list[str] = []

        for raw_name in neighbor_raw_names:
            neighbor = index.match(raw_name)
            if neighbor is not None:
                seatmate_members.append(neighbor)
                seatmate_canonical.append(neighbor.name)
            else:
                # Neighbor exists in the chart but wasn't matched to a member
                # (e.g. in dev mode with limited members).  Still record the
                # raw name so the field isn't misleadingly empty.
                seatmate_canonical.append(raw_name)

        member.seatmate_names = seatmate_canonical
        member.seatmate_affinity = _compute_seatmate_affinity(member, seatmate_members)
