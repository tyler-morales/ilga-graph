"""Normalize vote-event member names to canonical forms.

After scraping, vote PDFs yield mixed name formats:

- **Floor votes**: last name only (``"Murphy"``) or ``"Last,First"`` when
  duplicates exist (``"Davis,Jed"``).
- **Committee votes**: ``"Last, First"`` or ``"Last, First M"``
  (``"Curran, John F"``, ``"Sims Jr., Elgie R"``).

This module resolves those raw strings to the canonical ``Member.name``
(e.g. ``"Laura M. Murphy"``) when a match exists, and falls back to a
consistent display form otherwise.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from .models import Member, VoteEvent

LOGGER = logging.getLogger(__name__)

# Suffixes that appear after a last name (either in member.name or in vote PDFs).
_SUFFIX_RE = re.compile(
    r",?\s+(?:Jr\.?|Sr\.?|III|II|IV)\s*$", re.IGNORECASE
)
# Matches suffix when embedded *inside* a comma-delimited vote name, e.g.
# "Harris III, Napoleon" → suffix = "III", last = "Harris"
# "Sims Jr., Elgie R"   → suffix = "Jr.", last = "Sims"
_VOTE_SUFFIX_RE = re.compile(
    r"\s+(?:Jr\.?|Sr\.?|III|II|IV)$", re.IGNORECASE
)


# ── Key-building helpers ─────────────────────────────────────────────────────


def _norm_key(name: str) -> str:
    """Produce a lowercase key with periods stripped and whitespace collapsed.

    >>> _norm_key("Murphy, Laura M.")
    'murphy, laura m'
    >>> _norm_key("Davis,Jed")
    'davis,jed'
    """
    key = name.lower().replace(".", "")
    # collapse runs of whitespace (but preserve comma position)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _parse_member_name(full_name: str) -> tuple[str, str, str]:
    """Parse ``Member.name`` into ``(first_middle, last, suffix)``.

    ``Member.name`` is in "First [M.] Last [, Suffix]" order.
    Examples::

        "Laura M. Murphy"          -> ("Laura M.", "Murphy", "")
        "John F. Curran"           -> ("John F.", "Curran", "")
        "Elgie R. Sims, Jr."      -> ("Elgie R.", "Sims", "Jr.")
        "Marcus C. Evans, Jr."     -> ("Marcus C.", "Evans", "Jr.")
        "Neil Anderson"            -> ("Neil", "Anderson", "")
        'Emanuel "Chris" Welch'    -> ('Emanuel "Chris"', "Welch", "")
    """
    # Strip trailing suffix like ", Jr." / ", III"
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


def _parse_vote_name(raw: str) -> tuple[str, str, str]:
    """Parse a vote-PDF name into ``(last, first_middle, suffix)``.

    Vote PDFs come in two formats:

    - *No comma*: ``"Murphy"`` → last only, first_middle = "".
    - *Comma*: ``"Murphy, Laura M"`` or ``"Harris III, Napoleon"``
      → ``last, first_middle`` with optional suffix on the last-name part.

    Returns ``(last, first_middle, suffix)`` all stripped.
    """
    if "," not in raw:
        # Pure last name (floor vote) – no first name available
        return (raw.strip(), "", "")

    # Split on first comma only
    last_part, first_middle = raw.split(",", 1)
    last_part = last_part.strip()
    first_middle = first_middle.strip()

    # Check for suffix on last_part: "Harris III" or "Sims Jr."
    suffix = ""
    sm = _VOTE_SUFFIX_RE.search(last_part)
    if sm:
        suffix = sm.group().strip()
        last_part = last_part[: sm.start()].strip()

    return (last_part, first_middle, suffix)


# ── Variant map builder ──────────────────────────────────────────────────────


def _build_variant_map(
    members: list[Member],
) -> dict[tuple[str, str], str]:
    """Build ``{(chamber, norm_key): canonical_name}`` from the member list.

    For each member, we generate several key variants so that any format
    appearing in vote PDFs can resolve back to ``member.name``.
    """
    variant_map: dict[tuple[str, str], str] = {}

    # Count last-name occurrences per chamber so we only add bare-last-name
    # keys when there is no ambiguity.
    last_counts: dict[tuple[str, str], int] = Counter()
    parsed: list[tuple[Member, str, str, str]] = []

    for m in members:
        first_middle, last, suffix = _parse_member_name(m.name)
        parsed.append((m, first_middle, last, suffix))
        last_counts[(m.chamber, _norm_key(last))] += 1

    for m, first_middle, last, suffix in parsed:
        canonical = m.name
        chamber = m.chamber

        # 1. Full canonical name (normalized)
        variant_map[(chamber, _norm_key(canonical))] = canonical

        # 2. "Last, First M" variants (with and without space after comma)
        #    Strip periods from first_middle so "Laura M." matches "Laura M".
        fm_stripped = first_middle.replace(".", "").strip()
        if fm_stripped:
            # "Murphy, Laura M" (with space)
            v_spaced = f"{last}, {fm_stripped}"
            variant_map[(chamber, _norm_key(v_spaced))] = canonical

            # "Murphy,Laura M" (no space after comma — floor duplicate format)
            v_nospace = f"{last},{fm_stripped}"
            variant_map[(chamber, _norm_key(v_nospace))] = canonical

            # If there's a suffix, add variants with suffix on last name
            # e.g. "Sims Jr., Elgie R" or "Harris III, Napoleon"
            if suffix:
                v_suffix_spaced = f"{last} {suffix}, {fm_stripped}"
                variant_map[(chamber, _norm_key(v_suffix_spaced))] = canonical
                v_suffix_nospace = f"{last} {suffix},{fm_stripped}"
                variant_map[(chamber, _norm_key(v_suffix_nospace))] = canonical

            # Also try first-name-only variant (no middle initial)
            # e.g. for "Laura M. Murphy", add "Murphy, Laura" and "Murphy,Laura"
            first_only = fm_stripped.split()[0] if fm_stripped else ""
            if first_only and first_only != fm_stripped:
                v_first_spaced = f"{last}, {first_only}"
                key = (chamber, _norm_key(v_first_spaced))
                if key not in variant_map:
                    variant_map[key] = canonical
                v_first_nospace = f"{last},{first_only}"
                key = (chamber, _norm_key(v_first_nospace))
                if key not in variant_map:
                    variant_map[key] = canonical

                if suffix:
                    v_sf = f"{last} {suffix}, {first_only}"
                    key = (chamber, _norm_key(v_sf))
                    if key not in variant_map:
                        variant_map[key] = canonical
                    v_sfn = f"{last} {suffix},{first_only}"
                    key = (chamber, _norm_key(v_sfn))
                    if key not in variant_map:
                        variant_map[key] = canonical

        # 3. Bare last name — only when unambiguous in this chamber
        if last_counts[(chamber, _norm_key(last))] == 1:
            variant_map[(chamber, _norm_key(last))] = canonical

    return variant_map


# ── Single-name normalization ────────────────────────────────────────────────


def _display_fallback(raw: str) -> str:
    """Normalize a vote name to a consistent display form when no member match.

    - If raw has a comma, format as ``"Last, First"`` (space after comma,
      title-cased).
    - Otherwise return the raw string (already a last name).
    """
    if "," not in raw:
        return raw.strip()
    last_part, first_part = raw.split(",", 1)
    return f"{last_part.strip()}, {first_part.strip()}"


def _resolve_name(
    raw: str,
    chamber: str,
    variant_map: dict[tuple[str, str], str],
) -> str:
    """Resolve a single raw vote name to a canonical or display form."""
    key = _norm_key(raw)
    canonical = variant_map.get((chamber, key))
    if canonical is not None:
        return canonical
    return _display_fallback(raw)


# ── Entry point ──────────────────────────────────────────────────────────────


def normalize_vote_events(
    events: list[VoteEvent],
    member_lookup: dict[str, Member],
) -> None:
    """Normalize all vote names in *events* in place.

    Builds a variant map from the members in *member_lookup*, then replaces
    every name in ``yea_votes``, ``nay_votes``, ``present_votes``, and
    ``nv_votes`` with the canonical ``member.name`` when matched, or a
    consistently-formatted fallback otherwise.
    """
    members = list(member_lookup.values())
    variant_map = _build_variant_map(members)

    resolved = 0
    total = 0

    for event in events:
        chamber = event.chamber
        for attr in ("yea_votes", "nay_votes", "present_votes", "nv_votes"):
            names: list[str] = getattr(event, attr)
            for i, raw in enumerate(names):
                total += 1
                normalized = _resolve_name(raw, chamber, variant_map)
                if normalized != raw:
                    names[i] = normalized
                    resolved += 1

    LOGGER.info(
        "Vote-name normalization: %d/%d names resolved to canonical member names.",
        resolved,
        total,
    )
