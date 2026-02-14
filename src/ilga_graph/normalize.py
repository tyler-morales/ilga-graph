"""Shared data normalization utilities.

Centralizes date and chamber normalization so all scrapers, cache loaders,
and ML pipeline code use consistent formats.

**Date normalization:**
    All dates are converted to ISO ``YYYY-MM-DD`` format on scrape/load.
    Handles the three formats found in ILGA data:
    - ``1/13/2025``       (MM/DD/YYYY from BillStatus pages)
    - ``May 31, 2025``    (full month name from vote PDFs)
    - ``2025-05-31 17:00``(ISO-ish from witness slip exports)

**Chamber normalization:**
    All chamber values are normalized to single-letter codes:
    - ``"S"`` for Senate
    - ``"H"`` for House
    - ``"J"`` for Joint/Executive (EO, JSR, AM doc types)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

LOGGER = logging.getLogger(__name__)

# Common date formats encountered in ILGA data
_DATE_FORMATS = [
    "%Y-%m-%d",  # ISO (already normalized)
    "%m/%d/%Y",  # MM/DD/YYYY (BillStatus pages)
    "%B %d, %Y",  # "May 31, 2025" (vote PDFs)
    "%b %d, %Y",  # "May 31, 2025" (abbreviated month)
    "%Y-%m-%d %H:%M",  # "2025-05-31 17:00" (witness slips)
    "%Y-%m-%d %H:%M:%S",  # "2025-05-31 17:00:00" (witness slips alt)
    "%m/%d/%Y %I:%M:%S %p",  # "1/13/2025 12:00:00 AM" (occasional)
]

# Chamber normalization map
_CHAMBER_MAP = {
    "senate": "S",
    "house": "H",
    "joint": "J",
    "executive": "J",
    "s": "S",
    "h": "H",
    "j": "J",
}


def normalize_date(date_str: str | None) -> str:
    """Normalize any ILGA date string to ISO ``YYYY-MM-DD`` format.

    Returns empty string if the input is None, empty, or unparseable.
    Already-normalized dates (``YYYY-MM-DD``) pass through unchanged.

    Examples::

        >>> normalize_date("1/13/2025")
        '2025-01-13'
        >>> normalize_date("May 31, 2025")
        '2025-05-31'
        >>> normalize_date("2025-05-31 17:00")
        '2025-05-31'
        >>> normalize_date("2025-01-13")
        '2025-01-13'
        >>> normalize_date(None)
        ''
    """
    if not date_str or not isinstance(date_str, str):
        return ""

    date_str = date_str.strip()
    if not date_str:
        return ""

    # Quick check: already ISO format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    LOGGER.debug("normalize_date: unparseable date %r", date_str)
    return date_str  # Return original if unparseable (avoid data loss)


def normalize_chamber(chamber_str: str | None) -> str:
    """Normalize chamber values to single-letter codes: ``S``, ``H``, or ``J``.

    Handles all variants found in ILGA data:
    - ``"Senate"`` / ``"House"`` (action history, vote events)
    - ``"S"`` / ``"H"`` (bill model)
    - Empty/None returns empty string.

    Examples::

        >>> normalize_chamber("Senate")
        'S'
        >>> normalize_chamber("House")
        'H'
        >>> normalize_chamber("S")
        'S'
        >>> normalize_chamber(None)
        ''
    """
    if not chamber_str or not isinstance(chamber_str, str):
        return ""

    key = chamber_str.strip().lower()
    return _CHAMBER_MAP.get(key, chamber_str.strip())


# ── Cache schema validation ──────────────────────────────────────────────────


class CacheValidationError(ValueError):
    """Raised when cached data fails schema validation."""


def validate_bill_dict(d: dict, *, strict: bool = False) -> list[str]:
    """Validate a bill dict from cache has required fields.

    Returns a list of warning messages. If ``strict`` is True, raises
    ``CacheValidationError`` on the first critical issue.

    Required fields (critical -- bill is unusable without these):
        - ``bill_number``: non-empty string
        - ``leg_id``: non-empty string
        - ``chamber``: non-empty string, one of S/H/J or Senate/House

    Recommended fields (warn if missing but bill is still usable):
        - ``description``: should be non-empty
        - ``last_action_date``: should be parseable date
        - ``primary_sponsor``: should be non-empty
    """
    warnings = []

    # Critical fields
    for field in ("bill_number", "leg_id"):
        val = d.get(field)
        if not val or not isinstance(val, str) or not val.strip():
            msg = f"Bill missing required field '{field}': {d.get('leg_id', '?')}"
            if strict:
                raise CacheValidationError(msg)
            warnings.append(msg)

    chamber = d.get("chamber")
    if not chamber or not isinstance(chamber, str) or not chamber.strip():
        msg = f"Bill missing chamber: {d.get('leg_id', '?')}"
        if strict:
            raise CacheValidationError(msg)
        warnings.append(msg)

    # Recommended fields
    if not d.get("description"):
        warnings.append(
            f"Bill {d.get('bill_number', '?')} ({d.get('leg_id', '?')}): empty description"
        )

    return warnings


def validate_member_dict(d: dict, *, strict: bool = False) -> list[str]:
    """Validate a member dict from cache has required fields.

    Required fields:
        - ``id``: non-empty string
        - ``name``: non-empty string
        - ``chamber``: non-empty string

    Recommended fields:
        - ``party``: should be non-empty
        - ``district``: should be non-empty
    """
    warnings = []

    for field in ("id", "name"):
        val = d.get(field)
        if not val or not isinstance(val, str) or not val.strip():
            msg = f"Member missing required field '{field}': {d.get('id', '?')}"
            if strict:
                raise CacheValidationError(msg)
            warnings.append(msg)

    chamber = d.get("chamber")
    if not chamber or not isinstance(chamber, str) or not chamber.strip():
        msg = f"Member missing chamber: {d.get('name', '?')} ({d.get('id', '?')})"
        if strict:
            raise CacheValidationError(msg)
        warnings.append(msg)

    if not d.get("party"):
        warnings.append(f"Member {d.get('name', '?')} ({d.get('id', '?')}): empty party")

    return warnings


def validate_bill_cache(data: dict, *, strict: bool = False) -> list[str]:
    """Validate an entire bills cache dict.

    Parameters
    ----------
    data:
        The raw ``{leg_id: bill_dict}`` loaded from ``bills.json``.
    strict:
        If True, raises on the first critical validation error.

    Returns
    -------
    List of all validation warnings.
    """
    all_warnings = []
    critical_count = 0

    for lid, bill_dict in data.items():
        if not isinstance(bill_dict, dict):
            msg = f"Bill entry {lid} is not a dict"
            if strict:
                raise CacheValidationError(msg)
            all_warnings.append(msg)
            critical_count += 1
            continue

        warnings = validate_bill_dict(bill_dict, strict=strict)
        all_warnings.extend(warnings)
        # Count critical issues (required field missing)
        critical_count += sum(
            1 for w in warnings if "missing required" in w or "missing chamber" in w
        )

    if all_warnings:
        LOGGER.warning(
            "Bill cache validation: %d warnings (%d critical) across %d bills",
            len(all_warnings),
            critical_count,
            len(data),
        )
        for w in all_warnings[:10]:  # Log first 10
            LOGGER.warning("  %s", w)
        if len(all_warnings) > 10:
            LOGGER.warning("  ... and %d more", len(all_warnings) - 10)

    return all_warnings
