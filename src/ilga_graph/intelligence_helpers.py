"""Helpers for intelligence routes: witness-slip org normalization and bill description lookup."""

from __future__ import annotations

import re

from .app_state import state

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
