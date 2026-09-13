"""Money-intel lead capture helpers (lobbyist / adjacent buyer waitlist).

Separate from User / campaign-update subscribers: this list is a marketing
waitlist, not an auth account. Per Hardball Ch 7 (listservs/newsletters) and
Founding Sales Ch 6 (inbound lead capture): short form, opt-in, exportable.
"""

from __future__ import annotations

import re
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db_models import MoneyIntelLead

_EMAIL_MAX_LEN = 320
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", re.IGNORECASE)
_NAME_MAX_LEN = 120
_ORG_MAX_LEN = 200

MONEY_LEAD_ROLES: tuple[str, ...] = ("lobbyist", "lawyer", "nonprofit")
SignupResult = Literal["created", "already"]


def normalize_email(raw: str) -> str | None:
    """Return lowercase trimmed email if valid; else None."""
    s = (raw or "").strip().lower()
    if not s or len(s) > _EMAIL_MAX_LEN:
        return None
    if not _EMAIL_RE.match(s):
        return None
    return s


def normalize_optional_text(raw: str | None, max_len: int) -> str | None:
    """Return trimmed text if non-empty and within max_len; else None."""
    s = (raw or "").strip()
    if not s or len(s) > max_len:
        return None
    return s


def normalize_name(raw: str | None) -> str | None:
    """Return trimmed display name if valid."""
    return normalize_optional_text(raw, _NAME_MAX_LEN)


def normalize_org(raw: str | None) -> str | None:
    """Return trimmed org/firm if valid."""
    return normalize_optional_text(raw, _ORG_MAX_LEN)


def normalize_roles(roles: list[str] | None) -> str | None:
    """Return comma-joined allowlisted roles (sorted, unique) or None."""
    allowed = {r.strip().lower() for r in (roles or []) if r and r.strip()}
    chosen = sorted(allowed & set(MONEY_LEAD_ROLES))
    if not chosen:
        return None
    return ",".join(chosen)


async def persist_money_lead(
    db: AsyncSession,
    *,
    email: str,
    name: str | None,
    org: str | None,
    role: str | None,
) -> SignupResult:
    """Insert a new lead or enrich an existing one. Returns created | already."""
    result = await db.execute(select(MoneyIntelLead).where(MoneyIntelLead.email == email))
    lead = result.scalar_one_or_none()
    if lead:
        if name:
            lead.name = name
        if org:
            lead.org = org
        if role:
            lead.role = role
        await db.commit()
        return "already"
    db.add(MoneyIntelLead(email=email, name=name, org=org, role=role))
    await db.commit()
    return "created"
