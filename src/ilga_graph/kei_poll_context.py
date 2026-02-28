"""Kei poll context: initial state and sidebar context for poll form vs results."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .constants import KEI_STATUS_SLUGS
from .db_models import User

SIDEBAR_KEI_POLL_ID = "sidebar-kei-poll"
_KEI_POLL_IDS = frozenset(
    {"footer-kei-poll", "home-kei-poll", "updates-kei-poll", SIDEBAR_KEI_POLL_ID}
)
KEI_POLL_VOTED_COOKIE = "kei_poll_voted"
KEI_POLL_CHOICE_COOKIE = "kei_poll_choice"
KEI_POLL_VOTED_MAX_AGE = 365 * 24 * 60 * 60  # 1 year


def _validate_kei_status(slug: str | None) -> str | None:
    """Return slug if valid, else None."""
    if not slug or (s := slug.strip()) not in KEI_STATUS_SLUGS:
        return None
    return s


async def _get_kei_status_results(db: AsyncSession) -> dict[str, Any]:
    """Aggregate kei_status counts for verified users only (last_login_at IS NOT NULL)."""
    result = await db.execute(
        select(User.kei_status, func.count())
        .where(User.kei_status.isnot(None))
        .where(User.last_login_at.isnot(None))
        .group_by(User.kei_status)
    )
    by_status: dict[str, int] = {row[0]: row[1] for row in result.all()}
    total = sum(by_status.values())
    return {
        "by_status": {slug: by_status.get(slug, 0) for slug in KEI_STATUS_SLUGS},
        "total_responses": total,
    }


async def get_kei_poll_initial_state(
    request: Request,
    user: User | None,
    db: AsyncSession,
) -> dict[str, Any]:
    """Return context to show poll form vs results on initial load. If user has voted (logged-in
    kei_status or cookie for anonymous), show results; else show form."""
    voted_cookie = request.cookies.get(KEI_POLL_VOTED_COOKIE) == "1"
    logged_in_voted = user is not None and getattr(user, "kei_status", None) is not None
    show_results = logged_in_voted or voted_cookie
    if not show_results:
        return {"kei_poll_done": False}
    results = await _get_kei_status_results(db)
    selected = user.kei_status if user else None
    if not selected and voted_cookie:
        choice_cookie = request.cookies.get(KEI_POLL_CHOICE_COOKIE)
        selected = _validate_kei_status(choice_cookie) if choice_cookie else None
    return {
        "kei_poll_done": True,
        "kei_status_results": results,
        "kei_status_selected": selected,
        "kei_poll_initial_anon": not logged_in_voted,
    }


async def get_kei_poll_sidebar_context(
    request: Request,
    user: User | None,
    db: AsyncSession,
) -> dict[str, Any]:
    """Sidebar Kei poll (the-issue, legislator-brief, fact-sheet, glossary). Same poll_id."""
    state = await get_kei_poll_initial_state(request, user, db)
    state["poll_id"] = SIDEBAR_KEI_POLL_ID
    if not state.get("kei_poll_done"):
        results = await _get_kei_status_results(db)
        state["kei_status_total"] = results["total_responses"]
    return state


def get_kei_poll_ids() -> frozenset[str]:
    """Return the set of valid poll IDs (for routes that validate poll_id)."""
    return _KEI_POLL_IDS
