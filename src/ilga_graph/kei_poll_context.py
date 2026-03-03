"""Kei poll context: initial state and sidebar context for poll form vs results."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from sqlalchemy import cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import String

from .campaign_config import get_campaign_config, get_kei_poll_goal
from .constants import (
    KEI_IMPACT_ALL_OPTIONS,
    KEI_IMPACT_SLUG_COOKIE,
    KEI_OWNER_SLUGS,
    KEI_POLL_IMPACT_SLUGS,
    KEI_STATUS_SLUGS,
)
from .db_models import Poll, PollResponse, User

SIDEBAR_KEI_POLL_ID = "sidebar-kei-poll"
STANDALONE_KEI_POLL_ID = "standalone-kei-poll"
_KEI_POLL_IDS = frozenset(
    {
        "footer-kei-poll",
        "home-kei-poll",
        "updates-kei-poll",
        SIDEBAR_KEI_POLL_ID,
        STANDALONE_KEI_POLL_ID,
    }
)
KEI_POLL_VOTED_COOKIE = "kei_poll_voted"
KEI_POLL_CHOICE_COOKIE = "kei_poll_choice"
KEI_POLL_VOTED_MAX_AGE = 365 * 24 * 60 * 60  # 1 year


def _validate_kei_status(slug: str | None) -> str | None:
    """Return slug if valid, else None."""
    if not slug or (s := slug.strip()) not in KEI_STATUS_SLUGS:
        return None
    return s


def _validate_kei_poll_impact(slug: str | None) -> str | None:
    """Return slug if valid for main poll Q3 (universal impact), else None."""
    if not slug or (s := slug.strip()) not in KEI_POLL_IMPACT_SLUGS:
        return None
    return s


def _respondent_key_expr():
    """SQL expression: one key per respondent (user_id or session_id; id fallback)."""
    return func.coalesce(
        cast(PollResponse.user_id, String(64)),
        func.coalesce(PollResponse.session_id, cast(PollResponse.id, String(64))),
    )


async def get_distinct_respondent_count(db: AsyncSession, poll_id: int) -> int:
    """Distinct respondent count for a poll (one per person). Used by admin and progress bar."""
    r = await db.execute(
        select(func.count(func.distinct(_respondent_key_expr()))).where(
            PollResponse.poll_id == poll_id
        )
    )
    return r.scalar() or 0


async def _get_kei_status_results(db: AsyncSession) -> dict[str, Any]:
    """Aggregate kei_status counts from PollResponse for the campaign poll.
    total_responses = distinct respondents (one per person who completed the poll).
    by_status = response counts per option (for chart)."""
    poll_slug = get_campaign_config().poll_slug or "kei"
    poll = (await db.execute(select(Poll).where(Poll.slug == poll_slug))).scalar_one_or_none()
    if not poll:
        return {
            "by_status": {slug: 0 for slug in KEI_STATUS_SLUGS},
            "total_responses": 0,
        }
    result = await db.execute(
        select(PollResponse.option_slug, func.count())
        .where(PollResponse.poll_id == poll.id)
        .group_by(PollResponse.option_slug)
    )
    by_status: dict[str, int] = {row[0]: row[1] for row in result.all()}
    row_total = sum(by_status.values())
    # One response per person: count distinct respondents (progress bar "x of 1000")
    distinct_r = await db.execute(
        select(func.count(func.distinct(_respondent_key_expr()))).where(
            PollResponse.poll_id == poll.id
        )
    )
    total = distinct_r.scalar() or 0
    if total == 0 and row_total > 0:
        total = row_total  # fallback if distinct query returns 0
    return {
        "by_status": {slug: by_status.get(slug, 0) for slug in KEI_STATUS_SLUGS},
        "total_responses": total,
    }


def _impact_options_for_results(by_slug: dict[str, int]) -> list[tuple[str, str]]:
    """Full options list for impact results: KEI_IMPACT_ALL_OPTIONS plus any DB slugs not in it."""
    known = {s for s, _ in KEI_IMPACT_ALL_OPTIONS}
    extra = [(s, "Other") for s in by_slug if s not in known]
    return list(KEI_IMPACT_ALL_OPTIONS) + extra


async def _get_kei_impact_results(db: AsyncSession) -> dict[str, Any] | None:
    """Aggregate kei_impact poll counts for verified users. Returns None if kei_impact poll missing.
    Includes all impact options (status-specific + universal) for results display."""
    poll = (await db.execute(select(Poll).where(Poll.slug == "kei_impact"))).scalar_one_or_none()
    if not poll:
        return None
    result = await db.execute(
        select(PollResponse.option_slug, func.count())
        .join(User, PollResponse.user_id == User.id)
        .where(PollResponse.poll_id == poll.id)
        .where(User.last_login_at.isnot(None))
        .group_by(PollResponse.option_slug)
    )
    by_slug: dict[str, int] = {row[0]: row[1] for row in result.all()}
    total = sum(by_slug.values())
    options = _impact_options_for_results(by_slug)
    return {
        "by_status": {s: by_slug.get(s, 0) for s, _ in options},
        "total_responses": total,
        "options": options,
    }


async def get_kei_poll_initial_state(
    request: Request,
    user: User | None,
    db: AsyncSession,
) -> dict[str, Any]:
    """Return context to show poll form vs results on initial load. If user has voted (logged-in
    kei_status or cookie for anonymous), show results; else show form.
    Poll state is shared everywhere: cookie (anon) and user.kei_status (logged-in) so any page
    (home, /poll, /updates, sidebar) shows consistent 'voted' state after a vote."""
    voted_cookie = request.cookies.get(KEI_POLL_VOTED_COOKIE) == "1"
    logged_in_voted = user is not None and getattr(user, "kei_status", None) is not None
    show_results = logged_in_voted or voted_cookie
    if not show_results:
        return {"kei_poll_done": False, "kei_poll_goal": get_kei_poll_goal()}
    results = await _get_kei_status_results(db)
    selected = user.kei_status if user else None
    if not selected and voted_cookie:
        choice_cookie = request.cookies.get(KEI_POLL_CHOICE_COOKIE)
        selected = _validate_kei_status(choice_cookie) if choice_cookie else None
    impact_selected = getattr(user, "kei_impact_slug", None) if user else None
    if not impact_selected:
        impact_cookie = request.cookies.get(KEI_IMPACT_SLUG_COOKIE)
        impact_selected = _validate_kei_poll_impact(impact_cookie) if impact_cookie else None
    impact_results = await _get_kei_impact_results(db)
    return {
        "kei_poll_done": True,
        "kei_status_results": results,
        "kei_status_selected": selected,
        "kei_impact_selected": impact_selected,
        "kei_impact_results": impact_results,
        "kei_poll_initial_anon": not logged_in_voted,
        "kei_poll_is_owner": selected in KEI_OWNER_SLUGS if selected else False,
        "kei_poll_goal": get_kei_poll_goal(),
    }


def zip_known_for_user(user: User | None) -> bool:
    """True if we already have the user's ZIP (don't show zip panel in poll)."""
    return bool(user and (user.zip_code or "").strip())


async def get_kei_poll_sidebar_context(
    request: Request,
    user: User | None,
    db: AsyncSession,
) -> dict[str, Any]:
    """Sidebar Kei poll (the-issue, legislator-brief, fact-sheet, glossary). Same poll_id."""
    from .routers.content_constants import get_why_you_care_branch_for_selection

    state = await get_kei_poll_initial_state(request, user, db)
    state["poll_id"] = SIDEBAR_KEI_POLL_ID
    state["zip_known"] = zip_known_for_user(user)
    state["prefill_zip"] = (user.zip_code or "").strip() if user else ""
    if state.get("kei_poll_done") and state.get("kei_status_selected"):
        state["why_you_care_branch"] = get_why_you_care_branch_for_selection(
            state.get("kei_status_selected")
        )
    if not state.get("kei_poll_done"):
        results = await _get_kei_status_results(db)
        state["kei_status_total"] = results["total_responses"]
    return state


def get_kei_poll_ids() -> frozenset[str]:
    """Return the set of valid poll IDs (for routes that validate poll_id)."""
    return _KEI_POLL_IDS
