"""Community-sourced legislator emails: read path for drawer and wrap-up.

When a member has no public email (scraper), we use the best community email
from community_member_emails (most submitters; tie = most recent).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .db_models import CommunityMemberEmail
from .member_lookup import find_member_by_id

if TYPE_CHECKING:
    from .app_state import AppState


async def get_community_email_for_member(
    db: AsyncSession,
    member_id: str,
) -> tuple[str | None, int]:
    """Return (email, submitter_count) for the best community email for this member.

    Best = email with largest distinct submitter count; tie = most recent.
    Returns (None, 0) if no community submissions.
    """
    mid = (member_id or "").strip()
    if not mid:
        return None, 0
    stmt = (
        select(
            CommunityMemberEmail.email,
            func.count(distinct(CommunityMemberEmail.user_id)).label("n"),
        )
        .where(CommunityMemberEmail.member_id == mid)
        .group_by(CommunityMemberEmail.email)
        .order_by(
            func.count(distinct(CommunityMemberEmail.user_id)).desc(),
            func.max(CommunityMemberEmail.created_at).desc(),
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.one_or_none()
    if not row or not row[0]:
        return None, 0
    return row[0], row[1] or 0


async def get_effective_email_for_member(
    app_state: AppState,
    db: AsyncSession,
    member_id: str,
) -> tuple[str, str | None, str | None]:
    """Return (effective_email, email_source, community_verification).

    - effective_email: member.email or best community email or ""
    - email_source: "public" | "community" | None
    - community_verification: "verified_by_community" | "unverified" | None
    """
    member = find_member_by_id(app_state, member_id)
    if member and (member.email or "").strip():
        return (member.email or "").strip(), "public", None
    community_email, submitter_count = await get_community_email_for_member(db, member_id)
    if community_email:
        verification = "verified_by_community" if submitter_count >= 2 else "unverified"
        return community_email, "community", verification
    return "", None, None
