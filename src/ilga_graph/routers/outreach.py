"""Outreach recording router.

Tracks calls, emails, and no-answer events per authenticated user.
Anonymous users can still use advocacy but events are not persisted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..db_models import OutreachEvent, User
from ..dependencies import get_current_user_optional

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/outreach", tags=["outreach"])


def _parse_support_score(raw: str) -> int | None:
    """Parse support_score from form: 1-5 integer or empty."""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        n = int(s)
        if 1 <= n <= 5:
            return n
    except ValueError:
        pass
    return None


def _parse_constituent(raw: str) -> bool | None:
    """Parse constituent: '1'/'true'/'yes' -> True, '0'/'false'/'no' -> False, else None."""
    s = (raw or "").strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None


@router.post("/record")
async def record_outreach(
    member_id: str = Form(...),
    kind: str = Form(...),
    zip_code: str = Form(""),
    outcome: str = Form(""),
    notes: str = Form(""),
    contact_name: str = Form(""),
    support_score: str = Form(""),
    constituent: str = Form(""),
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Record an outreach event.  Requires authentication."""
    if user is None:
        return JSONResponse({"ok": False, "error": "Not authenticated"}, status_code=401)

    kind = kind.strip().lower()
    if kind not in ("call", "email", "no_answer"):
        return JSONResponse({"ok": False, "error": "Invalid kind"}, status_code=400)

    event = OutreachEvent(
        user_id=user.id,
        user_email=user.email,
        member_id=member_id.strip(),
        kind=kind,
        zip_code=zip_code.strip() or None,
        outcome=outcome.strip() or None,
        notes=notes.strip() or None,
        contact_name=contact_name.strip() or None,
        support_score=_parse_support_score(support_score),
        constituent=_parse_constituent(constituent),
    )
    db.add(event)
    await db.commit()
    LOGGER.info("Outreach recorded: user=%s member=%s kind=%s", user.email, member_id, kind)
    return {"ok": True, "event_id": event.id}


async def get_outreach_aggregate(db: AsyncSession) -> dict[str, int]:
    """Return global outreach counts for landing page ticker/social proof."""
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    # Total calls and emails (all time)
    result = await db.execute(
        select(OutreachEvent.kind, func.count())
        .where(OutreachEvent.kind.in_(["call", "email"]))
        .group_by(OutreachEvent.kind)
    )
    by_kind = {row[0]: row[1] for row in result.all()}
    calls_total = by_kind.get("call", 0)
    emails_total = by_kind.get("email", 0)
    # Calls this week
    week_result = await db.execute(
        select(func.count())
        .where(OutreachEvent.kind == "call")
        .where(OutreachEvent.created_at >= week_ago)
    )
    calls_this_week = week_result.scalar() or 0
    return {
        "calls_total": calls_total,
        "calls_this_week": calls_this_week,
        "emails_total": emails_total,
    }


@router.get("/aggregate")
async def outreach_aggregate(db: AsyncSession = Depends(get_db)):
    """Public global outreach counts for landing page social proof."""
    return await get_outreach_aggregate(db)


@router.get("/stats/{member_id}")
async def outreach_stats(
    member_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Public per-member outreach counts (all users aggregated)."""
    result = await db.execute(
        select(OutreachEvent.kind, func.count())
        .where(OutreachEvent.member_id == member_id.strip())
        .group_by(OutreachEvent.kind)
    )
    counts = {row[0]: row[1] for row in result.all()}
    total = sum(counts.values())
    return {
        "member_id": member_id,
        "calls": counts.get("call", 0),
        "emails": counts.get("email", 0),
        "no_answers": counts.get("no_answer", 0),
        "total": total,
    }


@router.get("/interest-poll/{member_id}")
async def interest_poll(
    member_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Aggregated interest-level (support_score) counts for callers who reported on this member.
    Used to show a small poll: how others rated this office. 1=Opposed … 5=Champion.
    """
    mid = member_id.strip()
    result = await db.execute(
        select(OutreachEvent.support_score, func.count())
        .where(OutreachEvent.member_id == mid)
        .where(OutreachEvent.kind == "call")
        .where(OutreachEvent.support_score.isnot(None))
        .group_by(OutreachEvent.support_score)
    )
    by_score: dict[int, int] = {row[0]: row[1] for row in result.all()}
    total = sum(by_score.values())
    return {
        "member_id": mid,
        "total_responses": total,
        "by_score": {
            "1": by_score.get(1, 0),
            "2": by_score.get(2, 0),
            "3": by_score.get(3, 0),
            "4": by_score.get(4, 0),
            "5": by_score.get(5, 0),
        },
    }


@router.get("/my-history")
async def my_history(
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's outreach history."""
    if user is None:
        return JSONResponse({"ok": False, "error": "Not authenticated"}, status_code=401)

    result = await db.execute(
        select(OutreachEvent)
        .where(OutreachEvent.user_id == user.id)
        .order_by(OutreachEvent.created_at.desc())
        .limit(100)
    )
    events = result.scalars().all()
    return {
        "events": [
            {
                "id": e.id,
                "member_id": e.member_id,
                "kind": e.kind,
                "zip_code": e.zip_code,
                "outcome": e.outcome,
                "notes": e.notes,
                "contact_name": e.contact_name,
                "support_score": e.support_score,
                "constituent": e.constituent,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ]
    }
