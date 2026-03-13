"""Outreach recording router.

Tracks calls, emails, and no-answer events per authenticated user.
Anonymous users can still use advocacy but events are not persisted.
When recording a call with legislator_email and member has no public email,
the email is stored in community_member_emails for the community layer.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..app_state import state
from ..campaign_helpers import get_active_campaign
from ..db import get_db
from ..db_models import CommunityMemberEmail, OutreachEvent, OutreachStepEvent, User
from ..dependencies import get_current_user_optional
from ..member_lookup import find_member_by_id
from ..outreach_steps import is_valid_step
from ..security import CSRF_COOKIE_NAME, validate_anon_session_id, validate_csrf_token

_LEGISLATOR_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", re.IGNORECASE)
_LEGISLATOR_EMAIL_MAX = 320

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


def _normalize_and_validate_legislator_email(raw: str) -> str | None:
    """Return normalized email if valid (format + length); else None."""
    s = (raw or "").strip().lower()
    if not s or len(s) > _LEGISLATOR_EMAIL_MAX:
        return None
    if not _LEGISLATOR_EMAIL_RE.match(s):
        return None
    local, _, domain = s.partition("@")
    if (
        len(local) < 1
        or len(local) > 64
        or len(domain) < 4
        or len(domain) > 253
        or "." not in domain
    ):
        return None
    return s


@router.post("/record")
async def record_outreach(
    request: Request,
    member_id: str = Form(...),
    kind: str = Form(...),
    zip_code: str = Form(""),
    outcome: str = Form(""),
    notes: str = Form(""),
    contact_name: str = Form(""),
    support_score: str = Form(""),
    constituent: str = Form(""),
    legislator_email: str = Form(""),
    campaign_id: str = Form(""),
    csrf_token: str | None = Form(None),
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Record an outreach event.  Requires authentication.
    When kind=call and legislator_email is provided and member has no public email,
    the email is stored in community_member_emails for future constituents.
    """
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        return JSONResponse(
            {"ok": False, "error": "Invalid or expired security token. Reload the page."},
            status_code=403,
        )
    if user is None:
        return JSONResponse({"ok": False, "error": "Not authenticated"}, status_code=401)

    kind = kind.strip().lower()
    if kind not in ("call", "email", "no_answer"):
        return JSONResponse({"ok": False, "error": "Invalid kind"}, status_code=400)

    mid = member_id.strip()[:32]
    if not mid:
        return JSONResponse({"ok": False, "error": "Missing legislator"}, status_code=400)
    member = find_member_by_id(state, mid)
    if member is None:
        return JSONResponse(
            {"ok": False, "error": "Legislator not found. Please refresh and try again."},
            status_code=400,
        )

    zip_val = zip_code.strip() or None
    campaign_id_val: int | None = None
    if campaign_id.strip():
        try:
            cid = int(campaign_id.strip())
            active = await get_active_campaign(db)
            if active and active.id == cid:
                campaign_id_val = cid
        except ValueError:
            pass
    event = OutreachEvent(
        user_id=user.id,
        user_email=user.email,
        member_id=mid,
        kind=kind,
        zip_code=zip_val,
        outcome=outcome.strip() or None,
        notes=notes.strip() or None,
        contact_name=contact_name.strip() or None,
        support_score=_parse_support_score(support_score),
        constituent=_parse_constituent(constituent),
        campaign_id=campaign_id_val,
    )
    db.add(event)

    if kind == "call" and not (getattr(member, "email", None) or "").strip():
        normalized = _normalize_and_validate_legislator_email(legislator_email)
        if normalized:
            try:
                async with db.begin_nested():
                    db.add(
                        CommunityMemberEmail(
                            member_id=mid,
                            email=normalized,
                            user_id=user.id,
                        )
                    )
                    await db.flush()
            except IntegrityError:
                pass

    # user.zip_code is set only by explicit zip commit (sidebar / Use location), not by outreach.

    # Record funnel step: call_recorded, email_recorded, or no_answer_recorded
    step_slug = {
        "call": "call_recorded",
        "email": "email_recorded",
        "no_answer": "no_answer_recorded",
    }[kind]
    db.add(
        OutreachStepEvent(
            user_id=user.id,
            member_id=mid,
            outreach_type="call" if kind in ("call", "no_answer") else "email",
            step_slug=step_slug,
        )
    )

    await db.commit()
    if state.ontology_sdk is not None:
        from ..ontology import outreach_event_to_action

        action = outreach_event_to_action(
            event_id=event.id,
            kind=kind,
            member_id=mid,
            user_id=user.id,
            created_at=event.created_at,
            outcome=outcome.strip() or None,
            campaign_id=campaign_id_val,
        )
        state.ontology_sdk.execute_action(action)
    LOGGER.info("Outreach recorded: user=%s member=%s kind=%s", user.email, member_id, kind)
    return {"ok": True, "event_id": event.id}


@router.post("/step")
async def record_outreach_step(
    request: Request,
    member_id: str = Form(""),
    outreach_type: str = Form(...),
    step_slug: str = Form(...),
    session_id: str | None = Form(None),
    csrf_token: str | None = Form(None),
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Record a checkpoint step in the call/email or WYC funnel.

    Authenticated: store user_id, session_id not persisted.
    Anonymous: accepted when session_id is present and valid; stored with user_id=NULL.
    For outreach_type=wyc, member_id is optional and stored as NULL.
    """
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        return JSONResponse(
            {"ok": False, "error": "Invalid or expired security token. Reload the page."},
            status_code=403,
        )

    outreach_type = outreach_type.strip().lower()
    if outreach_type not in ("call", "email", "wyc"):
        return JSONResponse({"ok": False, "error": "Invalid outreach_type"}, status_code=400)
    if not is_valid_step(outreach_type, step_slug.strip()):
        return JSONResponse({"ok": False, "error": "Invalid step"}, status_code=400)

    if outreach_type == "wyc":
        mid: str | None = None
    else:
        mid = member_id.strip()[:32] if member_id else ""
        if not mid:
            return JSONResponse({"ok": False, "error": "Missing legislator"}, status_code=400)
        if find_member_by_id(state, mid) is None:
            return JSONResponse(
                {"ok": False, "error": "Legislator not found. Please refresh and try again."},
                status_code=400,
            )

    if user is not None:
        db.add(
            OutreachStepEvent(
                user_id=user.id,
                session_id=None,
                member_id=mid,
                outreach_type=outreach_type,
                step_slug=step_slug.strip(),
            )
        )
    else:
        anon_sid = validate_anon_session_id(session_id)
        if anon_sid is None:
            if session_id is not None and str(session_id).strip():
                return JSONResponse(
                    {"ok": False, "error": "Invalid session_id format"},
                    status_code=400,
                )
            return JSONResponse(
                {"ok": False, "error": "Not authenticated"},
                status_code=401,
            )
        db.add(
            OutreachStepEvent(
                user_id=None,
                session_id=anon_sid,
                member_id=mid,
                outreach_type=outreach_type,
                step_slug=step_slug.strip(),
            )
        )

    await db.commit()
    return {"ok": True}


async def get_outreach_aggregate(db: AsyncSession) -> dict[str, int]:
    """Return global outreach counts for landing page ticker/social proof.

    Hero uses calls_total = total actions (each call and each email count
    separately); call + email for one member = 2.
    """
    total_result = await db.execute(
        select(func.count())
        .select_from(OutreachEvent)
        .where(OutreachEvent.kind.in_(["call", "email"]))
    )
    total_actions = total_result.scalar() or 0
    result = await db.execute(
        select(OutreachEvent.kind, func.count())
        .where(OutreachEvent.kind.in_(["call", "email"]))
        .group_by(OutreachEvent.kind)
    )
    by_kind = {row[0]: row[1] for row in result.all()}
    return {
        "calls_total": total_actions,
        "calls_this_week": 0,
        "emails_total": by_kind.get("email", 0),
    }


async def get_outreach_count_for_member(db: AsyncSession, member_id: str) -> int:
    """Return total call + email events for this member (for script social proof)."""
    if not (mid := (member_id or "").strip()):
        return 0
    r = await db.execute(
        select(func.count())
        .select_from(OutreachEvent)
        .where(
            OutreachEvent.member_id == mid,
            OutreachEvent.kind.in_(["call", "email"]),
        )
    )
    return r.scalar() or 0


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


@router.get("/my-stats")
async def my_stats(
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's outreach counts (calls, emails)."""
    if user is None:
        return JSONResponse({"ok": False, "error": "Not authenticated"}, status_code=401)
    result = await db.execute(
        select(OutreachEvent.kind, func.count())
        .where(OutreachEvent.user_id == user.id)
        .where(OutreachEvent.kind.in_(["call", "email"]))
        .group_by(OutreachEvent.kind)
    )
    by_kind = {row[0]: row[1] for row in result.all()}
    return {
        "calls": by_kind.get("call", 0),
        "emails": by_kind.get("email", 0),
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
