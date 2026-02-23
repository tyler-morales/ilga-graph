"""Admin/dev routes: logs dashboard, health check, dev bar API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..app_state import state
from ..db import get_db
from ..db_models import OutreachEvent, OutreachStepEvent, User
from ..run_log import get_log_path, load_recent_runs

router = APIRouter()
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


@router.get("/logs", include_in_schema=False)
async def logs_dashboard(request: Request):
    """Unified run log dashboard — scrape, ML, startup. Minimal 2000s-hacker UI."""
    runs = load_recent_runs(n=100)
    task_phases: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        if r.task not in task_phases:
            task_phases[r.task] = {}
        for p in r.phases:
            name = p.get("name", "?")
            if name not in task_phases[r.task]:
                task_phases[r.task][name] = []
            task_phases[r.task][name].append(p.get("duration_s") or 0)
    bottleneck: list[tuple[str, list[tuple[str, float]]]] = []
    for task, phases in task_phases.items():
        by_name = [(name, sum(durs) / len(durs) if durs else 0) for name, durs in phases.items()]
        by_name.sort(key=lambda x: x[1], reverse=True)
        bottleneck.append((task, by_name[:5]))
    return templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            "runs": runs,
            "bottleneck": bottleneck,
            "log_path": str(get_log_path()),
        },
    )


@router.get("/health", include_in_schema=True)
async def health() -> dict:
    """Service health check with data counts."""
    return {
        "status": "ok",
        "ready": len(state.members) > 0,
        "members": len(state.members),
        "bills": len(state.bills),
        "committees": len(state.committees),
        "vote_events": len(state.vote_events),
    }


@router.get("/api/dev/members", include_in_schema=False)
async def dev_members():
    """Return first 20 members as JSON for the dev bar member dropdown. Only active in DEV_MODE."""
    if not cfg.DEV_MODE:
        return JSONResponse(status_code=404, content={"detail": "Not available"})
    return [{"id": m.id, "name": m.name} for m in state.members[:20]]


CONVERSION_WINDOW_DAYS = 90


def _pct(denom: int, num: int) -> float:
    if denom == 0:
        return 0.0
    return round(100.0 * num / denom, 2)


@router.get("/admin/outreach/conversion", include_in_schema=False)
async def outreach_conversion(
    db: AsyncSession = Depends(get_db),
):
    """Conversion and volume report for advocacy funnel (last 90 days).

    Returns conversions (rates) and volumes (counts). Identity = user_id when set,
    else session_id. Protected: only when DEV_MODE is true (internal/admin or Metabase).
    """
    if not cfg.DEV_MODE:
        return JSONResponse(status_code=404, content={"detail": "Not available"})

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=CONVERSION_WINDOW_DAYS)
    identity_expr = func.coalesce(
        cast(OutreachStepEvent.user_id, String),
        OutreachStepEvent.session_id,
    )

    # --- Denominators (step-based) ---
    async def _count_identities_step(step_slug: str, outreach_type: str | None = None) -> int:
        q = (
            select(func.count(func.distinct(identity_expr)))
            .select_from(OutreachStepEvent)
            .where(OutreachStepEvent.step_slug == step_slug)
            .where(OutreachStepEvent.reached_at >= window_start)
            .where(OutreachStepEvent.reached_at <= now)
        )
        if outreach_type is not None:
            q = q.where(OutreachStepEvent.outreach_type == outreach_type)
        r = await db.execute(q)
        return r.scalar() or 0

    # Run step-based denominator queries (same window)
    denom_drawer = await _count_identities_step("drawer_opened")
    denom_phone_clicked = await _count_identities_step("phone_clicked", "call")

    # --- Numerators (outreach_events) ---
    async def _count_users_events(kinds: list[str]) -> int:
        q = (
            select(func.count(func.distinct(OutreachEvent.user_id)))
            .select_from(OutreachEvent)
            .where(OutreachEvent.kind.in_(kinds))
            .where(OutreachEvent.created_at >= window_start)
            .where(OutreachEvent.created_at <= now)
            .where(OutreachEvent.user_id.isnot(None))
        )
        r = await db.execute(q)
        return r.scalar() or 0

    users_call_or_email = await _count_users_events(["call", "email"])
    users_call_or_no_answer = await _count_users_events(["call", "no_answer"])
    users_email = await _count_users_events(["email"])

    # signed_in_to_outreach denominator: users with last_login in window
    signed_in_denom_q = (
        select(func.count(User.id))
        .select_from(User)
        .where(User.last_login_at.isnot(None))
        .where(User.last_login_at >= window_start)
        .where(User.last_login_at <= now)
    )
    signed_in_denom = (await db.execute(signed_in_denom_q)).scalar() or 0

    # --- Conversions (minimum set) ---
    conversions = {
        "drawer_to_outreach": {
            "denominator": denom_drawer,
            "numerator": users_call_or_email,
            "conversion_pct": _pct(denom_drawer, users_call_or_email),
        },
        "phone_to_call": {
            "denominator": denom_phone_clicked,
            "numerator": users_call_or_no_answer,
            "conversion_pct": _pct(denom_phone_clicked, users_call_or_no_answer),
        },
        "drawer_to_email": {
            "denominator": denom_drawer,
            "numerator": users_email,
            "conversion_pct": _pct(denom_drawer, users_email),
        },
        "signed_in_to_outreach": {
            "denominator": signed_in_denom,
            "numerator": users_call_or_email,
            "conversion_pct": _pct(signed_in_denom, users_call_or_email),
        },
    }

    # --- Volumes ---
    async def _count_events(kinds: list[str]) -> int:
        q = (
            select(func.count(OutreachEvent.id))
            .select_from(OutreachEvent)
            .where(OutreachEvent.kind.in_(kinds))
            .where(OutreachEvent.created_at >= window_start)
            .where(OutreachEvent.created_at <= now)
        )
        r = await db.execute(q)
        return r.scalar() or 0

    total_calls = await _count_events(["call"])
    total_emails = await _count_events(["email"])

    volumes = {
        "identities_opened_drawer": denom_drawer,
        "users_completed_outreach": users_call_or_email,
        "total_calls": total_calls,
        "total_emails": total_emails,
        "total_outreach_actions": total_calls + total_emails,
        "identities_clicked_phone": denom_phone_clicked,
    }

    return {
        "window_days": CONVERSION_WINDOW_DAYS,
        "window_start": window_start.isoformat(),
        "window_end": now.isoformat(),
        "conversions": conversions,
        "volumes": volumes,
    }
