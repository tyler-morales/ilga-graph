"""Admin/dev routes: logs dashboard, health check, dev bar API, mock switchboard."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import String, cast, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import advocacy_helpers as ah
from .. import config as cfg
from ..app_state import state
from ..constants import CATEGORY_COMMITTEES, GENERAL_COMMITTEE_CODES
from ..db import get_db
from ..db_models import OutreachEvent, OutreachStepEvent, User
from ..dependencies import get_current_user_optional
from ..member_lookup import find_member_by_district
from ..routers.content import STRATEGIC_FIVE_POINTS
from ..run_log import get_log_path, load_recent_runs
from ..security import validate_photo_url_for_drawer

router = APIRouter()
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["dev_available"] = cfg.DEV_MODE
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["site_name"] = cfg.SITE_NAME
templates.env.globals["meta_description"] = cfg.META_DESCRIPTION
templates.env.globals["og_image_url"] = cfg.OG_IMAGE_URL
templates.env.globals["umami_enabled"] = cfg.PROFILE == "prod" and bool(cfg.UMAMI_WEBSITE_ID)
templates.env.globals["umami_website_id"] = cfg.UMAMI_WEBSITE_ID
templates.env.globals["umami_script_url"] = cfg.UMAMI_SCRIPT_URL
templates.env.globals["show_beta_banner"] = cfg.BETA_BANNER
templates.env.globals["beta_banner_feedback_url"] = cfg.BETA_BANNER_REPORT_URL
templates.env.globals["footer_last_updated"] = cfg.FOOTER_LAST_UPDATED
templates.env.globals["footer_last_updated_iso"] = cfg.FOOTER_LAST_UPDATED_ISO
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS
templates.env.globals["features"] = cfg.get_client_features()

_ZIP_RE = re.compile(r"^\d{5}$")
MOCK_DEV_USER_EMAIL = "funky_mama11@gmail.com"
DEFAULT_MOCK_ZIP = "60007"


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


def _resolve_mock_legislators(zip_code: str) -> list[dict[str, Any]]:
    """Resolve ZIP to district legislators (Senator, Rep, Power Broker) for mock switchboard.

    Returns list of dicts: member_id, name, role_label, role_short, photo_url.
    """
    if not _ZIP_RE.match(zip_code) or zip_code not in state.zip_to_district:
        return []
    district_info = state.zip_to_district[zip_code]
    senate_district = district_info.il_senate
    house_district = district_info.il_house
    topic = "Transportation"
    committee_codes = CATEGORY_COMMITTEES.get(topic, [])
    committee_ids = ah.committee_member_ids(state, committee_codes) if committee_codes else None
    relevant_codes = list(dict.fromkeys(committee_codes + GENERAL_COMMITTEE_CODES))

    rows: list[dict[str, Any]] = []
    senator_member = (
        find_member_by_district(state, "senate", senate_district) if senate_district else None
    )
    if senator_member:
        card = ah.member_to_card(
            state,
            senator_member,
            why="",
            relevant_committee_codes=relevant_codes,
        )
        raw_url = (getattr(senator_member, "photo_url", None) or "") or ""
        photo_url = validate_photo_url_for_drawer(raw_url) if raw_url else ""
        rows.append(
            {
                "member_id": str(card["id"]),
                "name": card.get("name") or senator_member.name,
                "role_label": "Your Senator",
                "role_short": "Senator",
                "photo_url": photo_url,
            }
        )
    rep_member = find_member_by_district(state, "house", house_district) if house_district else None
    if rep_member:
        card = ah.member_to_card(
            state,
            rep_member,
            why="",
            relevant_committee_codes=relevant_codes,
        )
        raw_url = (getattr(rep_member, "photo_url", None) or "") or ""
        photo_url = validate_photo_url_for_drawer(raw_url) if raw_url else ""
        rows.append(
            {
                "member_id": str(card["id"]),
                "name": card.get("name") or rep_member.name,
                "role_label": "Your Representative",
                "role_short": "Rep",
                "photo_url": photo_url,
            }
        )
    broker_member, broker_why = ah.find_power_broker(
        state,
        exclude_senate_district=senate_district or "",
        exclude_house_district=house_district or "",
        committee_ids=committee_ids,
        committee_codes=committee_codes or None,
        category_name=topic,
    )
    if broker_member:
        card = ah.member_to_card(
            state,
            broker_member,
            why=broker_why or "",
            relevant_committee_codes=relevant_codes,
        )
        raw_url = (getattr(broker_member, "photo_url", None) or "") or ""
        photo_url = validate_photo_url_for_drawer(raw_url) if raw_url else ""
        rows.append(
            {
                "member_id": str(card["id"]),
                "name": card.get("name") or broker_member.name,
                "role_label": "Power Broker",
                "role_short": "Broker",
                "photo_url": photo_url,
            }
        )
    return rows


@router.get("/admin/mocks", include_in_schema=False)
async def mocks_control_panel(
    request: Request,
    user: User | None = Depends(get_current_user_optional),
    zip: str = "",
):
    """Mock switchboard control panel. DEV_MODE only. ?zip= pre-fills and resolves that ZIP."""
    if not cfg.DEV_MODE:
        return JSONResponse(status_code=404, content={"detail": "Not available"})
    zip_param = (zip or "").strip()
    if _ZIP_RE.match(zip_param) and zip_param in state.zip_to_district:
        zip_code = zip_param
    else:
        zip_code = DEFAULT_MOCK_ZIP
    legislators = _resolve_mock_legislators(zip_code) if zip_code in state.zip_to_district else []
    return templates.TemplateResponse(
        "admin_mocks.html",
        {
            "request": request,
            "zip_code": zip_code,
            "legislators": legislators,
            "current_user_email": user.email if user else None,
            "dev_user_email": MOCK_DEV_USER_EMAIL,
        },
    )


@router.get("/admin/mocks/resolve", include_in_schema=False)
async def mocks_resolve(request: Request, zip: str = ""):
    """HTMX partial: resolve ZIP to switchboard rows. DEV_MODE only."""
    if not cfg.DEV_MODE:
        return JSONResponse(status_code=404, content={"detail": "Not available"})
    zip_code = (zip or "").strip()
    if not zip_code or not _ZIP_RE.match(zip_code):
        return templates.TemplateResponse(
            "admin_mocks_grid.html",
            {
                "request": request,
                "zip_code": zip_code or "",
                "legislators": [],
                "error": "Enter a valid 5-digit ZIP.",
            },
        )
    if zip_code not in state.zip_to_district:
        return templates.TemplateResponse(
            "admin_mocks_grid.html",
            {
                "request": request,
                "zip_code": zip_code,
                "legislators": [],
                "error": f"ZIP {zip_code} not in district data.",
            },
        )
    legislators = _resolve_mock_legislators(zip_code)
    return templates.TemplateResponse(
        "admin_mocks_grid.html",
        {
            "request": request,
            "zip_code": zip_code,
            "legislators": legislators,
            "error": None,
        },
    )


@router.post("/admin/mocks/apply", include_in_schema=False)
async def mocks_apply(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Apply mock outreach state. DEV_MODE only. Uses current user if logged in, else dev user."""
    if not cfg.DEV_MODE:
        return JSONResponse(status_code=404, content={"detail": "Not available"})
    body = await request.json()
    zip_code = (body.get("zip") or "").strip()
    events_payload = body.get("events") or []
    heat_payload = body.get("heat") or {}
    apply_to = (body.get("apply_to") or "").strip().lower()

    if not _ZIP_RE.match(zip_code):
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid zip", "applied": False},
        )

    use_current = apply_to != "dev" and user is not None
    if use_current:
        target_user = user
    else:
        r = await db.execute(select(User).where(User.email == MOCK_DEV_USER_EMAIL))
        target_user = r.scalar_one_or_none()
        if not target_user:
            target_user = User(email=MOCK_DEV_USER_EMAIL)
            db.add(target_user)
            await db.flush()

    await db.execute(delete(OutreachEvent).where(OutreachEvent.user_id == target_user.id))
    await db.flush()

    now = datetime.now(timezone.utc)
    created = 0
    for ev in events_payload:
        member_id = (ev.get("member_id") or "").strip()
        if not member_id:
            continue
        support = ev.get("support_score")
        if support is not None and (support < 1 or support > 5):
            support = 4
        elif support is None:
            support = 4
        contact = (ev.get("contact_name") or "").strip() or None
        constituent = ev.get("constituent", True)
        if ev.get("call"):
            db.add(
                OutreachEvent(
                    user_id=target_user.id,
                    user_email=target_user.email,
                    member_id=member_id,
                    kind="call",
                    zip_code=zip_code,
                    outcome=None,
                    notes="Mock switchboard",
                    contact_name=contact,
                    support_score=support,
                    constituent=constituent,
                    created_at=now,
                )
            )
            created += 1
        if ev.get("email"):
            db.add(
                OutreachEvent(
                    user_id=target_user.id,
                    user_email=target_user.email,
                    member_id=member_id,
                    kind="email",
                    zip_code=zip_code,
                    outcome=None,
                    notes="Mock switchboard",
                    contact_name=contact,
                    support_score=support,
                    constituent=constituent,
                    created_at=now,
                )
            )
            created += 1

    heat_created = 0
    for member_id_str, count in heat_payload.items():
        member_id = (member_id_str or "").strip()
        if not member_id or not isinstance(count, int) or count < 1:
            continue
        for i in range(min(count, 20)):
            email = f"mock_heat_{i + 1}@example.com"
            r = await db.execute(select(User).where(User.email == email))
            heat_user = r.scalar_one_or_none()
            if not heat_user:
                heat_user = User(email=email)
                db.add(heat_user)
                await db.flush()
            db.add(
                OutreachEvent(
                    user_id=heat_user.id,
                    user_email=email,
                    member_id=member_id,
                    kind="call",
                    zip_code=zip_code,
                    outcome=None,
                    notes="Mock heat",
                    contact_name=None,
                    support_score=4,
                    constituent=True,
                    created_at=now - timedelta(days=min(i, 7)),
                )
            )
            heat_created += 1
    await db.commit()
    return {
        "applied": True,
        "applied_to": target_user.email,
        "events_created": created,
        "heat_created": heat_created,
    }
