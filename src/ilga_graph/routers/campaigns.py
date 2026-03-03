"""Admin CRUD for campaigns (action alerts)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..app_state import state
from ..campaign_config import get_campaign_config
from ..campaign_helpers import campaign_outreach_count, deactivate_other_campaigns
from ..constants import KEI_STATUS_OPTIONS
from ..db import get_db
from ..db_models import Campaign, OutreachEvent
from ..dependencies import require_admin
from ..routers.content import STRATEGIC_FIVE_POINTS
from ..session_schedule import (
    get_deadlines_for_campaigns,
    get_milestone_by_id,
    get_next_deadline_safe,
)

router = APIRouter()
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["dev_available"] = cfg.DEV_MODE
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["site_name"] = cfg.SITE_NAME
_campaign = get_campaign_config()
templates.env.globals["campaign_name"] = _campaign.campaign_name or cfg.SITE_NAME
templates.env.globals["primary_color"] = _campaign.primary_color or "#e55a1a"
templates.env.globals["issue_summary"] = _campaign.issue_summary
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

from ..campaign_helpers import get_current_action_campaign_for_template  # noqa: E402

templates.env.globals["get_current_action_campaign"] = get_current_action_campaign_for_template
templates.env.globals["get_milestone_by_id"] = get_milestone_by_id
templates.env.globals["get_next_deadline"] = get_next_deadline_safe
templates.env.globals["kei_status_options"] = KEI_STATUS_OPTIONS


def _district_options_from_zip_crosswalk() -> list[tuple[str, str]]:
    """Build (value, label) list for district multi-select from state.zip_to_district."""
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for info in state.zip_to_district.values():
        for prefix, num in [
            ("il_house", info.il_house),
            ("il_senate", info.il_senate),
        ]:
            key = f"{prefix}:{num}"
            if key not in seen:
                seen.add(key)
                label = f"IL {prefix.replace('_', ' ').title()} {num}"
                options.append((key, label))
    options.sort(key=lambda x: (x[0].startswith("il_senate"), x[0]))
    return options


async def _campaign_outreach_breakdown(db: AsyncSession, campaign_id: int) -> dict[str, Any]:
    """Return counts by kind and distinct users for a campaign."""
    total = await campaign_outreach_count(db, campaign_id)
    by_kind_q = (
        select(OutreachEvent.kind, func.count())
        .where(OutreachEvent.campaign_id == campaign_id)
        .group_by(OutreachEvent.kind)
    )
    r = await db.execute(by_kind_q)
    by_kind = dict(r.all())
    users_q = (
        select(func.count(func.distinct(OutreachEvent.user_id)))
        .where(OutreachEvent.campaign_id == campaign_id)
        .where(OutreachEvent.user_id.isnot(None))
    )
    r2 = await db.execute(users_q)
    unique_users = r2.scalar() or 0
    return {
        "total": total,
        "by_kind": by_kind,
        "unique_users": unique_users,
    }


@router.get("/admin/campaigns", include_in_schema=False)
async def admin_campaigns_list(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin_user: Any = Depends(require_admin),
):
    """List all campaigns (active first, then by created_at desc)."""
    q = select(Campaign).order_by(Campaign.is_active.desc(), Campaign.created_at.desc())
    result = await db.execute(q)
    campaigns = list(result.scalars().all())
    counts = {}
    for c in campaigns:
        counts[c.id] = await campaign_outreach_count(db, c.id)
    flash = request.query_params.get("flash", "")
    return templates.TemplateResponse(
        "admin_campaigns.html",
        {
            "request": request,
            "campaigns": campaigns,
            "counts": counts,
            "flash": flash,
        },
    )


def _campaign_milestones() -> list[dict]:
    """Session milestones for campaign dropdown (end-at deadline)."""
    try:
        return get_deadlines_for_campaigns()
    except (FileNotFoundError, ValueError):
        return []


@router.get("/admin/campaigns/new", include_in_schema=False)
async def admin_campaigns_new(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin_user: Any = Depends(require_admin),
):
    """Create campaign form."""
    district_options = _district_options_from_zip_crosswalk()
    return templates.TemplateResponse(
        "admin_campaign_form.html",
        {
            "request": request,
            "campaign": None,
            "district_options": district_options,
            "district_ids_plain": "",
            "member_ids_plain": "",
            "members": state.members,
            "campaign_milestones": _campaign_milestones(),
        },
    )


@router.post("/admin/campaigns", include_in_schema=False)
async def admin_campaigns_create(
    request: Request,
    title: str = Form(..., min_length=1, max_length=200),
    message: str = Form(..., min_length=1),
    ask: str = Form(..., min_length=1, max_length=100),
    target_type: str = Form("all"),
    target_district_ids: str = Form(""),
    target_member_ids: str = Form(""),
    is_active: str = Form("0"),
    start_at: str = Form(""),
    end_at: str = Form(""),
    session_milestone_id: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user: Any = Depends(require_admin),
):
    """Create a new campaign. If is_active=1, deactivate others."""
    title = title.strip()
    message = message.strip()
    ask = ask.strip()
    target_type = "by_district" if target_type == "by_district" else "all"
    district_json = None
    if target_type == "by_district" and target_district_ids.strip():
        raw = [x.strip() for x in target_district_ids.replace(",", "\n").split() if x.strip()]
        if raw:
            district_json = json.dumps(raw)
    member_json = None
    if target_member_ids.strip():
        raw = [x.strip() for x in target_member_ids.replace(",", "\n").split() if x.strip()]
        if raw:
            member_json = json.dumps(raw[:500])
    start_dt = _parse_naive_dt(start_at)
    end_dt = _parse_naive_dt(end_at)
    mid = session_milestone_id.strip() or None
    if mid and not end_dt:
        milestone = get_milestone_by_id(mid)
        if milestone:
            end_dt = _end_at_from_milestone_date(milestone.get("date"))
    active = is_active.strip() in ("1", "true", "on", "yes")
    campaign = Campaign(
        title=title,
        message=message,
        ask=ask,
        target_type=target_type,
        target_district_ids=district_json,
        target_member_ids=member_json,
        is_active=active,
        start_at=start_dt,
        end_at=end_dt,
        session_milestone_id=mid,
    )
    db.add(campaign)
    await db.flush()
    if active:
        await deactivate_other_campaigns(db, campaign.id)
    await db.commit()
    return RedirectResponse("/admin/campaigns?flash=created", status_code=303)


def _end_at_from_milestone_date(date_str: str) -> datetime | None:
    """Return end-of-day UTC for the given YYYY-MM-DD (campaign ends before this deadline)."""
    s = (date_str or "").strip()
    if not s or len(s) < 10:
        return None
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        return dt.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_naive_dt(s: str) -> datetime | None:
    """Parse optional date or datetime string; return timezone-aware UTC or None."""
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _campaign_district_ids_plain(campaign: Campaign) -> str:
    """Return target_district_ids as newline-separated string for textarea."""
    if not campaign or not campaign.target_district_ids:
        return ""
    try:
        data = json.loads(campaign.target_district_ids)
        return "\n".join(str(x) for x in data) if isinstance(data, list) else ""
    except (json.JSONDecodeError, TypeError):
        return ""


def _campaign_member_ids_plain(campaign: Campaign) -> str:
    """Return target_member_ids as newline-separated string for textarea."""
    if not campaign or not campaign.target_member_ids:
        return ""
    try:
        data = json.loads(campaign.target_member_ids)
        return "\n".join(str(x) for x in data) if isinstance(data, list) else ""
    except (json.JSONDecodeError, TypeError):
        return ""


@router.get("/admin/campaigns/{campaign_id:int}", include_in_schema=False)
async def admin_campaign_detail(
    request: Request,
    campaign_id: int,
    db: AsyncSession = Depends(get_db),
    admin_user: Any = Depends(require_admin),
):
    """Campaign detail/edit form with outreach count and breakdown."""
    result = await db.execute(select(Campaign).where(Campaign.id == campaign_id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        return RedirectResponse("/admin/campaigns?flash=not_found", status_code=303)
    breakdown = await _campaign_outreach_breakdown(db, campaign_id)
    district_options = _district_options_from_zip_crosswalk()
    flash = request.query_params.get("flash", "")
    return templates.TemplateResponse(
        "admin_campaign_form.html",
        {
            "request": request,
            "campaign": campaign,
            "breakdown": breakdown,
            "district_options": district_options,
            "district_ids_plain": _campaign_district_ids_plain(campaign),
            "member_ids_plain": _campaign_member_ids_plain(campaign),
            "members": state.members,
            "flash": flash,
            "campaign_milestones": _campaign_milestones(),
        },
    )


@router.post("/admin/campaigns/{campaign_id:int}", include_in_schema=False)
async def admin_campaign_update(
    request: Request,
    campaign_id: int,
    title: str = Form(..., min_length=1, max_length=200),
    message: str = Form(..., min_length=1),
    ask: str = Form(..., min_length=1, max_length=100),
    target_type: str = Form("all"),
    target_district_ids: str = Form(""),
    target_member_ids: str = Form(""),
    is_active: str = Form("0"),
    start_at: str = Form(""),
    end_at: str = Form(""),
    session_milestone_id: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user: Any = Depends(require_admin),
):
    """Update campaign. If is_active=1, deactivate others."""
    result = await db.execute(select(Campaign).where(Campaign.id == campaign_id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        return RedirectResponse("/admin/campaigns?flash=not_found", status_code=303)
    campaign.title = title.strip()
    campaign.message = message.strip()
    campaign.ask = ask.strip()
    campaign.target_type = "by_district" if target_type == "by_district" else "all"
    if campaign.target_type == "by_district" and target_district_ids.strip():
        raw = [x.strip() for x in target_district_ids.replace(",", "\n").split() if x.strip()]
        campaign.target_district_ids = json.dumps(raw) if raw else None
    else:
        campaign.target_district_ids = None
    if target_member_ids.strip():
        raw = [x.strip() for x in target_member_ids.replace(",", "\n").split() if x.strip()]
        campaign.target_member_ids = json.dumps(raw[:500]) if raw else None
    else:
        campaign.target_member_ids = None
    campaign.session_milestone_id = session_milestone_id.strip() or None
    end_dt = _parse_naive_dt(end_at)
    if campaign.session_milestone_id and not end_dt:
        milestone = get_milestone_by_id(campaign.session_milestone_id)
        if milestone:
            end_dt = _end_at_from_milestone_date(milestone.get("date"))
    campaign.start_at = _parse_naive_dt(start_at)
    campaign.end_at = end_dt
    active = is_active.strip() in ("1", "true", "on", "yes")
    campaign.is_active = active
    if active:
        await deactivate_other_campaigns(db, campaign.id)
    await db.commit()
    return RedirectResponse(f"/admin/campaigns/{campaign_id}?flash=updated", status_code=303)


@router.post("/admin/campaigns/{campaign_id:int}/activate", include_in_schema=False)
async def admin_campaign_activate(
    request: Request,
    campaign_id: int,
    db: AsyncSession = Depends(get_db),
    admin_user: Any = Depends(require_admin),
):
    """Set this campaign active and deactivate others."""
    result = await db.execute(select(Campaign).where(Campaign.id == campaign_id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        return RedirectResponse("/admin/campaigns?flash=not_found", status_code=303)
    campaign.is_active = True
    await deactivate_other_campaigns(db, campaign_id)
    await db.commit()
    return RedirectResponse(f"/admin/campaigns/{campaign_id}?flash=activated", status_code=303)


@router.post("/admin/campaigns/{campaign_id:int}/deactivate", include_in_schema=False)
async def admin_campaign_deactivate(
    request: Request,
    campaign_id: int,
    db: AsyncSession = Depends(get_db),
    admin_user: Any = Depends(require_admin),
):
    """Set this campaign inactive."""
    result = await db.execute(select(Campaign).where(Campaign.id == campaign_id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        return RedirectResponse("/admin/campaigns?flash=not_found", status_code=303)
    campaign.is_active = False
    await db.commit()
    return RedirectResponse(f"/admin/campaigns/{campaign_id}?flash=deactivated", status_code=303)
