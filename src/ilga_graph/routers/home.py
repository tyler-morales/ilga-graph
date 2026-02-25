"""Home page route: shared hero at / with 'Learn about the issue? / for legislators' link."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..db import get_db
from ..routers.advocacy import DEFAULT_HERO_ZIP, _hero_context
from ..routers.content import (
    STRATEGIC_FIVE_POINTS,
    STRATEGIC_MISSION,
    STRATEGIC_VISION,
)
from ..routers.outreach import get_outreach_aggregate

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

router = APIRouter()
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


@router.get("/", include_in_schema=False)
async def home(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Render home page: home hero, links to /the-issue and /legislator-brief; form → /advocacy/."""
    try:
        agg = await get_outreach_aggregate(db)
        calls_total = agg["calls_total"]
        calls_this_week = agg["calls_this_week"]
    except Exception:
        calls_total = 0
        calls_this_week = 0
    ctx: dict[str, Any] = {
        "request": request,
        "title": cfg.SITE_NAME,
        **_hero_context(),
        "calls_total": calls_total,
        "calls_this_week": calls_this_week,
        "zip": (cfg.DEV_MODE and DEFAULT_HERO_ZIP) or "",
        "strategic_mission": STRATEGIC_MISSION,
        "strategic_vision": STRATEGIC_VISION,
        "strategic_five_points": STRATEGIC_FIVE_POINTS,
    }
    return templates.TemplateResponse("home.html", ctx)
