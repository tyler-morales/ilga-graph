"""Content pages: The Issue and Legislator Brief."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from .. import config as cfg

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


@router.get("/the-issue", include_in_schema=False)
async def the_issue_page(request: Request):
    """Serve The Issue page: kei vehicle registration problem and how to help."""
    return templates.TemplateResponse(
        "the_issue.html",
        {"request": request},
    )


@router.get("/legislator-brief", include_in_schema=False)
async def legislator_brief_page(request: Request):
    """Serve the Legislator Brief: concise briefing for legislators and staff."""
    return templates.TemplateResponse(
        "legislator_brief.html",
        {"request": request},
    )
