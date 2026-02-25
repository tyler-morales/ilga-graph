"""Legal pages: Privacy policy and Terms of use."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from .. import config as cfg
from ..routers.content import STRATEGIC_FIVE_POINTS

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
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS
templates.env.globals["features"] = cfg.get_client_features()


@router.get("/privacy", include_in_schema=False)
async def privacy_page(request: Request):
    """Serve the Privacy policy page."""
    return templates.TemplateResponse(
        "privacy.html",
        {"request": request},
    )


@router.get("/terms", include_in_schema=False)
async def terms_page(request: Request):
    """Serve the Terms of use page."""
    return templates.TemplateResponse(
        "terms.html",
        {"request": request},
    )
