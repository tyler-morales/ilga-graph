"""Dev-only routes: component playground. All handlers return 404 when not DEV_MODE."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from .. import config as cfg
from ..campaign_config import get_campaign_config
from ..constants import KEI_STATUS_OPTIONS
from ..dev_playground_scenes import get_scene, get_scene_context, get_scenes
from ..routers.content import STRATEGIC_FIVE_POINTS
from ..session_schedule import get_milestone_by_id, get_next_deadline_safe

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
router = APIRouter()
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

from ..campaign_helpers import (  # noqa: E402
    get_current_action_campaign_for_template,
    get_poll_campaign_for_template,
)

templates.env.globals["get_current_action_campaign"] = get_current_action_campaign_for_template
templates.env.globals["get_poll_campaign_for_template"] = get_poll_campaign_for_template
templates.env.globals["get_milestone_by_id"] = get_milestone_by_id
templates.env.globals["get_next_deadline"] = get_next_deadline_safe
templates.env.globals["kei_status_options"] = KEI_STATUS_OPTIONS


def _playground_context(request: Request, scene_id: str | None):
    """Build context for dev_playground.html: scenes, scene_id, scene_html, trigger, scene_label."""
    scenes = get_scenes()
    scene_html = ""
    trigger = None
    scene_label = ""
    if scene_id:
        scene = get_scene(scene_id)
        if scene:
            ctx = get_scene_context(scene, request)
            scene_html = templates.env.get_template(scene["template"]).render(**ctx)
            trigger = scene.get("trigger")
            scene_label = scene.get("label", "")
    return {
        "request": request,
        "scenes": scenes,
        "scene_id": scene_id or "",
        "scene_html": scene_html,
        "trigger": trigger,
        "scene_label": scene_label,
    }


@router.get("/playground", include_in_schema=False)
async def dev_playground(request: Request, scene: str | None = None):
    """Render component playground; optional ?scene=<id>. Returns 404 when not DEV_MODE."""
    if not cfg.DEV_MODE:
        return JSONResponse(status_code=404, content={"detail": "Not available"})
    ctx = _playground_context(request, scene)
    return templates.TemplateResponse(request, "dev_playground.html", ctx)


@router.get("/playground/{scene_id}", include_in_schema=False)
async def dev_playground_scene(request: Request, scene_id: str):
    """Render playground with a specific scene (deep link). Returns 404 when not DEV_MODE."""
    if not cfg.DEV_MODE:
        return JSONResponse(status_code=404, content={"detail": "Not available"})
    ctx = _playground_context(request, scene_id)
    return templates.TemplateResponse(request, "dev_playground.html", ctx)
