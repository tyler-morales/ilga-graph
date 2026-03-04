"""Home and site-level routes: /, favicon, sitemap, robots, advocacy/intelligence redirects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..campaign_config import get_campaign_config
from ..constants import KEI_POLL_IMPACT_OPTIONS, KEI_STATUS_OPTIONS
from ..db import get_db
from ..db_models import User
from ..dependencies import get_current_user_optional
from ..kei_poll_context import (
    _get_kei_status_results,
    get_kei_poll_initial_state,
    zip_known_for_user,
)
from ..routers.advocacy import DEFAULT_HERO_ZIP, _hero_context
from ..routers.content import (
    HERO_URGENCY_LINE,
    KEI_POLL_WIDE_NET_LINE,
    STRATEGIC_FIVE_POINTS,
    STRATEGIC_MISSION,
    STRATEGIC_VISION,
    WHY_SHOULD_YOU_CARE_HEADING,
    WHY_SHOULD_YOU_CARE_INTRO,
    WHY_SHOULD_YOU_CARE_TEASER_ITEMS,
    WHY_SHOULD_YOU_CARE_VOICE,
    WHY_YOU_CARE_DEFAULT_CARDS,
    WHY_YOU_CARE_PRE_POLL_LINE,
    get_marquee_items,
    get_strategic_states_tooltips,
)
from ..routers.content_constants import get_why_you_care_branch_for_selection
from ..routers.outreach import get_outreach_aggregate
from ..session_schedule import get_milestone_by_id, get_next_deadline_safe

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_SITEMAP_PATHS = (
    "/",
    "/advocacy",
    "/intelligence/",
    "/explore",
    "/the-issue",
    "/legislator-brief",
    "/fact-sheet",
    "/glossary",
    "/privacy",
    "/terms",
)

router = APIRouter()


def _redirect_with_query(path: str, request: Request) -> RedirectResponse:
    """Redirect to path, preserving query string (e.g. ?zip=60614) so members load."""
    url = path if not request.url.query else f"{path}?{request.url.query}"
    return RedirectResponse(url=url, status_code=302)


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
templates.env.globals["hero_urgency_line"] = HERO_URGENCY_LINE
templates.env.globals["features"] = cfg.get_client_features()
templates.env.globals["marquee_items"] = []  # Overridden per-request when db available
templates.env.globals["why_should_you_care_heading"] = WHY_SHOULD_YOU_CARE_HEADING
templates.env.globals["why_should_you_care_intro"] = WHY_SHOULD_YOU_CARE_INTRO
templates.env.globals["why_should_you_care_teaser_items"] = WHY_SHOULD_YOU_CARE_TEASER_ITEMS
templates.env.globals["why_should_you_care_voice"] = WHY_SHOULD_YOU_CARE_VOICE

from ..campaign_helpers import get_current_action_campaign_for_template  # noqa: E402

templates.env.globals["get_current_action_campaign"] = get_current_action_campaign_for_template
templates.env.globals["get_milestone_by_id"] = get_milestone_by_id
templates.env.globals["get_next_deadline"] = get_next_deadline_safe
templates.env.globals["kei_status_options"] = KEI_STATUS_OPTIONS
templates.env.globals["kei_impact_options"] = KEI_POLL_IMPACT_OPTIONS
templates.env.globals["why_you_care_default_cards"] = WHY_YOU_CARE_DEFAULT_CARDS
templates.env.globals["why_you_care_pre_poll_line"] = WHY_YOU_CARE_PRE_POLL_LINE
templates.env.globals["turnstile_site_key"] = (
    "" if cfg.TURNSTILE_DISABLED else (cfg.TURNSTILE_SITE_KEY or "")
)


@router.get("/advocacy", include_in_schema=False)
def advocacy_trailing_slash_redirect(request: Request) -> RedirectResponse:
    """Ensure /advocacy is served: mounted router receives path ''; redirect so child sees '/'."""
    return _redirect_with_query("/advocacy/", request)


@router.get("/intelligence", include_in_schema=False)
def intelligence_trailing_slash_redirect(request: Request) -> RedirectResponse:
    """Redirect /intelligence → /intelligence/ so sitemap/bookmarks don't 404."""
    return _redirect_with_query("/intelligence/", request)


@router.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Letter avatar at /favicon.ico for email clients; site tab icon remains truck in base."""
    path = _STATIC_DIR / "favicon-email.svg"
    return FileResponse(path, media_type="image/svg+xml")


@router.get("/sitemap.xml", include_in_schema=False)
def sitemap_xml() -> Response:
    """Serve sitemap.xml for search engine discovery; URLs use APP_BASE_URL."""
    base = cfg.APP_BASE_URL
    urls = "".join(f"    <url><loc>{base}{path}</loc></url>\n" for path in _SITEMAP_PATHS)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}"
        "</urlset>\n"
    )
    return Response(content=xml, media_type="application/xml; charset=utf-8")


@router.get("/robots.txt", include_in_schema=False)
def robots_txt() -> PlainTextResponse:
    """Serve robots.txt (allow all, point to sitemap)."""
    body = f"User-agent: *\nAllow: /\nSitemap: {cfg.APP_BASE_URL}/sitemap.xml\n"
    return PlainTextResponse(content=body)


@router.get("/", include_in_schema=False)
async def home(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
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
        "strategic_mission": get_campaign_config().strategic_mission or STRATEGIC_MISSION,
        "strategic_vision": STRATEGIC_VISION,
        "strategic_five_points": STRATEGIC_FIVE_POINTS,
        "strategic_states_tooltips": get_strategic_states_tooltips(),
        "why_you_care_variant": "home",
    }
    poll_state = await get_kei_poll_initial_state(request, user, db)
    ctx.update(poll_state)
    ctx["poll_id"] = "home-kei-poll"
    ctx["marquee_items"] = await get_marquee_items(db)
    if ctx.get("kei_poll_done") and ctx.get("kei_status_selected"):
        slug = ctx["kei_status_selected"]
        branch_slug = "owner" if slug in ("registered", "revoked", "denied") else slug
        ctx["why_you_care_branch"] = get_why_you_care_branch_for_selection(slug)
        ctx["wyc_pill_icon_slug"] = (
            slug if slug in ("registered", "revoked", "denied") else branch_slug
        )
    if not ctx.get("kei_poll_done"):
        results = await _get_kei_status_results(db)
        ctx["kei_status_total"] = results["total_responses"]
    ctx["kei_poll_wide_net_line"] = KEI_POLL_WIDE_NET_LINE
    ctx["zip_known"] = zip_known_for_user(user)
    ctx["prefill_zip"] = (user.zip_code or "").strip() if user else ""
    return templates.TemplateResponse(request, "home.html", ctx)
