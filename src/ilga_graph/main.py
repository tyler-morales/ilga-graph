from __future__ import annotations

import logging
import random
import sys

import strawberry
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles as BaseStaticFiles
from strawberry.fastapi import GraphQLRouter

from . import config as cfg
from .app_state import state
from .campaign_config import get_campaign_config
from .constants import KEI_STATUS_OPTIONS
from .middleware import register_middleware
from .routers.account import router as _account_router
from .routers.admin import router as _admin_router
from .routers.advocacy import router as _advocacy_router
from .routers.auth import router as _auth_router
from .routers.bills import router as _bills_router
from .routers.campaigns import router as _campaigns_router
from .routers.content import (
    HERO_CLARITY_LINE,
    HERO_URGENCY_LINE,
    STRATEGIC_FIVE_POINTS,
)
from .routers.content import router as _content_router
from .routers.dev import router as _dev_router
from .routers.explore import router as _explore_router
from .routers.feedback import router as _feedback_router
from .routers.home import router as _home_router
from .routers.intelligence import router as _intelligence_router
from .routers.outreach import router as _outreach_router
from .routers.stories import router as _stories_router
from .routers.updates import router as _updates_router
from .session_schedule import get_milestone_by_id, get_next_deadline_safe

# ── Configure logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
    stream=sys.stderr,
    force=True,
)
LOGGER = logging.getLogger(__name__)

from pathlib import Path  # noqa: E402

# ── GraphQL schema (Query lives in graphql_query.py) ─────────────────────────
from strawberry.extensions import QueryDepthLimiter  # noqa: E402

from .graphql_query import (  # noqa: E402, F401 - _member_career_start for tests
    Query,
    _member_career_start,
)
from .loaders import create_loaders  # noqa: E402
from .startup import lifespan  # noqa: E402

# #region agent log
try:
    import json as _json
    import os as _os
    import time as _t

    _path = _os.path.join(_os.environ.get("TMPDIR", "/tmp"), "debug-d3a55a.log")
    with open(_path, "a") as _f:
        payload = {
            "sessionId": "d3a55a",
            "location": "main.py",
            "message": "main_loaded",
            "hypothesisId": "H3",
            "data": {},
            "timestamp": int(_t.time() * 1000),
        }
        _f.write(_json.dumps(payload) + "\n")
except Exception:
    pass
# #endregion


async def get_graphql_context() -> dict:
    """Request-scoped context with state and batch loaders for GraphQL."""
    return create_loaders(state)


schema = strawberry.Schema(
    query=Query,
    extensions=[QueryDepthLimiter(max_depth=10)],
)
graphql_app = GraphQLRouter(schema, context_getter=get_graphql_context)

app = FastAPI(title="ILGA Graph", lifespan=lifespan)

# ── Static files & Jinja2 templates ──────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATE_DIR = Path(__file__).parent / "templates"


class StaticFilesWithCache(BaseStaticFiles):
    """StaticFiles that sets Cache-Control for repeat visits. Unversioned assets use 1h."""

    async def get_response(self, path: str, scope: dict) -> Response:
        response = await super().get_response(path, scope)
        response.headers.setdefault("Cache-Control", "public, max-age=3600")
        return response


app.mount("/static", StaticFilesWithCache(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
app.state.templates = templates

# Dev bar is available when running in dev profile (never rendered in prod)
templates.env.globals["dev_available"] = cfg.DEV_MODE
# SEO, share cards, and analytics (base template uses these)
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["site_name"] = cfg.SITE_NAME
templates.env.globals["meta_description"] = cfg.META_DESCRIPTION
_campaign = get_campaign_config()
templates.env.globals["campaign_name"] = _campaign.campaign_name or cfg.SITE_NAME
templates.env.globals["primary_color"] = _campaign.primary_color or "#FF4500"
templates.env.globals["issue_summary"] = _campaign.issue_summary
templates.env.globals["og_image_url"] = cfg.OG_IMAGE_URL
# Umami script only in prod (and only when website ID is set)
templates.env.globals["umami_enabled"] = cfg.PROFILE == "prod" and bool(cfg.UMAMI_WEBSITE_ID)
templates.env.globals["umami_website_id"] = cfg.UMAMI_WEBSITE_ID
templates.env.globals["umami_script_url"] = cfg.UMAMI_SCRIPT_URL
templates.env.globals["show_beta_banner"] = cfg.BETA_BANNER
templates.env.globals["beta_banner_feedback_url"] = cfg.BETA_BANNER_REPORT_URL
templates.env.globals["footer_last_updated"] = cfg.FOOTER_LAST_UPDATED
templates.env.globals["footer_last_updated_iso"] = cfg.FOOTER_LAST_UPDATED_ISO
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS
templates.env.globals["hero_urgency_line"] = HERO_URGENCY_LINE
templates.env.globals["hero_clarity_line"] = HERO_CLARITY_LINE
templates.env.globals["features"] = cfg.get_client_features()
templates.env.globals["kei_status_options"] = KEI_STATUS_OPTIONS


templates.env.globals["get_next_deadline"] = get_next_deadline_safe
templates.env.globals["get_milestone_by_id"] = get_milestone_by_id


def _get_current_action_campaign(request: Request) -> object | None:
    """Return active campaign for the request (set by middleware) for base template top bar."""
    return getattr(request.state, "current_action_campaign", None)


templates.env.globals["get_current_action_campaign"] = _get_current_action_campaign


def _get_current_action_campaign(request: Request) -> object | None:
    """Return active campaign for the request (set by middleware) for base template top bar."""
    return getattr(request.state, "current_action_campaign", None)


templates.env.globals["get_current_action_campaign"] = _get_current_action_campaign


def _wants_html(request: Request) -> bool:
    """True if the client prefers an HTML response (e.g. browser navigation)."""
    accept = request.headers.get("accept", "")
    return "text/html" in accept


# Fun facts about Kei (軽) vehicles for the 404 page; one is chosen at random per request.
# Each fact is a dict: "text", and optionally "image" (URL), "image_alt", "image_credit" (a11y).
KEI_VEHICLE_FACTS: tuple[dict[str, str | None], ...] = (
    {
        "text": (
            "The 1999 generation of the Suzuki Carry introduced modern safety standards "
            "into the micro-truck class, including designated crumple zones, making them "
            "vastly safer than many older UTVs currently allowed on US roads."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/1999_Suzuki_Carry_1.3.jpg",
        "image_alt": "White 1999 Suzuki Carry truck.",
        "image_credit": "Rutger van der Maar, CC BY 2.0, Wikimedia Commons",
    },
    {
        "text": (
            "Because of their mid-engine layout, trucks like the Honda Acty have a "
            "remarkably low center of gravity, which prevents the massive rollover risk "
            "associated with modern lifted US pickup trucks."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/1999-present_Honda_Acty_(rear).jpg",
        "image_alt": "Rear angle of a white Honda Acty truck showing its low center of gravity.",
        "image_credit": "Niels de Wit, CC BY 2.0, Wikimedia Commons",
    },
    {
        "text": (
            "Despite their small 660cc engines, models like the Subaru Sambar are "
            "engineered to travel on Japanese expressways and can comfortably cruise at "
            "55 mph, easily handling local US traffic."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Subaru_SAMBAR_TRUCK_TA_(3BA-S510J).jpg",
        "image_alt": "Subaru Sambar Truck parked on a street.",
        "image_credit": "Tokumeigakarinoaoshima, CC0, Wikimedia Commons",
    },
    {
        "text": (
            "A standard Mitsubishi Minicab weighs around 1,500 pounds but has a legal "
            "payload capacity of 350 kg (771 lbs), meaning it can safely haul half its "
            "own weight without destroying municipal road infrastructure."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Mitsubishi_MINICAB_TRUCK_M_(DS16T)_front.JPG",
        "image_alt": "Front view of a white Mitsubishi Minicab Truck.",
        "image_credit": "Tokumeigakarinoaoshima, CC0, Wikimedia Commons",
    },
    {
        "text": (
            "Being only 4.2 feet wide, single-seater Kei trucks like the Daihatsu Midget II "
            "take up significantly less space, making them incredibly safe for pedestrians "
            "and cyclists in dense urban residential neighborhoods."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Daihatsu_Midget_II_(8101380599).jpg",
        "image_alt": "Green Daihatsu Midget II single-seater micro truck parked in a city.",
        "image_credit": "dave_7, CC BY 2.0, Wikimedia Commons",
    },
    {
        "text": (
            "Far from being unregulated, Kei engineering is perfectly legal in highly "
            "regulated regions with lower roadway fatality rates than the US. The Piaggio "
            "Porter is actually a European-built, street-legal version of the Daihatsu Hijet."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Piaggio_Porter_(31945694857).jpg",
        "image_alt": "Piaggio Porter (European Daihatsu Hijet variant) parked outdoors.",
        "image_credit": "Guillaume Vachey, CC0, Wikimedia Commons",
    },
    {
        "text": (
            "Because imported 25-year-old Kei trucks like the Honda Acty must have passed "
            "Japan's notoriously strict 'Shaken' inspections every two years, they arrive "
            "in the US meticulously maintained."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Honda_Acty,_EMS_2023,_Essen_(P1160528).jpg",
        "image_alt": "A heavily customized and well-maintained Honda Acty at an auto show.",
        "image_credit": "Matti Blume, CC BY-SA 4.0, Wikimedia Commons",
    },
    {
        "text": (
            "Across the US, 6th generation Mitsubishi Minicabs and similar Kei trucks are "
            "heavily utilized by universities and state parks for groundskeeping because "
            "they are fully enclosed, street-capable, and more robust than unregulated "
            "golf carts."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Mitsubishi_Minicab_truck_(sixth_generation).JPG",
        "image_alt": "Sixth-generation Mitsubishi Minicab utility truck.",
        "image_credit": "Tokumeigakarinoaoshima, CC0, Wikimedia Commons",
    },
    {
        "text": (
            "Thanks to their lightweight construction and small-displacement engines, "
            "vehicles like the Daihatsu Hijet typically achieve an average of 40 to 50 "
            "miles per gallon (mpg), offering a highly fuel-efficient and environmentally "
            "friendly alternative to full-size delivery vans."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Daihatsu_HiJet_(4500946659).jpg",
        "image_alt": "Daihatsu Hijet microvan parked.",
        "image_credit": "Brian Snelson, CC BY 2.0, Wikimedia Commons",
    },
    {
        "text": (
            "Most Kei-class 4x4s, such as the Suzuki Jimny, feature push-button 4WD, "
            "differential locks, and ultra-low crawler gears, making them exceptionally "
            "capable and safe in snow and rural mud."
        ),
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Suzuki_Jimny_2018_(04).jpg",
        "image_alt": "Modern Suzuki Jimny 4x4 off-road vehicle.",
        "image_credit": "Ery, CC BY-SA 4.0, Wikimedia Commons",
    },
)


def _error_page_context(request: Request) -> dict:
    """Context for error pages (404, 500): request plus a random Kei vehicle fact."""
    return {"request": request, "kei_fact": random.choice(KEI_VEHICLE_FACTS)}


def _404_context(request: Request) -> dict:
    """Context for the 404 template: request plus a random Kei vehicle fact."""
    return _error_page_context(request)


# ── Exception handlers (custom error pages + consistent JSON for API) ─────────
async def _http_exception_handler(request: Request, exc: HTTPException) -> Response:
    if _wants_html(request):
        if exc.status_code == 401:
            from urllib.parse import quote

            next_path = request.url.path
            if request.query_params:
                next_path = f"{next_path}?{request.query_params}"
            if next_path.startswith("/admin") and next_path != "/admin/login":
                return RedirectResponse(
                    url=f"/admin/login?next={quote(next_path, safe='')}", status_code=302
                )
            return RedirectResponse(url=f"/?next={quote(next_path, safe='')}", status_code=302)
        if exc.status_code == 403:
            if request.url.path.startswith("/admin"):
                return RedirectResponse(url="/admin/login?error=forbidden", status_code=302)
            return templates.TemplateResponse("403.html", {"request": request}, status_code=403)
        if exc.status_code == 404:
            return templates.TemplateResponse("404.html", _404_context(request), status_code=404)
        if exc.status_code >= 500:
            return templates.TemplateResponse(
                "500.html", _error_page_context(request), status_code=exc.status_code
            )
    detail = exc.detail if isinstance(exc.detail, (str, dict, list)) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Request validation failed: %s", exc.errors())
    if _wants_html(request):
        return templates.TemplateResponse("422.html", {"request": request}, status_code=422)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


async def _uncaught_exception_handler(request: Request, exc: Exception) -> Response:
    LOGGER.exception("Uncaught exception while handling %s %s", request.method, request.url.path)
    if _wants_html(request):
        return templates.TemplateResponse("500.html", _error_page_context(request), status_code=500)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.add_exception_handler(HTTPException, _http_exception_handler)
app.add_exception_handler(RequestValidationError, _validation_exception_handler)
app.add_exception_handler(Exception, _uncaught_exception_handler)


register_middleware(app)

app.include_router(graphql_app, prefix="/graphql")
app.include_router(_home_router)
app.include_router(_admin_router)
app.include_router(_campaigns_router)
app.include_router(_dev_router, prefix="/dev")

app.include_router(_advocacy_router, prefix="/advocacy")
app.include_router(_auth_router)
app.include_router(_account_router)
app.include_router(_content_router)
app.include_router(_stories_router)
app.include_router(_updates_router)
app.include_router(_feedback_router)
app.include_router(_bills_router, prefix="/api")
app.include_router(_explore_router)
app.include_router(_intelligence_router, prefix="/intelligence")
app.include_router(_outreach_router)


# ── Catch-all for unmatched paths (Starlette does not invoke 404 handler) ─────
@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def _catch_all_404(request: Request, full_path: str) -> Response:
    """Return custom 404 page or JSON for any path that did not match a route."""
    if _wants_html(request):
        return templates.TemplateResponse("404.html", _404_context(request), status_code=404)
    return JSONResponse(status_code=404, content={"detail": "Not found"})
