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

# Documents listed in the legislator brief sidebar (title, url, file_type for icon).
# Optional: available=False and note="..." for placeholders (disabled style, note under title).
BRIEF_DOCUMENTS = [
    {
        "title": "IL Kei Vehicle Registration Fix Brief",
        "url": "/static/advocacy/IL_Kei_Vehicle_Registration_Fix_Brief.pdf",
        "file_type": "pdf",
    },
    {
        "title": "1 pager sample bill",
        "url": "#",
        "file_type": "pdf",
        "available": False,
        "note": "Being developed.",
    },
]

# Bills in the brief sidebar: passed and current (title, url). From state table on the page.
BRIEF_BILLS_PASSED: list[dict[str, str]] = [
    {
        "title": "Texas SB 1816",
        "url": "https://capitol.texas.gov/BillLookup/History.aspx?LegSess=89R&Bill=SB1816",
    },
    {"title": "Colorado HB 25-1281", "url": "https://leg.colorado.gov/bills/hb25-1281"},
]
BRIEF_BILLS_CURRENT: list[dict[str, str]] = [
    {
        "title": "Oregon SB 1213",
        "url": "https://olis.oregonlegislature.gov/liz/2025R1/Measures/Overview/SB1213",
    },
    {
        "title": "Maine H.4053",
        "url": "https://legislature.maine.gov/legis/bills/display_ps.asp?ld=4053&num=H",
    },
]

# Sources for the brief: title, url. From table primary sources + Illinois statute.
BRIEF_SOURCES: list[dict[str, str]] = [
    {
        "title": "625 ILCS 5/3-401(c-1)",
        "url": "https://www.ilga.gov/legislation/ilcs/ilcs3.asp?ActID=2205&ChapterID=62",
    },
    {
        "title": "Texas SB 1816",
        "url": "https://capitol.texas.gov/BillLookup/History.aspx?LegSess=89R&Bill=SB1816",
    },
    {"title": "Colorado HB 25-1281", "url": "https://leg.colorado.gov/bills/hb25-1281"},
    {
        "title": "Massachusetts RMV",
        "url": "https://www.mass.gov/orgs/massachusetts-registry-of-motor-vehicles",
    },
    {"title": "Michigan SOS", "url": "https://www.michigan.gov/sos"},
    {
        "title": "Oregon SB 1213",
        "url": "https://olis.oregonlegislature.gov/liz/2025R1/Measures/Overview/SB1213",
    },
    {
        "title": "Maine H.4053",
        "url": "https://legislature.maine.gov/legis/bills/display_ps.asp?ld=4053&num=H",
    },
]


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
        {
            "request": request,
            "brief_documents": BRIEF_DOCUMENTS,
            "brief_bills_passed": BRIEF_BILLS_PASSED,
            "brief_bills_current": BRIEF_BILLS_CURRENT,
            "brief_sources": BRIEF_SOURCES,
        },
    )
