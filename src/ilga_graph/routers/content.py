"""Content pages: The Issue and Legislator Brief."""

from __future__ import annotations

import json
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

# Single source of truth for state table and map. bill_status: "passed" | "pending" | "none".
# state_abbr is two-letter lowercase for SVG map class matching.
BRIEF_STATE_STATUS: list[dict] = [
    {
        "state": "Texas",
        "state_abbr": "tx",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "SB 1816 — miniature vehicle statute; titling, registration, highway rules",
        "status": "Sep 2025",
        "notes": "",
        "bill_url": "https://capitol.texas.gov/BillLookup/History.aspx?LegSess=89R&Bill=SB1816",
        "bill_title": "Texas SB 1816",
    },
    {
        "state": "Colorado",
        "state_abbr": "co",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "HB 25-1281 — kei road-legal framework; 55 mph limit",
        "status": "Jul 2027",
        "notes": "",
        "bill_url": "https://leg.colorado.gov/bills/hb25-1281",
        "bill_title": "Colorado HB 25-1281",
    },
    {
        "state": "Massachusetts",
        "state_abbr": "ma",
        "bill_status": "pending",
        "policy": True,
        "mechanism": "RMV policy update + bill H.4053 (in committee)",
        "status": "Sep 2024",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
    },
    {
        "state": "Michigan",
        "state_abbr": "mi",
        "bill_status": "none",
        "policy": True,
        "mechanism": "SOS policy — title and registration allowed",
        "status": "Nov 2024",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
    },
    {
        "state": "Oregon",
        "state_abbr": "or",
        "bill_status": "pending",
        "policy": False,
        "mechanism": "SB 1213 — title/registration and operating rules (Transportation committee)",
        "status": "In committee",
        "notes": "",
        "bill_url": "https://olis.oregonlegislature.gov/liz/2025R1/Measures/Overview/SB1213",
        "bill_title": "Oregon SB 1213",
    },
    {
        "state": "Maine",
        "state_abbr": "me",
        "bill_status": "pending",
        "policy": False,
        "mechanism": "H.4053 — title/registration and working group (in committee)",
        "status": "In committee",
        "notes": "",
        "bill_url": "https://legislature.maine.gov/legis/bills/display_ps.asp?ld=4053&num=H",
        "bill_title": "Maine H.4053",
    },
]


# Derived for sidebar: passed and current bills (title, url).
def _brief_bills_passed() -> list[dict[str, str]]:
    return [
        {"title": s["bill_title"], "url": s["bill_url"]}
        for s in BRIEF_STATE_STATUS
        if s["bill_status"] == "passed" and s.get("bill_url")
    ]


def _brief_bills_current() -> list[dict[str, str]]:
    return [
        {"title": s["bill_title"], "url": s["bill_url"]}
        for s in BRIEF_STATE_STATUS
        if s["bill_status"] == "pending" and s.get("bill_url")
    ]


BRIEF_BILLS_PASSED: list[dict[str, str]] = _brief_bills_passed()
BRIEF_BILLS_CURRENT: list[dict[str, str]] = _brief_bills_current()

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

    # Map fill by state abbr: bill status takes priority, then policy, then gray.
    def _map_fill_status(s: dict) -> str:
        if s["bill_status"] in ("passed", "pending"):
            return s["bill_status"]
        if s.get("policy"):
            return "policy"
        return "none"

    brief_state_map_status = {s["state_abbr"]: _map_fill_status(s) for s in BRIEF_STATE_STATUS}
    return templates.TemplateResponse(
        "legislator_brief.html",
        {
            "request": request,
            "brief_documents": BRIEF_DOCUMENTS,
            "brief_state_status": BRIEF_STATE_STATUS,
            "brief_state_map_status_json": json.dumps(brief_state_map_status),
            "brief_bills_passed": BRIEF_BILLS_PASSED,
            "brief_bills_current": BRIEF_BILLS_CURRENT,
            "brief_sources": BRIEF_SOURCES,
        },
    )
