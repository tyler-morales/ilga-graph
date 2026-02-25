"""Content pages: The Issue and Legislator Brief."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from .. import config as cfg

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/ilga_graph/routers -> repo root
router = APIRouter()

# Canonical brief text files (repo root). Web pages source content from these.
CONSTITUENT_BRIEF_PATH = _REPO_ROOT / "Illinois_Kei_Vehicle_Registration_Constituent_Brief.txt"
LEGISLATOR_BRIEF_PATH = _REPO_ROOT / "IL_Kei_Vehicle_Registration_Fix_Brief 1.txt"


def _load_constituent_brief() -> dict | None:
    """Parse constituent brief .txt into title, subtitle, and sections. Returns None if file missing."""
    if not CONSTITUENT_BRIEF_PATH.is_file():
        return None
    text = CONSTITUENT_BRIEF_PATH.read_text(encoding="utf-8", errors="replace").strip()
    parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if len(parts) < 2:
        return None
    title = parts[0]
    subtitle = parts[1] if len(parts) > 1 else ""
    sections: list[dict] = []
    i = 2
    while i + 1 < len(parts):
        heading = parts[i]
        body = parts[i + 1]
        bullets: list[str] = []
        paragraphs: list[str] = []
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Bullet: starts with • (U+2022) or "- " / "• "
            if line.startswith("•") or re.match(r"^[-•]\s*", line):
                bullets.append(line.lstrip("••-	 ").strip())
            else:
                paragraphs.append(line)
        section: dict = {"heading": heading, "paragraphs": paragraphs}
        if bullets:
            section["bullets"] = bullets
        sections.append(section)
        i += 2
    return {"title": title, "subtitle": subtitle, "sections": sections}


def _load_legislator_brief() -> dict | None:
    """Parse legislator brief .txt into structured fields. Returns None if file missing."""
    if not LEGISLATOR_BRIEF_PATH.is_file():
        return None
    text = LEGISLATOR_BRIEF_PATH.read_text(encoding="utf-8", errors="replace").strip()
    lines = text.split("\n")
    out: dict = {
        "title": "",
        "subtitle": "",
        "issue_one_sentence": "",
        "core_ambiguity": "",
        "sections": [],
        "ask_list": [],
        "attachments": "",
        "statutory_ref": "",
        "point_of_contact": "",
    }
    i = 0
    if i < len(lines):
        out["title"] = lines[i].strip()
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines):
        out["subtitle"] = lines[i].strip()
        i += 1
    while i < len(lines):
        line = lines[i]
        if line.startswith("Issue in one sentence:"):
            out["issue_one_sentence"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.startswith("Core ambiguity:"):
            out["core_ambiguity"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.startswith("Illinois statutory reference:"):
            out["statutory_ref"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.startswith("Point of contact:"):
            out["point_of_contact"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.strip() == "Attachments":
            i += 1
            attach_lines = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not lines[i].startswith("Illinois ")
                and not lines[i].startswith("Point of ")
            ):
                attach_lines.append(lines[i].strip())
                i += 1
            out["attachments"] = "\n".join(attach_lines)
            continue
        if line.strip() == "What we are asking your office to do":
            i += 1
            ask_items = []
            while i < len(lines) and lines[i].strip():
                rest = re.sub(r"^\d\)\s*", "", lines[i].strip())
                if rest:
                    ask_items.append(rest)
                i += 1
            out["ask_list"] = ask_items
            continue
        # Section heading (title case, no colon at end)
        stripped = line.strip()
        if (
            stripped
            and i + 1 < len(lines)
            and not stripped.startswith(("Issue in", "Core ambiguity", "Illinois ", "Point of "))
        ):
            section_body: list[str] = []
            i += 1
            while (
                i < len(lines)
                and lines[i].strip()
                and not re.match(
                    r"^(What we are asking|Attachments|Illinois statutory|Point of contact)",
                    lines[i],
                )
            ):
                section_body.append(lines[i].strip())
                i += 1
            if section_body:
                out["sections"].append({"heading": stripped, "paragraphs": section_body})
            continue
        i += 1
    return out


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

# Strategic plan (Hardball Ch 5): single source of truth for mission, vision, 5-point message.
STRATEGIC_MISSION = "Fix the statutory gap that prevents road-legal kei vehicles in Illinois."
STRATEGIC_VISION = (
    "A narrow statutory clarification to 625 ILCS 5/3-401(c-1) so highway-built, federally lawful "
    "kei vehicles can be titled and registered in Illinois, consistent with 21+ other states."
)
STRATEGIC_FIVE_POINTS: list[str] = [
    "Kei vehicles are federally legal to import (25-year rule).",
    "21+ states already allow registration—Illinois is the outlier.",
    "The current Illinois statute has an ambiguity, not a prohibition.",
    "The fix is a narrow clarifying amendment—no new regulatory framework.",
    "This affects real Illinois residents who own legal vehicles they can't register.",
]
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS
templates.env.globals["features"] = cfg.get_client_features()
# How we measure advocacy success (things we control). Road-legal outcome is the campaign objective.
STRATEGIC_SUCCESS_MEASURE = (
    "We measure success by what we can control: constituent contacts, co-sponsors secured, "
    "witness slips filed, and a coalition ready to act when a bill moves."
)
STRATEGIC_SUCCESS_MEASURE_ITEMS: list[str] = [
    "Constituent contacts",
    "Co-sponsors secured",
    "Witness slips filed",
    "A coalition ready to act when a bill moves",
]

# Fact sheet for the base (Hardball Ch7; content matches docs/advocacy/focused-next-steps-1-2-4-5-6.md §5).
FACT_SHEET_ISSUE = (
    "Illinois is treating lawfully imported kei vehicles as off-highway, so owners cannot "
    "register them for normal road use. This is based on how Illinois interprets "
    "625 ILCS 5/3-401(c-1), not on federal law or missing paperwork."
)
FACT_SHEET_POSITION = (
    "We are asking for a narrow statutory clarification so that vehicles originally "
    "manufactured for highway use (in any jurisdiction) and lawfully importable under "
    "federal law may be titled and registered in Illinois under normal requirements "
    "(insurance, equipment, traffic laws). No weakening of safety or enforcement."
)
FACT_SHEET_SUPPORTERS_PLACEHOLDER = (
    "Add names or groups when you have them (e.g. Land of Kei Illinois advocacy group, "
    'local clubs, businesses). Leave blank or "Coalition forming" until you have a list.'
)

# Documents listed in the legislator brief sidebar (title, url, file_type for icon).
# Optional: available=False and note="..." for placeholders (disabled style, note under title).
BRIEF_DOCUMENTS = [
    {
        "title": "IL Kei Vehicle Registration Fix Brief",
        "url": "/static/advocacy/IL_Kei_Vehicle_Registration_Fix_Brief.pdf",
        "file_type": "pdf",
    },
    {
        "title": "IL Kei Vehicle Registration Fix Internal Summary",
        "url": "/static/IL_Kei_Vehicle_Registration_Fix_Internal_Summary.pdf",
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
# speed_limited: True if roads are restricted (e.g. 55 mph max or no Interstates).
# speed: Optional short text for Speed column (e.g. "55 mph or less", "45 mph max", "No Interstates"). If absent, column shows "Limited" when speed_limited else "—".
# aamva_fix: True if state had a ban (often AAMVA-driven) and reversed it via law/policy (post-2020).
# explicit_kei_law: True if state passed a new law explicitly naming kei/mini vehicles (not dependent on AAMVA interpretation).
# how: Optional. The law or policy that made Keis registrable (e.g. statute, bill number, "DMV policy"). If none, leave "".
# effective: Optional. The date the change or law came into effect (e.g. "Sep 2025"). If no date, leave "".
BRIEF_STATE_STATUS: list[dict] = [
    {
        "state": "Arizona",
        "state_abbr": "az",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Primary On-Road Use decal; legal on all roads including Interstates.",
        "status": "A.R.S. Title 28, Article 16",
        "how": "A.R.S. Title 28, Article 16",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "Arkansas",
        "state_abbr": "ar",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Allowed on roads 55 mph or less; prohibited on Interstates.",
        "status": "Ark. Code § 27-14-726",
        "how": "Ark. Code § 27-14-726",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "55 mph or less",
    },
    {
        "state": "Colorado",
        "state_abbr": "co",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "HB 25-1281 — kei road-legal framework.",
        "status": "Jul 2027",
        "how": "Colorado HB 25-1281",
        "effective": "Jul 2027",
        "notes": "",
        "bill_url": "https://leg.colorado.gov/bills/hb25-1281",
        "bill_title": "Colorado HB 25-1281",
        "speed_limited": True,
        "speed": "55 mph",
        "aamva_fix": True,
        "explicit_kei_law": True,
    },
    {
        "state": "Delaware",
        "state_abbr": "de",
        "bill_status": "none",
        "policy": True,
        "mechanism": "Standard registration possible; subject to strict safety inspection.",
        "status": "DMV Registration Policy",
        "how": "DMV Registration Policy",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "Idaho",
        "state_abbr": "id",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Treated as standard motor vehicle if >25 years old.",
        "status": "Idaho Code § 49-402",
        "how": "Idaho Code § 49-402",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "Indiana",
        "state_abbr": "in",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Mini-Truck Title Application & police inspection; no Interstates.",
        "status": "Ind. Code § 9-13-2-103",
        "how": "Ind. Code § 9-13-2-103",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "No Interstates",
    },
    {
        "state": "Louisiana",
        "state_abbr": "la",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Allowed on roads 55 mph or less.",
        "status": "La. R.S. 32:299",
        "how": "La. R.S. 32:299",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "55 mph or less",
    },
    {
        "state": "Maine",
        "state_abbr": "me",
        "bill_status": "pending",
        "policy": False,
        "mechanism": "H.4053 — title/registration and working group (in committee)",
        "status": "In committee",
        "how": "Maine H.4053",
        "effective": "",
        "notes": "",
        "bill_url": "https://legislature.maine.gov/legis/bills/display_ps.asp?ld=4053&num=H",
        "bill_title": "Maine H.4053",
        "speed_limited": False,
        "aamva_fix": True,
    },
    {
        "state": "Maryland",
        "state_abbr": "md",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Registered as Historic (20+ years); occasional use restrictions apply.",
        "status": "Md. Transp. Code § 13-936",
        "how": "Md. Transp. Code § 13-936",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "Massachusetts",
        "state_abbr": "ma",
        "bill_status": "pending",
        "policy": True,
        "mechanism": "Ban reversed Sept 2024; now registrable as standard auto.",
        "status": "Sep 2024",
        "how": "RMV policy reversal",
        "effective": "Sep 2024",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
        "aamva_fix": True,
    },
    {
        "state": "Michigan",
        "state_abbr": "mi",
        "bill_status": "none",
        "policy": True,
        "mechanism": "Ban reversed Nov 2024; registrable as Pickup/Station Wagon.",
        "status": "Nov 2024",
        "how": "SOS policy reversal",
        "effective": "Nov 2024",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
        "aamva_fix": True,
    },
    {
        "state": "Mississippi",
        "state_abbr": "ms",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Standard registration if federal import docs (Form 7501) provided.",
        "status": "Miss. Code § 27-19",
        "how": "Miss. Code § 27-19",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "Montana",
        "state_abbr": "mt",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Permanent registration available for vehicles 11+ years old.",
        "status": "Mont. Code § 61-3-321",
        "how": "Mont. Code § 61-3-321",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "Nebraska",
        "state_abbr": "ne",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Legal on all public roads except Interstates/Expressways.",
        "status": "Neb. Rev. Stat. § 60-339",
        "how": "Neb. Rev. Stat. § 60-339",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "No Interstates/Expressways",
    },
    {
        "state": "North Carolina",
        "state_abbr": "nc",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Allowed on roads with posted speed limits of 55 mph or less.",
        "status": "N.C. Gen. Stat. § 20-4.01",
        "how": "N.C. Gen. Stat. § 20-4.01",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "55 mph or less",
    },
    {
        "state": "North Dakota",
        "state_abbr": "nd",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Allowed on roads with posted speed limits of 55 mph or less.",
        "status": "N.D. Cent. Code § 39-29",
        "how": "N.D. Cent. Code § 39-29",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "55 mph or less",
    },
    {
        "state": "Oklahoma",
        "state_abbr": "ok",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Legal on state roads; prohibited on Interstates.",
        "status": "Okla. Stat. tit. 47 § 1151.3",
        "how": "Okla. Stat. tit. 47 § 1151.3",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "No Interstates",
    },
    {
        "state": "Oregon",
        "state_abbr": "or",
        "bill_status": "pending",
        "policy": False,
        "mechanism": "SB 1213 — title/registration and operating rules (Transportation committee)",
        "status": "In committee",
        "how": "Oregon SB 1213",
        "effective": "",
        "notes": "",
        "bill_url": "https://olis.oregonlegislature.gov/liz/2025R1/Measures/Overview/SB1213",
        "bill_title": "Oregon SB 1213",
        "speed_limited": False,
        "aamva_fix": True,
    },
    {
        "state": "South Carolina",
        "state_abbr": "sc",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Antique plates available for vehicles 25+ years old.",
        "status": "S.C. Code § 56-3",
        "how": "S.C. Code § 56-3",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "Tennessee",
        "state_abbr": "tn",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Registered as Antique (Class C); general road use allowed.",
        "status": "Tenn. Code § 55-4-111",
        "how": "Tenn. Code § 55-4-111",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "Texas",
        "state_abbr": "tx",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "SB 1816 — miniature vehicle statute; titling, registration, highway rules",
        "status": "Sep 2025",
        "how": "Texas SB 1816",
        "effective": "Sep 2025",
        "notes": "",
        "bill_url": "https://capitol.texas.gov/BillLookup/History.aspx?LegSess=89R&Bill=SB1816",
        "bill_title": "Texas SB 1816",
        "speed_limited": False,
        "aamva_fix": True,
        "explicit_kei_law": True,
    },
    {
        "state": "Washington",
        "state_abbr": "wa",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Road legal if safety equipment (lights/mirrors) is retrofitted.",
        "status": "RCW 46.16A.080",
        "how": "RCW 46.16A.080",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "West Virginia",
        "state_abbr": "wv",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Registered as Street-Legal SPV; max 20-mile range often waived.",
        "status": "W. Va. Code § 17A-13-1",
        "how": "W. Va. Code § 17A-13-1",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": False,
    },
    {
        "state": "Wyoming",
        "state_abbr": "wy",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Registered as MPV; prohibited on Interstates.",
        "status": "Wyo. Stat. § 31-2-232",
        "how": "Wyo. Stat. § 31-2-232",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "No Interstates",
    },
    # Restricted (antique/collector/speed or radius caps only).
    {
        "state": "Connecticut",
        "state_abbr": "ct",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Antique plates only; must be 20+ years old; limited use.",
        "status": "C.G.S. § 14-20",
        "how": "C.G.S. § 14-20",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "restricted": True,
    },
    {
        "state": "Missouri",
        "state_abbr": "mo",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Speed cap; requires local ordinance.",
        "status": "Mo. Rev. Stat. § 304.032",
        "how": "Mo. Rev. Stat. § 304.032",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "45 mph max",
        "restricted": True,
    },
    {
        "state": "New Hampshire",
        "state_abbr": "nh",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Radius cap: max 25 miles from home.",
        "status": "RSA 261:41-a",
        "how": "RSA 261:41-a",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "35 mph",
        "restricted": True,
    },
    {
        "state": "Pennsylvania",
        "state_abbr": "pa",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Antique plates only; stock condition required; no daily use.",
        "status": "PennDOT Fact Sheet",
        "how": "PennDOT policy",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "restricted": True,
    },
    {
        "state": "Utah",
        "state_abbr": "ut",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Speed cap; banned on Interstates.",
        "status": "Utah Code § 41-6a-1505",
        "how": "Utah Code § 41-6a-1505",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "speed": "50 mph max",
        "restricted": True,
    },
    {
        "state": "Virginia",
        "state_abbr": "va",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Antique/farm only; strict driving limits (e.g. car shows only).",
        "status": "Va. Code § 46.2-730",
        "how": "Va. Code § 46.2-730",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "restricted": True,
    },
    {
        "state": "Wisconsin",
        "state_abbr": "wi",
        "bill_status": "passed",
        "policy": False,
        "mechanism": "Collector plates only; owner must prove another daily driver.",
        "status": "Wis. Stat. § 341.266",
        "how": "Wis. Stat. § 341.266",
        "effective": "",
        "notes": "",
        "bill_url": "",
        "bill_title": "",
        "speed_limited": True,
        "restricted": True,
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

# FAQ for The Issue page (law/registration). Each item: id, question, answer, sources (list of {label, url}).
FAQ_LAW = {
    "title": "FAQ — Law & registration",
    "intro": (
        "Quick answers for Illinois residents taking action. "
        "Links go to primary sources (government or official statute text)."
    ),
    "items": [
        {
            "id": "a1",
            "question": "Are kei vehicles legal to import into the United States?",
            "answer": (
                "Often, yes. Under federal rules, many vehicles that are 25 years old or older can be "
                "lawfully imported without needing to meet current Federal Motor Vehicle Safety Standards "
                "(FMVSS). That does not automatically guarantee state registration—states control "
                "titling/registration eligibility."
            ),
            "sources": [
                {
                    "label": "NHTSA — Importing a Vehicle (overview)",
                    "url": "https://www.nhtsa.gov/importing-vehicle",
                },
                {
                    "label": "NHTSA — Importation & Certification FAQs",
                    "url": "https://www.nhtsa.gov/importing-vehicle/importation-and-certification-faqs",
                },
            ],
        },
        {
            "id": "a2",
            "question": "If it's legal to import, why won't Illinois register it?",
            "answer": (
                "Import legality and state registration are separate. Illinois registration decisions "
                "are being made under the Illinois Vehicle Code. The current barrier is the Secretary of "
                "State's interpretation of 625 ILCS 5/3-401(c-1) about whether a vehicle was "
                '"originally manufactured for operation on highways."'
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
        {
            "id": "a3",
            "question": "Is this about people trying to register off-road vehicles?",
            "answer": (
                "No. The advocacy ask is about kei vehicles that were built for highway use in their "
                "home jurisdiction and are lawfully imported. The goal is to stop treating them as "
                "off-highway/non-highway solely due to how Illinois reads 3-401(c-1). Illinois safety, "
                "insurance, equipment, and traffic enforcement would still apply."
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
        {
            "id": "a4",
            "question": "Are we asking for special treatment or exemptions from safety rules?",
            "answer": (
                "No. The request is for normal registration eligibility, while keeping normal Illinois "
                "requirements in place (titling documentation, insurance, equipment compliance, and "
                "traffic enforcement). This is a narrow statutory clarification request—not a blanket "
                "exemption."
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
        {
            "id": "a5",
            "question": "Does Illinois have to follow AAMVA guidance?",
            "answer": (
                "No. AAMVA is not a lawmaking body for Illinois. Illinois agencies and the General "
                "Assembly set Illinois policy through the Vehicle Code and formal agency policies. "
                "Your outreach is about how Illinois chooses to clarify its own statute and "
                "registration rules."
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
        {
            "id": "a6",
            "question": 'What should I say if someone claims "states can do whatever they want"?',
            "answer": (
                "States control registration, yes—but states must also apply their statutes consistently. "
                "The point here is that Illinois can fix this cleanly by clarifying how 3-401(c-1) "
                "applies to highway-built, federally lawful imports. That creates clarity for residents "
                "and reduces administrative conflict."
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
        {
            "id": "a7",
            "question": (
                "Can you show an example that state registration rules vary even for imports?"
            ),
            "answer": (
                "Yes. States publish their own eligibility rules. Many state DMVs publish guidance on "
                'how "foreign" or imported vehicles are handled and how FMVSS labeling/standards can '
                "affect title/registration eligibility. Illinois can clarify its own approach within "
                "that framework."
            ),
            "sources": [
                {
                    "label": "NHTSA — Importing a Vehicle",
                    "url": "https://www.nhtsa.gov/importing-vehicle",
                },
            ],
        },
    ],
}

# FAQ for The Issue page (advocacy process). Same shape: title, intro, items (id, question, answer, optional sources).
FAQ_ADVOCACY = {
    "title": "FAQ — Advocacy & how we work",
    "intro": (
        "How the advocacy effort is organized and how your outreach helps drive "
        "legal registration of kei vehicles in Illinois."
    ),
    "items": [
        {
            "id": "adv1",
            "question": "What is the goal of this advocacy?",
            "answer": (
                "To get kei vehicles legally registered for normal road use in Illinois via a narrow "
                "clarification to 625 ILCS 5/3-401(c-1). We focus on what we can control: constituent "
                "contacts, co-sponsors, witness slips, coalition readiness."
            ),
        },
        {
            "id": "adv1b",
            "question": "What does success look like for this advocacy?",
            "answer": (
                "Success is what we can control: constituent contacts, co-sponsors secured, witness "
                "slips filed, and a coalition ready to act when a bill moves. Road-legal status is the "
                "outcome we're working toward, but we measure success by these actions so we can see "
                "progress and stay motivated even when the legislature hasn't yet passed a fix."
            ),
        },
        {
            "id": "adv2",
            "question": "How does the advocacy group intend to achieve that goal?",
            "answer": (
                "By building awareness and constituent pressure—Illinois residents contact their "
                "legislators so they hear this is a real issue in their districts. When a bill "
                "exists, we support it (sponsor, committee, votes). Right now we are in the outreach "
                "stage: the more people who reach out by district or ZIP, the more clearly "
                "legislators see that constituents care and that the issue deserves a fix."
            ),
        },
        {
            "id": "adv3",
            "question": "Why is it important to voice my concerns?",
            "answer": (
                "Legislators prioritize issues they hear about from constituents. Your contact helps "
                "put the issue on the map and builds the case for a legislative fix."
            ),
        },
        {
            "id": "adv4",
            "question": "Where are we in the process?",
            "answer": (
                "No bill yet. We are in the outreach stage: constituents contact legislators and "
                "make it clear that kei registration is something Illinois residents care about. As "
                "momentum builds, we coordinate contact by district or ZIP to show where support exists."
            ),
        },
        {
            "id": "adv5",
            "question": "What are the steps or checkpoints in the process?",
            "answer": (
                "Right now we are at the no-bill stage. Step 1: Outreach to your legislators—get "
                "aware of the issue and make sure your senator and representative hear from you. "
                "Step 2: As momentum builds, we coordinate contact by district or ZIP so legislators "
                "see that this is a real issue in their district and that constituents want a fix. "
                "Later stages (when a bill exists) will include supporting the bill, committee "
                "contact, and votes. For now, your outreach is the main checkpoint."
            ),
        },
        {
            "id": "adv6",
            "question": "How do I find my legislators?",
            "answer": (
                "Use the advocacy tool on this site: enter your Illinois ZIP code at the Advocacy "
                "page to see your State Senator, your State Representative, and a recommended "
                "target (Power Broker). You can then call or email each one using the scripts and "
                "templates we provide."
            ),
        },
        {
            "id": "adv7",
            "question": "What should I say when I call or email?",
            "answer": (
                "Use the advocacy tool's script or email template: ask for support for a narrow "
                "statutory clarification to 625 ILCS 5/3-401(c-1) so kei vehicles that were built "
                "for highway use and are lawfully imported can be registered in Illinois. Key "
                "message: constituents care and want a fix."
            ),
        },
    ],
}

# FAQ for Legislator Brief page (legislators & staff). Same shape: title, intro, items (id, question, answer, sources).
FAQ_LEGISLATORS = {
    "title": "FAQ — For Legislators & Staff",
    "intro": (
        "Short, risk-aware answers for offices evaluating whether to sponsor or support "
        "a narrow statutory clarification."
    ),
    "items": [
        {
            "id": "l1",
            "question": "What exactly is being requested from the General Assembly?",
            "answer": (
                "A narrow clarification to 625 ILCS 5/3-401(c-1) (or a related definitional section) "
                "so that highway-built vehicles manufactured for on-road use in any jurisdiction—when "
                "lawfully importable under federal rules—can be titled/registered under normal Illinois "
                "requirements (insurance, equipment, traffic enforcement, documentation)."
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
        {
            "id": "l2",
            "question": "Is this an administrative dispute or a statutory issue?",
            "answer": (
                "It's statutory. The cited barrier is 625 ILCS 5/3-401(c-1). Under the SOS posture, "
                'eligibility turns on how "originally manufactured for operation on highways" is applied. '
                "If the statute is interpreted to exclude these vehicles, legislative clarification is "
                "the direct remedy."
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
        {
            "id": "l3",
            "question": "Does this weaken safety enforcement or create broad exemptions?",
            "answer": (
                "No. The concept preserves existing Illinois enforcement: insurance, equipment "
                "compliance, and traffic laws. The proposal is an eligibility clarification for normal "
                "registration—without rewriting enforcement authorities or creating a blanket carve-out."
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
        {
            "id": "l4",
            "question": "How does federal import legality relate to Illinois registration?",
            "answer": (
                "Federal law governs import eligibility; states govern title/registration. NHTSA "
                "provides the import framework (including exemptions commonly used for older vehicles). "
                "Illinois can keep its normal registration standards intact while clarifying how its "
                "statute applies to federally lawful imports."
            ),
            "sources": [
                {
                    "label": "NHTSA — Importing a Vehicle (overview)",
                    "url": "https://www.nhtsa.gov/importing-vehicle",
                },
                {
                    "label": "NHTSA — Importation & Certification FAQs",
                    "url": "https://www.nhtsa.gov/importing-vehicle/importation-and-certification-faqs",
                },
            ],
        },
        {
            "id": "l5",
            "question": "Does clarifying this open the door to other nonconforming vehicles?",
            "answer": (
                "Not if drafted narrowly. The concept can be limited to vehicles originally "
                "manufactured for highway use (in any jurisdiction) and lawfully importable under "
                "federal rules, while excluding off-road-only vehicles and preserving Illinois "
                "conditions for road use."
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
        {
            "id": "l6",
            "question": "What's the lowest-risk next step before a bill is filed?",
            "answer": (
                "A staff-level meeting with SOS legal/policy to confirm what statutory language would "
                "satisfy their interpretation of 3-401(c-1), followed by identifying the best sponsor "
                "path through the relevant transportation/vehicle code process."
            ),
            "sources": [
                {
                    "label": "Illinois Vehicle Code — 625 ILCS 5/3-401 (ILGA)",
                    "url": "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm",
                },
            ],
        },
    ],
}


def _brief_map_fill_status(s: dict) -> str:
    """Derive map fill key from BRIEF_STATE_STATUS row: passed | pending | policy | restricted | none."""
    if s.get("restricted"):
        return "restricted"
    if s["bill_status"] in ("passed", "pending"):
        return s["bill_status"]
    if s.get("policy"):
        return "policy"
    return "none"


def _brief_aamva_fix_state_abbrs() -> list[str]:
    """State abbrs where aamva_fix is True (reversed prior ban or passed explicit kei law)."""
    return [s["state_abbr"] for s in BRIEF_STATE_STATUS if s.get("aamva_fix")]


def _issue_sources_from_faq(faq: dict) -> list[dict[str, str]]:
    """Build deduplicated list of Issue page sources: statute first, then FAQ source links (by url)."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    statute_url = "https://www.ilga.gov/legislation/ilcs/documents/062500050K3-401.htm"
    statute_title = "625 ILCS 5/3-401(c-1)"
    out.append({"title": statute_title, "url": statute_url})
    seen.add(statute_url)
    for item in faq.get("items") or []:
        for src in item.get("sources") or []:
            url = src.get("url") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            out.append({"title": src.get("label") or url, "url": url})
    return out


ISSUE_SOURCES: list[dict[str, str]] = _issue_sources_from_faq(FAQ_LAW)

# Fact sheet document (PDF) linked from The Issue sidebar. Place the PDF at this path (e.g. print /fact-sheet to PDF).
FACT_SHEET_PDF_URL = "/static/advocacy/Kei_Registration_Fact_Sheet.pdf"


@router.get("/the-issue", include_in_schema=False)
async def the_issue_page(request: Request):
    """Serve The Issue page: kei vehicle registration problem and how to help. Content from canonical .txt when present."""
    brief_state_map_status = {
        s["state_abbr"]: _brief_map_fill_status(s) for s in BRIEF_STATE_STATUS
    }
    aamva_fix_abbrs = _brief_aamva_fix_state_abbrs()
    constituent_brief = _load_constituent_brief()
    return templates.TemplateResponse(
        "the_issue.html",
        {
            "request": request,
            "constituent_brief": constituent_brief,
            "fact_sheet_pdf_url": FACT_SHEET_PDF_URL,
            "faq_law": FAQ_LAW,
            "faq_advocacy": FAQ_ADVOCACY,
            "brief_state_status": BRIEF_STATE_STATUS,
            "brief_state_map_status_json": json.dumps(brief_state_map_status),
            "brief_aamva_fix_state_abbrs_json": json.dumps(aamva_fix_abbrs),
            "issue_sources": ISSUE_SOURCES,
            "strategic_mission": STRATEGIC_MISSION,
            "strategic_vision": STRATEGIC_VISION,
            "strategic_five_points": STRATEGIC_FIVE_POINTS,
        },
    )


@router.get("/legislator-brief", include_in_schema=False)
async def legislator_brief_page(request: Request):
    """Serve the Legislator Brief: concise briefing for legislators and staff. Content from canonical .txt when present."""
    brief_state_map_status = {
        s["state_abbr"]: _brief_map_fill_status(s) for s in BRIEF_STATE_STATUS
    }
    aamva_fix_abbrs = _brief_aamva_fix_state_abbrs()
    legislator_brief = _load_legislator_brief()
    return templates.TemplateResponse(
        "legislator_brief.html",
        {
            "request": request,
            "legislator_brief": legislator_brief,
            "strategic_five_points": STRATEGIC_FIVE_POINTS,
            "brief_documents": BRIEF_DOCUMENTS,
            "brief_state_status": BRIEF_STATE_STATUS,
            "brief_state_map_status_json": json.dumps(brief_state_map_status),
            "brief_aamva_fix_state_abbrs_json": json.dumps(aamva_fix_abbrs),
            "brief_bills_passed": BRIEF_BILLS_PASSED,
            "brief_bills_current": BRIEF_BILLS_CURRENT,
            "brief_sources": BRIEF_SOURCES,
            "faq": FAQ_LEGISLATORS,
        },
    )


@router.get("/fact-sheet", include_in_schema=False)
async def fact_sheet_page(request: Request):
    """Serve the one-page fact sheet for volunteers (Hardball Ch7; content from focused-next-steps doc §5)."""
    fact_sheet_faq_ids = ("adv1", "adv1b", "adv2", "adv3", "adv4", "adv7")
    fact_sheet_faq_items = [i for i in FAQ_ADVOCACY["items"] if i["id"] in fact_sheet_faq_ids]
    return templates.TemplateResponse(
        "fact_sheet.html",
        {
            "request": request,
            "strategic_five_points": STRATEGIC_FIVE_POINTS,
            "fact_sheet_issue": FACT_SHEET_ISSUE,
            "fact_sheet_position": FACT_SHEET_POSITION,
            "fact_sheet_supporters_placeholder": FACT_SHEET_SUPPORTERS_PLACEHOLDER,
            "fact_sheet_faq_items": fact_sheet_faq_items,
        },
    )


@router.get("/coalition", include_in_schema=False)
async def coalition_page(request: Request):
    """Serve the Supporting legislators / coalition page: recognized offices that engage."""
    return templates.TemplateResponse(
        "coalition.html",
        {"request": request},
    )
