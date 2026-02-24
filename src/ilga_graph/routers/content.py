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

# FAQ for The Issue page (advocates). Each item: id, question, answer, sources (list of {label, url}).
FAQ_ADVOCATES = {
    "title": "FAQ — For Advocates",
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
    """Derive map fill key from BRIEF_STATE_STATUS row: passed | pending | policy | none."""
    if s["bill_status"] in ("passed", "pending"):
        return s["bill_status"]
    if s.get("policy"):
        return "policy"
    return "none"


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


ISSUE_SOURCES: list[dict[str, str]] = _issue_sources_from_faq(FAQ_ADVOCATES)


@router.get("/the-issue", include_in_schema=False)
async def the_issue_page(request: Request):
    """Serve The Issue page: kei vehicle registration problem and how to help."""
    brief_state_map_status = {
        s["state_abbr"]: _brief_map_fill_status(s) for s in BRIEF_STATE_STATUS
    }
    return templates.TemplateResponse(
        "the_issue.html",
        {
            "request": request,
            "faq": FAQ_ADVOCATES,
            "brief_state_status": BRIEF_STATE_STATUS,
            "brief_state_map_status_json": json.dumps(brief_state_map_status),
            "issue_sources": ISSUE_SOURCES,
        },
    )


@router.get("/legislator-brief", include_in_schema=False)
async def legislator_brief_page(request: Request):
    """Serve the Legislator Brief: concise briefing for legislators and staff."""
    brief_state_map_status = {
        s["state_abbr"]: _brief_map_fill_status(s) for s in BRIEF_STATE_STATUS
    }
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
            "faq": FAQ_LEGISLATORS,
        },
    )
