"""Canonical content constants: strategic copy, FAQs, glossaries, timeline, brief data. No FastAPI or DB."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
CONSTITUENT_BRIEF_PATH = _REPO_ROOT / "Illinois_Kei_Vehicle_Registration_Constituent_Brief.txt"
LEGISLATOR_BRIEF_PATH = _REPO_ROOT / "IL_Kei_Vehicle_Registration_Fix_Brief 1.txt"

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

# Why should you care (Hardball Ch7: who benefits, why your voice matters). Canonical copy from constituent brief + FAQ_ADVOCACY + STRATEGIC_FIVE_POINTS.
WHY_SHOULD_YOU_CARE_HEADING = "Why should you care?"
WHY_SHOULD_YOU_CARE_TEASER_HEADING = "Three reasons"
WHY_SHOULD_YOU_CARE_INTRO = (
    "Clear law protects residents. When statutory language is ambiguous, regular people absorb the consequences. "
    "This issue is about fairness, predictability, and consistent application of Illinois law. "
    "This affects real Illinois residents who own legal vehicles they can't register—registrations denied or revoked, "
    "titles branded 'Not Eligible for Registration,' plates surrendered."
)
WHY_SHOULD_YOU_CARE_VOICE = (
    "Legislators prioritize issues they hear about from constituents. Your contact helps put the issue on the map "
    "and builds the case for a legislative fix."
)
WHY_SHOULD_YOU_CARE_TEASER_ITEMS: list[str] = [
    "This affects real Illinois residents who own legal vehicles they can't register and those receiving titles branded 'Not Eligible for Registration.'",
    "Even if you don't own one, it's about fair and consistent application of the law.",
    "Your voice helps legislators see the issue deserves a fix.",
]

# Why-you-care marquee: list of {"src", "alt", "name", "caption", "location"}.
# name = person's name (bold in blurb); caption = description; location = city only.
MARQUEE_IMAGES: list[dict[str, str]] = [
    {
        "src": "/static/images/tyler_morales.png",
        "alt": "Tyler Morales, kei vehicle owner affected by registration",
        "name": "Tyler Morales",
        "caption": "Followed all proper import rules; the DMV and Secretary of State keep denying his registration.",
        "location": "Chicago",
    },
    {
        "src": "/static/images/christian-eduardo-huerta.jpg",
        "alt": "Christian Eduardo Huerta, kei vehicle owner affected by registration",
        "name": "Christian Eduardo Huerta",
        "caption": "\"We use our Sambar for our mobile detailing. It's an amazing marketing tool and a gas saver.\" In Illinois, whether you get registration is inconsistent; he's one of the lucky ones who got plates.",
        "location": "Chicago",
    },
    {
        "src": "https://picsum.photos/seed/kei1/360/240",
        "alt": "Kei vehicle owner affected by registration — placeholder",
        "name": "Marcus",
        "caption": "Registration revoked after years of legal use",
        "location": "Chicago",
    },
    {
        "src": "https://picsum.photos/seed/kei2/360/240",
        "alt": "Kei owner unable to register — placeholder",
        "name": "Sarah",
        "caption": "Imported kei truck stuck, can't get plates",
        "location": "Peoria",
    },
    {
        "src": "https://picsum.photos/seed/kei3/360/240",
        "alt": "Affected kei owner — placeholder",
        "name": "James",
        "caption": "Van denied; same model registered in other states",
        "location": "Rockford",
    },
    {
        "src": "https://picsum.photos/seed/kei4/360/240",
        "alt": "Illinois kei vehicle registration issue — placeholder",
        "name": "Elena",
        "caption": "DMV turned away federally compliant vehicle",
        "location": "Springfield",
    },
    {
        "src": "https://picsum.photos/seed/kei5/360/240",
        "alt": "Kei owner story — placeholder",
        "name": "David",
        "caption": "Registration renewal rejected without notice",
        "location": "Naperville",
    },
    {
        "src": "https://picsum.photos/seed/kei6/360/240",
        "alt": "Vehicle registration affected owner — placeholder",
        "name": "Rachel",
        "caption": "Legal kei car can't be registered in Illinois",
        "location": "Champaign",
    },
]

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

# "Where we are" block on /updates. One-line phase label; campaign banner (when active) carries detail and CTA.
CAMPAIGN_STATUS = "We're in the outreach phase."

# Hero urgency and clarity (Hardball: timeline, clear ask). Switch to HERO_URGENCY_THIS_SESSION when pushing this session.
# Used in hero and session pill (expandable). Alternatives for pill: "Building support now for the 2027 session." / "2027 session ahead. Your outreach now builds momentum."
HERO_URGENCY_LINE = (
    "Next session starts early 2027. We're building constituent support now so we're ready."
)
HERO_URGENCY_THIS_SESSION = (
    "Spring session runs through May 31. We need your voice before key deadlines."
)
HERO_CLARITY_LINE = "Enter your ZIP — we'll show you who to call and what to say. Takes 2 min."

# Progress checklist: checkpoints from current phase to Keis be legal. Update achieved count as campaign advances.
PROGRESS_CHECKPOINTS: list[str] = [
    "Outreach & building contacts",
    "Sponsor identified",
    "Bill introduced",
    "Committee hearing",
    "Passes legislature",
    "Governor signature",
    "Keis be legal",
]
PROGRESS_ACHIEVED_COUNT: int = 1  # Steps 1..N at full opacity; rest at reduced opacity.


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


# BRIEF_STATE_STATUS
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


# FAQ_LAW
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


# FAQ_ADVOCACY
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
            "id": "success-measures",
            "question": "How we measure success?",
            "answer": STRATEGIC_SUCCESS_MEASURE,
            "answer_list": STRATEGIC_SUCCESS_MEASURE_ITEMS,
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

# FAQ for The Issue page: session calendar and deadlines. Single source of truth: reference/session_schedule.json.


# FAQ_SESSION
FAQ_SESSION = {
    "title": "FAQ — Session calendar & deadlines",
    "intro": (
        "We maintain the Illinois General Assembly House and Senate schedule as a single source "
        "of truth for session dates and key deadlines. All dates and reminders on this site come from it."
    ),
    "items": [
        {
            "id": "session1",
            "question": "Where can I find the legislative session calendar and key deadlines?",
            "answer": (
                "We use the official 104th General Assembly Spring 2026 schedule (House and Senate). "
                "Key deadlines—such as bill introduction, committee deadlines, and third reading—are listed below. "
                "Session and holiday dates are in our reference data; we update that file when the session calendar changes."
            ),
        },
    ],
}


# SESSION_SCHEDULE_TERMS
SESSION_SCHEDULE_TERMS = [
    {
        "id": "lrb",
        "term": "LRB",
        "definition": "Legislative Reference Bureau. Legislators request bill drafting from the LRB. The LRB request deadline is the last day to submit requests for the session; after that, the LRB blackout begins and no new bill requests are accepted until the next session.",
    },
    {
        "id": "committee-deadline",
        "term": "Committee deadline",
        "definition": "Final day for standing committees to report bills out of committee. Bills not reported by this date are re-referred to the gatekeeper (Senate: Committee on Assignments; House: Rules Committee)—not killed, but delayed. Senate Rule 2-10, House Rule 9.",
    },
    {
        "id": "third-reading",
        "term": "Third Reading",
        "definition": "Final reading of a bill before a floor vote. A bill must be read by title on three different days before passage. The Third Reading deadline is the last day the chamber may pass bills on third reading; after that, bills not passed are re-referred. Senate Rule 2-10(a)(5), House Rule 9(b)(5).",
    },
    {
        "id": "perfunctory-session",
        "term": "Perfunctory session",
        "definition": "A short session for procedural business (e.g. reading the journal, formalities). No substantive debate or votes on bills.",
    },
    {
        "id": "substantive-bills",
        "term": "Substantive bills",
        "definition": "Bills that change law or policy (as opposed to appropriation-only or purely procedural measures). Session deadlines often set separate dates for substantive bills vs. appropriation bills.",
    },
    {
        "id": "session",
        "term": "Session",
        "definition": "A day the chamber meets in Springfield. The schedule lists which days the House or Senate is in session.",
    },
    {
        "id": "adjournment",
        "term": "Adjournment",
        "definition": "End of the legislative session (sine die). After adjournment, no further action on bills until the next session.",
    },
]

# Domain glossary: app/organizational terms. Single source for docs/reference/glossary.md and public /glossary page.


# DOMAIN_GLOSSARY
DOMAIN_GLOSSARY: list[dict] = [
    {
        "id": "campaign",
        "term": "Campaign",
        "definition": "A single, time-bound action alert. At most one campaign is active at a time. Has title, message, ask, optional start/end dates; outreach recorded while active is attributed to it. Not the overall multi-year advocacy initiative (see Advocacy effort).",
        "category": "advocacy",
    },
    {
        "id": "advocacy-effort",
        "term": "Advocacy effort",
        "definition": "The overall multi-year initiative toward the legislative objective (e.g. kei vehicle registration fix). Not a DB object; encompasses all campaigns, updates, and organizing.",
        "category": "advocacy",
    },
    {
        "id": "ask",
        "term": "Ask",
        "definition": "The specific request to a legislator or constituent (noun). E.g. “Contact your rep” or “Support HB 1234.” Stored in Campaign.ask as CTA button text.",
        "category": "advocacy",
    },
    {
        "id": "coalition-advocacy",
        "term": "Coalition (advocacy)",
        "definition": "Organizations aligned on the issue; building coalition = recruiting orgs and stakeholders. Not ML voting coalition (legislators who vote together).",
        "category": "advocacy",
    },
    {
        "id": "advocate",
        "term": "Advocate",
        "definition": "A user who takes outreach action (call, email). May or may not be a constituent of the legislator contacted.",
        "category": "advocacy",
    },
    {
        "id": "constituent",
        "term": "Constituent",
        "definition": "An advocate who lives in a legislator’s district. Stored as a boolean on outreach events; “Constituent Brief” is the canonical document for the public.",
        "category": "advocacy",
    },
    {
        "id": "outreach",
        "term": "Outreach",
        "definition": "A call, email, or no-answer recorded against a legislator. One OutreachEvent per action.",
        "category": "advocacy",
    },
    {
        "id": "contact",
        "term": "Contact",
        "definition": "(Verb) To reach a legislator’s office (call or email). (Noun) The person at the office who answered the call (OutreachEvent.contact_name). “Contact period” = campaign duration.",
        "category": "advocacy",
    },
    {
        "id": "update",
        "term": "Update",
        "definition": "An email announcement sent to subscribers. Has title, body, type (Major/Minor/Other), optional image. DB: Update model.",
        "category": "advocacy",
    },
    {
        "id": "brief",
        "term": "Brief",
        "definition": "A canonical document: legislator brief (for offices) or constituent brief (for the public). The one-pager PDF is the print version of the legislator brief.",
        "category": "advocacy",
    },
    {
        "id": "session",
        "term": "Session",
        "definition": "A day the chamber meets in Springfield, or the full session period (e.g. 104th GA Spring 2026). Session schedule lists session days and deadlines.",
        "category": "legislative",
    },
    {
        "id": "session-milestone",
        "term": "Session milestone",
        "definition": "A legislative deadline with a date (e.g. committee deadline, third reading deadline). Used to set campaign end dates in admin; Campaign.session_milestone_id.",
        "category": "legislative",
    },
    {
        "id": "bill-stage",
        "term": "Bill stage",
        "definition": "Where a bill sits in the process: introduced, committee, floor, passed one chamber, passed both, signed. P(Advance) predicts chance of reaching a positive stage.",
        "category": "legislative",
    },
    {
        "id": "bill-action",
        "term": "Bill action",
        "definition": "A single procedural event on a bill (e.g. “Referred to Assignments”, “Do Pass”). Shown in bill action history.",
        "category": "legislative",
    },
    {
        "id": "voting-coalition",
        "term": "Voting coalition",
        "definition": "ML-discovered cluster of legislators who vote together. Used in Intelligence. Not an advocacy coalition (organizations aligned on the issue).",
        "category": "legislative",
    },
    {
        "id": "phase",
        "term": "Phase",
        "definition": "(Timeline) One of the four periods on the master timeline: Build, Intro, Committee & Floor, Governor. (Goal) “district” or “broker”—which set of outreach steps the user is on.",
        "category": "product",
    },
    {
        "id": "master-timeline",
        "term": "Master timeline",
        "definition": "The phased plan from now to bill signed, shown on /timeline. Source: TIMELINE_PHASES. Each phase has a date range and optional milestones.",
        "category": "product",
    },
    {
        "id": "progress-checklist",
        "term": "Progress checklist",
        "definition": "The short ordered list of stages on /updates (Outreach → … → Keis be legal). Achieved steps at full opacity, rest at reduced. Source: PROGRESS_CHECKPOINTS. Not dated; distinct from master timeline.",
        "category": "product",
    },
    {
        "id": "milestone",
        "term": "Milestone",
        "definition": "A dated checkpoint within a timeline phase (e.g. “Lock lead sponsor(s)”, “Bill introduced”). Shown on /timeline under each phase. Not session milestone (legislative deadline) or bill stage.",
        "category": "product",
    },
    {
        "id": "goal",
        "term": "Goal",
        "definition": "The user’s outreach task list: contact district legislators (4 actions), then Power Broker (2 actions). “Your goal” / “This week’s goal” in the sidebar. Not the advocacy objective (statutory fix).",
        "category": "product",
    },
    {
        "id": "drawer",
        "term": "Drawer",
        "definition": "The slide-out panel for call scripts and email templates. Opens from “Reach out” on a legislator card.",
        "category": "product",
    },
    {
        "id": "funnel",
        "term": "Funnel",
        "definition": "The user journey from page visit to completed outreach. Measured for conversion (e.g. % who opened drawer and completed at least one call/email).",
        "category": "product",
    },
]

# Kei vehicle and policy terms for the public glossary. Wording from canonical content (STRATEGIC_*, FAQ_*, briefs); no invented stats.


# KEI_GLOSSARY
KEI_GLOSSARY: list[dict] = [
    {
        "id": "kei-vehicle",
        "term": "Kei vehicle",
        "definition": "A small vehicle built to Japan’s kei (軽) class regulations—dimensions and engine size limits set by the Japanese government. Kei trucks, vans, and cars are federally legal to import into the US under the 25-year rule when they meet the age requirement.",
        "category": "kei",
    },
    {
        "id": "kei-class",
        "term": "Kei class",
        "definition": "Japanese vehicle category with strict size and engine limits (e.g. 660cc engine, maximum length/width/height). Vehicles in this class are built for highway use in Japan and are engineered for expressway travel.",
        "category": "kei",
    },
    {
        "id": "25-year-rule",
        "term": "25-year rule",
        "definition": "Federal rule that allows import into the US of vehicles that are at least 25 years old, without having to meet current US safety/emissions standards. Kei vehicles that meet this age requirement are federally legal to import.",
        "category": "kei",
    },
    {
        "id": "highway-built",
        "term": "Highway-built",
        "definition": "Originally manufactured for on-road use in any jurisdiction (e.g. Japan). The Illinois statutory fix we seek applies to highway-built, federally lawful imports—not off-road-only vehicles.",
        "category": "kei",
    },
    {
        "id": "625-ilcs-3-401",
        "term": "625 ILCS 5/3-401(c-1)",
        "definition": "Illinois statute that governs which vehicles are eligible for title and registration. The current language has an ambiguity about whether highway-built, federally lawful kei vehicles qualify. Our advocacy seeks a narrow clarifying amendment.",
        "category": "kei",
        "source": "625 ILCS 5/3-401 (ILGA)",
        "source_url": "https://www.ilga.gov/legislation/ilcs/fulltext.asp?DocName=062500050K3-401",
    },
    {
        "id": "shaken",
        "term": "Shaken",
        "definition": "Japan’s mandatory vehicle inspection program (every two years for most vehicles). Older kei vehicles imported under the 25-year rule have typically passed Shaken, so they arrive in the US well maintained.",
        "category": "kei",
    },
    {
        "id": "one-pager",
        "term": "One-pager",
        "definition": "The legislator brief: a single-page document for offices that summarizes the issue, the ask, and key points. Available as a PDF from the Legislator brief page.",
        "category": "kei",
    },
]


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


# FAQ_LEGISLATORS
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


# STRATEGIC_STATES_ICON_ABBR
STRATEGIC_STATES_ICON_ABBR = ("co", "mi", "tx")


# FACT_SHEET_PDF_URL, TIMELINE_PHASES
FACT_SHEET_PDF_URL = "/static/advocacy/Kei_Registration_Fact_Sheet.pdf"

# 2027 campaign master timeline: Feb 2026 → bill signed. Single source for /timeline page; update when session dates are known.
# start_date/end_date (YYYY-MM-DD) used to compute current phase for timeline node opacity.
TIMELINE_PHASES: list[dict] = [
    {
        "id": "build",
        "label": "Build",
        "date_range": "Feb 2026 – Dec 2026",
        "start_date": "2026-02-01",
        "end_date": "2026-12-31",
        "summary": "Lock sponsor(s), draft bill, grow list, and build coalition so we're ready for the 2027 session.",
        "milestones": [
            {
                "date": "Mar – May 2026",
                "title": "Lock lead sponsor(s)",
                "description": "Secure sponsor in the chamber you want to start in. Share draft concept so they're ready to file in January.",
            },
            {
                "date": "Jun – Aug 2026",
                "title": "Bill draft with LRB",
                "description": "Work with Legislative Reference Bureau or sponsor's staff. Nail down one-sentence ask and one-pager.",
            },
            {
                "date": "Sep – Nov 2026",
                "title": "Co-sponsors and coalition",
                "description": "Co-sponsor asks; recruit orgs; brief stakeholders. Plan pre-session pushes.",
            },
            {
                "date": "Dec 2026",
                "title": "Finalize intro plan",
                "description": "Bill number and intro plan with sponsor. Prep first-session campaign.",
            },
        ],
    },
    {
        "id": "intro",
        "label": "Session convenes & intro",
        "date_range": "Jan – Feb 2027",
        "start_date": "2027-01-01",
        "end_date": "2027-02-28",
        "summary": "105th GA convenes; bill introduced by the introduction deadline.",
        "milestones": [
            {
                "date": "Early Jan 2027",
                "title": "Session convenes",
                "description": "105th GA perfunctory/session days. LRB request deadline ~Jan 16.",
            },
            {
                "date": "Jan – early Feb 2027",
                "title": "Bill introduced",
                "description": 'File as soon as practical. Once filed, you have a bill number for "Support HB/SB XXXX" campaigns.',
            },
            {
                "date": "~Feb 6, 2027",
                "title": "Introduction deadline",
                "description": "House and Senate bill introduction deadline. Bill must be introduced by this date.",
            },
        ],
    },
    {
        "id": "committee-floor",
        "label": "Committee & floor",
        "date_range": "Feb – May 2027",
        "start_date": "2027-02-01",
        "end_date": "2027-05-31",
        "summary": "Hearings, committee votes, third reading in each chamber. Session adjourns late May.",
        "milestones": [
            {
                "date": "Feb 2027",
                "title": "Committee assignments",
                "description": "First hearings possible. Push witness slips and constituent contacts to committee members.",
            },
            {
                "date": "~Mar 13 / Mar 27, 2027",
                "title": "Committee deadlines",
                "description": "SB committee deadline ~Mar 13; HB committee deadline ~Mar 27 (substantive bills out of committee).",
            },
            {
                "date": "~Apr 17, 2027",
                "title": "Third reading (first chamber)",
                "description": "HB 3rd reading deadline (House); SB 3rd reading deadline (Senate).",
            },
            {
                "date": "~May 8 / May 22, 2027",
                "title": "Crossover and second chamber",
                "description": "House bills in Senate: committee ~May 8, 3rd reading ~May 22. Senate bills in House: same pattern.",
            },
            {
                "date": "~May 31, 2027",
                "title": "Session adjournment",
                "description": "If the bill passed both chambers, it goes to the governor.",
            },
        ],
    },
    {
        "id": "governor",
        "label": "Governor",
        "date_range": "Jun – Aug 2027",
        "start_date": "2027-06-01",
        "end_date": "2027-08-31",
        "summary": "Governor has 60 days from passage to sign or veto. Bill signed into law.",
        "milestones": [
            {
                "date": "Jun – Jul 2027",
                "title": "Governor review",
                "description": "60 days from passage to sign or veto. Signing often within a few weeks.",
            },
            {
                "date": "Jun – Aug 2027",
                "title": "Bill signed into law",
                "description": "Effective date is usually upon signing or Jan 1 of the next year, per the bill.",
            },
        ],
    },
]
