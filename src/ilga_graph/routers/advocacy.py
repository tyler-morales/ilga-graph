"""SSR advocacy routes: landing, drawer (call/email), search, letter template."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import advocacy_helpers as ah
from .. import config as cfg
from ..app_state import state
from ..config import DEV_MODE, SEED_MODE
from ..constants import CATEGORY_CHOICES, CATEGORY_COMMITTEES
from ..db import get_db
from ..db_models import OutreachEvent, User
from ..dependencies import get_current_user_optional
from ..member_lookup import find_member_by_district, find_member_by_id
from ..routers.outreach import get_outreach_aggregate

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_LETTER_PDF_PATH = (
    Path(__file__).resolve().parent.parent / "static" / "advocacy" / "letter-template.pdf"
)
_BRIEF_PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "static"
    / "advocacy"
    / "IL_Kei_Vehicle_Registration_Fix_Brief.pdf"
)

router = APIRouter()
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["dev_available"] = DEV_MODE
# SEO, share cards, analytics (base.html uses these; same as main.py globals)
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["site_name"] = cfg.SITE_NAME
templates.env.globals["meta_description"] = cfg.META_DESCRIPTION
templates.env.globals["og_image_url"] = cfg.OG_IMAGE_URL
templates.env.globals["umami_enabled"] = cfg.PROFILE == "prod" and bool(cfg.UMAMI_WEBSITE_ID)
templates.env.globals["umami_website_id"] = cfg.UMAMI_WEBSITE_ID
templates.env.globals["umami_script_url"] = cfg.UMAMI_SCRIPT_URL

# Kei truck facts for loading animation: informative, fun trivia (one shown per search).
_KEI_LOADING_FACTS: list[str] = [
    "Suzuki Carry, Honda Acty, Daihatsu Hijet — classic kei trucks.",
    "'Kei' means 'light' in Japanese — the vehicle class is kei jidōsha.",
    "Texas made kei trucks legal for on-road use in 2025.",
    "25+ year old kei vehicles are federally legal to import; states set registration rules.",
    "Kei engines are capped at 660cc — that's why they're so efficient.",
    "Subaru Sambar: so small it fits Japan's strict kei parking spaces.",
    "Mini trucks were never sold new in the US; the 25-year import rule is why you see them now.",
    "Illinois SOS sometimes brands titles 'Not Eligible for Registration' — we want clarity.",
    "Kei jidōsha = 'light vehicle' — Japan's most popular vehicle class.",
    "Mazda Scrum, Mitsubishi Minicab — more kei truck names to know.",
]


def _loading_facts(member_count: int, zip_count: int) -> list[str]:
    """Return kei truck facts for loading animation (one random fact shown per button press)."""
    return list(_KEI_LOADING_FACTS)


async def _build_search_results_context(
    zip_code: str,
    category: str,
    db: AsyncSession,
    user: User | None,
) -> dict[str, Any]:
    """Build context for the results partial. Assumes zip_code is in state.zip_to_district."""
    district_info = state.zip_to_district[zip_code]
    senate_district = district_info.il_senate
    house_district = district_info.il_house
    warnings: list[str] = []

    committee_codes = CATEGORY_COMMITTEES.get(category, [])
    committee_ids = ah.committee_member_ids(state, committee_codes) if committee_codes else None
    category_label = category if category else ""

    senator_member = (
        find_member_by_district(state, "senate", senate_district) if senate_district else None
    )
    senator_card = None
    if senator_member:
        senator_card = ah.member_to_card(
            state,
            senator_member,
            why=f"Represents IL Senate District {senate_district}, which contains ZIP {zip_code}.",
        )
        senator_card["script_hint"] = ah.build_script_hint_senator(
            senator_card, zip_code, senate_district
        )
        senator_card["script_sections"] = ah.build_script_sections_senator(
            senator_card, zip_code, senate_district
        )
        senator_card["email_subject"] = ah.build_email_subject(zip_code)
        senator_card["email_body"] = ah.build_email_body(
            senator_member.name,
            senator_card["script_hint"],
            has_public_email=bool(senator_member.email),
        )
    elif senate_district:
        warnings.append(
            f"Senate District {senate_district} (for ZIP {zip_code}) — "
            "senator not in current data (dev/seed mode has limited members)."
        )

    rep_member = find_member_by_district(state, "house", house_district) if house_district else None
    rep_card = None
    if rep_member:
        rep_card = ah.member_to_card(
            state,
            rep_member,
            why=f"Represents IL House District {house_district}, which contains ZIP {zip_code}.",
        )
        rep_card["script_hint"] = ah.build_script_hint_rep(rep_card, zip_code, house_district)
        rep_card["script_sections"] = ah.build_script_sections_rep(
            rep_card, zip_code, house_district
        )
        rep_card["email_subject"] = ah.build_email_subject(zip_code)
        rep_card["email_body"] = ah.build_email_body(
            rep_member.name,
            rep_card["script_hint"],
            has_public_email=bool(rep_member.email),
        )
    elif house_district:
        warnings.append(
            f"House District {house_district} (for ZIP {zip_code}) — "
            "representative not in current data (dev/seed mode has limited members)."
        )

    your_legislators: list[dict[str, Any]] = []
    for card, role_label, role_class in [
        (senator_card, "Your Senator", "role-senator"),
        (rep_card, "Your Representative", "role-rep"),
    ]:
        if card is None:
            continue
        your_legislators.append({"card": card, "role_label": role_label, "role_class": role_class})
    your_legislators.sort(
        key=lambda x: x["card"].get("moneyball_score") or 0,
        reverse=True,
    )

    exclude_dist = senate_district or ""
    broker_member, broker_why = ah.find_power_broker(
        state,
        exclude_dist,
        committee_ids=committee_ids,
        committee_codes=committee_codes or None,
        category_name=category_label,
    )

    broker_card = None
    if broker_member:
        broker_card = ah.member_to_card(state, broker_member, why=broker_why)
        broker_card["script_hint"] = ah.build_script_hint_broker(broker_card, broker_why)
        broker_card["script_sections"] = ah.build_script_sections_broker(broker_card, broker_why)
        broker_card["email_subject"] = ah.build_email_subject(zip_code)
        broker_card["email_body"] = ah.build_email_body(
            broker_member.name,
            broker_card["script_hint"],
            has_public_email=bool(broker_member.email),
        )

    error = "; ".join(warnings) if warnings else None
    result_member_ids: list[str] = []
    for item in your_legislators:
        result_member_ids.append(item["card"]["id"])
    for card in (senator_card, rep_card, broker_card):
        if card is not None:
            result_member_ids.append(card["id"])
    result_member_ids = list(dict.fromkeys(result_member_ids))

    user_called_member_ids: set[str] = set()
    user_emailed_member_ids: set[str] = set()
    if user and result_member_ids:
        outreach_result = await db.execute(
            select(OutreachEvent.member_id, OutreachEvent.kind)
            .where(OutreachEvent.user_id == user.id)
            .where(OutreachEvent.member_id.in_(result_member_ids))
            .where(OutreachEvent.kind.in_(["call", "email"]))
        )
        for mid, kind in outreach_result.all():
            if kind == "call":
                user_called_member_ids.add(mid)
            elif kind == "email":
                user_emailed_member_ids.add(mid)

    outreach_heat: dict[str, int] = {}
    if result_member_ids:
        heat_result = await db.execute(
            select(OutreachEvent.member_id, func.count(func.distinct(OutreachEvent.user_id)))
            .where(OutreachEvent.member_id.in_(result_member_ids))
            .where(OutreachEvent.kind.in_(["call", "email"]))
            .group_by(OutreachEvent.member_id)
        )
        outreach_heat = {str(mid): int(cnt) for mid, cnt in heat_result.all()}

    return {
        "seed_mode": cfg.SEED_MODE,
        "member_count": len(state.members),
        "zip_count": len(state.zip_to_district),
        "zip": zip_code,
        "category": category,
        "senate_district": senate_district,
        "house_district": house_district,
        "your_legislators": your_legislators,
        "senator": senator_card,
        "representative": rep_card,
        "broker": broker_card,
        "error": error,
        "user_called_member_ids": user_called_member_ids,
        "user_emailed_member_ids": user_emailed_member_ids,
        "outreach_heat": outreach_heat,
    }


@router.get("/")
async def advocacy_index(
    request: Request,
    zip: str = "",
    member_id: str = "",
    view: str = "",
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Render the advocacy search page. Accepts dev deep-link params when ?dev is present."""
    zip_param = (zip or "").strip()
    in_district = zip_param in state.zip_to_district if zip_param else False
    member_count = len(state.members)
    zip_count = len(state.zip_to_district)
    try:
        agg = await get_outreach_aggregate(db)
        calls_total = agg["calls_total"]
        calls_this_week = agg["calls_this_week"]
    except Exception:
        calls_total = 0
        calls_this_week = 0
    ctx: dict[str, Any] = {
        "request": request,
        "title": "Kei Truck Freedom",
        "hero_headline": "Don't Let Springfield Ban Our Kei Trucks.",
        "hero_subhead": (
            "The state wants them off the road. Enter your ZIP code to find your rep, "
            "get a custom script, and tell them why they need to protect our trucks. "
            "It takes 60 seconds."
        ),
        "categories": CATEGORY_CHOICES,
        "member_count": member_count,
        "zip_count": zip_count,
        "category": "Transportation",
        "calls_total": calls_total,
        "calls_this_week": calls_this_week,
        "features": cfg.get_client_features(),
        "loading_facts": _loading_facts(member_count, zip_count),
    }
    if zip:
        ctx["zip"] = zip
    elif cfg.DEV_MODE:
        ctx["zip"] = "60601"
    elif cfg.SEED_MODE:
        ctx["zip"] = "60601"
    if zip_param and in_district:
        results_ctx = await _build_search_results_context(zip_param, "Transportation", db, user)
        ctx.update(results_ctx)
    return templates.TemplateResponse("index.html", ctx)


@router.get("/test")
async def advocacy_test(request: Request):
    """Dev back door: jump to any advocacy feature without clicking through."""
    test_members = ah.test_member_list(state)
    default_zip = "60601"
    return templates.TemplateResponse(
        "advocacy_test.html",
        {
            "request": request,
            "test_members": test_members,
            "default_zip": default_zip,
        },
    )


@router.get("/letter-template")
async def advocacy_letter_template(request: Request):
    """Letter template HTML (print to PDF) — fallback if PDF not provided."""
    return templates.TemplateResponse(
        "letter_template.html",
        {"request": request},
    )


@router.get("/letter-template.pdf")
async def advocacy_letter_template_pdf():
    """Download constituent letter template PDF (static/advocacy/letter-template.pdf)."""
    if not _LETTER_PDF_PATH.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Letter template PDF not found. Add static/advocacy/letter-template.pdf."
            },
        )
    return FileResponse(
        path=str(_LETTER_PDF_PATH),
        media_type="application/pdf",
        filename="letter-template.pdf",
        headers={"Content-Disposition": "attachment; filename=letter-template.pdf"},
    )


@router.get("/IL_Kei_Vehicle_Registration_Fix_Brief.pdf")
async def advocacy_brief_pdf():
    """Download Kei vehicle registration fix brief PDF from static/advocacy."""
    if not _BRIEF_PDF_PATH.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "detail": (
                    "Brief PDF not found. Add static/advocacy/"
                    "IL_Kei_Vehicle_Registration_Fix_Brief.pdf."
                ),
            },
        )
    return FileResponse(
        path=str(_BRIEF_PDF_PATH),
        media_type="application/pdf",
        filename="IL_Kei_Vehicle_Registration_Fix_Brief.pdf",
        headers={
            "Content-Disposition": "attachment; filename=IL_Kei_Vehicle_Registration_Fix_Brief.pdf"  # noqa: E501
        },
    )


@router.get("/drawer")
async def advocacy_drawer(
    request: Request,
    view: str = "call",
    member_id: str = "",
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Return drawer body: view=call (script + form) or view=email (template)."""
    zip_code = (request.query_params.get("zip") or "").strip()
    photo_url_param = (request.query_params.get("photo_url") or "").strip()
    target_type_param = (request.query_params.get("target_type") or "").strip().upper()
    member_id_stripped = member_id.strip() if member_id else ""
    member = find_member_by_id(state, member_id_stripped) if member_id_stripped else None
    if member_id_stripped and member is None:
        return JSONResponse(
            {"detail": "Legislator not found."},
            status_code=404,
        )
    # Constituent checkbox: checked only when selected member is user's rep or senator for this zip
    is_constituent = False
    if zip_code and member:
        district_info = state.zip_to_district.get(zip_code)
        if district_info:
            senator_member = (
                find_member_by_district(state, "senate", district_info.il_senate)
                if district_info.il_senate
                else None
            )
            rep_member = (
                find_member_by_district(state, "house", district_info.il_house)
                if district_info.il_house
                else None
            )
            is_constituent = (senator_member and member.id == senator_member.id) or (
                rep_member and member.id == rep_member.id
            )
    legislator_name = member.name if member else ""
    phone = None
    if member:
        for office in member.offices:
            if office.phone:
                phone = office.phone
                break
    has_public_email = bool(member and member.email)
    recipient_email = (member.email or "") if member else ""

    if view == "email":
        show_call_nudge = True
        if user and member_id:
            r = await db.execute(
                select(func.count())
                .select_from(OutreachEvent)
                .where(
                    OutreachEvent.user_id == user.id,
                    OutreachEvent.member_id == member_id.strip(),
                    OutreachEvent.kind == "call",
                )
            )
            if (r.scalar() or 0) > 0:
                show_call_nudge = False

        target_type = "POWER_BROKER" if target_type_param == "POWER_BROKER" else "NON_COMMITTEE"
        chamber = getattr(member, "chamber", None) if member else None
        district = getattr(member, "district", None) if member else None
        subject_constituent = ah.build_email_subject_line(zip_code, variant="constituent")
        subject_general = ah.build_email_subject_line(zip_code, variant="general")
        body = ah.build_email_first_body(
            legislator_name,
            zip_code,
            chamber=chamber,
            district=district,
            target_type=target_type,
        )
        body_followup = ah.build_after_call_email_body(
            "",
            legislator_name,
            zip_code,
            chamber=chamber,
            district=district,
            target_type=target_type,
            call_date="",
        )
        legislator_display_name = ah.get_legislator_display_name(legislator_name, chamber, district)
        party_abbr = ""
        if member and (member.party or "").lower():
            if "republican" in (member.party or "").lower():
                party_abbr = "R"
            elif "democrat" in (member.party or "").lower():
                party_abbr = "D"
            else:
                party_abbr = (member.party or "")[:1]
        return templates.TemplateResponse(
            "_advocacy_drawer_email.html",
            {
                "request": request,
                "drawer_view": "email_first",
                "legislator_name": legislator_name,
                "legislator_display_name": legislator_display_name,
                "recipient_email": recipient_email,
                "contact_name": "",
                "has_public_email": has_public_email,
                "subject": subject_constituent,
                "subject_constituent": subject_constituent,
                "subject_general": subject_general,
                "body": body,
                "body_followup": body_followup,
                "body_first": body,
                "show_call_nudge": show_call_nudge,
                "show_go_to_call": not has_public_email,
                "zip_code": zip_code,
                "is_constituent": is_constituent,
                "party_abbr": party_abbr,
            },
        )

    photo_url = photo_url_param or (getattr(member, "photo_url", "") or "" if member else "")
    if photo_url and not photo_url.startswith(("http://", "https://")):
        photo_url = urljoin("https://www.ilga.gov/", photo_url)
    member_public_email = (member.email or "").strip() if member else ""
    target_type = "POWER_BROKER" if target_type_param == "POWER_BROKER" else "NON_COMMITTEE"
    drawer_ctx = ah.legislator_drawer_context(member)
    response = templates.TemplateResponse(
        "_advocacy_drawer_call.html",
        {
            "request": request,
            "legislator_name": legislator_name,
            "zip_code": zip_code,
            "is_constituent": is_constituent,
            "phone": phone or "",
            "member_id": member_id or "",
            "photo_url": photo_url,
            "member_public_email": member_public_email,
            "target_type": target_type,
            **drawer_ctx,
        },
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@router.post("/call/{call_id}/wrapup")
async def advocacy_call_wrapup(request: Request, call_id: str):
    """Wrap-up from call: swap drawer to Email view (prefilled or copy-only)."""
    form = await request.form()
    zip_code = (form.get("zip") or "").strip()
    staffer_name = (form.get("staffer_name") or "").strip()
    email_address = (form.get("email_address") or "").strip()
    next_step = (form.get("next_step") or "").strip()
    member_id = call_id.strip()
    member = find_member_by_id(state, member_id) if member_id else None
    legislator_name = member.name if member else ""
    recipient = (email_address or "").strip() or (member.email if member else "") or ""

    staffer = (staffer_name or "").strip() or ""
    target_type_form = (form.get("target_type") or "").strip().upper()
    target_type = "POWER_BROKER" if target_type_form == "POWER_BROKER" else "NON_COMMITTEE"
    call_date = (form.get("call_date") or "").strip()
    chamber = getattr(member, "chamber", None) if member else None
    district = getattr(member, "district", None) if member else None
    is_constituent = False
    if zip_code and member:
        district_info = state.zip_to_district.get(zip_code)
        if district_info:
            senator_member = (
                find_member_by_district(state, "senate", district_info.il_senate)
                if district_info.il_senate
                else None
            )
            rep_member = (
                find_member_by_district(state, "house", district_info.il_house)
                if district_info.il_house
                else None
            )
            is_constituent = (senator_member and member.id == senator_member.id) or (
                rep_member and member.id == rep_member.id
            )
    subject_constituent = ah.build_email_subject_line(zip_code, variant="constituent")
    subject_general = ah.build_email_subject_line(zip_code, variant="general")
    body = ah.build_after_call_email_body(
        staffer,
        legislator_name,
        zip_code,
        chamber=chamber,
        district=district,
        target_type=target_type,
        call_date=call_date,
    )
    body_first = ah.build_email_first_body(
        legislator_name,
        zip_code,
        chamber=chamber,
        district=district,
        target_type=target_type,
    )

    contact_name = staffer or ""
    legislator_display_name = ah.get_legislator_display_name(legislator_name, chamber, district)
    party_abbr = ""
    if member and (member.party or "").lower():
        if "republican" in (member.party or "").lower():
            party_abbr = "R"
        elif "democrat" in (member.party or "").lower():
            party_abbr = "D"
        else:
            party_abbr = (member.party or "")[:1]
    if recipient:
        return templates.TemplateResponse(
            "_advocacy_drawer_email.html",
            {
                "request": request,
                "drawer_view": "after_call",
                "legislator_name": legislator_name,
                "legislator_display_name": legislator_display_name,
                "recipient_email": recipient,
                "contact_name": contact_name,
                "has_public_email": True,
                "subject": subject_constituent,
                "subject_constituent": subject_constituent,
                "subject_general": subject_general,
                "body": body,
                "body_followup": body,
                "body_first": body_first,
                "show_call_nudge": False,
                "show_go_to_call": False,
                "copy_only_mode": False,
                "zip_code": zip_code,
                "is_constituent": is_constituent,
                "party_abbr": party_abbr,
            },
        )

    return templates.TemplateResponse(
        "_advocacy_drawer_email.html",
        {
            "request": request,
            "drawer_view": "after_call",
            "legislator_name": legislator_name,
            "legislator_display_name": legislator_display_name,
            "recipient_email": "",
            "contact_name": contact_name,
            "has_public_email": False,
            "subject": subject_constituent,
            "subject_constituent": subject_constituent,
            "subject_general": subject_general,
            "body": body,
            "body_followup": body,
            "body_first": body_first,
            "instructions": next_step,
            "show_call_nudge": False,
            "show_go_to_call": True,
            "copy_only_mode": True,
            "zip_code": zip_code,
            "is_constituent": is_constituent,
            "party_abbr": party_abbr,
        },
    )


@router.post("/call/{call_id}/no-answer")
async def advocacy_call_no_answer(request: Request, call_id: str):
    """No-answer / voicemail outcome: return guidance partial with next-step CTAs."""
    form = await request.form()
    zip_code = (form.get("zip") or "").strip()
    outcome = (form.get("outcome") or "no_answer").strip()
    member_id = call_id.strip()
    member = find_member_by_id(state, member_id) if member_id else None
    legislator_name = member.name if member else ""
    return templates.TemplateResponse(
        "_advocacy_drawer_no_answer.html",
        {
            "request": request,
            "legislator_name": legislator_name,
            "member_id": member_id,
            "zip_code": zip_code,
            "outcome": outcome,
        },
    )


@router.get("/api/check-constituent")
async def check_constituent(member_id: str = "", zip: str = ""):
    """Return whether the given ZIP is in the given member's district (constituent checkbox)."""
    zip_code = (zip or "").strip()
    member_id_stripped = (member_id or "").strip()
    if not member_id_stripped or not zip_code:
        return JSONResponse({"is_constituent": False})
    member = find_member_by_id(state, member_id_stripped)
    if not member:
        return JSONResponse({"is_constituent": False})
    district_info = state.zip_to_district.get(zip_code)
    if not district_info:
        return JSONResponse({"is_constituent": False})
    senator_member = (
        find_member_by_district(state, "senate", district_info.il_senate)
        if district_info.il_senate
        else None
    )
    rep_member = (
        find_member_by_district(state, "house", district_info.il_house)
        if district_info.il_house
        else None
    )
    is_constituent = (senator_member and member.id == senator_member.id) or (
        rep_member and member.id == rep_member.id
    )
    return JSONResponse({"is_constituent": is_constituent})


@router.post("/search")
async def advocacy_search(
    request: Request,
    zip_code: str = Form(...),
    category: str = Form(""),
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Look up advocacy targets for a given ZIP code and optional policy category.

    Returns up to three cards:

    1. **Your Senator** — IL Senate member for this ZIP's district.
    2. **Your Representative** — IL House member for this ZIP's district.
    3. **Power Broker** — highest Moneyball score in the Senate (different district).

    When *category* is provided, Power Broker is filtered to members who sit
    on a committee in that policy area.

    When the request comes from htmx (``HX-Request`` header), only the
    results partial is returned.
    """
    zip_code = zip_code.strip()
    category = category.strip()
    is_htmx = request.headers.get("HX-Request") == "true"

    district_info = state.zip_to_district.get(zip_code)
    if district_info is None:
        error = (
            f"ZIP code {zip_code!r} not found in Illinois district data. "
            "Please enter a valid 5-digit Illinois ZIP code."
        )
        if SEED_MODE and state.zip_to_district:
            sample = sorted(state.zip_to_district.keys())[:6]
            error += f" In dev mode, try ZIPs such as: {', '.join(sample)}."
        tpl = "_results_partial.html" if is_htmx else "index.html"
        ctx_error: dict[str, Any] = {
            "request": request,
            "title": "Kei Truck Freedom",
            "hero_headline": "Don't Let Springfield Ban Our Kei Trucks.",
            "hero_subhead": (
                "The state wants them off the road. Enter your ZIP code to find your rep, "
                "get a custom script, and tell them why they need to protect our trucks. "
                "It takes 60 seconds."
            ),
            "categories": CATEGORY_CHOICES,
            "zip": zip_code,
            "category": category or "Transportation",
            "error": error,
        }
        if not is_htmx:
            try:
                agg = await get_outreach_aggregate(db)
                ctx_error["calls_total"] = agg["calls_total"]
                ctx_error["calls_this_week"] = agg["calls_this_week"]
            except Exception:
                ctx_error["calls_total"] = 0
                ctx_error["calls_this_week"] = 0
        return templates.TemplateResponse(tpl, ctx_error)

    results_ctx = await _build_search_results_context(zip_code, category, db, user)
    tpl = "_results_partial.html" if is_htmx else "results.html"
    return templates.TemplateResponse(
        tpl,
        {
            "request": request,
            "title": "Kei Truck Freedom",
            "categories": CATEGORY_CHOICES,
            **results_ctx,
        },
    )
