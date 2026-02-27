"""SSR advocacy routes: landing, drawer (call/email), search, letter template."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import advocacy_helpers as ah
from .. import config as cfg
from ..app_state import state
from ..campaign_helpers import get_active_campaign, is_campaign_visible_to_zip
from ..community_email import get_effective_email_for_member
from ..config import DEV_MODE
from ..constants import CATEGORY_CHOICES, CATEGORY_COMMITTEES, GENERAL_COMMITTEE_CODES
from ..data_source import is_using_mocks
from ..db import get_db
from ..db_models import OutreachEvent, User
from ..dependencies import get_current_user_optional, require_user
from ..member_lookup import (
    find_member_by_district,
    find_member_by_id,
    is_constituent_for_zip_member,
)
from ..routers.content import (
    HERO_CLARITY_LINE,
    HERO_URGENCY_LINE,
    STRATEGIC_FIVE_POINTS,
)
from ..routers.outreach import get_outreach_aggregate
from ..security import (
    CSRF_COOKIE_NAME,
    validate_csrf_token,
    validate_photo_url_for_drawer,
)

_ZIP_RE = re.compile(r"^\d{5}$")
# Pre-fill hero ZIP in dev/mocks; must exist in state.zip_to_district.
DEFAULT_HERO_ZIP = "60007"

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
templates.env.globals["show_beta_banner"] = cfg.BETA_BANNER
templates.env.globals["beta_banner_feedback_url"] = cfg.BETA_BANNER_REPORT_URL
templates.env.globals["footer_last_updated"] = cfg.FOOTER_LAST_UPDATED
templates.env.globals["footer_last_updated_iso"] = cfg.FOOTER_LAST_UPDATED_ISO
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS
templates.env.globals["hero_urgency_line"] = HERO_URGENCY_LINE
templates.env.globals["hero_clarity_line"] = HERO_CLARITY_LINE
templates.env.globals["features"] = cfg.get_client_features()

from ..campaign_helpers import get_current_action_campaign_for_template  # noqa: E402

templates.env.globals["get_current_action_campaign"] = get_current_action_campaign_for_template

_HERO_SUBHEAD = (
    "Illinois is treating lawfully imported kei vehicles as off-highway, so owners cannot "
    "register them for normal road use. You can help fix it—contact your legislator with a "
    "pre-written script in under a minute."
)

# Two-line subhead: break after "below"; second line starts with "to".
_HERO_SUBHEAD_ADVOCACY_LINE1 = "We're building constituent support and identifying a sponsor."
_HERO_SUBHEAD_ADVOCACY_LINE2 = (
    " No bill yet — your rep needs to hear from you now so we're ready when legislation moves."
)


def _hero_context() -> dict[str, Any]:
    """Shared hero headline/subhead for home page (issue-focused)."""
    return {
        "hero_headline": "Fix the statutory gap. Allow kei vehicle registration.",
        "hero_headline_line1": "Fix the statutory gap.",
        "hero_headline_line1_prefix": "",
        "hero_headline_line1_highlight": "Fix",
        "hero_headline_line1_suffix": " the statutory gap.",
        "hero_headline_line2": "Allow kei vehicle registration.",
        "hero_headline_line2_prefix": "Allow ",
        "hero_headline_highlight": "kei vehicle",
        "hero_headline_line2_suffix": " registration.",
        "hero_subhead": _HERO_SUBHEAD,
    }


def _hero_context_advocacy() -> dict[str, Any]:
    """Advocacy-page hero: advocate-focused headline (find legislators, take action)."""
    return {
        "hero_headline": "Find your legislators. Build support for the Kei vehicle registration.",
        "hero_headline_line1": "Contact your legislators.",
        "hero_headline_line1_prefix": "",
        "hero_headline_line1_highlight": "",
        "hero_headline_line1_suffix": "",
        "hero_headline_line2": "Build support for the Kei registration fix",
        "hero_headline_line2_prefix": "",
        "hero_headline_highlight": "Build support",
        "hero_headline_line2_suffix": " for Kei vehicle registration.",
        "hero_subhead_line1": _HERO_SUBHEAD_ADVOCACY_LINE1,
        "hero_subhead_line2": _HERO_SUBHEAD_ADVOCACY_LINE2,
    }


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

    # Power Broker: default topic to Transportation when no category selected.
    topic_for_broker = category or "Transportation"
    committee_codes = CATEGORY_COMMITTEES.get(topic_for_broker, [])
    committee_ids = ah.committee_member_ids(state, committee_codes) if committee_codes else None
    category_label = category if category else ""
    # Topic + general committees (Appropriations, Assignments) for "Why we recommend" chair chips.
    relevant_committee_codes = list(dict.fromkeys(committee_codes + GENERAL_COMMITTEE_CODES))

    senator_member = (
        find_member_by_district(state, "senate", senate_district) if senate_district else None
    )
    senator_card = None
    if senator_member:
        senator_card = ah.member_to_card(
            state,
            senator_member,
            why=f"Represents IL Senate District {senate_district}, which contains ZIP {zip_code}.",
            relevant_committee_codes=relevant_committee_codes,
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
            chamber=senator_member.chamber,
            one_pager_points=STRATEGIC_FIVE_POINTS,
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
            relevant_committee_codes=relevant_committee_codes,
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
            chamber=rep_member.chamber,
            one_pager_points=STRATEGIC_FIVE_POINTS,
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

    broker_member, broker_why = ah.find_power_broker(
        state,
        exclude_senate_district=senate_district or "",
        exclude_house_district=house_district or "",
        committee_ids=committee_ids,
        committee_codes=committee_codes or None,
        category_name=category_label,
    )

    broker_card = None
    if broker_member:
        broker_card = ah.member_to_card(
            state,
            broker_member,
            why=broker_why,
            relevant_committee_codes=relevant_committee_codes,
        )
        broker_card["script_hint"] = ah.build_script_hint_broker(broker_card, broker_why)
        broker_card["script_sections"] = ah.build_script_sections_broker(broker_card, broker_why)
        broker_card["email_subject"] = ah.build_email_subject(zip_code)
        broker_card["email_body"] = ah.build_email_body(
            broker_member.name,
            broker_card["script_hint"],
            has_public_email=bool(broker_member.email),
            chamber=broker_member.chamber,
            one_pager_points=STRATEGIC_FIVE_POINTS,
        )

    error = "; ".join(warnings) if warnings else None
    result_member_ids: list[str] = []
    for item in your_legislators:
        result_member_ids.append(item["card"]["id"])
    for card in (senator_card, rep_card, broker_card):
        if card is not None:
            result_member_ids.append(card["id"])
    result_member_ids = list(dict.fromkeys(result_member_ids))
    result_member_ids_str = [str(mid) for mid in result_member_ids]

    user_called_member_ids: set[str] = set()
    user_emailed_member_ids: set[str] = set()
    if user and result_member_ids_str:
        outreach_result = await db.execute(
            select(OutreachEvent.member_id, OutreachEvent.kind)
            .where(OutreachEvent.user_id == user.id)
            .where(OutreachEvent.member_id.in_(result_member_ids_str))
            .where(OutreachEvent.kind.in_(["call", "email"]))
        )
        for mid, kind in outreach_result.all():
            mid_str = str(mid)
            if kind == "call":
                user_called_member_ids.add(mid_str)
            elif kind == "email":
                user_emailed_member_ids.add(mid_str)

    senator_id = str(senator_card["id"]) if senator_card else None
    rep_id = str(rep_card["id"]) if rep_card else None
    senator_called = senator_id is not None and senator_id in user_called_member_ids
    rep_called = rep_id is not None and rep_id in user_called_member_ids
    senator_emailed = senator_id is not None and senator_id in user_emailed_member_ids
    rep_emailed = rep_id is not None and rep_id in user_emailed_member_ids
    district_called_count = (1 if senator_called else 0) + (1 if rep_called else 0)
    district_goal_done = (
        (1 if senator_called else 0)
        + (1 if senator_emailed else 0)
        + (1 if rep_called else 0)
        + (1 if rep_emailed else 0)
    )
    district_goal_total = 2 * (1 if senator_card else 0) + 2 * (1 if rep_card else 0)
    both_district_members_called = (senator_card is None or senator_called) and (
        rep_card is None or rep_called
    )

    broker_id = str(broker_card["id"]) if broker_card else None
    broker_called = broker_id is not None and broker_id in user_called_member_ids
    broker_emailed = broker_id is not None and broker_id in user_emailed_member_ids
    broker_goal_done = (1 if broker_called else 0) + (1 if broker_emailed else 0)
    broker_goal_total = 2 if broker_card else 0
    district_goal_complete = district_goal_done == district_goal_total and district_goal_total > 0
    in_broker_phase = district_goal_complete and broker_card is not None

    # District steps (for phase 1 or for "completed goals" in phase 2).
    district_steps: list[dict[str, Any]] = []
    for item in your_legislators:
        card = item["card"]
        mid = str(card["id"])
        role_short = "Senator" if "Senator" in item["role_label"] else "Rep"
        district_steps.append(
            {
                "member_id": mid,
                "role_label": role_short,
                "action": "call",
                "done": mid in user_called_member_ids,
            }
        )
        district_steps.append(
            {
                "member_id": mid,
                "role_label": role_short,
                "action": "email",
                "done": mid in user_emailed_member_ids,
            }
        )

    broker_goal_steps: list[dict[str, Any]] = []
    if broker_card:
        broker_goal_steps = [
            {
                "member_id": broker_id,
                "role_label": "Power Broker",
                "action": "call",
                "done": broker_called,
            },
            {
                "member_id": broker_id,
                "role_label": "Power Broker",
                "action": "email",
                "done": broker_emailed,
            },
        ]

    if in_broker_phase:
        goal_phase = "broker"
        current_goal_label = "Contact the Power Broker"
        goal_steps = broker_goal_steps
        goal_done = broker_goal_done
        goal_total = broker_goal_total
        completed_goal_steps = [{**s, "done": True} for s in district_steps]
    else:
        goal_phase = "district"
        current_goal_label = "Contact your district legislators"
        goal_steps = district_steps
        goal_done = district_goal_done
        goal_total = district_goal_total
        completed_goal_steps = []

    goal_next_step: dict[str, Any] | None = None
    for s in goal_steps:
        if not s["done"]:
            goal_next_step = {
                "action": s["action"],
                "member_id": s["member_id"],
                "role_label": s["role_label"],
            }
            break

    outreach_heat: dict[str, int] = {}
    if result_member_ids_str:
        heat_result = await db.execute(
            select(OutreachEvent.member_id, func.count(func.distinct(OutreachEvent.user_id)))
            .where(OutreachEvent.member_id.in_(result_member_ids_str))
            .where(OutreachEvent.kind.in_(["call", "email"]))
            .group_by(OutreachEvent.member_id)
        )
        outreach_heat = {str(mid): int(cnt) for mid, cnt in heat_result.all()}

    outreach_sidebar: list[dict[str, Any]] = []
    outreach_calls_count = 0
    outreach_emails_count = 0
    if user and getattr(state, "member_lookup_by_id", None):
        sidebar_result = await db.execute(
            select(OutreachEvent.member_id, OutreachEvent.kind)
            .where(OutreachEvent.user_id == user.id)
            .where(OutreachEvent.kind.in_(["call", "email"]))
            .order_by(OutreachEvent.created_at.desc())
        )
        member_kinds: dict[str, set[str]] = {}
        # #region agent log
        _raw_events: list[list[str]] = []
        _total_call_events = 0
        _total_email_events = 0
        # #endregion
        for mid, kind in sidebar_result.all():
            if kind == "call":
                _total_call_events += 1
            elif kind == "email":
                _total_email_events += 1
            mid_str = str(mid)
            if mid_str not in member_kinds:
                member_kinds[mid_str] = set()
            member_kinds[mid_str].add(kind)
            # #region agent log
            _raw_events.append([mid_str, kind])
            # #endregion
        distinct_members_called = sum(1 for k in member_kinds.values() if "call" in k)
        distinct_members_emailed = sum(1 for k in member_kinds.values() if "email" in k)
        outreach_calls_count = distinct_members_called
        outreach_emails_count = distinct_members_emailed
        # #region agent log
        try:
            _log_path = Path("/Users/tyler/Projects/Code/ilga_graph_poc/.cursor/debug-40cdc3.log")
            with open(_log_path, "a") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "40cdc3",
                            "hypothesisId": "H1_H2_H3",
                            "runId": "post-fix",
                            "location": "advocacy.py:sidebar_counts",
                            "message": "outreach sidebar events and counts",
                            "data": {
                                "user_id": user.id,
                                "raw_events": _raw_events,
                                "total_call_events": _total_call_events,
                                "total_email_events": _total_email_events,
                                "outreach_calls_count_displayed": outreach_calls_count,
                                "outreach_emails_count_displayed": outreach_emails_count,
                            },
                            "timestamp": __import__("time").time() * 1000,
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        for member_id in member_kinds:
            member = state.member_lookup_by_id.get(member_id)
            if member is None:
                continue
            kinds = member_kinds[member_id]
            photo_url = getattr(member, "photo_url", "") or ""
            if photo_url and not photo_url.startswith(("http://", "https://")):
                photo_url = urljoin("https://www.ilga.gov/", photo_url.lstrip("/"))
            outreach_sidebar.append(
                {
                    "id": member_id,
                    "name": getattr(member, "name", "") or "",
                    "district": getattr(member, "district", "") or "",
                    "chamber": getattr(member, "chamber", "") or "",
                    "photo_url": photo_url,
                    "called": "call" in kinds,
                    "emailed": "email" in kinds,
                    "is_constituent": is_constituent_for_zip_member(state, zip_code, member),
                }
            )

    return {
        "seed_mode": is_using_mocks(),
        "member_count": len(state.members),
        "zip_count": len(state.zip_to_district),
        "zip": zip_code,
        "category": category,
        "relevant_committee_codes": relevant_committee_codes,
        "senate_district": senate_district,
        "house_district": house_district,
        "your_legislators": your_legislators,
        "senator": senator_card,
        "representative": rep_card,
        "broker": broker_card,
        "error": error,
        "user_called_member_ids": user_called_member_ids,
        "user_emailed_member_ids": user_emailed_member_ids,
        "senator_called": senator_called,
        "rep_called": rep_called,
        "senator_emailed": senator_emailed,
        "rep_emailed": rep_emailed,
        "district_called_count": district_called_count,
        "district_goal_done": district_goal_done,
        "district_goal_total": district_goal_total,
        "both_district_members_called": both_district_members_called,
        "broker_id": broker_id,
        "broker_called": broker_called,
        "broker_emailed": broker_emailed,
        "goal_phase": goal_phase,
        "current_goal_label": current_goal_label,
        "goal_done": goal_done,
        "goal_total": goal_total,
        "completed_goal_steps": completed_goal_steps,
        "district_steps": district_steps,
        "broker_goal_steps": broker_goal_steps,
        "goal_steps": goal_steps,
        "goal_next_step": goal_next_step,
        "outreach_heat": outreach_heat,
        "outreach_sidebar": outreach_sidebar,
        "outreach_calls_count": outreach_calls_count,
        "outreach_emails_count": outreach_emails_count,
        "show_my_outreach": user is not None,
    }


@router.get("")
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
    if zip_param and not _ZIP_RE.match(zip_param):
        zip_param = ""
    if not zip_param and user and getattr(user, "zip_code", None):
        saved = (user.zip_code or "").strip()
        if _ZIP_RE.match(saved) and saved in state.zip_to_district:
            zip_param = saved
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
    hero_ctx = _hero_context_advocacy()
    ctx: dict[str, Any] = {
        "request": request,
        "title": cfg.SITE_NAME,
        **hero_ctx,
        "categories": CATEGORY_CHOICES,
        "member_count": member_count,
        "zip_count": zip_count,
        "category": "Transportation",
        "calls_total": calls_total,
        "calls_this_week": calls_this_week,
        "features": cfg.get_client_features(),
    }
    if zip_param:
        ctx["zip"] = zip_param
    elif cfg.DEV_MODE or is_using_mocks():
        ctx["zip"] = DEFAULT_HERO_ZIP
    if zip_param and in_district:
        results_ctx = await _build_search_results_context(zip_param, "Transportation", db, user)
        ctx.update(results_ctx)
    elif zip_param and not in_district:
        ctx["error"] = (
            f"ZIP code {zip_param!r} not found in Illinois district data. "
            "Please enter a valid 5-digit Illinois ZIP code."
        )
        if is_using_mocks() and state.zip_to_district:
            sample = sorted(state.zip_to_district.keys())[:6]
            ctx["error"] += f" In dev mode, try ZIPs such as: {', '.join(sample)}."
        ctx["zip"] = DEFAULT_HERO_ZIP

    active_campaign = await get_active_campaign(db)
    if active_campaign:
        if active_campaign.target_type == "all":
            ctx["active_campaign"] = active_campaign
        elif zip_param and is_campaign_visible_to_zip(
            active_campaign, zip_param, state.zip_to_district
        ):
            ctx["active_campaign"] = active_campaign
        else:
            ctx["active_campaign"] = None
    else:
        ctx["active_campaign"] = None

    return templates.TemplateResponse("index.html", ctx)


@router.get("/test")
async def advocacy_test(request: Request):
    """Dev back door: jump to any advocacy feature. Only when DEV_MODE is on (local/dev)."""
    if not DEV_MODE:
        raise HTTPException(status_code=404, detail="Not found")
    test_members = ah.test_member_list(state)
    default_zip = DEFAULT_HERO_ZIP
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


def _role_label_for_member(member: Any) -> str | None:
    """'Senator' or 'Representative' based on the member's chamber. None for unknown."""
    chamber = (getattr(member, "chamber", None) or "").strip().lower()
    if chamber == "senate":
        return "Senator"
    if chamber == "house":
        return "Representative"
    return None


@router.get("/drawer")
async def advocacy_drawer(
    request: Request,
    view: str = "call",
    member_id: str = "",
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Return drawer body: view=call (script + form) or view=email (template)."""
    raw_zip = (request.query_params.get("zip") or "").strip()
    zip_code = raw_zip if _ZIP_RE.match(raw_zip) else ""
    photo_url_param = (request.query_params.get("photo_url") or "").strip()
    photo_url_validated = validate_photo_url_for_drawer(photo_url_param or None)
    target_type_param = (request.query_params.get("target_type") or "").strip().upper()
    member_id_stripped = member_id.strip() if member_id else ""
    member = find_member_by_id(state, member_id_stripped) if member_id_stripped else None
    if member_id_stripped and member is None:
        return JSONResponse(
            {"detail": "Legislator not found."},
            status_code=404,
        )
    is_constituent = is_constituent_for_zip_member(state, zip_code, member)
    legislator_name = member.name if member else ""
    phone = ah.get_preferred_phone_for_member(member)
    effective_email, email_source, community_verification = await get_effective_email_for_member(
        state, db, member_id_stripped
    )
    has_public_email = bool(effective_email)
    recipient_email = effective_email

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
            one_pager_points=STRATEGIC_FIVE_POINTS,
        )
        body_followup = ah.build_after_call_email_body(
            "",
            legislator_name,
            zip_code,
            chamber=chamber,
            district=district,
            target_type=target_type,
            call_date="",
            one_pager_points=STRATEGIC_FIVE_POINTS,
        )
        legislator_display_name = ah.get_legislator_display_name(legislator_name, chamber, district)
        party_abbr = ah.party_abbr_for_member(member)
        current_role_label = _role_label_for_member(member)
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
                "email_source": email_source,
                "community_verification": community_verification,
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
                "current_member_role_label": current_role_label,
                "current_member_already_called": not show_call_nudge,
            },
        )

    photo_url = photo_url_validated or (getattr(member, "photo_url", "") or "" if member else "")
    if photo_url and not photo_url.startswith(("http://", "https://")):
        photo_url = urljoin("https://www.ilga.gov/", photo_url)
    member_public_email = effective_email
    target_type = "POWER_BROKER" if target_type_param == "POWER_BROKER" else "NON_COMMITTEE"
    drawer_ctx = ah.legislator_drawer_context(member)

    call_completed = False
    call_notes = ""
    call_contact_name = ""
    call_support_score: int | None = None
    if user and member_id_stripped:
        r = await db.execute(
            select(OutreachEvent)
            .where(
                OutreachEvent.user_id == user.id,
                OutreachEvent.member_id == member_id_stripped,
                OutreachEvent.kind == "call",
            )
            .order_by(OutreachEvent.created_at.desc())
            .limit(1)
        )
        last_call = r.scalar_one_or_none()
        if last_call:
            call_completed = True
            call_notes = (last_call.notes or "").strip()
            call_contact_name = (last_call.contact_name or "").strip()
            if last_call.support_score is not None and 1 <= last_call.support_score <= 5:
                call_support_score = last_call.support_score

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
            "email_source": email_source,
            "community_verification": community_verification,
            "call_completed": call_completed,
            "call_notes": call_notes,
            "call_contact_name": call_contact_name,
            "call_support_score": call_support_score,
            **drawer_ctx,
        },
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@router.post("/call/{call_id}/wrapup")
async def advocacy_call_wrapup(
    request: Request,
    call_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Wrap-up from call: swap drawer to Email view (prefilled or copy-only)."""
    form = await request.form()
    raw_zip = (form.get("zip") or "").strip()
    zip_code = raw_zip if _ZIP_RE.match(raw_zip) else ""
    staffer_name = (form.get("staffer_name") or "").strip()
    email_address = (form.get("email_address") or "").strip()
    next_step = (form.get("next_step") or "").strip()
    member_id = call_id.strip()
    member = find_member_by_id(state, member_id) if member_id else None
    legislator_name = member.name if member else ""
    effective_email, email_source, community_verification = await get_effective_email_for_member(
        state, db, member_id
    )
    recipient = (email_address or "").strip() or effective_email or ""

    staffer = (staffer_name or "").strip() or ""
    target_type_form = (form.get("target_type") or "").strip().upper()
    target_type = "POWER_BROKER" if target_type_form == "POWER_BROKER" else "NON_COMMITTEE"
    call_date = (form.get("call_date") or "").strip()
    chamber = getattr(member, "chamber", None) if member else None
    district = getattr(member, "district", None) if member else None
    is_constituent = is_constituent_for_zip_member(state, zip_code, member)
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
        one_pager_points=STRATEGIC_FIVE_POINTS,
    )
    body_first = ah.build_email_first_body(
        legislator_name,
        zip_code,
        chamber=chamber,
        district=district,
        target_type=target_type,
        one_pager_points=STRATEGIC_FIVE_POINTS,
    )

    contact_name = staffer or ""
    legislator_display_name = ah.get_legislator_display_name(legislator_name, chamber, district)
    party_abbr = ah.party_abbr_for_member(member)
    used_community_recipient = not (email_address or "").strip() and bool(effective_email)
    wrapup_email_source = email_source if used_community_recipient else None
    wrapup_community_verification = (
        community_verification if used_community_recipient and email_source == "community" else None
    )
    wrapup_role_label = _role_label_for_member(member)
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
                "email_source": wrapup_email_source,
                "community_verification": wrapup_community_verification,
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
                "current_member_role_label": wrapup_role_label,
                "current_member_already_called": True,
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
            "email_source": None,
            "community_verification": None,
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
            "current_member_role_label": wrapup_role_label,
            "current_member_already_called": True,
        },
    )


@router.post("/call/{call_id}/no-answer")
async def advocacy_call_no_answer(request: Request, call_id: str):
    """No-answer / voicemail outcome: return guidance partial with next-step CTAs."""
    form = await request.form()
    raw_zip = (form.get("zip") or "").strip()
    zip_code = raw_zip if _ZIP_RE.match(raw_zip) else ""
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


def _reverse_geocode_to_zip(lat: float, lon: float) -> str | None:
    """Call Nominatim reverse geocoder; return 5-digit US ZIP or None."""
    import requests

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    headers = {"User-Agent": "ILGAAdvocacy/1.0 (Illinois legislator lookup)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        postcode = (data.get("address") or {}).get("postcode")
        if not postcode:
            return None
        digits = "".join(c for c in str(postcode) if c.isdigit())
        return digits[:5] if len(digits) >= 5 else None
    except Exception:
        return None


@router.get("/api/zip-from-coords")
async def zip_from_coords(lat: float = 0, lon: float = 0):
    """Reverse-geocode lat/lon to a 5-digit US ZIP. For use with browser geolocation."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return JSONResponse({"zip": None, "error": "Invalid coordinates"}, status_code=400)
    zip_code = await asyncio.to_thread(_reverse_geocode_to_zip, lat, lon)
    if not zip_code or not _ZIP_RE.match(zip_code):
        return JSONResponse({"zip": None, "error": "No ZIP found for this location"})
    return JSONResponse({"zip": zip_code})


@router.get("/api/check-constituent")
async def check_constituent(member_id: str = "", zip: str = ""):
    """Return whether the given ZIP is in the given member's district (constituent checkbox)."""
    zip_code = (zip or "").strip()
    member_id_stripped = (member_id or "").strip()
    if not member_id_stripped or not zip_code or not _ZIP_RE.match(zip_code):
        return JSONResponse({"is_constituent": False})
    member = find_member_by_id(state, member_id_stripped)
    is_constituent = is_constituent_for_zip_member(state, zip_code, member)
    return JSONResponse({"is_constituent": is_constituent})


@router.patch("/api/me/zip")
async def update_my_zip(
    request: Request,
    zip_code: str = Form(...),
    csrf_token: str | None = Form(None),
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Update current user's stored ZIP (hero zip / Use location commit only)."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        return JSONResponse(
            {"ok": False, "error": "Invalid or expired security token. Reload the page."},
            status_code=403,
        )
    zip_param = (zip_code or "").strip()
    if not _ZIP_RE.match(zip_param) or zip_param not in state.zip_to_district:
        return JSONResponse(
            {"ok": False, "error": "Invalid or unsupported Illinois ZIP code."},
            status_code=400,
        )
    user.zip_code = zip_param
    await db.commit()
    return JSONResponse({"ok": True})


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
    3. **Power Broker** — committee chair for the topic (default Transportation),
       or highest Moneyball senator or representative outside your district.

    When *category* is empty, topic defaults to Transportation for chair lookup.
    When *category* is provided, Power Broker is filtered to that policy area.

    When the request comes from htmx (``HX-Request`` header), only the
    results partial is returned.
    """
    zip_code = zip_code.strip()
    category = category.strip()
    is_htmx = request.headers.get("HX-Request") == "true"

    if not _ZIP_RE.match(zip_code):
        error = "Please enter a valid 5-digit Illinois ZIP code."
        tpl = "_results_partial.html" if is_htmx else "index.html"
        hero_ctx_err = _hero_context_advocacy()
        ctx_error: dict[str, Any] = {
            "request": request,
            "title": cfg.SITE_NAME,
            **hero_ctx_err,
            "categories": CATEGORY_CHOICES,
            "zip": "",
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

    district_info = state.zip_to_district.get(zip_code)
    if district_info is None:
        error = (
            f"ZIP code {zip_code!r} not found in Illinois district data. "
            "Please enter a valid 5-digit Illinois ZIP code."
        )
        if is_using_mocks() and state.zip_to_district:
            sample = sorted(state.zip_to_district.keys())[:6]
            error += f" In dev mode, try ZIPs such as: {', '.join(sample)}."
        tpl = "_results_partial.html" if is_htmx else "index.html"
        hero_ctx_err2 = _hero_context_advocacy()
        ctx_error = {
            "request": request,
            "title": cfg.SITE_NAME,
            **hero_ctx_err2,
            "categories": CATEGORY_CHOICES,
            "zip": DEFAULT_HERO_ZIP,
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
            "title": cfg.SITE_NAME,
            "categories": CATEGORY_CHOICES,
            **results_ctx,
        },
    )
