"""SSR advocacy routes: landing, drawer (call/email), search, letter template."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import advocacy_helpers as ah
from .. import config as cfg
from ..advocacy_helpers import CallerProfile
from ..app_state import state
from ..campaign_config import get_campaign_config
from ..campaign_helpers import get_active_campaign, is_campaign_visible_to_zip
from ..community_email import get_effective_email_for_member
from ..config import DEV_MODE
from ..constants import (
    ADV_CALL_PREF_COOKIE,
    ADV_CALL_PREF_MAX_AGE,
    ADV_CALL_PREF_VALUES,
    CATEGORY_CHOICES,
    CATEGORY_COMMITTEES,
    GENERAL_COMMITTEE_CODES,
    KEI_FIRST_OPTIONS,
    KEI_IMPACT_OPTIONS,
    KEI_IMPACT_SLUG_COOKIE,
    KEI_STATUS_BY_FIRST,
    KEI_STATUS_OPTIONS,
    KEI_STATUS_SLUGS,
)
from ..data_source import is_using_mocks
from ..db import get_db
from ..db_models import OutreachEvent, User
from ..dependencies import get_current_user_optional, require_user
from ..kei_poll_context import (
    KEI_POLL_CHOICE_COOKIE,
    KEI_POLL_VOTED_COOKIE,
    KEI_POLL_VOTED_MAX_AGE,
    get_kei_poll_sidebar_context,
)
from ..member_lookup import (
    find_member_by_district,
    find_member_by_id,
    is_constituent_for_zip_member,
)
from ..routers.content import (
    HERO_URGENCY_LINE,
    INTRO_CARD_WHY_CALL,
    STRATEGIC_FIVE_POINTS,
)
from ..routers.outreach import get_outreach_aggregate, get_outreach_count_for_member
from ..security import (
    CSRF_COOKIE_NAME,
    validate_csrf_token,
    validate_photo_url_for_drawer,
)
from ..session_schedule import get_milestone_by_id, get_next_deadline_safe

_ZIP_RE = re.compile(r"^\d{5}$")
# Pre-fill hero ZIP in dev/mocks; must exist in state.zip_to_district.
DEFAULT_HERO_ZIP = "60007"


def _visible_steps(steps: list[dict[str, Any]], user_call_pref: str | None) -> list[dict[str, Any]]:
    """Filter by pref: email-only see email steps; call_only/elevator see call steps."""
    if user_call_pref == "no":
        return [s for s in steps if s["action"] != "call"]
    if user_call_pref in ("call_only", "elevator"):
        return [s for s in steps if s["action"] != "email"]
    return steps


def _member_fits_carousel_pref(
    has_email: bool, has_phone: bool, user_call_pref: str | None
) -> bool:
    """True if member appears in carousel for this pref.
    When unset, require both so 'up next' doesn't force a choice."""
    if user_call_pref == "no":
        return has_email
    if user_call_pref in ("call_only", "elevator"):
        return has_phone
    if user_call_pref is not None:
        return has_phone or has_email
    return has_email and has_phone


def _build_district_steps(
    your_legislators: list[dict[str, Any]],
    user_called_member_ids: set[str],
    user_emailed_member_ids: set[str],
) -> list[dict[str, Any]]:
    """Build district goal steps (call + email only when member has effective email)."""
    district_steps: list[dict[str, Any]] = []
    for item in your_legislators:
        card = item["card"]
        mid = str(card["id"])
        role_short = "Senator" if "Senator" in item["role_label"] else "Rep"
        has_email = bool((card.get("email") or "").strip())
        district_steps.append(
            {
                "member_id": mid,
                "role_label": role_short,
                "action": "call",
                "done": mid in user_called_member_ids,
            }
        )
        if has_email:
            district_steps.append(
                {
                    "member_id": mid,
                    "role_label": role_short,
                    "action": "email",
                    "done": mid in user_emailed_member_ids,
                }
            )
    return district_steps


# Cookie names for script personalization (anon users and skip flag).
KEI_PERSONAL_NOTE_COOKIE = "kei_personal_note"
KEI_PERSONALIZATION_SKIPPED_COOKIE = "kei_personalization_skipped"
PERSONALIZATION_COOKIE_MAX_AGE = 365 * 24 * 60 * 60  # 1 year


def _caller_profile_from_request(request: Request, user: User | None) -> CallerProfile:
    """Build CallerProfile from user record or cookies (anon)."""
    if user:
        return CallerProfile(
            kei_status=getattr(user, "kei_status", None),
            kei_impact_slug=getattr(user, "kei_impact_slug", None),
            kei_personal_note=getattr(user, "kei_personal_note", None),
        )
    choice = request.cookies.get(KEI_POLL_CHOICE_COOKIE)
    kei_status = choice.strip() if choice and choice.strip() in KEI_STATUS_SLUGS else None
    impact = request.cookies.get(KEI_IMPACT_SLUG_COOKIE)
    note = request.cookies.get(KEI_PERSONAL_NOTE_COOKIE)
    return CallerProfile(
        kei_status=kei_status,
        kei_impact_slug=impact.strip() if impact and impact.strip() else None,
        kei_personal_note=note.strip() if note and note.strip() else None,
    )


def _caller_profile_complete(caller: CallerProfile, request: Request) -> bool:
    """True if we show script directly (skip cookie, or poll answered, or impact/note captured)."""
    if request.cookies.get(KEI_PERSONALIZATION_SKIPPED_COOKIE) == "1":
        return True
    if caller.kei_status:
        return True
    return bool(caller.kei_impact_slug or caller.kei_personal_note)


def _validate_kei_impact_slug(slug: str | None, kei_status: str | None) -> str | None:
    """Return slug if valid for the given kei_status, else None. 'other' is always allowed."""
    if not slug or not (s := slug.strip()):
        return None
    if s == "other":
        return s
    options = KEI_IMPACT_OPTIONS.get(kei_status or "", []) if kei_status else []
    valid = {opt[0] for opt in options}
    return s if s in valid else None


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_LETTER_PDF_PATH = _STATIC_DIR / "advocacy" / "letter-template.pdf"


def _brief_pdf_path() -> Path:
    """Legislator brief PDF path from campaign config."""
    cfg_campaign = get_campaign_config()
    return _STATIC_DIR / "advocacy" / cfg_campaign.brief_pdf_filename


router = APIRouter()
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["dev_available"] = DEV_MODE
# SEO, share cards, analytics (base.html uses these; same as main.py globals)
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


def _one_pager_points() -> list[str]:
    """One-pager points from campaign config; fallback to content constants."""
    points = get_campaign_config().one_pager_points
    return points if points else list(STRATEGIC_FIVE_POINTS)


def _default_topic() -> str:
    """Default policy topic for Power Broker / category from campaign config."""
    return get_campaign_config().default_topic


templates.env.globals["strategic_five_points"] = _one_pager_points()
templates.env.globals["hero_urgency_line"] = HERO_URGENCY_LINE
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
templates.env.globals["turnstile_site_key"] = (
    "" if cfg.TURNSTILE_DISABLED else (cfg.TURNSTILE_SITE_KEY or "")
)


def _hero_context() -> dict[str, Any]:
    """Shared hero headline/subhead for home page (issue-focused). From campaign config."""
    c = get_campaign_config()
    return {
        "hero_headline": c.hero_headline,
        "hero_headline_line1": c.hero_headline_line1,
        "hero_headline_line1_prefix": c.hero_headline_line1_prefix,
        "hero_headline_line1_highlight": c.hero_headline_line1_highlight,
        "hero_headline_line1_suffix": c.hero_headline_line1_suffix,
        "hero_headline_line2": c.hero_headline_line2,
        "hero_headline_line2_prefix": c.hero_headline_line2_prefix,
        "hero_headline_highlight": c.hero_headline_highlight,
        "hero_headline_line2_suffix": c.hero_headline_line2_suffix,
        "hero_subhead": c.hero_subhead,
    }


def _hero_context_advocacy() -> dict[str, Any]:
    """Advocacy-page hero: advocate-focused headline. From campaign config."""
    c = get_campaign_config()
    return {
        "hero_headline": c.advocacy_hero_headline,
        "hero_headline_line1": c.advocacy_hero_headline_line1,
        "hero_headline_line1_prefix": c.advocacy_hero_headline_line1_prefix,
        "hero_headline_line1_highlight": c.advocacy_hero_headline_line1_highlight,
        "hero_headline_line1_suffix": c.advocacy_hero_headline_line1_suffix,
        "hero_headline_line2": c.advocacy_hero_headline_line2,
        "hero_headline_line2_prefix": c.advocacy_hero_headline_line2_prefix,
        "hero_headline_highlight": c.advocacy_hero_headline_highlight,
        "hero_headline_line2_suffix": c.advocacy_hero_headline_line2_suffix,
        "hero_subhead_line1": c.advocacy_hero_subhead_line1,
        "hero_subhead_line2": c.advocacy_hero_subhead_line2,
    }


async def _build_search_results_context(
    request: Request,
    zip_code: str,
    category: str,
    db: AsyncSession,
    user: User | None,
    *,
    calls_total: int = 0,
) -> dict[str, Any]:
    """Build context for the results partial. Assumes zip_code is in state.zip_to_district."""
    caller = _caller_profile_from_request(request, user)
    user_call_pref: str | None = None
    if user and (p := getattr(user, "call_pref", None)) and p in ADV_CALL_PREF_VALUES:
        user_call_pref = p
    if user_call_pref is None:
        user_call_pref = request.cookies.get(ADV_CALL_PREF_COOKIE)
        if user_call_pref not in ADV_CALL_PREF_VALUES:
            user_call_pref = None
    district_info = state.zip_to_district[zip_code]
    senate_district = district_info.il_senate
    house_district = district_info.il_house
    warnings: list[str] = []

    # Power Broker: default topic from campaign config when no category selected.
    topic_for_broker = category or get_campaign_config().default_topic
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
            senator_card, zip_code, senate_district, calls_total=calls_total
        )
        senator_card["script_sections"] = ah.build_script_sections_senator(
            senator_card, zip_code, senate_district, calls_total=calls_total
        )
        senator_card["email_subject"] = ah.build_email_subject(zip_code, district=senate_district)
        senator_card["email_body"] = ah.build_email_body(
            senator_member.name,
            senator_card["script_hint"],
            has_public_email=bool(senator_member.email),
            chamber=senator_member.chamber,
            district=senate_district,
            one_pager_points=_one_pager_points(),
            calls_total=calls_total,
            caller=caller,
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
        rep_card["script_hint"] = ah.build_script_hint_rep(
            rep_card, zip_code, house_district, calls_total=calls_total
        )
        rep_card["script_sections"] = ah.build_script_sections_rep(
            rep_card, zip_code, house_district, calls_total=calls_total
        )
        rep_card["email_subject"] = ah.build_email_subject(zip_code, district=house_district)
        rep_card["email_body"] = ah.build_email_body(
            rep_member.name,
            rep_card["script_hint"],
            has_public_email=bool(rep_member.email),
            chamber=rep_member.chamber,
            district=house_district,
            one_pager_points=_one_pager_points(),
            calls_total=calls_total,
            caller=caller,
        )
    elif house_district:
        warnings.append(
            f"House District {house_district} (for ZIP {zip_code}) — "
            "representative not in current data (dev/seed mode has limited members)."
        )

    # District legislators in Moneyball order (higher first) so progress bar and carousel match.
    your_legislators: list[dict[str, Any]] = []
    for card, role_label, role_class in [
        (senator_card, "Your Senator", "role-senator"),
        (rep_card, "Your Representative", "role-rep"),
    ]:
        if card is None:
            continue
        your_legislators.append({"card": card, "role_label": role_label, "role_class": role_class})
    your_legislators.sort(
        key=lambda x: float(x["card"].get("moneyball_score") or 0),
        reverse=True,
    )

    power_brokers = ah.find_power_brokers(
        state,
        exclude_senate_district=senate_district or "",
        exclude_house_district=house_district or "",
        committee_ids=committee_ids,
        committee_codes=committee_codes or None,
        category_name=category_label,
    )

    broker_cards: list[dict[str, Any]] = []
    for broker_member, broker_why in power_brokers:
        broker_card = ah.member_to_card(
            state,
            broker_member,
            why=broker_why,
            relevant_committee_codes=relevant_committee_codes,
        )
        broker_card["script_hint"] = ah.build_script_hint_broker(
            broker_card, broker_why, calls_total=calls_total
        )
        broker_card["script_sections"] = ah.build_script_sections_broker(
            broker_card, broker_why, calls_total=calls_total
        )
        broker_card["email_subject"] = ah.build_email_subject(
            zip_code, district=getattr(broker_member, "district", None)
        )
        broker_card["email_body"] = ah.build_email_body(
            broker_member.name,
            broker_card["script_hint"],
            has_public_email=bool(broker_member.email),
            chamber=broker_member.chamber,
            district=getattr(broker_member, "district", None),
            one_pager_points=_one_pager_points(),
            calls_total=calls_total,
            caller=caller,
        )
        broker_cards.append(broker_card)

    # Resolve effective email (public + community) per card so template shows correct lock state.
    for card, member in [
        (senator_card, senator_member),
        (rep_card, rep_member),
    ]:
        if card is not None and member is not None:
            effective_email, _, _ = await get_effective_email_for_member(state, db, str(member.id))
            card["email"] = effective_email or ""
    for broker_card, (broker_member, _) in zip(broker_cards, power_brokers):
        effective_email, _, _ = await get_effective_email_for_member(
            state, db, str(broker_member.id)
        )
        broker_card["email"] = effective_email or ""

    error = "; ".join(warnings) if warnings else None
    result_member_ids: list[str] = []
    for item in your_legislators:
        result_member_ids.append(item["card"]["id"])
    for card in (senator_card, rep_card, *broker_cards):
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
    both_district_members_called = (senator_card is None or senator_called) and (
        rep_card is None or rep_called
    )

    broker_id = str(broker_cards[0]["id"]) if broker_cards else None
    broker_called = (
        any(str(c["id"]) in user_called_member_ids for c in broker_cards) if broker_cards else False
    )
    broker_emailed = (
        any(str(c["id"]) in user_emailed_member_ids for c in broker_cards)
        if broker_cards
        else False
    )

    # Which members fit the user's pref (same logic as carousel). Goal steps only include these.
    call_only_or_elevator_pref = user_call_pref in ("call_only", "elevator")
    carousel_member_ids: set[str] = set()
    for item in your_legislators:
        card = item["card"]
        he = bool((card.get("email") or "").strip())
        hp = bool((card.get("phone") or "").strip())
        if _member_fits_carousel_pref(he, hp, user_call_pref):
            carousel_member_ids.add(str(card["id"]))
    for broker_card in broker_cards:
        he = bool((broker_card.get("email") or "").strip())
        hp = bool((broker_card.get("phone") or "").strip())
        if _member_fits_carousel_pref(he, hp, user_call_pref):
            carousel_member_ids.add(str(broker_card["id"]))

    # District steps (phase 1 or "completed goals" in phase 2). Only steps for members in carousel.
    district_steps = _build_district_steps(
        your_legislators, user_called_member_ids, user_emailed_member_ids
    )
    district_steps = [s for s in district_steps if s["member_id"] in carousel_member_ids]
    district_goal_done = sum(1 for s in district_steps if s["done"])
    district_goal_total = len(district_steps)

    broker_goal_steps = []
    for broker_card in broker_cards:
        bid = str(broker_card["id"])
        if bid not in carousel_member_ids:
            continue
        broker_goal_steps.append(
            {
                "member_id": bid,
                "role_label": "Power Broker",
                "action": "call",
                "done": bid in user_called_member_ids,
            }
        )
        if bool((broker_card.get("email") or "").strip()):
            broker_goal_steps.append(
                {
                    "member_id": bid,
                    "role_label": "Power Broker",
                    "action": "email",
                    "done": bid in user_emailed_member_ids,
                }
            )

    broker_goal_done = sum(1 for s in broker_goal_steps if s["done"])
    broker_goal_total = len(broker_goal_steps)

    # Visible steps: same filter as template (dots); used for bar position and "Now" next step.
    visible_district_steps = _visible_steps(district_steps, user_call_pref)
    visible_district_done = sum(1 for s in visible_district_steps if s["done"])
    visible_district_total = len(visible_district_steps)
    visible_broker_steps = _visible_steps(broker_goal_steps, user_call_pref)
    visible_broker_done = sum(1 for s in visible_broker_steps if s["done"])
    visible_broker_total = len(visible_broker_steps)

    # District phase complete when all visible steps are done (email-only: 2 steps; call+email: 4).
    district_goal_complete = (
        visible_district_done == visible_district_total and visible_district_total > 0
    )
    in_broker_phase = district_goal_complete and len(broker_cards) > 0

    def _checkpoint_fill_pct(done: int, total: int) -> float:
        """Fill width so bar extends to the last completed step."""
        if total <= 1:
            return 100.0 if done else 0.0
        idx = min(done, total - 1)
        return round(100.0 * idx / (total - 1), 1)

    def _truck_on_checkpoint_pct(done: int, total: int) -> float:
        """Truck on next checkpoint (upcoming action); moves to next dot once a goal completes."""
        if total <= 1:
            return 100.0 if done else 0.0
        if done >= total:
            return 100.0
        idx = min(done, total - 1)  # next action index (0-based)
        return round(100.0 * idx / (total - 1), 1)

    district_fill_pct = _checkpoint_fill_pct(visible_district_done, visible_district_total)
    district_truck_pct = _truck_on_checkpoint_pct(visible_district_done, visible_district_total)
    broker_fill_pct = _checkpoint_fill_pct(visible_broker_done, visible_broker_total)
    broker_truck_pct = _truck_on_checkpoint_pct(visible_broker_done, visible_broker_total)

    # District-phase goal label: dynamic count by contact preference.
    # Email-only: legislators with email; phone: both district members (caller can ask for email).
    if user_call_pref == "no":
        district_legislator_count = sum(
            1
            for card in (senator_card, rep_card)
            if card is not None and bool((card.get("email") or "").strip())
        )
    else:
        district_legislator_count = (1 if senator_card else 0) + (1 if rep_card else 0)
    if user_call_pref == "no":
        district_goal_label = (
            "Email 1 legislator"
            if district_legislator_count == 1
            else "Email 2 legislators"
            if district_legislator_count >= 2
            else "Email your district legislators"
        )
    else:
        district_goal_label = (
            "Contact 1 legislator"
            if district_legislator_count == 1
            else "Contact 2 legislators"
            if district_legislator_count >= 2
            else "Contact your district legislators"
        )

    if in_broker_phase:
        goal_phase = "broker"
        current_goal_label = (
            "Contact the Power Brokers" if len(broker_cards) > 1 else "Contact the Power Broker"
        )
        goal_steps = broker_goal_steps
        goal_done = broker_goal_done
        goal_total = broker_goal_total
        completed_goal_steps = [{**s, "done": True} for s in district_steps]
        visible_goal_steps = visible_broker_steps
    else:
        goal_phase = "district"
        current_goal_label = district_goal_label
        goal_steps = district_steps
        goal_done = district_goal_done
        goal_total = district_goal_total
        completed_goal_steps = []
        visible_goal_steps = visible_district_steps

    goal_next_step: dict[str, Any] | None = None
    for s in visible_goal_steps:
        if not s["done"]:
            goal_next_step = {
                "action": s["action"],
                "member_id": s["member_id"],
                "role_label": s["role_label"],
            }
            break

    outreach_heat: dict[str, int] = {}
    total_advocates = 0
    if result_member_ids_str:
        heat_result = await db.execute(
            select(OutreachEvent.member_id, func.count(func.distinct(OutreachEvent.user_id)))
            .where(OutreachEvent.member_id.in_(result_member_ids_str))
            .where(OutreachEvent.kind.in_(["call", "email"]))
            .group_by(OutreachEvent.member_id)
        )
        outreach_heat = {str(mid): int(cnt) for mid, cnt in heat_result.all()}
        total_advocates = sum(outreach_heat.values())

    # Carousel: only show members that fit the user's outreach preference (same set as goal steps).
    members_for_carousel = []
    for item in your_legislators:
        card = item["card"]
        has_email = bool((card.get("email") or "").strip())
        has_phone = bool((card.get("phone") or "").strip())
        if not _member_fits_carousel_pref(has_email, has_phone, user_call_pref):
            continue
        mid = str(card["id"])
        if user_call_pref == "no":
            completed = mid in user_emailed_member_ids
        elif call_only_or_elevator_pref:
            completed = mid in user_called_member_ids
        else:
            completed = mid in user_called_member_ids and mid in user_emailed_member_ids
        members_for_carousel.append(
            {
                "card": card,
                "role_label": item["role_label"],
                "role_class": item["role_class"],
                "completed": completed,
            }
        )
    for broker_card in broker_cards:
        has_email = bool((broker_card.get("email") or "").strip())
        has_phone = bool((broker_card.get("phone") or "").strip())
        if _member_fits_carousel_pref(has_email, has_phone, user_call_pref):
            mid = str(broker_card["id"])
            if user_call_pref == "no":
                completed = mid in user_emailed_member_ids
            elif user_call_pref in ("call_only", "elevator"):
                completed = mid in user_called_member_ids
            else:
                completed = mid in user_called_member_ids and mid in user_emailed_member_ids
            members_for_carousel.append(
                {
                    "card": broker_card,
                    "role_label": "Power Broker",
                    "role_class": "role-broker",
                    "completed": completed,
                }
            )

    # Completed members at the back so the next actionable card is first.
    members_for_carousel.sort(key=lambda m: m["completed"])

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
        for mid, kind in sidebar_result.all():
            mid_str = str(mid)
            if mid_str not in member_kinds:
                member_kinds[mid_str] = set()
            member_kinds[mid_str].add(kind)
        distinct_members_called = sum(1 for k in member_kinds.values() if "call" in k)
        distinct_members_emailed = sum(1 for k in member_kinds.values() if "email" in k)
        outreach_calls_count = distinct_members_called
        outreach_emails_count = distinct_members_emailed
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
        "broker": broker_cards[0] if broker_cards else None,
        "broker_cards": broker_cards,
        "members": members_for_carousel,
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
        "district_fill_pct": district_fill_pct,
        "district_truck_pct": district_truck_pct,
        "visible_district_done": visible_district_done,
        "visible_district_total": visible_district_total,
        "both_district_members_called": both_district_members_called,
        "broker_id": broker_id,
        "broker_ids": [str(c["id"]) for c in broker_cards],
        "broker_goal_label": (
            "Contact the Power Brokers" if len(broker_cards) > 1 else "Contact the Power Broker"
        ),
        "broker_called": broker_called,
        "broker_emailed": broker_emailed,
        "goal_phase": goal_phase,
        "current_goal_label": current_goal_label,
        "goal_done": goal_done,
        "goal_total": goal_total,
        "broker_fill_pct": broker_fill_pct,
        "broker_truck_pct": broker_truck_pct,
        "visible_broker_done": visible_broker_done,
        "visible_broker_total": visible_broker_total,
        "completed_goal_steps": completed_goal_steps,
        "district_steps": district_steps,
        "broker_goal_steps": broker_goal_steps,
        "goal_steps": goal_steps,
        "goal_next_step": goal_next_step,
        "outreach_heat": outreach_heat,
        "total_advocates": total_advocates,
        "outreach_sidebar": outreach_sidebar,
        "outreach_calls_count": outreach_calls_count,
        "outreach_emails_count": outreach_emails_count,
        "show_my_outreach": user is not None,
        "calls_total": calls_total,
        "user_call_pref": user_call_pref,
        "intro_card_why_call": INTRO_CARD_WHY_CALL,
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
        "user": user,
        **hero_ctx,
        "categories": CATEGORY_CHOICES,
        "member_count": member_count,
        "zip_count": zip_count,
        "category": _default_topic(),
        "calls_total": calls_total,
        "calls_this_week": calls_this_week,
        "features": cfg.get_client_features(),
    }
    if zip_param:
        ctx["zip"] = zip_param
    elif cfg.DEV_MODE or is_using_mocks():
        ctx["zip"] = DEFAULT_HERO_ZIP
    if zip_param and in_district:
        results_ctx = await _build_search_results_context(
            request, zip_param, _default_topic(), db, user, calls_total=calls_total
        )
        ctx.update(results_ctx)
        ctx.update(await get_kei_poll_sidebar_context(request, user, db))
        ctx["poll_on_advocacy_page"] = True
        ctx["hide_outreach_cta"] = True
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

    return templates.TemplateResponse(request, "index.html", ctx)


@router.get("/test")
async def advocacy_test(request: Request):
    """Dev back door: jump to any advocacy feature. Only when DEV_MODE is on (local/dev)."""
    if not DEV_MODE:
        raise HTTPException(status_code=404, detail="Not found")
    test_members = ah.test_member_list(state)
    default_zip = DEFAULT_HERO_ZIP
    return templates.TemplateResponse(
        request,
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
        request,
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


@router.post("/set-call-pref")
async def set_call_pref(
    request: Request,
    pref: str = Form(...),
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Set call preference. Cookie always; DB when logged in. Returns HTMX fragment.
    When pref is 'none' (user declined every contact method in the tree), we do not persist
    and return the tree with a message to select at least one. pref='no' means email-only (valid).
    """
    # #region agent log
    try:
        _dbg = open("/Users/tyler/Projects/Code/hardball/.cursor/debug-4332d0.log", "a")
        _dbg.write(
            '{"sessionId":"4332d0","hypothesisId":"H1","location":"advocacy.py:set_call_pref",'
            '"message":"pref received","data":{"pref":' + repr(pref) + "},"
            '"timestamp":' + str(__import__("time").time_ns() // 1_000_000) + "}\n"
        )
        _dbg.close()
    except Exception:
        pass
    # #endregion
    if pref == "none":
        return templates.TemplateResponse(
            request,
            "_advocacy_intro_pref_choose_one.html",
            {"request": request},
        )
    if pref not in ADV_CALL_PREF_VALUES:
        raise HTTPException(
            status_code=422,
            detail="pref must be one of: no, yes, call_only, elevator",
        )
    # #region agent log
    try:
        _dbg3 = open("/Users/tyler/Projects/Code/hardball/.cursor/debug-4332d0.log", "a")
        _dbg3.write(
            '{"sessionId":"4332d0","hypothesisId":"H1","location":"advocacy.py:set_call_pref",'
            '"message":"branch: persisting pref (success)","data":{"pref":' + repr(pref) + "},"
            '"timestamp":' + str(__import__("time").time_ns() // 1_000_000) + "}\n"
        )
        _dbg3.close()
    except Exception:
        pass
    # #endregion
    if user:
        user.call_pref = pref
        await db.commit()
    res = templates.TemplateResponse(
        request,
        "_advocacy_intro_pref_saved.html",
        {"request": request, "pref": pref},
    )
    res.set_cookie(
        ADV_CALL_PREF_COOKIE,
        pref,
        max_age=ADV_CALL_PREF_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    res.headers["HX-Trigger"] = "refreshResults"
    return res


def _serve_brief_pdf() -> FileResponse | JSONResponse:
    """Serve legislator brief PDF from campaign config path."""
    path = _brief_pdf_path()
    if not path.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "detail": (
                    "Brief PDF not found. Add static/advocacy/"
                    f"{get_campaign_config().brief_pdf_filename}"
                ),
            },
        )
    filename = get_campaign_config().brief_pdf_filename
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/brief.pdf")
async def advocacy_brief_pdf_canonical():
    """Download legislator brief PDF (campaign-configured). Use this URL in app links."""
    return _serve_brief_pdf()


@router.get("/IL_Kei_Vehicle_Registration_Fix_Brief.pdf")
async def advocacy_brief_pdf_legacy():
    """Legacy URL for Kei brief; serves campaign-configured brief for backward compatibility."""
    return _serve_brief_pdf()


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
    effective_email, email_source, community_verification = await get_effective_email_for_member(
        state, db, member_id_stripped
    )
    has_public_email = bool(effective_email)
    recipient_email = effective_email

    caller = _caller_profile_from_request(request, user)
    show_personalization = (
        view == "call"
        and member_id_stripped
        and member is not None
        and not _caller_profile_complete(caller, request)
    )

    if view == "email":
        if not has_public_email:
            return templates.TemplateResponse(
                request,
                "_advocacy_drawer_no_email.html",
                {
                    "request": request,
                    "member_id": member_id_stripped,
                    "zip_code": zip_code,
                },
            )
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
        # When user chose email-only, do not show call nudge in drawer.
        user_call_pref_drawer: str | None = None
        if user and (p := getattr(user, "call_pref", None)) and p in ADV_CALL_PREF_VALUES:
            user_call_pref_drawer = p
        if user_call_pref_drawer is None:
            user_call_pref_drawer = request.cookies.get(ADV_CALL_PREF_COOKIE)
            if user_call_pref_drawer not in ADV_CALL_PREF_VALUES:
                user_call_pref_drawer = None
        if user_call_pref_drawer == "no":
            show_call_nudge = False
        target_type = "POWER_BROKER" if target_type_param == "POWER_BROKER" else "NON_COMMITTEE"
        chamber = getattr(member, "chamber", None) if member else None
        district = getattr(member, "district", None) if member else None
        try:
            agg = await get_outreach_aggregate(db)
            drawer_calls_total = agg["calls_total"]
        except Exception:
            drawer_calls_total = 0
        subject_constituent = ah.build_email_subject_line(
            zip_code, variant="constituent", district=district
        )
        subject_general = ah.build_email_subject_line(zip_code, variant="general")
        body = ah.build_email_first_body(
            legislator_name,
            zip_code,
            chamber=chamber,
            district=district,
            target_type=target_type,
            one_pager_points=_one_pager_points(),
            calls_total=drawer_calls_total,
            caller=caller,
        )
        body_followup = ah.build_after_call_email_body(
            "",
            legislator_name,
            zip_code,
            chamber=chamber,
            district=district,
            target_type=target_type,
            call_date="",
            one_pager_points=_one_pager_points(),
            calls_total=drawer_calls_total,
            caller=caller,
        )
        legislator_display_name = ah.get_legislator_display_name(legislator_name, chamber, district)
        party_abbr = ah.party_abbr_for_member(member)
        current_role_label = _role_label_for_member(member)
        c = get_campaign_config()
        return templates.TemplateResponse(
            request,
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
                "brief_pdf_url": c.brief_pdf_url_path,
                "brief_pdf_download_name": c.brief_pdf_filename,
            },
        )

    if show_personalization:
        photo_url_for_personalize = photo_url_validated or (
            (getattr(member, "photo_url", "") or "") if member else ""
        )
        if photo_url_for_personalize and not photo_url_for_personalize.startswith(
            ("http://", "https://")
        ):
            photo_url_for_personalize = urljoin("https://www.ilga.gov/", photo_url_for_personalize)
        # Flat list (slug, label, first_choice) for step 2; template uses data-first to show 3 or 2.
        kei_status_with_first = [
            (slug, label, first_key)
            for first_key, opts in KEI_STATUS_BY_FIRST.items()
            for slug, label in opts
        ]
        return templates.TemplateResponse(
            request,
            "_kei_personalize_drawer.html",
            {
                "request": request,
                "member_id": member_id_stripped,
                "zip_code": zip_code,
                "photo_url": photo_url_for_personalize,
                "target_type": target_type_param or "NON_COMMITTEE",
                "legislator_name": legislator_name,
                "is_constituent": is_constituent,
                "kei_first_options": KEI_FIRST_OPTIONS,
                "kei_status_options": KEI_STATUS_OPTIONS,
                "kei_status_with_first": kei_status_with_first,
                "kei_status_selected": caller.kei_status,
                "kei_impact_options": KEI_IMPACT_OPTIONS.get(caller.kei_status or "", []),
                "kei_impact_options_by_status": KEI_IMPACT_OPTIONS,
            },
        )

    return await _render_call_drawer(
        request,
        member_id_stripped,
        zip_code,
        photo_url_validated,
        target_type_param,
        user,
        db,
        caller,
    )


@router.post("/personalize-poll")
async def advocacy_personalize_poll(
    request: Request,
    kei_status: str | None = Form(None),
    kei_impact_slug: str | None = Form(None),
    csrf_token: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Persist poll choice (step 2/3) when user answers; called from JS, no HTML swap."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token")
    updated_caller = _caller_profile_from_request(request, user)
    raw_status = (kei_status or "").strip()
    kei_status_val = raw_status if raw_status in KEI_STATUS_SLUGS else None
    impact_val = _validate_kei_impact_slug(
        (kei_impact_slug or "").strip() or None, updated_caller.kei_status or kei_status_val
    )
    # Only update user/cookies here. Responses are counted only after Turnstile-verified
    # submit at /updates/kei-status (last question), not when saving preferences.
    cookies_to_set: list[dict[str, Any]] = []
    if user:
        if kei_status_val:
            user.kei_status = kei_status_val
        if impact_val is not None:
            user.kei_impact_slug = impact_val
        if kei_status_val or impact_val is not None:
            await db.commit()
    else:
        if kei_status_val:
            cookies_to_set.append(
                {
                    "key": KEI_POLL_CHOICE_COOKIE,
                    "value": kei_status_val,
                    "max_age": PERSONALIZATION_COOKIE_MAX_AGE,
                    "path": "/",
                    "httponly": False,
                    "samesite": "lax",
                }
            )
            cookies_to_set.append(
                {
                    "key": KEI_POLL_VOTED_COOKIE,
                    "value": "1",
                    "max_age": KEI_POLL_VOTED_MAX_AGE,
                    "path": "/",
                    "httponly": False,
                    "samesite": "lax",
                }
            )
        if impact_val:
            cookies_to_set.append(
                {
                    "key": KEI_IMPACT_SLUG_COOKIE,
                    "value": impact_val,
                    "max_age": PERSONALIZATION_COOKIE_MAX_AGE,
                    "path": "/",
                    "httponly": False,
                    "samesite": "lax",
                }
            )
    resp = JSONResponse({"ok": True})
    for params in cookies_to_set:
        resp.set_cookie(**params)
    return resp


@router.post("/personalize")
async def advocacy_personalize(
    request: Request,
    member_id: str = Form(""),
    zip_code: str = Form(""),
    photo_url: str = Form(""),
    target_type: str = Form("NON_COMMITTEE"),
    kei_status: str | None = Form(None),
    kei_impact_slug: str | None = Form(None),
    kei_personal_note: str | None = Form(None),
    skip: str | None = Form(None),
    constituent: str = Form(""),
    csrf_token: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Save script personalization (impact + note); return call drawer HTML for HTMX swap."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token")
    mid = (member_id or "").strip()
    zip_val = (zip_code or "").strip() if _ZIP_RE.match((zip_code or "").strip()) else ""
    member = find_member_by_id(state, mid) if mid else None
    if not mid or not member:
        raise HTTPException(status_code=404, detail="Legislator not found")

    cookies_to_set: list[dict[str, Any]] = []
    updated_caller = _caller_profile_from_request(request, user)

    if (skip or "").strip() == "1":
        cookies_to_set.append(
            {
                "key": KEI_PERSONALIZATION_SKIPPED_COOKIE,
                "value": "1",
                "max_age": PERSONALIZATION_COOKIE_MAX_AGE,
                "path": "/",
                "httponly": False,
                "samesite": "lax",
            }
        )
    else:
        raw_s = (kei_status or "").strip()
        kei_status_val = raw_s if raw_s in KEI_STATUS_SLUGS else None
        impact_val = _validate_kei_impact_slug(
            (kei_impact_slug or "").strip() or None,
            updated_caller.kei_status or kei_status_val,
        )
        raw_note = (kei_personal_note or "").strip()
        note_val = raw_note[:200] if raw_note else None
        # Only update user/cookies; responses counted only after Turnstile at /updates/kei-status.
        if user:
            if kei_status_val:
                user.kei_status = kei_status_val
            if impact_val is not None:
                user.kei_impact_slug = impact_val
            if note_val is not None:
                user.kei_personal_note = note_val
            await db.commit()
        else:
            if kei_status_val:
                cookies_to_set.append(
                    {
                        "key": KEI_POLL_CHOICE_COOKIE,
                        "value": kei_status_val,
                        "max_age": PERSONALIZATION_COOKIE_MAX_AGE,
                        "path": "/",
                        "httponly": False,
                        "samesite": "lax",
                    }
                )
                cookies_to_set.append(
                    {
                        "key": KEI_POLL_VOTED_COOKIE,
                        "value": "1",
                        "max_age": KEI_POLL_VOTED_MAX_AGE,
                        "path": "/",
                        "httponly": False,
                        "samesite": "lax",
                    }
                )
            if impact_val:
                cookies_to_set.append(
                    {
                        "key": KEI_IMPACT_SLUG_COOKIE,
                        "value": impact_val,
                        "max_age": PERSONALIZATION_COOKIE_MAX_AGE,
                        "path": "/",
                        "httponly": False,
                        "samesite": "lax",
                    }
                )
            if note_val is not None:
                cookies_to_set.append(
                    {
                        "key": KEI_PERSONAL_NOTE_COOKIE,
                        "value": note_val,
                        "max_age": PERSONALIZATION_COOKIE_MAX_AGE,
                        "path": "/",
                        "httponly": False,
                        "samesite": "lax",
                    }
                )
        updated_caller = CallerProfile(
            kei_status=kei_status_val or updated_caller.kei_status,
            kei_impact_slug=impact_val or updated_caller.kei_impact_slug,
            kei_personal_note=note_val
            if note_val is not None
            else updated_caller.kei_personal_note,
        )

    photo_url_validated = validate_photo_url_for_drawer((photo_url or "").strip() or None)
    target_type_param = (target_type or "NON_COMMITTEE").strip().upper()
    constituent_override: bool | None = None
    if (constituent or "").strip().lower() in ("1", "true", "yes"):
        constituent_override = True
    elif (constituent or "").strip().lower() in ("0", "false", "no"):
        constituent_override = False
    response = await _render_call_drawer(
        request,
        mid,
        zip_val,
        photo_url_validated,
        target_type_param,
        user,
        db,
        updated_caller,
        is_constituent_override=constituent_override,
    )
    for params in cookies_to_set:
        response.set_cookie(**params)
    response.headers["HX-Trigger"] = "pollVoted"
    return response


async def _render_call_drawer(
    request: Request,
    member_id_stripped: str,
    zip_code: str,
    photo_url_validated: str | None,
    target_type_param: str,
    user: User | None,
    db: AsyncSession,
    caller: CallerProfile,
    *,
    is_constituent_override: bool | None = None,
) -> Response:
    """Build and return call drawer (script + wrap-up). Used by GET drawer and POST personalize."""
    member = find_member_by_id(state, member_id_stripped)
    if not member:
        return JSONResponse({"detail": "Legislator not found."}, status_code=404)
    is_constituent = (
        is_constituent_override
        if is_constituent_override is not None
        else is_constituent_for_zip_member(state, zip_code, member)
    )
    legislator_name = member.name or ""
    phone = ah.get_preferred_phone_for_member(member)
    effective_email, email_source, community_verification = await get_effective_email_for_member(
        state, db, member_id_stripped
    )
    photo_url = photo_url_validated or (getattr(member, "photo_url", "") or "")
    if photo_url and not photo_url.startswith(("http://", "https://")):
        photo_url = urljoin("https://www.ilga.gov/", photo_url)
    target_type = "POWER_BROKER" if target_type_param == "POWER_BROKER" else "NON_COMMITTEE"
    drawer_ctx = ah.legislator_drawer_context(member)
    try:
        agg = await get_outreach_aggregate(db)
        call_drawer_calls_total = agg["calls_total"]
    except Exception:
        call_drawer_calls_total = 0
    contact_count_this_office = 0
    try:
        contact_count_this_office = await get_outreach_count_for_member(db, member_id_stripped)
    except Exception:
        pass
    script_sections = None
    if member and zip_code:
        district_info = state.zip_to_district.get(zip_code)
        senate_district = district_info.il_senate if district_info else ""
        house_district = district_info.il_house if district_info else ""
        card = ah.member_to_card(state, member, why="")
        chamber = (getattr(member, "chamber", None) or "").strip().lower()
        if chamber == "senate":
            script_sections = ah.build_script_sections_senator(
                card,
                zip_code,
                senate_district,
                calls_total=call_drawer_calls_total,
                contact_count_this_office=contact_count_this_office,
            )
        elif chamber == "house":
            script_sections = ah.build_script_sections_rep(
                card,
                zip_code,
                house_district,
                calls_total=call_drawer_calls_total,
                contact_count_this_office=contact_count_this_office,
            )
        else:
            script_sections = ah.build_script_sections_broker(
                card,
                "This senator has high influence.",
                calls_total=call_drawer_calls_total,
                contact_count_this_office=contact_count_this_office,
            )
        if script_sections and caller and (caller.kei_status or caller.kei_impact_slug):
            script_sections = dict(script_sections)
            script_sections["opening"] = ah.build_personalized_opening(
                caller, is_constituent, drawer_ctx["title_label"], drawer_ctx["legislator_last"]
            )
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
    resp = templates.TemplateResponse(
        request,
        "_advocacy_drawer_call.html",
        {
            "request": request,
            "legislator_name": legislator_name,
            "zip_code": zip_code,
            "is_constituent": is_constituent,
            "phone": phone or "",
            "member_id": member_id_stripped,
            "photo_url": photo_url,
            "member_public_email": effective_email,
            "target_type": target_type,
            "email_source": email_source,
            "community_verification": community_verification,
            "call_completed": call_completed,
            "call_notes": call_notes,
            "call_contact_name": call_contact_name,
            "call_support_score": call_support_score,
            "calls_total": call_drawer_calls_total,
            "script_sections": script_sections,
            **drawer_ctx,
        },
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@router.post("/call/{call_id}/wrapup")
async def advocacy_call_wrapup(
    request: Request,
    call_id: str,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Wrap-up from call: swap drawer to Email view (prefilled or copy-only)."""
    form = await request.form()
    raw_zip = (form.get("zip") or "").strip()
    zip_code = raw_zip if _ZIP_RE.match(raw_zip) else ""
    staffer_name = (form.get("staffer_name") or "").strip()
    email_address = (form.get("email_address") or "").strip()
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
    try:
        agg = await get_outreach_aggregate(db)
        wrapup_calls_total = agg["calls_total"]
    except Exception:
        wrapup_calls_total = 0
    subject_constituent = ah.build_email_subject_line(
        zip_code, variant="constituent", district=district
    )
    subject_general = ah.build_email_subject_line(zip_code, variant="general")
    wrapup_caller = _caller_profile_from_request(request, user)
    body = ah.build_after_call_email_body(
        staffer,
        legislator_name,
        zip_code,
        chamber=chamber,
        district=district,
        target_type=target_type,
        call_date=call_date,
        one_pager_points=_one_pager_points(),
        calls_total=wrapup_calls_total,
        caller=wrapup_caller,
    )
    body_first = ah.build_email_first_body(
        legislator_name,
        zip_code,
        chamber=chamber,
        district=district,
        target_type=target_type,
        one_pager_points=_one_pager_points(),
        calls_total=wrapup_calls_total,
        caller=wrapup_caller,
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
    c = get_campaign_config()
    if recipient:
        return templates.TemplateResponse(
            request,
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
                "brief_pdf_url": c.brief_pdf_url_path,
                "brief_pdf_download_name": c.brief_pdf_filename,
            },
        )

    return templates.TemplateResponse(
        request,
        "_advocacy_drawer_no_email.html",
        {
            "request": request,
            "member_id": member_id,
            "zip_code": zip_code,
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
        request,
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
            "category": category or _default_topic(),
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
        return templates.TemplateResponse(request, tpl, ctx_error)

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
            "category": category or _default_topic(),
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
        return templates.TemplateResponse(request, tpl, ctx_error)

    try:
        agg = await get_outreach_aggregate(db)
        calls_total = agg["calls_total"]
    except Exception:
        calls_total = 0
    results_ctx = await _build_search_results_context(
        request, zip_code, category, db, user, calls_total=calls_total
    )
    poll_ctx = await get_kei_poll_sidebar_context(request, user, db)
    tpl = "_results_partial.html" if is_htmx else "results.html"
    return templates.TemplateResponse(
        request,
        tpl,
        {
            "request": request,
            "title": cfg.SITE_NAME,
            "categories": CATEGORY_CHOICES,
            **results_ctx,
            **poll_ctx,
            "poll_on_advocacy_page": True,
            "hide_outreach_cta": True,
        },
    )
