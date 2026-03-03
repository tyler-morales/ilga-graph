"""Updates router: public /updates page, subscribe/unsubscribe, admin compose and send."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import markdown
import requests
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import advocacy_helpers as ah
from .. import config as cfg
from ..app_state import state
from ..campaign_config import get_campaign_config, get_kei_poll_goal
from ..campaign_helpers import get_active_campaign
from ..constants import (
    CATEGORY_COMMITTEES,
    KEI_IMPACT_SLUG_COOKIE,
    KEI_OWNER_SLUGS,
    KEI_POLL_IMPACT_OPTIONS,
    KEI_STATUS_OPTIONS,
)
from ..db import async_session_factory, get_db
from ..db_models import KeiPollResponse, OutreachEvent, Poll, PollResponse, Update, User
from ..dependencies import get_current_user_optional, require_admin, require_user
from ..email_utils import send_email, send_welcome_email
from ..file_validation import magic_matches_image_content_type
from ..kei_poll_context import (
    KEI_POLL_CHOICE_COOKIE,
    KEI_POLL_VOTED_COOKIE,
    KEI_POLL_VOTED_MAX_AGE,
    STANDALONE_KEI_POLL_ID,
    _get_kei_impact_results,
    _get_kei_status_results,
    _validate_kei_poll_impact,
    _validate_kei_status,
    get_kei_poll_ids,
    get_kei_poll_initial_state,
    get_kei_poll_sidebar_context,
    zip_known_for_user,
)
from ..member_lookup import find_member_by_district
from ..routers.content import (
    CAMPAIGN_STATUS,
    KEI_POLL_WIDE_NET_LINE,
    PROGRESS_ACHIEVED_COUNT,
    PROGRESS_CHECKPOINTS,
    STRATEGIC_FIVE_POINTS,
    WHY_YOU_CARE_CTA_NUDGE,
    WHY_YOU_CARE_DEFAULT_CARDS,
    WHY_YOU_CARE_PRE_POLL_LINE,
    get_marquee_items,
)
from ..routers.content_constants import get_why_you_care_branch_for_selection
from ..security import (
    CSRF_COOKIE_NAME,
    rate_limit_kei_status,
    rate_limit_subscribe_email,
    validate_anon_session_id,
    validate_csrf_token,
)
from ..session_schedule import get_milestone_by_id, get_next_deadline_safe

LOGGER = logging.getLogger(__name__)

_TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

# Email validation for public subscribe (no auth code). Max length matches User.email.
_EMAIL_MAX_LEN = 320
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", re.IGNORECASE)
_ZIP_RE = re.compile(r"^\d{5}$")


def _client_ip(request: Request) -> str:
    """Return client IP from X-Forwarded-For or direct connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _verify_turnstile_sync(token: str, remote_ip: str) -> bool:
    """Blocking Turnstile verification for poll; run via run_in_executor from async."""
    try:
        resp = requests.post(
            _TURNSTILE_VERIFY_URL,
            data={
                "secret": cfg.TURNSTILE_SECRET_KEY,
                "response": token,
                "remoteip": remote_ip,
            },
            timeout=10,
        )
        return resp.json().get("success") is True
    except Exception:
        LOGGER.exception("Turnstile siteverify failed")
        return False


async def _verify_turnstile(token: str | None, remote_ip: str) -> bool:
    """Verify Turnstile token for poll submission. Skipped when TURNSTILE_DISABLED (e.g. dev)."""
    if cfg.TURNSTILE_DISABLED:
        return True
    if not cfg.TURNSTILE_SECRET_KEY:
        return True
    if not token or not token.strip():
        return False
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _verify_turnstile_sync, token.strip(), remote_ip)


def _normalize_and_validate_subscribe_email(raw: str) -> str | None:
    """Return normalized email if valid; else None."""
    s = (raw or "").strip().lower()
    if not s or len(s) > _EMAIL_MAX_LEN:
        return None
    if not _EMAIL_RE.match(s):
        return None
    return s


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
router = APIRouter()
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["dev_available"] = cfg.DEV_MODE
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["site_name"] = cfg.SITE_NAME
_campaign = get_campaign_config()
templates.env.globals["campaign_name"] = _campaign.campaign_name or cfg.SITE_NAME
templates.env.globals["primary_color"] = _campaign.primary_color or "#FF4500"
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
templates.env.globals["features"] = cfg.get_client_features()
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS

from ..campaign_helpers import get_current_action_campaign_for_template  # noqa: E402

templates.env.globals["get_current_action_campaign"] = get_current_action_campaign_for_template
templates.env.globals["get_milestone_by_id"] = get_milestone_by_id
templates.env.globals["get_next_deadline"] = get_next_deadline_safe
templates.env.globals["kei_status_options"] = KEI_STATUS_OPTIONS
templates.env.globals["kei_impact_options"] = KEI_POLL_IMPACT_OPTIONS
templates.env.globals["turnstile_site_key"] = (
    "" if cfg.TURNSTILE_DISABLED else (cfg.TURNSTILE_SITE_KEY or "")
)

# Email update types: slug -> display label. Default for new drafts is "other".
UPDATE_TYPES = [("major", "Major"), ("minor", "Minor"), ("other", "Other")]
UPDATE_TYPE_SLUGS = {s for s, _ in UPDATE_TYPES}
DEFAULT_UPDATE_TYPE = "other"
# Phrase shown above title on public page (e.g. "Major update", "Update").
UPDATE_TYPE_PHRASES: dict[str, str] = {
    "major": "Major update",
    "minor": "Minor update",
    "other": "Update",
}

# Optional image upload for updates (admin compose).
_UPDATE_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
_UPDATE_IMAGE_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _update_type_label(slug: str) -> str:
    """Return display label for an update type slug; fallback to slug if unknown."""
    for s, label in UPDATE_TYPES:
        if s == slug:
            return label
    return slug


def _normalize_update_type(value: str | None) -> str:
    """Validate form update_type; return slug or DEFAULT_UPDATE_TYPE."""
    if value and value.strip() in UPDATE_TYPE_SLUGS:
        return value.strip()
    return DEFAULT_UPDATE_TYPE


async def _save_update_image(upload: UploadFile, update_id: int) -> str | None:
    """Validate and save one image; return relative path (e.g. updates/1_abc.jpg) or None."""
    if not upload.filename or not upload.content_type:
        return None
    if upload.content_type not in _UPDATE_IMAGE_TYPES:
        LOGGER.warning("Update image rejected: content_type=%s", upload.content_type)
        return None
    ext = _UPDATE_IMAGE_EXT.get(upload.content_type, ".bin")
    content = b""
    while True:
        chunk = await upload.read(1024 * 256)
        if not chunk:
            break
        content += chunk
        if len(content) > cfg.UPDATE_MAX_IMAGE_BYTES:
            LOGGER.warning("Update image rejected: too large")
            return None
    if not content:
        return None
    if not magic_matches_image_content_type(content, upload.content_type):
        LOGGER.warning(
            "Update image rejected: magic bytes do not match content_type=%s",
            upload.content_type,
        )
        return None
    upload_dir = _STATIC_DIR / cfg.UPDATE_IMAGE_UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{update_id}_{uuid.uuid4().hex}{ext}"
    (upload_dir / fname).write_bytes(content)
    return f"{cfg.UPDATE_IMAGE_UPLOAD_DIR}/{fname}"


# Unsubscribe tokens valid 1 year (links in emails).
_UNSUB_MAX_AGE = 365 * 24 * 60 * 60
_signer = URLSafeTimedSerializer(cfg.AUTH_SECRET)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _create_unsubscribe_token(user_id: int) -> str:
    return _signer.dumps({"uid": user_id, "action": "unsub"})


def _decode_unsubscribe_token(token: str) -> int | None:
    try:
        data = _signer.loads(token, max_age=_UNSUB_MAX_AGE)
        if data.get("action") == "unsub":
            return data.get("uid")
    except (BadSignature, SignatureExpired):
        pass
    return None


def _plain_to_html(plain: str) -> str:
    """Wrap plain text in HTML paragraphs. Fallback when markdown is not used."""
    if not plain.strip():
        return ""
    return "".join(f"<p>{html.escape(p.strip())}</p>" for p in plain.split("\n\n") if p.strip())


def _markdown_to_html(plain: str) -> str:
    """Convert Markdown to HTML for update body. Used when creating/saving updates."""
    if not plain.strip():
        return ""
    return markdown.markdown(plain, extensions=["nl2br"])


def _render_update_email_html(
    title: str, body_html: str, unsub_url: str, image_url: str | None = None
) -> str:
    """Render the update email HTML (header, optional image, body, footer with unsubscribe)."""
    site = cfg.SITE_NAME
    updates_url = (cfg.APP_BASE_URL or "").rstrip("/") + "/updates"
    tmpl = templates.env.get_template("_update_email.html")
    return tmpl.render(
        site_name=site,
        title=title,
        body_html=body_html,
        unsub_url=unsub_url,
        image_url=image_url,
        updates_url=updates_url,
    )


async def _get_subscriber_list(
    db: AsyncSession, recipient_ids: list[int] | None
) -> tuple[list[User], bool]:
    """Resolve list of users to send to. Returns (users, used_dev_fallback)."""
    if recipient_ids is not None:
        if not recipient_ids:
            return [], False
        result = await db.execute(select(User).where(User.id.in_(recipient_ids)))
        return list(result.scalars().all()), False
    result = await db.execute(select(User).where(User.wants_updates.is_(True)))
    subscribers = list(result.scalars().all())
    if not subscribers and cfg.DEV_MODE:
        fallback = await db.execute(select(User).where(User.email.isnot(None), User.email != ""))
        subscribers = list(fallback.scalars().all())
        if subscribers:
            LOGGER.info(
                "DEV: no wants_updates=True; sending to all %d users for testing", len(subscribers)
            )
        return subscribers, True
    return subscribers, False


def _user_has_sendable_email(user: User) -> bool:
    """True if user has a non-empty email we can send to."""
    return bool((user.email or "").strip())


async def _send_to_users(update: Update, users: list[User], db: AsyncSession) -> int:
    """Send this update to the given users. Sets sent_at/sent_count and commits. Returns count."""
    base_url = (cfg.APP_BASE_URL or "").rstrip("/")
    body_html = update.body_html or _plain_to_html(update.body_plain)
    image_url = f"{base_url}/static/{update.image_path}" if base_url and update.image_path else None
    count = 0
    for user in users:
        if not _user_has_sendable_email(user):
            continue
        try:
            token = _create_unsubscribe_token(user.id)
            unsub_url = f"{base_url}/updates/unsubscribe?token={token}"
            html_content = _render_update_email_html(
                update.title, body_html, unsub_url, image_url=image_url
            )
            if await send_email(
                user.email,
                f"Update: {update.title}",
                update.body_plain,
                html_content,
            ):
                count += 1
        except Exception:
            LOGGER.exception("Failed to send update to %s", user.email)
    update.sent_at = _utcnow()
    update.sent_count = count
    await db.commit()
    LOGGER.info("Update id=%s sent to %d subscribers", update.id, count)
    return count


async def send_update_to_subscribers(
    update: Update, db: AsyncSession, recipient_ids: list[int] | None = None
) -> int:
    """Send to wants_updates=True (or recipient_ids if provided). Sets sent_at/sent_count."""
    users, _ = await _get_subscriber_list(db, recipient_ids)
    return await _send_to_users(update, users, db)


async def run_send_loop(
    app: Any, job_id: str, update_id: int, recipient_ids: list[int] | None
) -> None:
    """Background task: load update and users, send to each, update job progress and persist."""
    jobs = getattr(app.state, "send_jobs", None)
    if jobs is None:
        return
    job = jobs.get(job_id)
    if job is None:
        return
    try:
        async with async_session_factory() as db:
            result = await db.execute(select(Update).where(Update.id == update_id))
            update = result.scalar_one_or_none()
            if not update or update.sent_at is not None:
                job["done"] = True
                job["error"] = "Update not found or already sent"
                return
            users, _ = await _get_subscriber_list(db, recipient_ids)
            sendable = [u for u in users if _user_has_sendable_email(u)]
            job["total"] = len(sendable)
            if not sendable:
                job["done"] = True
                return
            base_url = (cfg.APP_BASE_URL or "").rstrip("/")
            body_html = update.body_html or _plain_to_html(update.body_plain)
            image_url = (
                f"{base_url}/static/{update.image_path}" if base_url and update.image_path else None
            )
            sent = 0
            failed = 0
            for user in sendable:
                try:
                    token = _create_unsubscribe_token(user.id)
                    unsub_url = f"{base_url}/updates/unsubscribe?token={token}"
                    html_content = _render_update_email_html(
                        update.title, body_html, unsub_url, image_url=image_url
                    )
                    if await send_email(
                        user.email,
                        f"Update: {update.title}",
                        update.body_plain,
                        html_content,
                    ):
                        sent += 1
                except Exception:
                    failed += 1
                    LOGGER.exception("Failed to send update to %s", user.email)
                job["sent"] = sent
                job["failed"] = failed
            update.sent_at = _utcnow()
            update.sent_count = sent
            await db.commit()
            LOGGER.info("Update id=%s sent to %d subscribers (%d failed)", update.id, sent, failed)
    except Exception as e:
        LOGGER.exception("Send loop failed for job %s", job_id)
        job["error"] = str(e)
    finally:
        job["done"] = True


# ─── Priority card status (logged-in user outreach) ────────────────────────────


def _priority_card_cta_href(zip_code: str | None) -> str:
    """Link to advocacy; include zip so user lands on their results."""
    if zip_code:
        return f"/advocacy?zip={zip_code}"
    return "/advocacy#zip_code"


async def _get_priority_card_status(
    user: User,
    active_campaign: object,
    db: AsyncSession,
) -> dict[str, Any] | None:
    """Build status message and CTA for the priority callout from advocacy next-step logic.

    Uses user.zip_code + district to compute the same goal steps as the advocacy page (district
    call/email, then Power Broker when district is complete). CTA shows the actual next step
    (e.g. "Call your Senator", "Email your Rep") instead of generic "Contact your rep".
    """
    campaign_id = getattr(active_campaign, "id", None)
    if not campaign_id:
        return None

    zip_code = (user.zip_code or "").strip() or None
    senator = None
    rep = None
    senate_district = None
    house_district = None
    if zip_code and getattr(state, "zip_to_district", None) and zip_code in state.zip_to_district:
        district_info = state.zip_to_district[zip_code]
        senate_district = district_info.il_senate if district_info else None
        house_district = district_info.il_house if district_info else None
        senator = (
            find_member_by_district(state, "senate", senate_district) if senate_district else None
        )
        rep = find_member_by_district(state, "house", house_district) if house_district else None

    member_ids: list[str] = []
    if senator:
        member_ids.append(str(senator.id))
    if rep:
        member_ids.append(str(rep.id))

    if not member_ids:
        # No district: fallback if user has any outreach
        any_result = await db.execute(
            select(OutreachEvent.id)
            .where(OutreachEvent.user_id == user.id)
            .where(OutreachEvent.kind.in_(["call", "email"]))
            .limit(1)
        )
        if any_result.scalar() is not None:
            return {
                "status_message": (
                    "You've recorded outreach. Enter your ZIP on the advocacy page "
                    "to see your full progress."
                ),
                "goal_complete": False,
                "cta_text": "Go to advocacy",
                "cta_href": "/advocacy",
            }
        return None

    # Same outreach lookup as advocacy: call/email per member
    outreach_result = await db.execute(
        select(OutreachEvent.member_id, OutreachEvent.kind)
        .where(OutreachEvent.user_id == user.id)
        .where(OutreachEvent.member_id.in_(member_ids))
        .where(OutreachEvent.kind.in_(["call", "email"]))
    )
    called: set[str] = set()
    emailed: set[str] = set()
    for mid, kind in outreach_result.all():
        mid_str = str(mid)
        if kind == "call":
            called.add(mid_str)
        elif kind == "email":
            emailed.add(mid_str)

    # District steps: same order as advocacy (call then email per member; senator then rep)
    role_by_id: dict[str, str] = {}
    if senator:
        role_by_id[str(senator.id)] = "Senator"
    if rep:
        role_by_id[str(rep.id)] = "Rep"
    district_steps: list[dict[str, Any]] = []
    for mid in member_ids:
        role = role_by_id.get(mid, "Rep")
        district_steps.append(
            {"member_id": mid, "role_label": role, "action": "call", "done": mid in called}
        )
        district_steps.append(
            {"member_id": mid, "role_label": role, "action": "email", "done": mid in emailed}
        )

    district_done = sum(1 for s in district_steps if s["done"])
    district_total = len(district_steps)
    district_complete = district_done == district_total and district_total > 0

    # Next step: first undone district step, or broker step if district complete
    goal_next_step: dict[str, Any] | None = None
    for s in district_steps:
        if not s["done"]:
            goal_next_step = {
                "action": s["action"],
                "member_id": s["member_id"],
                "role_label": s["role_label"],
            }
            break

    if district_complete and senate_district is not None and house_district is not None:
        committee_codes = CATEGORY_COMMITTEES.get("Transportation", [])
        power_brokers = ah.find_power_brokers(
            state,
            exclude_senate_district=senate_district or "",
            exclude_house_district=house_district or "",
            committee_codes=committee_codes or None,
            category_name="Transportation",
        )
        broker_member_ids = [str(m.id) for m, _ in power_brokers]
        if broker_member_ids:
            broker_result = await db.execute(
                select(OutreachEvent.member_id, OutreachEvent.kind)
                .where(OutreachEvent.user_id == user.id)
                .where(OutreachEvent.member_id.in_(broker_member_ids))
                .where(OutreachEvent.kind.in_(["call", "email"]))
            )
            broker_done: dict[str, dict[str, bool]] = {}
            for mid, kind in broker_result.all():
                mid_str = str(mid)
                if mid_str not in broker_done:
                    broker_done[mid_str] = {"call": False, "email": False}
                if kind == "call":
                    broker_done[mid_str]["call"] = True
                elif kind == "email":
                    broker_done[mid_str]["email"] = True
            broker_steps = []
            for mid in broker_member_ids:
                done = broker_done.get(mid, {"call": False, "email": False})
                broker_steps.append(
                    {
                        "member_id": mid,
                        "role_label": "Power Broker",
                        "action": "call",
                        "done": done["call"],
                    }
                )
                broker_steps.append(
                    {
                        "member_id": mid,
                        "role_label": "Power Broker",
                        "action": "email",
                        "done": done["email"],
                    }
                )
            if goal_next_step is None:
                for s in broker_steps:
                    if not s["done"]:
                        goal_next_step = {
                            "action": s["action"],
                            "member_id": s["member_id"],
                            "role_label": s["role_label"],
                        }
                        break

    # "What you did" message
    if district_complete and goal_next_step is None:
        status_message = "You've contacted your rep — thank you!"
        cta_text = "See your progress"
        goal_complete = True
    elif district_complete and goal_next_step is not None:
        status_message = "You've contacted your district legislators."
        cta_text = f"{goal_next_step['action'].capitalize()} your {goal_next_step['role_label']}"
        goal_complete = False
    elif district_done > 0:
        status_message = f"You've completed {district_done} of {district_total} steps."
        cta_text = (
            f"{goal_next_step['action'].capitalize()} your {goal_next_step['role_label']}"
            if goal_next_step
            else getattr(active_campaign, "ask", "Contact your rep")
        )
        goal_complete = False
    else:
        return None

    return {
        "status_message": status_message,
        "goal_complete": goal_complete,
        "cta_text": cta_text,
        "cta_href": _priority_card_cta_href(zip_code),
    }


# ─── Public routes ────────────────────────────────────────────────────────────


def _updates_page_ctx(
    request: Request,
    sent_updates: list[Update],
    user: User | None,
) -> dict[str, Any]:
    """Build template context for updates page (all updates on one page, sidebar TOC)."""
    progress_checklist = [
        {"label": label, "achieved": i < PROGRESS_ACHIEVED_COUNT}
        for i, label in enumerate(PROGRESS_CHECKPOINTS)
    ]
    return {
        "request": request,
        "campaign_status": CAMPAIGN_STATUS,
        "progress_checklist": progress_checklist,
        "updates": sent_updates,
        "user": user,
        "wants_updates": user.wants_updates if user else None,
        "update_type_labels": dict(UPDATE_TYPES),
        "update_type_phrases": UPDATE_TYPE_PHRASES,
    }


@router.get("/poll", include_in_schema=False)
async def poll_standalone_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Shareable poll-only page: minimal UI, kei poll then results + Go to full site CTA."""
    state = await get_kei_poll_initial_state(request, user, db)
    show_results = state.get("kei_poll_done") or request.query_params.get("submitted") == "1"
    ctx: dict[str, Any] = {
        "request": request,
        "user": user,
        "poll_id": STANDALONE_KEI_POLL_ID,
        "show_results": show_results,
        "show_go_to_site": True,
        "standalone_poll": True,
    }
    ctx.update(state)
    ctx["zip_known"] = False
    ctx["prefill_zip"] = (user.zip_code or "").strip() if user else ""
    if show_results and state.get("kei_status_selected"):
        ctx["why_you_care_branch"] = get_why_you_care_branch_for_selection(
            state.get("kei_status_selected")
        )
    if not show_results:
        results = await _get_kei_status_results(db)
        ctx["kei_status_total"] = results["total_responses"]
        ctx["kei_impact_options"] = KEI_POLL_IMPACT_OPTIONS
    ctx["marquee_items"] = await get_marquee_items(db)
    ctx["marquee_title"] = "Stories from Kei truck owners"
    return templates.TemplateResponse(request, "poll_standalone.html", ctx)


@router.get("/updates", include_in_schema=False)
async def updates_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Public updates page: campaign status, subscribe, past updates sidebar; main shows latest."""
    result = await db.execute(
        select(Update).where(Update.sent_at.isnot(None)).order_by(Update.sent_at.desc())
    )
    sent_updates = list(result.scalars().all())
    ctx = _updates_page_ctx(request, sent_updates, user)
    active_campaign = await get_active_campaign(db)
    ctx["active_campaign"] = active_campaign
    if user and active_campaign:
        ctx["priority_card_status"] = await _get_priority_card_status(user, active_campaign, db)
    else:
        ctx["priority_card_status"] = None
    q = request.query_params
    prompt_q = get_campaign_config().poll_prompt_query or "kei"
    ctx["prompt_kei"] = q.get("prompt") == prompt_q
    ctx["kei_submitted"] = q.get("submitted") == "1"
    ctx["kei_error"] = q.get("error")
    poll_state = await get_kei_poll_initial_state(request, user, db)
    ctx.update(poll_state)
    ctx["poll_id"] = "updates-kei-poll"
    if ctx.get("kei_poll_done") and ctx.get("kei_status_selected"):
        ctx["why_you_care_branch"] = get_why_you_care_branch_for_selection(
            ctx.get("kei_status_selected")
        )
    ctx["zip_known"] = zip_known_for_user(user)
    ctx["prefill_zip"] = (user.zip_code or "").strip() if user else ""
    return templates.TemplateResponse(request, "updates.html", ctx)


@router.get("/updates/unsubscribe", include_in_schema=False)
async def unsubscribe_page(
    request: Request,
    token: str = "",
    db: AsyncSession = Depends(get_db),
):
    """One-click unsubscribe via token from email. Renders confirmation."""
    if not token:
        return RedirectResponse("/updates", status_code=303)
    user_id = _decode_unsubscribe_token(token)
    if user_id is None:
        return templates.TemplateResponse(
            request,
            "updates_unsubscribe.html",
            {"request": request, "success": False, "message": "Invalid or expired link."},
        )
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user:
        user.wants_updates = False
        await db.commit()
        LOGGER.info("User id=%s unsubscribed from updates", user.id)
    return templates.TemplateResponse(
        request,
        "updates_unsubscribe.html",
        {
            "request": request,
            "success": True,
            "message": "Unsubscribed from campaign updates.",
        },
    )


@router.post("/updates/subscribe", include_in_schema=False)
async def subscribe_post(
    request: Request,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Turn subscription on for current user; send welcome if not yet. HTMX returns fragment."""
    user.wants_updates = True
    await db.commit()
    now = datetime.now(timezone.utc)
    if getattr(user, "welcome_email_sent_at", None) is None:
        try:
            base_url = (cfg.APP_BASE_URL or "").rstrip("/")
            unsub_url = f"{base_url}/updates/unsubscribe?token={_create_unsubscribe_token(user.id)}"
            sent = await send_welcome_email(user.email, unsub_url=unsub_url)
            if sent:
                user.welcome_email_sent_at = now
                await db.commit()
        except Exception:
            LOGGER.exception("Failed to send welcome email to %s", user.email)
    if request.headers.get("HX-Request"):
        return HTMLResponse(
            '<p class="subscribe-email-success poll-signup-success" '
            'role="status" aria-live="polite">'
            "You're on the list. We'll send one email when we move.</p>"
        )
    return RedirectResponse("/updates", status_code=303)


@router.get("/updates/kei-status-results", include_in_schema=False)
async def kei_status_results(
    db: AsyncSession = Depends(get_db),
):
    """Poll results: counts by kei_status for users who have signed in at least once."""
    data = await _get_kei_status_results(db)
    return JSONResponse(data)


@router.get("/updates/kei-poll-sidebar-fragment", include_in_schema=False)
async def kei_poll_sidebar_fragment(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Return sidebar kei poll fragment (form or results) for HTMX refresh after vote in drawer."""
    ctx = await get_kei_poll_sidebar_context(request, user, db)
    return templates.TemplateResponse(
        request,
        "_sidebar_kei_poll.html",
        ctx,
    )


@router.get("/updates/kei-poll-form", include_in_schema=False)
async def kei_poll_form(
    request: Request,
    poll_id: str = "footer-kei-poll",
    show_email: bool = False,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Return poll form HTML for HTMX swap (e.g. change your answer). Includes total and impact."""
    if poll_id not in get_kei_poll_ids():
        poll_id = "footer-kei-poll"
    results = await _get_kei_status_results(db)
    state = await get_kei_poll_initial_state(request, user, db)
    return templates.TemplateResponse(
        request,
        "_kei_poll_form_partial.html",
        {
            "poll_id": poll_id,
            "show_email": show_email,
            "show_intro": poll_id != STANDALONE_KEI_POLL_ID,
            "kei_status_total": results["total_responses"],
            "kei_poll_goal": get_kei_poll_goal(),
            "kei_impact_options": KEI_POLL_IMPACT_OPTIONS,
            "kei_status_selected": state.get("kei_status_selected"),
            "kei_impact_selected": state.get("kei_impact_selected"),
            "zip_known": zip_known_for_user(user),
            "prefill_zip": (user.zip_code or "").strip() if user else "",
        },
    )


def _why_you_care_flow_ctx(
    request: Request,
    kei_status_total: int,
    kei_error_message: str | None = None,
    user: User | None = None,
) -> dict[str, Any]:
    """Context for _why_you_care_flow_ambient.html (initial load or error re-render)."""
    ctx: dict[str, Any] = {
        "request": request,
        "why_you_care_default_cards": WHY_YOU_CARE_DEFAULT_CARDS,
        "why_you_care_pre_poll_line": WHY_YOU_CARE_PRE_POLL_LINE,
        "kei_status_total": kei_status_total,
        "kei_poll_goal": get_kei_poll_goal(),
        "kei_poll_wide_net_line": KEI_POLL_WIDE_NET_LINE,
        "zip_known": zip_known_for_user(user),
        "prefill_zip": (user.zip_code or "").strip() if user else "",
    }
    if kei_error_message:
        ctx["kei_error_message"] = kei_error_message
    return ctx


def _kei_status_redirect_url(poll_id: str, prompt_q: str, query: str) -> str:
    """Redirect URL after kei-status POST: /poll for standalone, else /updates?prompt=..."""
    if poll_id == STANDALONE_KEI_POLL_ID:
        return f"/poll?{query}"
    return f"/updates?prompt={prompt_q}&{query}"


@router.get("/updates/why-you-care-flow", include_in_schema=False)
async def why_you_care_flow(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Return ambient Why-you-care flow HTML for HTMX swap into #why-you-care-flow."""
    results = await _get_kei_status_results(db)
    return templates.TemplateResponse(
        request,
        "_why_you_care_flow_ambient.html",
        _why_you_care_flow_ctx(request, results["total_responses"], user=user),
    )


@router.get("/updates/kei-poll-results", include_in_schema=False)
async def kei_poll_results(
    request: Request,
    poll_id: str = "footer-kei-poll",
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_user),
):
    """Logged-in poll results fragment for HTMX swap after sign-in (removes 'not counted' nudge)."""
    if poll_id not in get_kei_poll_ids():
        poll_id = "footer-kei-poll"
    results = await _get_kei_status_results(db)
    impact_results = await _get_kei_impact_results(db)
    return templates.TemplateResponse(
        request,
        "_kei_poll_logged_in_success.html",
        {
            "kei_status_results": results,
            "kei_status_options": KEI_STATUS_OPTIONS,
            "kei_status_selected": user.kei_status,
            "kei_impact_selected": user.kei_impact_slug,
            "kei_impact_results": impact_results,
            "kei_impact_options": KEI_POLL_IMPACT_OPTIONS,
            "kei_poll_goal": get_kei_poll_goal(),
            "kei_poll_is_owner": user.kei_status in KEI_OWNER_SLUGS if user.kei_status else False,
            "poll_id": poll_id,
        },
    )


def _normalize_poll_zip(raw: str | None) -> str | None:
    """Return 5-digit ZIP if valid (and in state.zip_to_district when available), else None."""
    s = (raw or "").strip()
    if not s or not _ZIP_RE.match(s):
        return None
    if getattr(state, "zip_to_district", None) and s not in state.zip_to_district:
        return None
    return s


@router.post("/updates/kei-status", include_in_schema=False)
async def kei_status_post(
    request: Request,
    kei_status: str = Form(..., max_length=32),
    kei_impact_slug: str | None = Form(None, max_length=32),
    zip_code: str | None = Form(None, max_length=10),
    email: str | None = Form(None, max_length=_EMAIL_MAX_LEN),
    poll_id: str = Form("footer-kei-poll", max_length=64),
    session_id: str | None = Form(None, max_length=64),
    csrf_token: str | None = Form(None),
    cf_turnstile_response: str | None = Form(None, alias="cf-turnstile-response"),
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Set kei status and impact (Q3). Insert responses; set user fields if logged in."""
    prompt_q = get_campaign_config().poll_prompt_query or "kei"
    token = csrf_token or request.headers.get("X-XSRF-TOKEN")
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(token, cookie_token):
        if request.headers.get("HX-Request") and poll_id == "home-kei-poll":
            results = await _get_kei_status_results(db)
            resp = templates.TemplateResponse(
                request,
                "_why_you_care_flow_ambient.html",
                _why_you_care_flow_ctx(
                    request,
                    results["total_responses"],
                    "Invalid or expired security token. Reload the page and try again.",
                    user=user,
                ),
            )
            resp.status_code = 403
            return resp
        if request.headers.get("HX-Request"):
            return HTMLResponse(
                '<p class="kei-status-error" role="alert">Invalid or expired security token. '
                "Reload the page and try again.</p>",
                status_code=403,
            )
        return RedirectResponse(
            _kei_status_redirect_url(poll_id, prompt_q, "error=csrf"), status_code=303
        )
    if user is None:
        client_ip = _client_ip(request)
        if not rate_limit_kei_status(client_ip):
            if request.headers.get("HX-Request") and poll_id == "home-kei-poll":
                results = await _get_kei_status_results(db)
                resp = templates.TemplateResponse(
                    request,
                    "_why_you_care_flow_ambient.html",
                    _why_you_care_flow_ctx(
                        request,
                        results["total_responses"],
                        "Too many responses from this device. Try again later.",
                        user=user,
                    ),
                )
                resp.status_code = 429
                return resp
            if request.headers.get("HX-Request"):
                return HTMLResponse(
                    '<p class="kei-status-error" role="alert">Too many responses from this '
                    "device. Try again later.</p>",
                    status_code=429,
                )
            return RedirectResponse(
                _kei_status_redirect_url(poll_id, prompt_q, "error=rate"), status_code=303
            )
    # Require Turnstile for everyone so a response is only counted after verification.
    client_ip = _client_ip(request)
    if not await _verify_turnstile(cf_turnstile_response, client_ip):
        if request.headers.get("HX-Request") and poll_id == "home-kei-poll":
            results = await _get_kei_status_results(db)
            resp = templates.TemplateResponse(
                request,
                "_why_you_care_flow_ambient.html",
                _why_you_care_flow_ctx(
                    request,
                    results["total_responses"],
                    "Verification failed. Complete the security check and try again.",
                    user=user,
                ),
            )
            resp.status_code = 400
            return resp
        if request.headers.get("HX-Request"):
            return HTMLResponse(
                '<p class="kei-status-error" role="alert">Verification failed. Complete the '
                "security check and try again.</p>",
                status_code=400,
            )
        return RedirectResponse(
            _kei_status_redirect_url(poll_id, prompt_q, "error=verify"), status_code=303
        )
    if poll_id not in get_kei_poll_ids():
        poll_id = "footer-kei-poll"
    validated = _validate_kei_status(kei_status)
    if not validated:
        if request.headers.get("HX-Request") and poll_id == "home-kei-poll":
            results = await _get_kei_status_results(db)
            resp = templates.TemplateResponse(
                request,
                "_why_you_care_flow_ambient.html",
                _why_you_care_flow_ctx(
                    request,
                    results["total_responses"],
                    "Please choose an option.",
                    user=user,
                ),
            )
            resp.status_code = 400
            return resp
        if request.headers.get("HX-Request"):
            return HTMLResponse(
                '<p class="kei-status-error" role="alert">Please choose an option.</p>',
                status_code=400,
            )
        return RedirectResponse(
            _kei_status_redirect_url(poll_id, prompt_q, "error=invalid"), status_code=303
        )
    impact_val = _validate_kei_poll_impact((kei_impact_slug or "").strip() or None)
    if not impact_val:
        if request.headers.get("HX-Request") and poll_id == "home-kei-poll":
            results = await _get_kei_status_results(db)
            resp = templates.TemplateResponse(
                request,
                "_why_you_care_flow_ambient.html",
                _why_you_care_flow_ctx(
                    request,
                    results["total_responses"],
                    "Please choose how it affects you.",
                    user=user,
                ),
            )
            resp.status_code = 400
            return resp
        if request.headers.get("HX-Request"):
            return HTMLResponse(
                '<p class="kei-status-error" role="alert">Please choose how it affects you.</p>',
                status_code=400,
            )
        return RedirectResponse(
            _kei_status_redirect_url(poll_id, prompt_q, "error=invalid"), status_code=303
        )
    # Standalone poll requires a valid ZIP so we can show district results.
    if poll_id == STANDALONE_KEI_POLL_ID:
        zip_val = _normalize_poll_zip(zip_code)
        if not zip_val:
            if request.headers.get("HX-Request"):
                return HTMLResponse(
                    '<p class="kei-status-error" role="alert">'
                    "Please enter a valid 5-digit Illinois ZIP code.</p>",
                    status_code=400,
                )
            return RedirectResponse(
                _kei_status_redirect_url(poll_id, prompt_q, "error=zip"), status_code=303
            )
    # Idempotent: if user already voted (e.g. anon attributed on sign-in), return success.
    if user and getattr(user, "kei_status", None) is not None:
        existing_status = user.kei_status
        existing_impact = getattr(user, "kei_impact_slug", None) or impact_val
        if request.headers.get("HX-Request"):
            results = await _get_kei_status_results(db)
            impact_results = await _get_kei_impact_results(db)
            if poll_id == "home-kei-poll":
                why_you_care_branch = get_why_you_care_branch_for_selection(existing_status)
                wyc_pill_icon_slug = (
                    "owner"
                    if existing_status in ("registered", "revoked", "denied")
                    else existing_status
                )
                return templates.TemplateResponse(
                    request,
                    "_why_you_care_branch.html",
                    {
                        "why_you_care_branch": why_you_care_branch,
                        "why_you_care_cta_nudge": WHY_YOU_CARE_CTA_NUDGE,
                        "wyc_pill_icon_slug": wyc_pill_icon_slug,
                        "kei_status_results": results,
                        "kei_status_options": KEI_STATUS_OPTIONS,
                        "kei_status_selected": existing_status,
                        "kei_impact_selected": existing_impact,
                        "kei_impact_results": impact_results,
                        "kei_impact_options": KEI_POLL_IMPACT_OPTIONS,
                        "kei_poll_goal": get_kei_poll_goal(),
                        "kei_poll_initial_anon": False,
                        "poll_id": poll_id,
                    },
                )
            return templates.TemplateResponse(
                request,
                "_kei_poll_logged_in_success.html",
                {
                    "kei_status_results": results,
                    "kei_status_options": KEI_STATUS_OPTIONS,
                    "kei_status_selected": existing_status,
                    "kei_impact_selected": existing_impact,
                    "kei_impact_results": impact_results,
                    "kei_impact_options": KEI_POLL_IMPACT_OPTIONS,
                    "kei_poll_goal": get_kei_poll_goal(),
                    "kei_poll_is_owner": (
                        existing_status in KEI_OWNER_SLUGS if existing_status else False
                    ),
                    "why_you_care_branch": get_why_you_care_branch_for_selection(existing_status),
                    "poll_id": poll_id,
                },
            )
        return RedirectResponse(
            _kei_status_redirect_url(poll_id, prompt_q, "submitted=1"), status_code=303
        )
    anon_sid = validate_anon_session_id(session_id) if session_id else None
    zip_val = _normalize_poll_zip(zip_code)
    response_row = KeiPollResponse(
        user_id=user.id if user else None,
        session_id=anon_sid,
        kei_status=validated,
        zip_code=zip_val,
    )
    db.add(response_row)
    if user:
        user.kei_status = validated
        user.kei_impact_slug = impact_val
        if zip_val and not (user.zip_code or "").strip():
            user.zip_code = zip_val
    # Dual-write to poll_responses so admin Polls list and per-poll results stay in sync.
    poll_slug = get_campaign_config().poll_slug or "kei"
    campaign_poll = (
        await db.execute(select(Poll).where(Poll.slug == poll_slug))
    ).scalar_one_or_none()
    if campaign_poll:
        db.add(
            PollResponse(
                poll_id=campaign_poll.id,
                user_id=user.id if user else None,
                session_id=anon_sid,
                option_slug=validated,
            )
        )
    impact_poll = (
        await db.execute(select(Poll).where(Poll.slug == "kei_impact"))
    ).scalar_one_or_none()
    if impact_poll:
        db.add(
            PollResponse(
                poll_id=impact_poll.id,
                user_id=user.id if user else None,
                session_id=anon_sid,
                option_slug=impact_val,
            )
        )
    await db.commit()
    LOGGER.info(
        "Kei poll response id=%s user_id=%s kei_status=%s",
        response_row.id,
        user.id if user else None,
        validated,
    )
    if user:
        if request.headers.get("HX-Request"):
            results = await _get_kei_status_results(db)
            impact_results = await _get_kei_impact_results(db)
            if poll_id == "home-kei-poll":
                why_you_care_branch = get_why_you_care_branch_for_selection(validated)
                branch_slug = (
                    "owner" if validated in ("registered", "revoked", "denied") else validated
                )
                wyc_pill_icon_slug = (
                    validated if validated in ("registered", "revoked", "denied") else branch_slug
                )
                return templates.TemplateResponse(
                    request,
                    "_why_you_care_branch.html",
                    {
                        "why_you_care_branch": why_you_care_branch,
                        "why_you_care_cta_nudge": WHY_YOU_CARE_CTA_NUDGE,
                        "wyc_pill_icon_slug": wyc_pill_icon_slug,
                        "kei_status_results": results,
                        "kei_status_options": KEI_STATUS_OPTIONS,
                        "kei_status_selected": validated,
                        "kei_impact_selected": impact_val,
                        "kei_impact_results": impact_results,
                        "kei_impact_options": KEI_POLL_IMPACT_OPTIONS,
                        "kei_poll_goal": get_kei_poll_goal(),
                        "kei_poll_initial_anon": False,
                        "poll_id": poll_id,
                    },
                )
            return templates.TemplateResponse(
                request,
                "_kei_poll_logged_in_success.html",
                {
                    "kei_status_results": results,
                    "kei_status_options": KEI_STATUS_OPTIONS,
                    "kei_status_selected": validated,
                    "kei_impact_selected": impact_val,
                    "kei_impact_results": impact_results,
                    "kei_impact_options": KEI_POLL_IMPACT_OPTIONS,
                    "kei_poll_goal": get_kei_poll_goal(),
                    "kei_poll_is_owner": validated in KEI_OWNER_SLUGS,
                    "why_you_care_branch": get_why_you_care_branch_for_selection(validated),
                    "poll_id": poll_id,
                },
            )
        return RedirectResponse(
            _kei_status_redirect_url(poll_id, prompt_q, "submitted=1"), status_code=303
        )
    # Anonymous: set cookies for results/selection on next visit; return fragment or redirect
    results = await _get_kei_status_results(db)
    impact_results = await _get_kei_impact_results(db)
    cookie_opts = {"max_age": KEI_POLL_VOTED_MAX_AGE, "path": "/", "samesite": "lax"}
    cookies_to_set = [
        {"key": KEI_POLL_VOTED_COOKIE, "value": "1", **cookie_opts},
        {"key": KEI_POLL_CHOICE_COOKIE, "value": validated, **cookie_opts},
        {"key": KEI_IMPACT_SLUG_COOKIE, "value": impact_val, **cookie_opts},
    ]
    if request.headers.get("HX-Request"):
        if poll_id == "home-kei-poll":
            why_you_care_branch = get_why_you_care_branch_for_selection(validated)
            branch_slug = "owner" if validated in ("registered", "revoked", "denied") else validated
            wyc_pill_icon_slug = (
                validated if validated in ("registered", "revoked", "denied") else branch_slug
            )
            resp = templates.TemplateResponse(
                request,
                "_why_you_care_branch.html",
                {
                    "why_you_care_branch": why_you_care_branch,
                    "why_you_care_cta_nudge": WHY_YOU_CARE_CTA_NUDGE,
                    "wyc_pill_icon_slug": wyc_pill_icon_slug,
                    "kei_status_results": results,
                    "kei_status_options": KEI_STATUS_OPTIONS,
                    "kei_status_selected": validated,
                    "kei_impact_selected": impact_val,
                    "kei_impact_results": impact_results,
                    "kei_impact_options": KEI_POLL_IMPACT_OPTIONS,
                    "kei_poll_goal": get_kei_poll_goal(),
                    "kei_poll_initial_anon": True,
                    "poll_id": poll_id,
                },
            )
        else:
            resp = templates.TemplateResponse(
                request,
                "_kei_poll_anonymous_success.html",
                {
                    "kei_status_results": results,
                    "kei_status_options": KEI_STATUS_OPTIONS,
                    "kei_status_selected": validated,
                    "kei_impact_selected": impact_val,
                    "kei_impact_results": impact_results,
                    "kei_impact_options": KEI_POLL_IMPACT_OPTIONS,
                    "kei_poll_goal": get_kei_poll_goal(),
                    "kei_poll_is_owner": validated in KEI_OWNER_SLUGS,
                    "why_you_care_branch": get_why_you_care_branch_for_selection(validated),
                    "dev_available": cfg.DEV_MODE,
                    "poll_id": poll_id,
                    "show_go_to_site": poll_id == STANDALONE_KEI_POLL_ID,
                },
            )
        for params in cookies_to_set:
            resp.set_cookie(**params)
        return resp
    redir = RedirectResponse(
        _kei_status_redirect_url(poll_id, prompt_q, "submitted=1"), status_code=303
    )
    for params in cookies_to_set:
        redir.set_cookie(**params)
    return redir


@router.post("/updates/subscribe-email", include_in_schema=False)
async def subscribe_email_post(
    request: Request,
    email: str = Form(..., max_length=_EMAIL_MAX_LEN),
    csrf_token: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Public email-only subscription: create/update user wants_updates=True. No auth code."""
    token = csrf_token or request.headers.get("X-XSRF-TOKEN")
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(token, cookie_token):
        if request.headers.get("HX-Request"):
            return HTMLResponse(
                '<p class="subscribe-email-error" role="alert">Invalid or expired security '
                "token. Reload the page and try again.</p>",
                status_code=403,
            )
        return RedirectResponse("/updates?subscribe=csrf", status_code=303)
    client_ip = _client_ip(request)
    if not rate_limit_subscribe_email(client_ip):
        if request.headers.get("HX-Request"):
            return HTMLResponse(
                '<p class="subscribe-email-error" role="alert">Too many signup attempts. '
                "Try again later.</p>",
                status_code=429,
            )
        return RedirectResponse("/updates?subscribe=rate", status_code=303)
    normalized = _normalize_and_validate_subscribe_email(email)
    if not normalized:
        if request.headers.get("HX-Request"):
            return HTMLResponse(
                '<p class="subscribe-email-error" role="alert">'
                "Please enter a valid email address.</p>",
                status_code=400,
            )
        return RedirectResponse("/updates?subscribe=invalid", status_code=303)
    result = await db.execute(select(User).where(User.email == normalized))
    user = result.scalar_one_or_none()
    if user:
        user.wants_updates = True
        await db.commit()
        LOGGER.info("User id=%s re-subscribed to updates (email-only)", user.id)
    else:
        user = User(email=normalized, wants_updates=True)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        LOGGER.info("New subscriber via email-only signup: id=%s", user.id)
    if request.headers.get("HX-Request"):
        return HTMLResponse(
            '<p class="subscribe-email-success" role="status">'
            "You're subscribed. We'll send one email when the bill moves.</p>"
        )
    return RedirectResponse("/updates?subscribed=1", status_code=303)


@router.get("/updates/{update_id:int}", include_in_schema=False)
async def update_detail_page(
    update_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Redirect to updates page with anchor for backward compatibility and deep links."""
    result = await db.execute(
        select(Update).where(Update.id == update_id, Update.sent_at.isnot(None))
    )
    update = result.scalar_one_or_none()
    if not update:
        raise HTTPException(status_code=404, detail="Update not found")
    return RedirectResponse(f"/updates#update-{update_id}", status_code=303)


# ─── Admin routes ────────────────────────────────────────────────────────────


@router.get("/admin/updates", include_in_schema=False)
async def admin_updates_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Compose form and list of drafts/sent updates."""
    result = await db.execute(select(Update).order_by(Update.created_at.desc()))
    all_updates = list(result.scalars().all())
    flash = request.query_params.get("flash", "")
    ctx = {
        "request": request,
        "updates": all_updates,
        "flash": flash,
        "update_types": UPDATE_TYPES,
        "default_update_type": DEFAULT_UPDATE_TYPE,
        "update_type_labels": dict(UPDATE_TYPES),
    }
    return templates.TemplateResponse(request, "admin_updates.html", ctx)


@router.post("/admin/updates", include_in_schema=False)
async def admin_updates_create(
    request: Request,
    title: str = Form(..., min_length=1, max_length=256),
    body_plain: str = Form(..., min_length=1),
    update_type: str | None = Form(default=None),
    image: UploadFile | None = File(None),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Create a new draft update. Optional image is shown on the updates page and in the email."""
    title = title.strip()
    body_plain = body_plain.strip()
    if not title or not body_plain:
        return RedirectResponse("/admin/updates?flash=error", status_code=303)
    kind = _normalize_update_type(update_type)
    update = Update(
        title=title,
        body_plain=body_plain,
        body_html=_markdown_to_html(body_plain),
        update_type=kind,
    )
    db.add(update)
    await db.commit()
    await db.refresh(update)
    if image and image.filename:
        image_path = await _save_update_image(image, update.id)
        if image_path:
            update.image_path = image_path
            await db.commit()
    return RedirectResponse("/admin/updates?flash=draft_saved", status_code=303)


@router.get("/admin/updates/{update_id:int}/recipients", include_in_schema=False)
async def admin_updates_recipients(
    request: Request,
    update_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Choose recipients for this send. All subscribers listed with checkboxes (all checked)."""
    result = await db.execute(select(Update).where(Update.id == update_id))
    update = result.scalar_one_or_none()
    if not update:
        return RedirectResponse("/admin/updates?flash=not_found", status_code=303)
    if update.sent_at is not None:
        return RedirectResponse("/admin/updates?flash=already_sent", status_code=303)
    subscribers, _ = await _get_subscriber_list(db, None)
    flash = request.query_params.get("flash", "")
    ctx = {
        "request": request,
        "update": update,
        "subscribers": subscribers,
        "flash": flash,
    }
    return templates.TemplateResponse(request, "admin_recipients.html", ctx)


@router.get("/admin/updates/{update_id:int}/preview", include_in_schema=False)
async def admin_updates_preview(
    request: Request,
    update_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Preview the email as it would appear (with a sample unsubscribe link)."""
    result = await db.execute(select(Update).where(Update.id == update_id))
    update = result.scalar_one_or_none()
    if not update:
        return RedirectResponse("/admin/updates", status_code=303)
    sample_token = _create_unsubscribe_token(0)
    base_url = (cfg.APP_BASE_URL or "").rstrip("/")
    unsub_url = f"{base_url}/updates/unsubscribe?token={sample_token}"
    body_html = update.body_html or _plain_to_html(update.body_plain)
    image_url = f"{base_url}/static/{update.image_path}" if update.image_path else None
    html_content = _render_update_email_html(
        update.title, body_html, unsub_url, image_url=image_url
    )
    return HTMLResponse(html_content)


@router.post("/admin/updates/{update_id:int}/send", include_in_schema=False)
async def admin_updates_send(
    request: Request,
    update_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Start background send and redirect to status page. recipient_ids from recipients form."""
    form_data = await request.form()
    from_recipients_page = form_data.get("from_recipients_page") == "1"
    raw_ids = form_data.getlist("recipient_ids") if hasattr(form_data, "getlist") else []
    recipient_ids_parsed: list[int] = []
    for v in raw_ids:
        try:
            recipient_ids_parsed.append(int(v))
        except (TypeError, ValueError):
            continue
    # If from recipients page, use submitted list (empty = no recipients). Else send to all.
    ids_param: list[int] | None = recipient_ids_parsed if from_recipients_page else None
    if from_recipients_page and len(recipient_ids_parsed) == 0:
        return RedirectResponse(
            f"/admin/updates/{update_id}/recipients?flash=no_recipients", status_code=303
        )
    result = await db.execute(select(Update).where(Update.id == update_id))
    update = result.scalar_one_or_none()
    if not update:
        return RedirectResponse("/admin/updates?flash=not_found", status_code=303)
    if update.sent_at is not None:
        return RedirectResponse("/admin/updates?flash=already_sent", status_code=303)
    job_id = str(uuid.uuid4())
    jobs = getattr(request.app.state, "send_jobs", None)
    if jobs is None:
        request.app.state.send_jobs = {}
        jobs = request.app.state.send_jobs
    jobs[job_id] = {"total": 0, "sent": 0, "failed": 0, "done": False, "error": None}
    background_tasks.add_task(run_send_loop, request.app, job_id, update_id, ids_param)
    return RedirectResponse(f"/admin/updates/{update_id}/send/status?job={job_id}", status_code=303)


@router.get("/admin/updates/send/status/{job_id}", include_in_schema=False)
async def admin_send_status_json(
    request: Request,
    job_id: str,
    admin: User = Depends(require_admin),
):
    """JSON progress for status page polling. Returns total, sent, failed, done, error."""
    jobs = getattr(request.app.state, "send_jobs", None) or {}
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(
        {
            "total": job.get("total", 0),
            "sent": job.get("sent", 0),
            "failed": job.get("failed", 0),
            "done": job.get("done", False),
            "error": job.get("error"),
        }
    )


@router.get("/admin/updates/{update_id:int}/send/status", include_in_schema=False)
async def admin_send_status_page(
    request: Request,
    update_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Status page: progress (X of Y) and link back to admin when done."""
    job_id = request.query_params.get("job", "")
    if not job_id:
        return RedirectResponse("/admin/updates?flash=status_no_job", status_code=303)
    jobs = getattr(request.app.state, "send_jobs", None) or {}
    job = jobs.get(job_id)
    if job is None:
        ctx = {
            "request": request,
            "update_id": update_id,
            "job_id": job_id,
            "job_not_found": True,
        }
        return templates.TemplateResponse(request, "admin_send_status.html", ctx)
    result = await db.execute(select(Update).where(Update.id == update_id))
    update = result.scalar_one_or_none()
    ctx = {
        "request": request,
        "update_id": update_id,
        "job_id": job_id,
        "job_not_found": False,
        "update_title": update.title if update else "",
    }
    return templates.TemplateResponse(request, "admin_send_status.html", ctx)
