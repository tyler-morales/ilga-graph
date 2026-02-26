"""Updates router: public /updates page, subscribe/unsubscribe, admin compose and send."""

from __future__ import annotations

import html
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import markdown
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

from .. import config as cfg
from ..db import async_session_factory, get_db
from ..db_models import Update, User
from ..dependencies import get_current_user_optional, require_admin, require_user
from ..email_utils import send_email
from ..routers.content import (
    CAMPAIGN_STATUS,
    CAMPAIGN_TIMELINE_ACHIEVED_COUNT,
    CAMPAIGN_TIMELINE_CHECKPOINTS,
    STRATEGIC_FIVE_POINTS,
)

LOGGER = logging.getLogger(__name__)

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
templates.env.globals["footer_last_updated"] = cfg.FOOTER_LAST_UPDATED
templates.env.globals["footer_last_updated_iso"] = cfg.FOOTER_LAST_UPDATED_ISO
templates.env.globals["features"] = cfg.get_client_features()
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS

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
    site = cfg.SITE_NAME or "The Land of Kei"
    tmpl = templates.env.get_template("_update_email.html")
    return tmpl.render(
        site_name=site,
        title=title,
        body_html=body_html,
        unsub_url=unsub_url,
        image_url=image_url,
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


# ─── Public routes ────────────────────────────────────────────────────────────


def _updates_page_ctx(
    request: Request,
    sent_updates: list[Update],
    user: User | None,
) -> dict[str, Any]:
    """Build template context for updates page (all updates on one page, sidebar TOC)."""
    campaign_timeline = [
        {"label": label, "achieved": i < CAMPAIGN_TIMELINE_ACHIEVED_COUNT}
        for i, label in enumerate(CAMPAIGN_TIMELINE_CHECKPOINTS)
    ]
    return {
        "request": request,
        "campaign_status": CAMPAIGN_STATUS,
        "campaign_timeline": campaign_timeline,
        "updates": sent_updates,
        "user": user,
        "wants_updates": user.wants_updates if user else None,
        "update_type_labels": dict(UPDATE_TYPES),
        "update_type_phrases": UPDATE_TYPE_PHRASES,
    }


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
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Turn subscription on for the current user; redirect back to /updates."""
    user.wants_updates = True
    await db.commit()
    return RedirectResponse("/updates", status_code=303)


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
