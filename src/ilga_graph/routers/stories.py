"""Community Stories: user-submitted photos + stories for the home page marquee."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..constants import KEI_STATUS_OPTIONS
from ..db import get_db
from ..db_models import CommunityStory, KeiInterestStatement, User
from ..dependencies import require_admin, require_user
from ..email_utils import send_statement_review_email, send_story_review_email
from ..routers.content import STRATEGIC_FIVE_POINTS
from ..security import CSRF_COOKIE_NAME, validate_csrf_token
from ..session_schedule import get_milestone_by_id, get_next_deadline_safe

LOGGER = logging.getLogger(__name__)

router = APIRouter()
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
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
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS
templates.env.globals["features"] = cfg.get_client_features()
templates.env.globals["kei_status_options"] = KEI_STATUS_OPTIONS

from ..campaign_helpers import get_current_action_campaign_for_template  # noqa: E402

templates.env.globals["get_current_action_campaign"] = get_current_action_campaign_for_template
templates.env.globals["get_milestone_by_id"] = get_milestone_by_id
templates.env.globals["get_next_deadline"] = get_next_deadline_safe

_STORY_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
_STORY_IMAGE_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
# Magic bytes for content-type verification (reject spoofed types).
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_WEBP_RIFF = b"RIFF"
_WEBP_WEBP = b"WEBP"  # at offset 8
_STORY_MAX_LEN = 500
_STATEMENT_MAX_LEN = 500
_NAME_MAX_LEN = 120
_LOCATION_MAX_LEN = 100
_KEI_OWNER_SLUGS = frozenset({"registered", "revoked", "denied"})
_KEI_NON_OWNER_SLUGS = frozenset({"would_want", "would_not_want"})


def _magic_matches_content_type(content: bytes, content_type: str) -> bool:
    """Return True if file magic bytes match the declared content type."""
    if not content:
        return False
    if content_type == "image/jpeg":
        return content.startswith(_JPEG_MAGIC)
    if content_type == "image/png":
        return content.startswith(_PNG_MAGIC)
    if content_type == "image/webp":
        return len(content) >= 12 and content.startswith(_WEBP_RIFF) and content[8:12] == _WEBP_WEBP
    return False


async def _save_story_image(upload: UploadFile, user_id: int) -> str | None:
    """Validate and save one image; return relative path or None."""
    if not upload.filename or not upload.content_type:
        return None
    if upload.content_type not in _STORY_IMAGE_TYPES:
        LOGGER.warning("Story image rejected: content_type=%s", upload.content_type)
        return None
    ext = _STORY_IMAGE_EXT.get(upload.content_type, ".bin")
    content = b""
    while True:
        chunk = await upload.read(1024 * 256)
        if not chunk:
            break
        content += chunk
        if len(content) > cfg.STORY_MAX_IMAGE_BYTES:
            LOGGER.warning("Story image rejected: too large")
            return None
    if not content:
        return None
    if not _magic_matches_content_type(content, upload.content_type):
        LOGGER.warning(
            "Story image rejected: magic bytes do not match content_type=%s",
            upload.content_type,
        )
        return None
    upload_dir = _STATIC_DIR / cfg.STORY_IMAGE_UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{user_id}_{uuid.uuid4().hex}{ext}"
    (upload_dir / fname).write_bytes(content)
    return f"{cfg.STORY_IMAGE_UPLOAD_DIR}/{fname}"


@router.post("/community-stories", response_class=HTMLResponse, include_in_schema=False)
async def community_stories_submit(
    request: Request,
    name: str = Form(..., min_length=1, max_length=_NAME_MAX_LEN),
    location: str = Form(..., min_length=1, max_length=_LOCATION_MAX_LEN),
    story: str = Form(..., min_length=1),
    consent: str | None = Form(None),
    csrf_token: str | None = Form(None),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_user),
):
    """Submit a community story (photo + name + story). Auth and consent required."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired security token. Reload the page and try again.",
        )
    if not consent or consent.strip().lower() not in ("on", "true", "1", "yes"):
        raise HTTPException(
            status_code=400,
            detail="You must agree to the terms (consent checkbox) to submit your story.",
        )
    name = name.strip()
    location = location.strip()
    story = story.strip()
    if len(story) > _STORY_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Story must be at most {_STORY_MAX_LEN} characters.",
        )
    if not name or not location or not story:
        raise HTTPException(status_code=400, detail="Name, location, and story are required.")

    image_path = await _save_story_image(image, user.id)
    if not image_path:
        raise HTTPException(
            status_code=400,
            detail="A valid image (JPEG, PNG, or WebP, max 5MB) is required.",
        )

    story_row = CommunityStory(
        user_id=user.id,
        name=name,
        email=user.email,
        location=location,
        story=story,
        image_path=image_path,
        consent=True,
        status="pending",
    )
    db.add(story_row)
    await db.commit()
    await db.refresh(story_row)

    return templates.TemplateResponse(
        request,
        "_story_submit_success.html",
        {"story": story_row},
    )


@router.post("/community-statements", response_class=HTMLResponse, include_in_schema=False)
async def community_statements_submit(
    request: Request,
    name: str = Form(..., min_length=1, max_length=_NAME_MAX_LEN),
    location: str = Form(..., min_length=1, max_length=_LOCATION_MAX_LEN),
    statement: str = Form(..., min_length=1),
    consent: str | None = Form(None),
    csrf_token: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_user),
):
    """Submit text-only statement (non-owners). Requires kei_status would_want/would_not_want."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired security token. Reload the page and try again.",
        )
    kei_status = getattr(user, "kei_status", None)
    if kei_status not in _KEI_NON_OWNER_SLUGS:
        raise HTTPException(
            status_code=403,
            detail="Non-owners only (would want/wouldn't want). Use Share your story for photo.",
        )
    if not consent or consent.strip().lower() not in ("on", "true", "1", "yes"):
        raise HTTPException(
            status_code=400,
            detail="You must agree to the terms (consent checkbox) to submit.",
        )
    name = name.strip()
    location = location.strip()
    statement = statement.strip()
    if len(statement) > _STATEMENT_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Statement must be at most {_STATEMENT_MAX_LEN} characters.",
        )
    if not name or not location or not statement:
        raise HTTPException(status_code=400, detail="Name, location, and statement are required.")

    row = KeiInterestStatement(
        user_id=user.id,
        name=name,
        email=user.email,
        location=location,
        statement=statement,
        consent=True,
        status="pending",
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    return templates.TemplateResponse(
        request,
        "_statement_submit_success.html",
        {"statement": row},
    )


@router.get("/admin/stories", include_in_schema=False)
async def admin_stories_list(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin),
):
    """List all community story submissions (pending, approved, denied)."""
    result = await db.execute(select(CommunityStory).order_by(CommunityStory.created_at.desc()))
    stories = list(result.scalars().all())
    pending = [s for s in stories if s.status == "pending"]
    approved = [s for s in stories if s.status == "approved"]
    denied = [s for s in stories if s.status == "denied"]
    return templates.TemplateResponse(
        request,
        "admin_stories.html",
        {
            "pending": pending,
            "approved": approved,
            "denied": denied,
        },
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@router.post("/admin/stories/{story_id:int}/review", include_in_schema=False)
async def admin_stories_review(
    request: Request,
    story_id: int,
    action: str = Form(...),
    message: str | None = Form(None),
    csrf_token: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin),
):
    """Approve or deny a community story; send email to submitter with optional message."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        return RedirectResponse("/admin/stories?flash=csrf", status_code=303)
    result = await db.execute(select(CommunityStory).where(CommunityStory.id == story_id))
    story = result.scalar_one_or_none()
    if not story:
        return RedirectResponse("/admin/stories?flash=not_found", status_code=303)
    if story.status != "pending":
        return RedirectResponse("/admin/stories?flash=already_reviewed", status_code=303)
    action = (action or "").strip().lower()
    if action not in ("approve", "deny"):
        return RedirectResponse("/admin/stories?flash=invalid_action", status_code=303)

    story.status = "approved" if action == "approve" else "denied"
    story.reviewed_at = _utcnow()
    story.admin_message = (message or "").strip() or None
    await db.commit()

    try:
        await send_story_review_email(story.email, action == "approve", story.admin_message)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("Story review email failed (story_id=%s): %s", story_id, e)

    flash = "approved" if action == "approve" else "denied"
    return RedirectResponse(f"/admin/stories?flash={flash}", status_code=303)


@router.get("/admin/statements", include_in_schema=False)
async def admin_statements_list(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin),
):
    """List all interest statement submissions (pending, approved, denied)."""
    result = await db.execute(
        select(KeiInterestStatement).order_by(KeiInterestStatement.created_at.desc())
    )
    statements = list(result.scalars().all())
    pending = [s for s in statements if s.status == "pending"]
    approved = [s for s in statements if s.status == "approved"]
    denied = [s for s in statements if s.status == "denied"]
    return templates.TemplateResponse(
        request,
        "admin_statements.html",
        {
            "pending": pending,
            "approved": approved,
            "denied": denied,
        },
    )


@router.post("/admin/statements/{statement_id:int}/review", include_in_schema=False)
async def admin_statements_review(
    request: Request,
    statement_id: int,
    action: str = Form(...),
    message: str | None = Form(None),
    csrf_token: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin),
):
    """Approve or deny an interest statement; send email to submitter with optional message."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        return RedirectResponse("/admin/statements?flash=csrf", status_code=303)
    result = await db.execute(
        select(KeiInterestStatement).where(KeiInterestStatement.id == statement_id)
    )
    stmt = result.scalar_one_or_none()
    if not stmt:
        return RedirectResponse("/admin/statements?flash=not_found", status_code=303)
    if stmt.status != "pending":
        return RedirectResponse("/admin/statements?flash=already_reviewed", status_code=303)
    action = (action or "").strip().lower()
    if action not in ("approve", "deny"):
        return RedirectResponse("/admin/statements?flash=invalid_action", status_code=303)

    stmt.status = "approved" if action == "approve" else "denied"
    stmt.reviewed_at = _utcnow()
    stmt.admin_message = (message or "").strip() or None
    await db.commit()

    try:
        await send_statement_review_email(stmt.email, action == "approve", stmt.admin_message)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("Statement review email failed (statement_id=%s): %s", statement_id, e)

    flash = "approved" if action == "approve" else "denied"
    return RedirectResponse(f"/admin/statements?flash={flash}", status_code=303)
