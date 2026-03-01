"""Account router: minimal user profile page (view and edit zip, newsletter, answers)."""

from __future__ import annotations

import re
from datetime import datetime

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..app_state import state
from ..constants import KEI_IMPACT_ALL_OPTIONS, KEI_STATUS_OPTIONS
from ..db import get_db
from ..db_models import User
from ..dependencies import require_user
from ..security import CSRF_COOKIE_NAME, validate_csrf_token

router = APIRouter(tags=["account"])
_ZIP_RE = re.compile(r"^\d{5}$")


def _kei_status_label(slug: str | None) -> str:
    """Return display label for kei_status slug."""
    if not slug:
        return ""
    for s, label in KEI_STATUS_OPTIONS:
        if s == slug:
            return label
    return slug


def _kei_impact_label(slug: str | None) -> str:
    """Return display label for kei_impact_slug."""
    if not slug:
        return ""
    for s, label in KEI_IMPACT_ALL_OPTIONS:
        if s == slug:
            return label
    return slug


def _account_context(user: User) -> dict:
    """Build template context for account page."""
    created_at = getattr(user, "created_at", None)
    return {
        "user": user,
        "email": user.email,
        "zip_code": getattr(user, "zip_code", None) or "",
        "wants_updates": getattr(user, "wants_updates", True),
        "kei_status": getattr(user, "kei_status", None),
        "kei_status_label": _kei_status_label(getattr(user, "kei_status", None)),
        "kei_impact_slug": getattr(user, "kei_impact_slug", None),
        "kei_impact_label": _kei_impact_label(getattr(user, "kei_impact_slug", None)),
        "kei_personal_note": getattr(user, "kei_personal_note", None) or "",
        "created_at": created_at,
        "created_at_iso": created_at.isoformat() if isinstance(created_at, datetime) else None,
    }


@router.get("/account")
async def account_page(
    request: Request,
    user: User = Depends(require_user),
):
    """Render account profile page (auth required; 401 redirects to home)."""
    templates = request.app.state.templates
    ctx = _account_context(user)
    ctx["request"] = request
    return templates.TemplateResponse(request, "account.html", ctx)


@router.post("/account")
async def account_update(
    request: Request,
    zip_code: str = Form(""),
    wants_updates: str | None = Form(None),
    csrf_token: str | None = Form(None),
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Update zip and/or newsletter preference; redirect back to /account."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        return RedirectResponse("/account?error=csrf", status_code=303)
    zip_param = (zip_code or "").strip()
    if zip_param:
        if not _ZIP_RE.match(zip_param) or zip_param not in getattr(state, "zip_to_district", {}):
            return RedirectResponse("/account?error=zip", status_code=303)
        user.zip_code = zip_param
    else:
        user.zip_code = None
    wants_true = ("1", "true", "on", "yes")
    user.wants_updates = (
        wants_updates is not None and wants_updates.lower() in wants_true
    )
    await db.commit()
    return RedirectResponse("/account?saved=1", status_code=303)
