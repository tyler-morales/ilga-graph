"""Auth router: passwordless email-code authentication.

Endpoints:
    POST /auth/request-code   — send a 6-digit code to the given email
    POST /auth/verify-code    — verify code, set session cookie, return user email
    POST /auth/logout         — clear session cookie
    GET  /auth/me             — return current user email (or 401)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sys
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..db import get_db
from ..db_models import AuthCode, OutreachStepEvent, User
from ..dependencies import create_session_token, get_current_user_optional
from ..email_utils import send_email
from ..security import (
    CSRF_COOKIE_NAME,
    rate_limit_request_code,
    rate_limit_verify_code,
    validate_anon_session_id,
    validate_csrf_token,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_CODE_TTL = timedelta(minutes=10)


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _verification_email_html(code: str) -> str:
    """Return HTML body for the verification code email."""
    return f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: sans-serif;">
        <p>Your verification code is: <strong>{code}</strong></p>
        <p>This code expires in 10 minutes.</p>
        <p>If you didn't request this, you can ignore this email.</p>
    </body>
    </html>
    """


async def _send_code_email(email: str, code: str) -> None:
    """Send the verification code via SMTP, or log it in dev."""
    subject = f"Your {cfg.SITE_NAME} verification code: {code}"
    plain = (
        f"Your verification code is: {code}\n\n"
        f"This code expires in 10 minutes.\n\n"
        f"If you didn't request this, you can ignore this email."
    )
    sent = await send_email(email, subject, plain, _verification_email_html(code))
    if not sent:
        banner = (
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  AUTH CODE (no SMTP — check this terminal for sign-in)  ║\n"
            f"║  Email: {email[:44]:<44} ║\n"
            f"║  Code:  {code:<44} ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
        )
        print(banner, file=sys.stderr, flush=True)
        LOGGER.warning("Auth code for %s (no SMTP): %s", email, code)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


@router.post("/request-code")
async def request_code(
    request: Request,
    email: str = Form(...),
    csrf_token: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Generate a 6-digit code and send it to the user's email."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        return JSONResponse(
            {"ok": False, "error": "Invalid or expired security token. Reload the page."},
            status_code=403,
        )
    client_ip = _client_ip(request)
    if not rate_limit_request_code(client_ip, email):
        return JSONResponse(
            {"ok": False, "error": "Too many attempts. Try again later."},
            status_code=429,
        )
    email = email.strip().lower()
    if not email or "@" not in email:
        return JSONResponse({"ok": False, "error": "Invalid email"}, status_code=400)

    code = f"{secrets.randbelow(10**6):06d}"
    auth_code = AuthCode(
        email=email,
        code_hash=_hash_code(code),
        expires_at=datetime.now(timezone.utc) + _CODE_TTL,
    )
    db.add(auth_code)
    await db.commit()

    try:
        await _send_code_email(email, code)
    except Exception:
        LOGGER.exception("Failed to send auth code to %s", email)
        # Help debug 535: Brevo needs SMTP login (username) from SMTP tab, not account email.
        if cfg.SMTP_HOST and "brevo" in cfg.SMTP_HOST.lower():
            LOGGER.info(
                "Brevo SMTP: ILGA_SMTP_USER = SMTP login from Settings → SMTP & API; "
                "ILGA_SMTP_PASS = SMTP key from same page."
            )
        return JSONResponse(
            {"ok": False, "error": "Could not send email. Try again."},
            status_code=500,
        )

    return {"ok": True}


@router.post("/verify-code")
async def verify_code(
    request: Request,
    email: str = Form(...),
    code: str = Form(...),
    anon_session_id: str | None = Form(None),
    csrf_token: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Verify the 6-digit code. On success: create/get user, set session cookie.

    If anon_session_id is provided and valid, outreach_step_events rows with that
    session_id are attributed to the user (user_id set, session_id cleared).
    """
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        return JSONResponse(
            {"ok": False, "error": "Invalid or expired security token. Reload the page."},
            status_code=403,
        )
    client_ip = _client_ip(request)
    if not rate_limit_verify_code(client_ip):
        return JSONResponse(
            {"ok": False, "error": "Too many attempts. Try again later."},
            status_code=429,
        )
    email = email.strip().lower()
    code = code.strip()
    now = datetime.now(timezone.utc)

    result = await db.execute(
        select(AuthCode)
        .where(
            AuthCode.email == email,
            AuthCode.code_hash == _hash_code(code),
            AuthCode.used == False,  # noqa: E712
            AuthCode.expires_at > now,
        )
        .order_by(AuthCode.created_at.desc())
        .limit(1)
    )
    auth_code = result.scalar_one_or_none()

    if auth_code is None:
        return JSONResponse(
            {"ok": False, "error": "Invalid or expired code"},
            status_code=400,
        )

    auth_code.used = True

    user_result = await db.execute(select(User).where(User.email == email))
    user = user_result.scalar_one_or_none()
    if user is None:
        user = User(email=email)
        db.add(user)
        await db.flush()
    user.last_login_at = now

    anon_sid = validate_anon_session_id(anon_session_id)
    if anon_sid:
        await db.execute(
            update(OutreachStepEvent)
            .where(OutreachStepEvent.session_id == anon_sid)
            .values(user_id=user.id, session_id=None)
        )

    await db.commit()

    token = create_session_token(user.id)
    response = JSONResponse({"ok": True, "email": user.email})
    response.set_cookie(
        key=cfg.AUTH_COOKIE_NAME,
        value=token,
        max_age=cfg.AUTH_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=cfg.PROFILE == "prod",
    )
    LOGGER.info("User authenticated: %s (id=%d)", user.email, user.id)
    return response


@router.post("/logout")
async def logout():
    """Clear the session cookie."""
    response = JSONResponse({"ok": True})
    response.delete_cookie(key=cfg.AUTH_COOKIE_NAME)
    return response


@router.get("/me")
async def me(user: User | None = Depends(get_current_user_optional)):
    """Return the current user's email. Always 200; use body.authenticated to check session."""
    if user is None:
        return {"authenticated": False}
    return {"authenticated": True, "email": user.email}
