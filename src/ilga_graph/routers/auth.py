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
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..db import get_db
from ..db_models import AuthCode, User
from ..dependencies import create_session_token, get_current_user_optional

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
    if not cfg.SMTP_HOST:
        LOGGER.warning(
            "╔══════════════════════════════════════════╗\n"
            "║  AUTH CODE for %-26s ║\n"
            "║  Code: %-34s ║\n"
            "╚══════════════════════════════════════════╝",
            email,
            code,
        )
        return

    import aiosmtplib

    msg = EmailMessage()
    msg["Subject"] = f"Your ILGA Graph verification code: {code}"
    msg["From"] = cfg.SMTP_FROM
    msg["To"] = email
    plain = (
        f"Your verification code is: {code}\n\n"
        f"This code expires in 10 minutes.\n\n"
        f"If you didn't request this, you can ignore this email."
    )
    msg.set_content(plain)
    msg.add_alternative(_verification_email_html(code), subtype="html")

    # Port 587 = STARTTLS (plain then upgrade). Port 465 = immediate TLS (mutually exclusive).
    use_tls = cfg.SMTP_USE_TLS and cfg.SMTP_PORT == 465
    start_tls = cfg.SMTP_USE_TLS and cfg.SMTP_PORT == 587

    await aiosmtplib.send(
        msg,
        hostname=cfg.SMTP_HOST,
        port=cfg.SMTP_PORT,
        username=cfg.SMTP_USER or None,
        password=cfg.SMTP_PASS or None,
        use_tls=use_tls,
        start_tls=start_tls,
    )
    LOGGER.info("Auth code sent to %s via %s:%d", email, cfg.SMTP_HOST, cfg.SMTP_PORT)


@router.post("/request-code")
async def request_code(
    email: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Generate a 6-digit code and send it to the user's email."""
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
    email: str = Form(...),
    code: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Verify the 6-digit code.  On success: create/get user, set session cookie."""
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
