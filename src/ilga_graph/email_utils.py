"""Shared email sending via configured SMTP (Brevo). No-op when SMTP not configured."""

from __future__ import annotations

import logging
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart

from . import config as cfg

LOGGER = logging.getLogger(__name__)

_MessageT = EmailMessage | MIMEMultipart


async def send_message(msg: _MessageT) -> bool:
    """Send one email via SMTP. Returns True if sent, False if skipped (no SMTP)."""
    if not cfg.SMTP_HOST:
        return False
    import aiosmtplib

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
    to = msg.get("To", "")
    LOGGER.info("Email sent to %s via %s:%d", to, cfg.SMTP_HOST, cfg.SMTP_PORT)
    return True


def _is_valid_recipient(to: str) -> bool:
    """Return True if to is a non-empty string that looks like an email address."""
    if not to or not isinstance(to, str):
        return False
    t = to.strip()
    return bool(t and "@" in t and len(t) <= 320)


async def send_email(to: str, subject: str, plain: str, html: str) -> bool:
    """Build and send one plain+HTML email.
    True if sent or dev mock; False if no SMTP or invalid to.
    """
    if not _is_valid_recipient(to):
        LOGGER.warning("send_email skipped: invalid or empty recipient")
        return False
    if cfg.SMTP_HOST and not (cfg.SMTP_FROM or "").strip():
        LOGGER.warning("send_email skipped: SMTP configured but SMTP_FROM empty")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.SMTP_FROM or ""
    msg["To"] = to.strip()
    msg.set_content(plain)
    msg.add_alternative(html, subtype="html")
    if not cfg.SMTP_HOST and cfg.DEV_MODE:
        snippet = (plain[:200] + "…") if len(plain) > 200 else plain
        LOGGER.info(
            "[DEV mock] Campaign email would send to %s | subject=%s | body: %s",
            to,
            subject,
            snippet.replace("\n", " "),
        )
        return True
    return await send_message(msg)
