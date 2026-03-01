"""Shared email sending via configured SMTP (Brevo). No-op when SMTP not configured."""

from __future__ import annotations

import html
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


def _welcome_email_plain(
    site_name: str,
    advocacy_url: str,
    kei_poll_url: str,
    unsub_url: str | None = None,
) -> str:
    """Plain text body for welcome email."""
    body = (
        f"Hello! My name is Tyler and I'm the founder of Land of Kei. "
        f"Whether you daydream of riding in a Honda Acty or are frustrated with the legal status of your kei vehicle, you're in the right place. \n\n"
        f"Thanks for signing in. You're now part of the effort to fix kei vehicle registration in Illinois.\n\n"
        f"Next step: Enter your ZIP to see who represents you and get a 2-minute call "
        f"script and email template: {advocacy_url}\n\n"
        f"Your input helps us explain to legislators who's affected (numbers speak louder): "
        f"{kei_poll_url}\n\n"
        f"You're receiving this because you just signed in to {site_name}.\n"
    )
    if unsub_url:
        body += f"\nUnsubscribe: {unsub_url}\n"
    body += "\n\nTyler Morales\nFounder of Land of Kei\n773-318-8539\n"
    return body


def _welcome_email_html(
    site_name: str,
    advocacy_url: str,
    kei_poll_url: str,
    unsub_url: str | None = None,
) -> str:
    """HTML body for welcome email."""
    s = html.escape(site_name)
    unsub_block = ""
    if unsub_url:
        unsub_block = f'\n    <p style="font-size: 0.85em;"><a href="{html.escape(unsub_url)}">Unsubscribe</a></p>'
    return f"""<!DOCTYPE html>
<html>
<body style="font-family: sans-serif; max-width: 600px; margin: 0 auto;">
    <p style="font-size: 0.9em; color: #666;">{s}</p>
    <h2 style="margin-top: 1em;">Welcome</h2>
    <p>Thanks for signing in. You're now part of the effort to fix kei vehicle registration in Illinois.</p>
    <p><strong>Next step:</strong> <a href="{html.escape(advocacy_url)}">Enter your ZIP</a> to see who represents you and get a 2-minute call script and email template.</p>
    <p>Your input helps us explain to legislators who's affected — <a href="{html.escape(kei_poll_url)}">tell us your kei status</a> (one quick question). Numbers speak louder than words.</p>
    <hr style="border: none; border-top: 1px solid #eee; margin: 2em 0;">
    <p style="font-size: 0.85em; color: #666;">You're receiving this because you just signed in to {s}.</p>{unsub_block}
    <p style="margin-top: 1.5em; font-size: 0.95em;">Tyler Morales<br>Founder of Land of Kei<br><a href="tel:+17733188539">773-318-8539</a></p>
</body>
</html>"""


async def send_welcome_email(
    email: str,
    site_name: str | None = None,
    unsub_url: str | None = None,
) -> bool:
    """Send welcome email to a newly signed-in user. Returns True if sent or dev mock."""
    base = (cfg.APP_BASE_URL or "").rstrip("/") or "https://landofkei.com"
    site = site_name or cfg.SITE_NAME or "Land of Kei"
    advocacy_url = f"{base}/advocacy"
    kei_poll_url = f"{base}/updates?prompt=kei"
    subject = f"Welcome to {site}"
    plain = _welcome_email_plain(site, advocacy_url, kei_poll_url, unsub_url)
    html_body = _welcome_email_html(site, advocacy_url, kei_poll_url, unsub_url)
    sent = await send_email(email, subject, plain, html_body)
    if sent:
        LOGGER.info("Welcome email sent to %s", email)
    return sent


def _story_review_plain(approved: bool, site_name: str, admin_message: str | None) -> str:
    """Plain text for community story approval/denial email."""
    site = site_name or "Land of Kei"
    if approved:
        line = "Your story is now live on the site."
        if admin_message and admin_message.strip():
            line += f"\n\n{admin_message.strip()}"
    else:
        line = "Thanks for submitting. We weren't able to include your story at this time."
        if admin_message and admin_message.strip():
            line += f"\n\n{admin_message.strip()}"
    return f"{line}\n\n— {site}\n"


def _story_review_html(approved: bool, site_name: str, admin_message: str | None) -> str:
    """HTML body for community story approval/denial email."""
    site = html.escape(site_name or "Land of Kei")
    if approved:
        line = "<p>Your story is now live on the site.</p>"
        if admin_message and admin_message.strip():
            line += f"<p>{html.escape(admin_message.strip())}</p>"
    else:
        line = "<p>Thanks for submitting. We weren't able to include your story at this time.</p>"
        if admin_message and admin_message.strip():
            line += f"<p>{html.escape(admin_message.strip())}</p>"
    return f"""<!DOCTYPE html><html><body style="font-family: sans-serif; max-width: 600px;">
    <p style="color: #666;">{site}</p>
    {line}
    <p style="font-size: 0.85em; color: #666;">— {site}</p>
</body></html>"""


async def send_story_review_email(
    to: str, approved: bool, admin_message: str | None = None
) -> bool:
    """Notify user after admin approves or denies their community story submission."""
    site = cfg.SITE_NAME or "Land of Kei"
    subject = "Your community story — update" if approved else "Your community story submission"
    plain = _story_review_plain(approved, site, admin_message)
    html_body = _story_review_html(approved, site, admin_message)
    sent = await send_email(to, subject, plain, html_body)
    if sent:
        LOGGER.info("Story review email sent to %s (approved=%s)", to, approved)
    return sent


def _statement_review_plain(approved: bool, site_name: str, admin_message: str | None) -> str:
    """Plain text for interest statement approval/denial email."""
    site = site_name or "Land of Kei"
    if approved:
        line = "Your statement is now live on the site."
        if admin_message and admin_message.strip():
            line += f"\n\n{admin_message.strip()}"
    else:
        line = "Thanks for submitting. We weren't able to include your statement at this time."
        if admin_message and admin_message.strip():
            line += f"\n\n{admin_message.strip()}"
    return f"{line}\n\n— {site}\n"


def _statement_review_html(approved: bool, site_name: str, admin_message: str | None) -> str:
    """HTML body for interest statement approval/denial email."""
    site = html.escape(site_name or "Land of Kei")
    if approved:
        line = "<p>Your statement is now live on the site.</p>"
        if admin_message and admin_message.strip():
            line += f"<p>{html.escape(admin_message.strip())}</p>"
    else:
        line = (
            "<p>Thanks for submitting. We weren't able to include your statement at this time.</p>"
        )
        if admin_message and admin_message.strip():
            line += f"<p>{html.escape(admin_message.strip())}</p>"
    return f"""<!DOCTYPE html><html><body style="font-family: sans-serif; max-width: 600px;">
    <p style="color: #666;">{site}</p>
    {line}
    <p style="font-size: 0.85em; color: #666;">— {site}</p>
</body></html>"""


async def send_statement_review_email(
    to: str, approved: bool, admin_message: str | None = None
) -> bool:
    """Notify user after admin approves or denies their interest statement submission."""
    site = cfg.SITE_NAME or "Land of Kei"
    subject = "Your statement — update" if approved else "Your statement submission"
    plain = _statement_review_plain(approved, site, admin_message)
    html_body = _statement_review_html(approved, site, admin_message)
    sent = await send_email(to, subject, plain, html_body)
    if sent:
        LOGGER.info("Statement review email sent to %s (approved=%s)", to, approved)
    return sent
