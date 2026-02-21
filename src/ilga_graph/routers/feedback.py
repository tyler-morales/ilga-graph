"""In-app bug report form (no GitHub or external service required)."""

from __future__ import annotations

import html
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..db import get_db
from ..db_models import BugReport
from ..security import (
    CSRF_COOKIE_NAME,
    rate_limit_bug_report,
    validate_csrf_token,
    validate_page_url,
)

LOGGER = logging.getLogger(__name__)

BUG_REPORT_DESCRIPTION_MIN_LENGTH = 20

# Optional email: local@domain.tld (one @, dot in domain, length limits).
_EMAIL_LOCAL_MAX = 64
_EMAIL_DOMAIN_MAX = 253
_EMAIL_RE = re.compile(
    r"^[^\s@]+@[^\s@]+\.[^\s@]+$",
    re.IGNORECASE,
)


def _is_valid_email(addr: str) -> bool:
    """Return True if addr is a non-empty string with valid email format."""
    if not addr or len(addr) > 320:
        return False
    if not _EMAIL_RE.match(addr):
        return False
    local, _, domain = addr.partition("@")
    if len(local) < 1 or len(local) > _EMAIL_LOCAL_MAX:
        return False
    if len(domain) < 4 or len(domain) > _EMAIL_DOMAIN_MAX or "." not in domain:
        return False
    return True


_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_EXT_FROM_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
_EXT_TO_MIME = {ext: ct for ct, ext in _EXT_FROM_TYPE.items()}

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
router = APIRouter()

_TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
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


def _mime_type_for_filename(filename: str) -> tuple[str, str]:
    """Return (maintype, subtype) for add_attachment from a stored filename."""
    ext = Path(filename).suffix.lower()
    ct = _EXT_TO_MIME.get(ext, "application/octet-stream")
    maintype, _, subtype = ct.partition("/")
    return maintype, subtype or "octet-stream"


_SCREENSHOT_PLACEHOLDER = "__SCREENSHOT_BLOCK__"


def _build_bug_report_bodies(
    description: str,
    reporter_email: str | None,
    page_url: str | None,
    attachment_urls: list[str] | None,
    timestamp: str,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> tuple[str, str]:
    """Return (plain_text_body, html_body) for bug report email. Screenshot block is placeholder."""
    if attachment_urls:
        screenshot_plain = "\n".join(f"  {u}" for u in attachment_urls)
    else:
        screenshot_plain = "No image sent."

    plain = (
        f"Timestamp: {timestamp}\n"
        f"Email: {reporter_email or '—'}\n"
        f"Issue: {description}\n"
        f"Page: {page_url or '—'}\n"
        f"Screenshot: {screenshot_plain}\n"
    )
    if client_ip:
        plain += f"IP: {client_ip}\n"
    if user_agent:
        plain += f"User-Agent: {user_agent}\n"

    desc_esc = html.escape(description)
    screenshot_html = _SCREENSHOT_PLACEHOLDER
    html_parts = [
        "<p><strong>Timestamp:</strong> ",
        html.escape(timestamp),
        "</p>",
        "<p><strong>Email:</strong> ",
        html.escape(reporter_email or "—"),
        "</p>",
        "<p><strong>Issue:</strong></p><p>",
        desc_esc,
        "</p>",
        "<p><strong>Page:</strong> ",
        (f'<a href="{html.escape(page_url)}">{html.escape(page_url)}</a>' if page_url else "—"),
        "</p>",
        "<p><strong>Screenshot:</strong> ",
        screenshot_html,
        "</p>",
    ]
    if client_ip:
        html_parts.append(f"<p><strong>IP:</strong> {html.escape(client_ip)}</p>")
    if user_agent:
        html_parts.append(f"<p><strong>User-Agent:</strong> {html.escape(user_agent)}</p>")
    return plain, "".join(html_parts)


def _screenshot_block_html(
    attachment_urls: list[str],
    image_count: int,
) -> str:
    """Build HTML for screenshot block: inline img cid refs plus fallback links when available."""
    inline = "".join(
        f'<img src="cid:screenshot{i}" alt="Screenshot {i + 1}" style="max-width:100%;" />'
        for i in range(image_count)
    )
    if attachment_urls:
        links = " ".join(
            f'<a href="{html.escape(u)}">View screenshot {i + 1}</a>'
            for i, u in enumerate(attachment_urls)
        )
        return inline + "<br><small>(" + links + ")</small>"
    return inline if image_count else "No image sent."


async def _send_bug_report_notification(
    description: str,
    reporter_email: str | None,
    page_url: str | None,
    attachment_paths: list[str] | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Email BETA_BANNER_EMAIL when a bug report is submitted (if SMTP configured)."""
    if not cfg.SMTP_HOST or not cfg.BETA_BANNER_EMAIL:
        return
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    import aiosmtplib

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    base_url = (cfg.APP_BASE_URL or "").rstrip("/")
    attachment_urls: list[str] = []
    if attachment_paths and base_url:
        attachment_urls = [f"{base_url}/report-bug/attachments/{f}" for f in attachment_paths]

    plain_body, html_body = _build_bug_report_bodies(
        description,
        reporter_email,
        page_url,
        attachment_urls if attachment_urls else None,
        timestamp=timestamp,
        client_ip=client_ip,
        user_agent=user_agent,
    )

    upload_dir = Path(cfg.BUG_REPORT_UPLOAD_DIR) if cfg.BUG_REPORT_UPLOAD_DIR else None
    image_parts: list[tuple[str, bytes, str, str]] = []  # (fname, data, maintype, subtype)
    if attachment_paths and upload_dir and upload_dir.is_dir():
        for fname in attachment_paths:
            if not fname or ".." in fname or "/" in fname:
                continue
            path = (upload_dir / fname).resolve()
            try:
                path.relative_to(upload_dir.resolve())
            except ValueError:
                continue
            if not path.is_file():
                continue
            maintype, subtype = _mime_type_for_filename(fname)
            image_parts.append((fname, path.read_bytes(), maintype, subtype))

    if image_parts:
        screenshot_block = _screenshot_block_html(attachment_urls, len(image_parts))
    elif attachment_urls:
        screenshot_block = " ".join(
            f'<a href="{html.escape(u)}">View screenshot</a>' for u in attachment_urls
        )
    else:
        screenshot_block = "No image sent."

    html_body = html_body.replace(_SCREENSHOT_PLACEHOLDER, screenshot_block)

    if image_parts:
        msg = MIMEMultipart("related")
        msg["Subject"] = f"Bug report from {cfg.SITE_NAME or 'Land of Kei'}"
        msg["From"] = cfg.SMTP_FROM
        msg["To"] = cfg.BETA_BANNER_EMAIL
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_body, "plain"))
        alt.attach(MIMEText(html_body, "html"))
        msg.attach(alt)
        for i, (fname, data, maintype, subtype) in enumerate(image_parts):
            img = MIMEImage(data, _subtype=subtype)
            img.add_header("Content-Disposition", "inline", filename=fname)
            img.add_header("Content-ID", f"<screenshot{i}>")
            msg.attach(img)
    else:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Bug report from {cfg.SITE_NAME or 'Land of Kei'}"
        msg["From"] = cfg.SMTP_FROM
        msg["To"] = cfg.BETA_BANNER_EMAIL
        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

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
    LOGGER.info("Bug report notification sent to %s", cfg.BETA_BANNER_EMAIL)


@router.get("/report-bug/attachments/{filename:path}")
async def report_bug_attachment(filename: str):
    """Serve a stored bug-report image (for viewing screenshots). No auth; dir is not listable."""
    if not cfg.BUG_REPORT_UPLOAD_DIR or not filename or ".." in filename or "/" in filename:
        return RedirectResponse("/report-bug", status_code=303)
    base = Path(cfg.BUG_REPORT_UPLOAD_DIR).resolve()
    path = (base / filename).resolve()
    if not path.is_file():
        return RedirectResponse("/report-bug", status_code=303)
    try:
        path.relative_to(base)
    except ValueError:
        return RedirectResponse("/report-bug", status_code=303)
    return FileResponse(path)


def _client_ip(request: Request) -> str:
    """Best-effort client IP (proxy headers or direct)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return ""


def _verify_turnstile(token: str | None, remote_ip: str) -> bool:
    """Verify Turnstile token with Cloudflare. Returns True only if success."""
    if not token or not cfg.TURNSTILE_SECRET_KEY:
        return False
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
        data = resp.json()
        return data.get("success") is True
    except Exception:
        LOGGER.exception("Turnstile siteverify failed")
        return False


@router.get("/report-bug")
async def report_bug_page(request: Request):
    """Show the in-app bug report form. Sets csrf_token for the form."""
    submitted = request.query_params.get("submitted") == "1"
    csrf_token = getattr(request.state, "csrf_token", None) or ""
    turnstile_site_key = cfg.TURNSTILE_SITE_KEY or None
    return templates.TemplateResponse(
        "report_bug.html",
        {
            "request": request,
            "submitted": submitted,
            "csrf_token": csrf_token,
            "turnstile_site_key": turnstile_site_key,
            "description_min_length": BUG_REPORT_DESCRIPTION_MIN_LENGTH,
        },
    )


async def _save_bug_report_image(upload: UploadFile) -> str | None:
    """Validate and save one image; return stored filename or None."""
    if not upload.filename or not upload.content_type:
        return None
    if upload.content_type not in _ALLOWED_IMAGE_TYPES:
        LOGGER.warning("Bug report image rejected: content_type=%s", upload.content_type)
        return None
    ext = _EXT_FROM_TYPE.get(upload.content_type, ".bin")
    # Read with size limit
    content = b""
    while True:
        chunk = await upload.read(1024 * 256)
        if not chunk:
            break
        content += chunk
        if len(content) > cfg.BUG_REPORT_MAX_IMAGE_BYTES:
            LOGGER.warning("Bug report image rejected: too large")
            return None
    if not content:
        return None
    upload_dir = Path(cfg.BUG_REPORT_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    fname = uuid.uuid4().hex + ext
    (upload_dir / fname).write_bytes(content)
    return fname


@router.post("/report-bug")
async def report_bug_submit(
    request: Request,
    description: str = Form(..., min_length=1, max_length=10000),
    reporter_email: str | None = Form(None, max_length=320),
    page_url: str | None = Form(None, max_length=2048),
    image: UploadFile | None = File(None),
    csrf_token: str | None = Form(None),
    cf_turnstile_response: str | None = Form(None, alias="cf-turnstile-response"),
    db: AsyncSession = Depends(get_db),
):
    """Accept bug report form: store in DB, optional image; optionally email."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(csrf_token, cookie_token):
        return RedirectResponse("/report-bug?error=csrf", status_code=303)
    client_ip = _client_ip(request)
    if not rate_limit_bug_report(client_ip):
        return RedirectResponse("/report-bug?error=rate", status_code=303)
    if cfg.TURNSTILE_SECRET_KEY and not _verify_turnstile(cf_turnstile_response, client_ip):
        return RedirectResponse("/report-bug?error=captcha", status_code=303)
    description = description.strip()
    if not description or len(description) < BUG_REPORT_DESCRIPTION_MIN_LENGTH:
        return RedirectResponse("/report-bug?error=description_short", status_code=303)
    reporter_email = (reporter_email or "").strip() or None
    if reporter_email and not _is_valid_email(reporter_email):
        return RedirectResponse("/report-bug?error=email_invalid", status_code=303)
    page_url = validate_page_url((page_url or "").strip() or None)

    attachment_paths: list[str] = []
    if cfg.BUG_REPORT_UPLOAD_DIR and image:
        saved = await _save_bug_report_image(image)
        if saved:
            attachment_paths.append(saved)

    report = BugReport(
        description=description,
        reporter_email=reporter_email,
        page_url=page_url,
        attachment_paths=json.dumps(attachment_paths) if attachment_paths else None,
    )
    db.add(report)
    await db.commit()
    LOGGER.info("Bug report saved id=%s attachments=%s", report.id, len(attachment_paths))

    try:
        await _send_bug_report_notification(
            description,
            reporter_email,
            page_url,
            attachment_paths or None,
            client_ip=client_ip,
            user_agent=request.headers.get("User-Agent"),
        )
    except Exception:
        LOGGER.exception("Failed to send bug report notification email")

    return RedirectResponse("/report-bug?submitted=1", status_code=303)
