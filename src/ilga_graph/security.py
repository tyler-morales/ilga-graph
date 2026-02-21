"""CSRF and rate limiting for form and API submissions.

- CSRF: double-submit cookie (token in cookie + body/header); validated on state-changing POSTs.
- Rate limiting: in-memory sliding window per key (IP, or IP+email) to prevent abuse.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from itsdangerous import BadSignature, URLSafeTimedSerializer

from . import config as cfg

LOGGER = logging.getLogger(__name__)

# Cookie name for CSRF token (read by JS for fetch; validated against body/header).
CSRF_COOKIE_NAME = "XSRF-TOKEN"
CSRF_FORM_FIELD = "csrf_token"
CSRF_HEADER_NAME = "X-XSRF-TOKEN"
CSRF_MAX_AGE_SECONDS = 60 * 60  # 1 hour

_csrf_signer = URLSafeTimedSerializer(cfg.AUTH_SECRET)

# In-memory rate limit: key -> list of timestamps (pruned each check).
_rate_entries: defaultdict[str, list[float]] = defaultdict(list)


def generate_csrf_token() -> str:
    """Return a signed, time-limited token for CSRF protection."""
    payload = {"t": int(time.time())}
    return _csrf_signer.dumps(payload)


def validate_csrf_token(token: str | None, cookie_value: str | None) -> bool:
    """Return True if token is present, matches cookie, and is valid (signature + not expired)."""
    if not token or not cookie_value or token != cookie_value:
        return False
    try:
        _csrf_signer.loads(token, max_age=CSRF_MAX_AGE_SECONDS)
        return True
    except BadSignature:
        return False


def rate_limit_check(key: str, window_seconds: int, max_count: int) -> bool:
    """Return True if the key is within limit (allow request); False if over limit (reject).

    Sliding window (wall-clock): prune timestamps older than window_seconds, then check count.
    """
    now = time.time()
    cutoff = now - window_seconds
    entries = _rate_entries[key]
    while entries and entries[0] < cutoff:
        entries.pop(0)
    if len(entries) >= max_count:
        return False
    entries.append(now)
    return True


def rate_limit_bug_report(client_ip: str) -> bool:
    """Allow if under bug-report limit (per IP)."""
    return rate_limit_check(
        f"bug_report:{client_ip}",
        window_seconds=3600,
        max_count=cfg.RATE_LIMIT_BUG_REPORT_PER_HOUR,
    )


def rate_limit_request_code(client_ip: str, email: str) -> bool:
    """Allow if under request-code limit (per IP and per email)."""
    if not rate_limit_check(
        f"request_code_ip:{client_ip}",
        window_seconds=900,
        max_count=cfg.RATE_LIMIT_REQUEST_CODE_PER_15MIN,
    ):
        return False
    email_key = email.strip().lower() if email else ""
    if not email_key:
        return True
    return rate_limit_check(
        f"request_code_email:{email_key}",
        window_seconds=900,
        max_count=cfg.RATE_LIMIT_REQUEST_CODE_PER_15MIN,
    )


def rate_limit_verify_code(client_ip: str) -> bool:
    """Allow if under verify-code limit (per IP)."""
    return rate_limit_check(
        f"verify_code:{client_ip}",
        window_seconds=900,
        max_count=cfg.RATE_LIMIT_VERIFY_CODE_PER_15MIN,
    )


def validate_page_url(url: str | None) -> str | None:
    """Return sanitized page_url if valid (http/https, same-site optional); else None."""
    if not url or not url.strip():
        return None
    u = url.strip()
    if len(u) > 2048:
        return None
    lower = u.lower()
    if not (lower.startswith("http://") or lower.startswith("https://")):
        return None
    # Reject javascript:, data:, etc. (already blocked by http/https check)
    return u
