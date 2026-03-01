"""Centralized configuration for the ILGA Graph application.

All settings live here so they can be overridden by environment variables or a
``.env`` file without touching source code.

**Profile system:** Set ``ILGA_PROFILE=dev`` (default) or ``ILGA_PROFILE=prod``
to get sensible defaults for each environment.  Any individual ``ILGA_*`` var
still overrides the profile value.

Usage::

    from ilga_graph.config import GA_ID, SESSION_ID

    url = f"...&GaId={GA_ID}&SessionId={SESSION_ID}"
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Load .env from current working directory (project root when running make dev / uvicorn)
load_dotenv()

LOGGER = logging.getLogger(__name__)

# ── Profile: one knob for the whole environment ──────────────────────────────
# "dev" = lightweight local mode, "prod" = production-ready defaults.
# Individual vars always override the profile.

PROFILE: str = os.getenv("ILGA_PROFILE", "dev").lower().strip()

_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "dev": {
        "ILGA_DEV_MODE": "1",
        "ILGA_SEED_MODE": "1",
        "ILGA_INCREMENTAL": "0",
        "ILGA_CORS_ORIGINS": "*",
        "ILGA_MEMBER_LIMIT": "0",
        "ILGA_DB_PATH": "data/ilga_dev.db",  # sandbox DB; mock outreach data when seeding
    },
    "prod": {
        "ILGA_DEV_MODE": "0",
        "ILGA_SEED_MODE": "0",
        "ILGA_INCREMENTAL": "0",
        "ILGA_CORS_ORIGINS": "",  # empty → must be explicitly set
        "ILGA_MEMBER_LIMIT": "0",
        "ILGA_DB_PATH": "data/ilga.db",  # live DB; real backlog only when seeding
    },
}

if PROFILE not in _PROFILE_DEFAULTS:
    LOGGER.warning("Unknown ILGA_PROFILE=%r, falling back to 'dev'.", PROFILE)
    PROFILE = "dev"

_defaults = _PROFILE_DEFAULTS[PROFILE]


def _env(key: str, fallback: str = "") -> str:
    """Read an env var, falling back to profile default then *fallback*."""
    return os.getenv(key, _defaults.get(key, fallback))


# ── ILGA session identifiers ─────────────────────────────────────────────────
# 104th General Assembly (2025-2026): GaId=18, SessionId=114.
GA_ID: int = int(_env("ILGA_GA_ID", "18"))
SESSION_ID: int = int(_env("ILGA_SESSION_ID", "114"))
# General Assembly number (used for direct PDF URL construction).
# Mapping: GA_NUMBER = GA_ID + 86  →  GAID 18 = 104th GA.
GA_NUMBER: int = GA_ID + 86

# ── Base URLs ────────────────────────────────────────────────────────────────
BASE_URL: str = _env("ILGA_BASE_URL", "https://www.ilga.gov/").rstrip("/") + "/"
# Public URL of this app (startup banner, logs). Set in production e.g. https://landofkei.org
APP_BASE_URL: str = _env("ILGA_APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
# Optional; docs site URL for startup banner when different from app (e.g. same host).
DOCS_BASE_URL: str = _env("ILGA_DOCS_BASE_URL", "").strip().rstrip("/")

# ── SEO & share (Open Graph, canonical, meta description) ─────────────────────
SITE_NAME: str = _env("ILGA_SITE_NAME", "The Land of Kei").strip() or "The Land of Kei"
META_DESCRIPTION: str = _env(
    "ILGA_META_DESCRIPTION",
    "Find your Illinois legislators and advocate with The Land of Kei for a statutory fix so "
    "highway-built Kei vehicles can be titled and registered. 625 ILCS 5/3-401(c-1).",
).strip()
# Optional absolute URL for share card image (1200×630). Unset → APP_BASE_URL/static/og-image.png.
_OG_IMAGE_OVERRIDE: str = _env("ILGA_OG_IMAGE_URL", "").strip()
OG_IMAGE_URL: str = (
    _OG_IMAGE_OVERRIDE if _OG_IMAGE_OVERRIDE else f"{APP_BASE_URL}/static/og-image.png"
)

# ── Beta banner (site-wide notice) ───────────────────────────────────────────
# When 1, base template shows beta banner. Report-a-bug link remains in footer only.
BETA_BANNER: bool = _env("ILGA_BETA_BANNER", "0") == "1"
# URL for "Report a bug" in footer and drawer nudges (e.g. Google Form, GitHub Issues, /report-bug).
BETA_BANNER_FEEDBACK_URL: str = _env("ILGA_BETA_BANNER_FEEDBACK_URL", "").strip()
# When someone submits the in-app form at /report-bug, an email is sent here if SMTP is configured.
# Banner link is unaffected; it goes to /report-bug (or FEEDBACK_URL if set).
BETA_BANNER_EMAIL: str = _env("ILGA_BETA_BANNER_EMAIL", "").strip()


def _beta_banner_report_url() -> str:
    """Report a bug link: external URL or in-app form at /report-bug."""
    if BETA_BANNER_FEEDBACK_URL:
        return BETA_BANNER_FEEDBACK_URL
    return "/report-bug"


# Resolved URL passed to templates.
BETA_BANNER_REPORT_URL: str = _beta_banner_report_url()


def _footer_last_updated_from_git() -> tuple[str, str] | None:
    """Return (human_date, iso_date) from last git commit, or None if unavailable.
    Used when ILGA_FOOTER_LAST_UPDATED / ILGA_FOOTER_LAST_UPDATED_ISO are not set,
    so dev and prod (deploy does git pull) show the last-commit date automatically.
    """
    try:
        cfg_dir = Path(__file__).resolve().parent
        root = cfg_dir
        for _ in range(10):
            if (root / ".git").exists():
                break
            parent = root.parent
            if parent == root:
                return None
            root = parent
        else:
            return None
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=format:%B %d, %Y"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode != 0 or not out.stdout:
            return None
        human = out.stdout.strip()
        out_iso = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        iso = out_iso.stdout.strip() if out_iso.returncode == 0 and out_iso.stdout else ""
        return (human, iso) if iso else None
    except Exception as e:
        LOGGER.debug("Footer date from git skipped: %s", e)
        return None


_FOOTER_DEFAULT_HUMAN = "February 24, 2026"
_FOOTER_DEFAULT_ISO = "2026-02-24"

_env_footer = os.getenv("ILGA_FOOTER_LAST_UPDATED")
_env_footer_iso = os.getenv("ILGA_FOOTER_LAST_UPDATED_ISO")
if _env_footer is None and _env_footer_iso is None:
    _git_dates = _footer_last_updated_from_git()
    if _git_dates:
        FOOTER_LAST_UPDATED, FOOTER_LAST_UPDATED_ISO = _git_dates
    else:
        FOOTER_LAST_UPDATED = _FOOTER_DEFAULT_HUMAN
        FOOTER_LAST_UPDATED_ISO = _FOOTER_DEFAULT_ISO
else:
    FOOTER_LAST_UPDATED = _env("ILGA_FOOTER_LAST_UPDATED", _FOOTER_DEFAULT_HUMAN).strip()
    FOOTER_LAST_UPDATED_ISO = _env("ILGA_FOOTER_LAST_UPDATED_ISO", _FOOTER_DEFAULT_ISO).strip()
# Bug report image uploads (optional). Empty = no uploads. Dir created on first report with image.
BUG_REPORT_UPLOAD_DIR: str = _env("ILGA_BUG_REPORT_UPLOAD_DIR", "data/bug_report_uploads").strip()
BUG_REPORT_MAX_IMAGE_BYTES: int = int(_env("ILGA_BUG_REPORT_MAX_IMAGE_MB", "5")) * 1024 * 1024
# Update image uploads: subdir under static (e.g. updates → /static/updates/). Created on first use.
UPDATE_IMAGE_UPLOAD_DIR: str = _env("ILGA_UPDATE_IMAGE_UPLOAD_DIR", "updates").strip()
UPDATE_MAX_IMAGE_BYTES: int = int(_env("ILGA_UPDATE_MAX_IMAGE_MB", "5")) * 1024 * 1024
# Community story uploads: subdir under static (e.g. uploads/stories).
STORY_IMAGE_UPLOAD_DIR: str = _env("ILGA_STORY_IMAGE_UPLOAD_DIR", "uploads/stories").strip()
STORY_MAX_IMAGE_BYTES: int = int(_env("ILGA_STORY_MAX_IMAGE_MB", "5")) * 1024 * 1024

# Rate limits (per key: IP or IP+email). In-memory; resets on process restart.
RATE_LIMIT_BUG_REPORT_PER_HOUR: int = int(_env("ILGA_RATE_LIMIT_BUG_REPORT_PER_HOUR", "10"))
RATE_LIMIT_REQUEST_CODE_PER_15MIN: int = int(_env("ILGA_RATE_LIMIT_REQUEST_CODE_PER_15MIN", "3"))
RATE_LIMIT_VERIFY_CODE_PER_15MIN: int = int(_env("ILGA_RATE_LIMIT_VERIFY_CODE_PER_15MIN", "10"))
RATE_LIMIT_SUBSCRIBE_EMAIL_PER_HOUR: int = int(
    _env("ILGA_RATE_LIMIT_SUBSCRIBE_EMAIL_PER_HOUR", "10")
)
RATE_LIMIT_KEI_STATUS_ANON_PER_HOUR: int = int(
    _env("ILGA_RATE_LIMIT_KEI_STATUS_ANON_PER_HOUR", "20")
)
RATE_LIMIT_STORY_SUBMIT_PER_HOUR: int = int(_env("ILGA_RATE_LIMIT_STORY_SUBMIT_PER_HOUR", "5"))
RATE_LIMIT_STATEMENT_SUBMIT_PER_HOUR: int = int(
    _env("ILGA_RATE_LIMIT_STATEMENT_SUBMIT_PER_HOUR", "5")
)

# Cloudflare Turnstile (optional). Free tier: 1M requests/month. When both keys are set,
# the bug report form shows the widget and server verifies the token.
# Dashboard: https://dash.cloudflare.com/?to=/:account/turnstile
TURNSTILE_SITE_KEY: str = _env("ILGA_TURNSTILE_SITE_KEY", "").strip()
TURNSTILE_SECRET_KEY: str = _env("ILGA_TURNSTILE_SECRET_KEY", "").strip()

# ── Analytics (Umami) ───────────────────────────────────────────────────────
# When set (e.g. in prod), base template injects the script. Get ID from Umami Cloud → Add website.
UMAMI_WEBSITE_ID: str = _env("ILGA_UMAMI_WEBSITE_ID", "").strip()
UMAMI_SCRIPT_URL: str = (
    _env(
        "ILGA_UMAMI_SCRIPT_URL",
        "https://cloud.umami.is/script.js",
    ).strip()
    or "https://cloud.umami.is/script.js"
)

# ── Directories ──────────────────────────────────────────────────────────────
# Dev uses cache/dev/ so it never touches the full scraped data in cache/.
_CACHE_BASE: Path = Path(_env("ILGA_CACHE_DIR", "cache"))
CACHE_DIR: Path = _CACHE_BASE / "dev" if PROFILE == "dev" else _CACHE_BASE
MOCK_DEV_DIR: Path = Path(_env("ILGA_MOCK_DIR", "mocks/dev"))

# ── Mode flags ───────────────────────────────────────────────────────────────
DEV_MODE: bool = _env("ILGA_DEV_MODE") == "1"
SEED_MODE: bool = _env("ILGA_SEED_MODE") == "1"
INCREMENTAL: bool = _env("ILGA_INCREMENTAL") == "1"
# When true, API startup only loads from cache (no scraping). Set for fast start.
LOAD_ONLY: bool = _env("ILGA_LOAD_ONLY") == "1"

# ── Feature flags (single source of truth; profile defaults, ILGA_FEATURE_* overrides) ─
# Each entry: key (JS/template), env_var, dev_default, prod_default, expose_to_client.
_FEATURE_REGISTRY: list[dict[str, str | bool]] = [
    {
        "key": "message_marquee",
        "env_var": "ILGA_FEATURE_MESSAGE_MARQUEE",
        "dev_default": "1",
        "prod_default": "0",
        "expose_to_client": True,
    },
    {
        "key": "images_marquee",
        "env_var": "ILGA_FEATURE_IMAGES_MARQUEE",
        "dev_default": "1",
        "prod_default": "0",
        "expose_to_client": True,
    },
]


def _feature_value(entry: dict[str, str | bool]) -> bool:
    """Resolve one flag: env override else profile default."""
    env_var = str(entry["env_var"])
    raw = os.getenv(env_var)
    if raw is not None:
        return raw.strip() == "1"
    key = "prod_default" if PROFILE == "prod" else "dev_default"
    default = str(entry.get(key, "0"))
    return default == "1"


def get_client_features() -> dict[str, bool]:
    """Return { key: value } for all flags with expose_to_client=True (for templates/JS)."""
    return {
        str(e["key"]): _feature_value(e)
        for e in _FEATURE_REGISTRY
        if e.get("expose_to_client") is True
    }


# ── Scrape / export limits ───────────────────────────────────────────────────
MEMBER_LIMIT: int = int(_env("ILGA_MEMBER_LIMIT", "0"))
TEST_MEMBER_URL: str = _env("ILGA_TEST_MEMBER_URL").strip()
TEST_MEMBER_CHAMBER: str = _env("ILGA_TEST_MEMBER_CHAMBER", "Senate").strip() or "Senate"

# ── Security / network ──────────────────────────────────────────────────────
CORS_ORIGINS: str = _env("ILGA_CORS_ORIGINS").strip()
API_KEY: str = _env("ILGA_API_KEY").strip()

# CSP: report-only by default; set ILGA_CSP_ENFORCE=1 to send enforcing header.
CSP_ENFORCE: bool = _env("ILGA_CSP_ENFORCE", "0") == "1"
# Optional endpoint for CSP violation reports (report-uri or report-to).
CSP_REPORT_URI: str = _env("ILGA_CSP_REPORT_URI", "").strip()
# HSTS: only set when site is fully served over HTTPS (e.g. behind TLS-terminating proxy).
HSTS_ENABLED: bool = _env("ILGA_HSTS_ENABLED", "0") == "1"

# ── Production guard: warn if CORS is wide-open or API_KEY is missing ────────
if PROFILE == "prod":
    if CORS_ORIGINS in ("*", ""):
        LOGGER.warning(
            "ILGA_PROFILE=prod but ILGA_CORS_ORIGINS=%r. "
            "Set it to your front-end origin(s) for security.",
            CORS_ORIGINS,
        )
    if not API_KEY:
        LOGGER.warning(
            "ILGA_PROFILE=prod but ILGA_API_KEY is empty. GraphQL endpoint is unprotected."
        )
    if APP_BASE_URL.strip().lower().startswith("http://"):
        LOGGER.warning(
            "ILGA_APP_BASE_URL is not HTTPS; set it to your public https:// URL in production "
            "for correct canonical, Open Graph, and cookie behavior."
        )

# ── Auth + SMTP ──────────────────────────────────────────────────────────────
# Session cookie signing key (generate a random string for prod).
AUTH_SECRET: str = _env("ILGA_AUTH_SECRET", "dev-secret-change-me")
# Cookie max-age in seconds (default 30 days).
AUTH_COOKIE_MAX_AGE: int = int(_env("ILGA_AUTH_COOKIE_MAX_AGE", str(60 * 60 * 24 * 30)))
AUTH_COOKIE_NAME: str = "ilga_session"
# Comma-separated emails allowed to access admin (e.g. compose/send updates).
ADMIN_EMAILS: list[str] = [
    e.strip().lower() for e in _env("ILGA_ADMIN_EMAILS", "").split(",") if e.strip()
]

# SMTP for sending auth codes.  When empty, codes are logged to console (dev).
SMTP_HOST: str = _env("ILGA_SMTP_HOST", "").strip()
SMTP_PORT: int = int(_env("ILGA_SMTP_PORT", "587"))
SMTP_USER: str = _env("ILGA_SMTP_USER", "").strip()
SMTP_PASS: str = _env("ILGA_SMTP_PASS", "").strip()
SMTP_FROM: str = _env("ILGA_SMTP_FROM", "").strip() or SMTP_USER
SMTP_USE_TLS: bool = _env("ILGA_SMTP_TLS", "1") == "1"

# ── Bill status URLs (votes + witness slips) ─────────────────────────────────
DEFAULT_BILL_STATUS_URLS: list[str] = [
    # Senate bills
    f"{BASE_URL}Legislation/BillStatus?DocNum=852&GAID={GA_ID}&DocTypeID=SB&LegId=158575&SessionID={SESSION_ID}",
    f"{BASE_URL}Legislation/BillStatus?DocNum=8&GAID={GA_ID}&DocTypeID=SB&LegId=157098&SessionID={SESSION_ID}",
    f"{BASE_URL}Legislation/BillStatus?DocNum=9&GAID={GA_ID}&DocTypeID=SB&LegId=157099&SessionID={SESSION_ID}",
    # House bills (HB0034 has high-volume slips)
    f"{BASE_URL}Legislation/BillStatus?DocNum=576&GAID={GA_ID}&DocTypeID=HB&LegId=156254&SessionID={SESSION_ID}",
    f"{BASE_URL}Legislation/BillStatus?DocNum=34&GAID={GA_ID}&DocTypeID=HB&LegId=155692&SessionID={SESSION_ID}",
]


def get_bill_status_urls() -> list[str]:
    """Return bill status URLs from env or defaults.

    Used by both the FastAPI lifespan and ``scripts/scrape.py`` so the same
    bills are scraped for vote events **and** witness slips.
    """
    custom = _env("ILGA_VOTE_BILL_URLS").strip()
    if custom:
        return [u.strip() for u in custom.split(",") if u.strip()]
    return list(DEFAULT_BILL_STATUS_URLS)
