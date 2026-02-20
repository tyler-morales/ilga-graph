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
        "key": "zip_loading_animation",
        "env_var": "ILGA_FEATURE_ZIP_LOADING_ANIMATION",
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

# ── Auth + SMTP ──────────────────────────────────────────────────────────────
# Session cookie signing key (generate a random string for prod).
AUTH_SECRET: str = _env("ILGA_AUTH_SECRET", "dev-secret-change-me")
# Cookie max-age in seconds (default 30 days).
AUTH_COOKIE_MAX_AGE: int = int(_env("ILGA_AUTH_COOKIE_MAX_AGE", str(60 * 60 * 24 * 30)))
AUTH_COOKIE_NAME: str = "ilga_session"

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
