"""Smoke tests: public-facing GET pages must not return 5xx or error pages.

Ensures that the set of public URLs (home, content, advocacy, updates, etc.)
always return 200 or 302—never 500 or an error response. Add new public routes
to PUBLIC_GET_PAGES when they are introduced.

Single source for "what is public" is aligned with home._SITEMAP_PATHS plus
other public GET endpoints (poll, report-bug, health, favicon, sitemap, robots).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Allowed statuses for "success" (no error page).
ALLOWED_STATUSES = (200, 302)


def _make_client(**env_overrides: str) -> TestClient:
    """Build TestClient against the real app with optional env overrides."""
    with patch.dict(os.environ, env_overrides, clear=False):
        import importlib

        import ilga_graph.config as _cfg_mod
        import ilga_graph.main as _main_mod

        importlib.reload(_cfg_mod)
        importlib.reload(_main_mod)
        return TestClient(_main_mod.app, raise_server_exceptions=False)


# Public GET paths that must not return 5xx or an error page.
# 302 is allowed (e.g. /advocacy -> /advocacy/). Order matches sitemap + extras.
PUBLIC_GET_PAGES = [
    "/",
    "/advocacy",
    "/advocacy/",
    "/intelligence",
    "/intelligence/",
    "/explore",
    "/the-issue",
    "/legislator-brief",
    "/fact-sheet",
    "/coalition",
    "/timeline",
    "/glossary",
    "/privacy",
    "/terms",
    "/updates",
    "/poll",
    "/report-bug",
    "/favicon.ico",
    "/sitemap.xml",
    "/robots.txt",
    "/health",
]


@pytest.fixture
def client() -> TestClient:
    """TestClient with dev profile so public pages run without prod requirements."""
    return _make_client(ILGA_PROFILE="dev", ILGA_API_KEY="")


@pytest.mark.parametrize("path", PUBLIC_GET_PAGES)
def test_public_get_returns_success_not_error_page(client: TestClient, path: str) -> None:
    """Each public GET URL must return 200 or 302—never 500 or an error page."""
    headers = {"Accept": "text/html,application/xml,image/*,*/*"}
    resp = client.get(path, headers=headers)
    assert resp.status_code in ALLOWED_STATUSES, (
        f"GET {path} returned {resp.status_code}; expected one of {ALLOWED_STATUSES}. "
        "Public pages must not return error responses (4xx/5xx)."
    )
