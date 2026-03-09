"""Tests for /intelligence/legislator-twitter (Legislator Twitter tab)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _make_client(**env_overrides: str) -> TestClient:
    with patch.dict(os.environ, env_overrides, clear=False):
        import importlib

        import ilga_graph.config as _cfg_mod
        import ilga_graph.main as _main_mod

        importlib.reload(_cfg_mod)
        importlib.reload(_main_mod)
        return TestClient(_main_mod.app, raise_server_exceptions=False)


@pytest.fixture
def client() -> TestClient:
    return _make_client(ILGA_PROFILE="dev", ILGA_API_KEY="")


def test_legislator_twitter_returns_200(client: TestClient) -> None:
    """GET /intelligence/legislator-twitter returns 200."""
    resp = client.get("/intelligence/legislator-twitter", headers={"Accept": "text/html"})
    assert resp.status_code == 200


def test_legislator_twitter_response_structure(client: TestClient) -> None:
    """Response contains Legislator Twitter section and either empty state or table."""
    resp = client.get("/intelligence/legislator-twitter", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "Legislator Twitter" in body or "Twitter / X" in body
    assert "legislator_twitter_handles.json" in body or "legislator-twitter-table" in body
