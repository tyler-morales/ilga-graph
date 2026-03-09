"""Tests for /intelligence/productive-days (most productive days for bills)."""

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


def test_productive_days_html_returns_200(client: TestClient) -> None:
    """GET /intelligence/productive-days returns 200 and page title or empty state."""
    resp = client.get("/intelligence/productive-days", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "Most Productive Days" in body or "Productive Days" in body
    # Either we have a table or an empty/error message
    assert "intel-nav" in body


def test_productive_days_json_returns_structure(client: TestClient) -> None:
    """GET ...?format=json returns JSON with session_label, productive_days, error."""
    resp = client.get("/intelligence/productive-days", params={"format": "json"})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_label" in data
    assert "productive_days" in data
    assert isinstance(data["productive_days"], list)
    assert "error" in data
    # If no parquet, error is set; if parquet exists, rows may be present
    for row in data["productive_days"]:
        assert "date" in row
        assert "total_actions" in row
        assert "unique_bills" in row
        assert "by_chamber" in row
        assert "House" in row["by_chamber"]
        assert "Senate" in row["by_chamber"]
