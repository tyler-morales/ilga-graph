"""Tests for /intelligence/activity-calendar (Activity by Day tab: heatmap and breakdown)."""

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


def test_activity_calendar_returns_200(client: TestClient) -> None:
    """GET /intelligence/activity-calendar returns 200."""
    resp = client.get("/intelligence/activity-calendar", headers={"Accept": "text/html"})
    assert resp.status_code == 200


def test_activity_calendar_response_contains_expected_structure(client: TestClient) -> None:
    """Response contains Activity by Day heading and either heatmap/table or empty state."""
    resp = client.get("/intelligence/activity-calendar", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "Activity by Day" in body
    # Either heatmap/table or empty/error message
    assert (
        "activity-calendar-heatmap" in body
        or "activity-days-table" in body
        or "intel-empty" in body
        or "No Data Available" in body
        or "No Action Data" in body
    )
    # When heatmap is present, month timeline row is present for temporal orientation
    if "activity-calendar-heatmap" in body:
        assert "activity-calendar-month-row" in body


def test_activity_calendar_day_invalid_date_returns_400(client: TestClient) -> None:
    """GET /intelligence/activity-calendar/day with invalid date returns 400."""
    resp = client.get("/intelligence/activity-calendar/day", params={"date": "not-a-date"})
    assert resp.status_code == 400


def test_activity_calendar_day_valid_date_returns_json(client: TestClient) -> None:
    """GET .../activity-calendar/day with valid date returns 200 and JSON with expected keys."""
    resp = client.get(
        "/intelligence/activity-calendar/day",
        params={"date": "2025-01-15"},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["date"] == "2025-01-15"
    assert "total_actions" in data
    assert "unique_bills" in data
    assert "by_chamber" in data
    assert "is_session_day" in data
    assert "actions" in data
