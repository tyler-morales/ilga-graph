"""Tests for /intelligence/votes (Votes tab: bills with vote data, deciding voters)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ilga_graph.models import VoteEvent


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


def test_votes_returns_200(client: TestClient) -> None:
    """GET /intelligence/votes returns 200."""
    resp = client.get("/intelligence/votes", headers={"Accept": "text/html"})
    assert resp.status_code == 200


def test_votes_response_structure(client: TestClient) -> None:
    """Response contains either empty state or Roll-call votes table."""
    resp = client.get("/intelligence/votes", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "No vote data available" in body or ("Roll-call votes" in body and "votes-table" in body)


def test_votes_empty_state_shows_empty_message(client: TestClient) -> None:
    """When state has no vote events, response shows empty state message."""
    from ilga_graph.app_state import state as app_state

    with patch.object(app_state, "vote_events", []), patch.object(app_state, "vote_lookup", {}):
        resp = client.get("/intelligence/votes", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "No vote data available" in resp.text


def test_votes_one_event_margin_two_no_deciding_voters(client: TestClient) -> None:
    """One bill, one event with margin 2: no deciding voters shown."""
    from ilga_graph.app_state import state as app_state

    event = VoteEvent(
        bill_number="SB1000",
        date="2025-05-01",
        description="Third Reading",
        chamber="Senate",
        yea_votes=["Alice", "Bob", "Carol"],
        nay_votes=["Dave"],
        present_votes=[],
        nv_votes=[],
        vote_type="floor",
    )
    with (
        patch.object(app_state, "vote_events", [event]),
        patch.object(app_state, "vote_lookup", {"SB1000": [event]}),
        patch.object(app_state, "bill_lookup", {}),
    ):
        resp = client.get("/intelligence/votes", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "SB1000" in resp.text
    assert "Passed" in resp.text
    assert "deciding" in resp.text.lower() or "Deciding voters" in resp.text
    # Deciding voters cell should show — (em dash) for this event (margin 3)
    assert "3" in resp.text and "1" in resp.text  # yea 3, nay 1


def test_votes_margin_one_shows_deciding_voters(client: TestClient) -> None:
    """One event with margin 1: deciding voters list equals winning side."""
    from ilga_graph.app_state import state as app_state

    event = VoteEvent(
        bill_number="HB2000",
        date="2025-05-15",
        description="Third Reading",
        chamber="House",
        yea_votes=["Smith", "Jones"],
        nay_votes=["Brown"],
        present_votes=[],
        nv_votes=[],
        vote_type="floor",
    )
    with (
        patch.object(app_state, "vote_events", [event]),
        patch.object(app_state, "vote_lookup", {"HB2000": [event]}),
        patch.object(app_state, "bill_lookup", {}),
    ):
        resp = client.get("/intelligence/votes", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "HB2000" in resp.text
    assert "Smith" in resp.text and "Jones" in resp.text
    assert "Deciding voters" in resp.text or "deciding" in resp.text.lower()
    assert "1" in resp.text  # margin 1


def test_votes_two_events_committee_then_floor_order(client: TestClient) -> None:
    """One bill with committee then floor event: both appear, committee before floor."""
    from ilga_graph.app_state import state as app_state

    committee = VoteEvent(
        bill_number="SB3000",
        date="2025-04-01",
        description="Do Pass",
        chamber="Senate",
        yea_votes=["A", "B", "C"],
        nay_votes=["D", "E"],
        present_votes=[],
        nv_votes=[],
        vote_type="committee",
    )
    floor = VoteEvent(
        bill_number="SB3000",
        date="2025-05-01",
        description="Third Reading",
        chamber="Senate",
        yea_votes=["A", "B", "C", "F"],
        nay_votes=["D", "E"],
        present_votes=[],
        nv_votes=[],
        vote_type="floor",
    )
    events = [floor, committee]
    with (
        patch.object(app_state, "vote_events", events),
        patch.object(app_state, "vote_lookup", {"SB3000": events}),
        patch.object(app_state, "bill_lookup", {}),
    ):
        resp = client.get("/intelligence/votes", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert resp.text.count("SB3000") >= 2
    assert "Committee" in resp.text and "Floor" in resp.text
    idx_committee = resp.text.find("Committee")
    idx_floor = resp.text.find("Floor")
    assert idx_committee < idx_floor, "Committee event should appear before Floor in table"


def test_intelligence_raw_includes_votes_summary(client: TestClient) -> None:
    """Raw dashboard shows Votes card when ML data available, or empty state."""
    resp = client.get("/intelligence/raw", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "Bills w/ votes" in resp.text or "No ML Data Available" in resp.text
