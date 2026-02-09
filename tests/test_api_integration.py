"""Integration tests for the FastAPI / GraphQL API.

Uses FastAPI's TestClient (backed by httpx) to exercise the health endpoint,
CORS headers, API key auth, and GraphQL queries against real schema resolution.

These tests do NOT start the lifespan (no scraping / ETL).  They operate on
an empty AppState where members/bills/committees lists are all [].
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_client(**env_overrides: str) -> TestClient:
    """Build a fresh TestClient, optionally with env var overrides.

    We reimport ``config`` then ``main`` each time to pick up changed
    env vars for API_KEY, CORS_ORIGINS, etc.  We skip the lifespan so
    there is no real startup.
    """
    with patch.dict(os.environ, env_overrides):
        import importlib

        import ilga_graph.config as _cfg_mod
        import ilga_graph.main as _main_mod

        importlib.reload(_cfg_mod)
        importlib.reload(_main_mod)
        return TestClient(_main_mod.app, raise_server_exceptions=False)


@pytest.fixture()
def client() -> TestClient:
    """TestClient with default env (no API key required)."""
    return _make_client(ILGA_API_KEY="")


# ── Health endpoint ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "members" in body
        assert "bills" in body
        assert "committees" in body
        assert "vote_events" in body

    def test_ready_is_false_when_empty(self, client: TestClient) -> None:
        resp = client.get("/health")
        # With no lifespan, state is empty → ready = False
        assert resp.json()["ready"] is False


# ── CORS headers ──────────────────────────────────────────────────────────────


class TestCORS:
    def test_cors_headers_present(self, client: TestClient) -> None:
        resp = client.options(
            "/graphql",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # CORSMiddleware should add access-control-allow-origin
        assert "access-control-allow-origin" in resp.headers


# ── API key authentication ────────────────────────────────────────────────────


class TestAPIKeyAuth:
    def test_no_key_required_when_env_empty(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_graphql_accessible_without_key(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={"query": "{ __typename }"},
        )
        assert resp.status_code == 200

    def test_rejects_without_key(self) -> None:
        secured = _make_client(ILGA_API_KEY="secret-key-123")
        resp = secured.post(
            "/graphql",
            json={"query": "{ __typename }"},
        )
        assert resp.status_code == 401
        assert "Invalid or missing API key" in resp.json()["detail"]

    def test_accepts_correct_key(self) -> None:
        secured = _make_client(ILGA_API_KEY="secret-key-123")
        resp = secured.post(
            "/graphql",
            json={"query": "{ __typename }"},
            headers={"X-API-Key": "secret-key-123"},
        )
        assert resp.status_code == 200

    def test_rejects_wrong_key(self) -> None:
        secured = _make_client(ILGA_API_KEY="secret-key-123")
        resp = secured.post(
            "/graphql",
            json={"query": "{ __typename }"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_health_exempt_from_auth(self) -> None:
        secured = _make_client(ILGA_API_KEY="secret-key-123")
        resp = secured.get("/health")
        assert resp.status_code == 200


# ── GraphQL queries (against empty state) ─────────────────────────────────────


class TestGraphQLQueries:
    def test_members_returns_connection(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={
                "query": """
                {
                    members {
                        items { name }
                        pageInfo { totalCount hasNextPage hasPreviousPage }
                    }
                }
                """,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]["members"]
        assert data["items"] == []
        assert data["pageInfo"]["totalCount"] == 0

    def test_bills_returns_connection(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={
                "query": """
                {
                    bills {
                        items { billNumber }
                        pageInfo { totalCount }
                    }
                }
                """,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]["bills"]
        assert data["items"] == []

    def test_committees_returns_connection(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={
                "query": """
                {
                    committees {
                        items { code name }
                        pageInfo { totalCount }
                    }
                }
                """,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]["committees"]
        assert data["items"] == []

    def test_member_not_found(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={"query": '{ member(name: "Nobody") { name } }'},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["member"] is None

    def test_bill_not_found(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={"query": '{ bill(number: "XX0000") { billNumber } }'},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["bill"] is None

    def test_committee_not_found(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={"query": '{ committee(code: "XXXX") { code } }'},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["committee"] is None

    def test_all_vote_events_empty(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={
                "query": """
                {
                    allVoteEvents {
                        items { billNumber }
                        pageInfo { totalCount }
                    }
                }
                """,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]["allVoteEvents"]
        assert data["items"] == []

    def test_witness_slips_empty(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={
                "query": """
                {
                    witnessSlips(billNumber: "HB0001") {
                        items { name position }
                        pageInfo { totalCount }
                    }
                }
                """,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]["witnessSlips"]
        assert data["items"] == []

    def test_moneyball_leaderboard_empty(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={
                "query": """
                {
                    moneyballLeaderboard {
                        items { name }
                        pageInfo { totalCount }
                    }
                }
                """,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]["moneyballLeaderboard"]
        assert data["items"] == []

    def test_chamber_enum_accepted(self, client: TestClient) -> None:
        resp = client.post(
            "/graphql",
            json={
                "query": """
                {
                    members(chamber: HOUSE) {
                        pageInfo { totalCount }
                    }
                }
                """,
            },
        )
        assert resp.status_code == 200

    def test_invalid_date_handled_gracefully(self, client: TestClient) -> None:
        """Bad dateFrom should not crash -- just returns empty results."""
        resp = client.post(
            "/graphql",
            json={
                "query": """
                {
                    bills(dateFrom: "not-a-date") {
                        items { billNumber }
                        pageInfo { totalCount }
                    }
                }
                """,
            },
        )
        assert resp.status_code == 200
        # Should still return a valid connection (empty since no bills)
        data = resp.json()["data"]["bills"]
        assert data["items"] == []
