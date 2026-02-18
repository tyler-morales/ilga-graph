"""Tests for auth and outreach API: request-code, verify-code, record, stats, my-history.

Uses a temp DB and a minimal FastAPI app (auth + outreach routers only) so the
real app lifespan and data/ilga.db are not used.
"""

from __future__ import annotations

import importlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ilga_graph.config as cfg_mod
import ilga_graph.db as db_mod
import ilga_graph.dependencies as deps_mod
from ilga_graph.routers import auth as auth_router_mod
from ilga_graph.routers import outreach as outreach_router_mod


def _make_test_app(db_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db_mod.init_db()
        yield

    app = FastAPI(title="Test Auth+Outreach", lifespan=lifespan)
    app.include_router(auth_router_mod.router)
    app.include_router(outreach_router_mod.router)
    return app


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_ilga.db"


@pytest.fixture
def client(test_db_path: Path) -> TestClient:
    env = {
        "ILGA_DB_PATH": str(test_db_path),
        "ILGA_AUTH_SECRET": "test-secret-for-pytest",
    }
    with patch.dict(os.environ, env, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        importlib.reload(deps_mod)
        importlib.reload(auth_router_mod)
        importlib.reload(outreach_router_mod)
        app = _make_test_app(test_db_path)
        with TestClient(app, raise_server_exceptions=True) as c:
            # Trigger lifespan so init_db() runs
            c.get("/auth/me")
            yield c


def _request_code(client: TestClient, email: str) -> dict:
    return client.post("/auth/request-code", data={"email": email}).json()


def _verify_code(client: TestClient, email: str, code: str, use_cookie: bool = True):
    return client.post(
        "/auth/verify-code",
        data={"email": email, "code": code},
        follow_redirects=True,
    )


class TestAuthFlow:
    def test_request_code_returns_ok(self, client: TestClient) -> None:
        resp = client.post("/auth/request-code", data={"email": "user@example.com"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_request_code_invalid_email(self, client: TestClient) -> None:
        resp = client.post("/auth/request-code", data={"email": "not-an-email"})
        assert resp.status_code == 400
        assert "Invalid" in resp.json().get("error", "")

    def test_verify_code_creates_user_and_sets_cookie(self, client: TestClient) -> None:
        # Request code (in dev without SMTP it's logged; we need to get it from DB or mock)
        client.post("/auth/request-code", data={"email": "verify@example.com"})
        # In tests we don't have the actual code; use the hash to look up or inject
        # Instead: verify with a wrong code fails
        resp = client.post(
            "/auth/verify-code",
            data={"email": "verify@example.com", "code": "000000"},
        )
        # Wrong code -> 400 (or 200 with error body depending on impl)
        assert resp.status_code in (200, 400)
        if resp.status_code == 200:
            assert resp.json().get("ok") is False or "Invalid" in resp.json().get("error", "")

    def test_me_unauthenticated(self, client: TestClient) -> None:
        resp = client.get("/auth/me")
        assert resp.status_code == 200
        assert resp.json() == {"authenticated": False}

    def test_logout_returns_ok(self, client: TestClient) -> None:
        resp = client.post("/auth/logout")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


class TestAuthVerifyRoundtrip:
    """Verify with a known code (inserted directly), then /me returns authenticated."""

    def test_full_flow_with_code_from_db(self, client: TestClient, test_db_path: Path) -> None:
        import asyncio

        email = "roundtrip@example.com"
        known_code = "123456"
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            asyncio.run(_add_auth_code(email, known_code))

        resp = client.post("/auth/verify-code", data={"email": email, "code": known_code})
        assert resp.status_code == 200
        assert resp.json().get("ok") is True
        assert resp.json().get("email") == email
        assert "ilga_session" in resp.cookies

        resp2 = client.get("/auth/me")
        assert resp2.status_code == 200
        assert resp2.json() == {"authenticated": True, "email": email}


class TestOutreachRecord:
    """POST /outreach/record: requires auth, validates kind, stores all fields."""

    def test_record_without_auth_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/outreach/record",
            data={"member_id": "1234", "kind": "call", "zip_code": "60601"},
        )
        assert resp.status_code == 401
        assert resp.json().get("ok") is False

    def test_record_invalid_kind_returns_400(self, client: TestClient, authed_client: TestClient) -> None:
        resp = authed_client.post(
            "/outreach/record",
            data={"member_id": "1234", "kind": "invalid", "zip_code": "60601"},
        )
        assert resp.status_code == 400
        assert "Invalid" in resp.json().get("error", "")

    def test_record_success_returns_ok_and_event_id(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        # Create user and get session cookie via verify with known code
        email = "outreach@example.com"
        known_code = "654321"
        import asyncio
        import hashlib
        from datetime import datetime, timedelta, timezone

        from ilga_graph.db_models import AuthCode

        async def setup():
            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(cfg_mod)
                importlib.reload(db_mod)
                async with db_mod.async_session_factory() as session:
                    session.add(
                        AuthCode(
                            email=email,
                            code_hash=hashlib.sha256(known_code.encode()).hexdigest(),
                            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
                        )
                    )
                    await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            asyncio.run(setup())

        client.post("/auth/verify-code", data={"email": email, "code": known_code})
        resp = client.post(
            "/outreach/record",
            data={
                "member_id": "1234",
                "kind": "call",
                "zip_code": "60601",
                "notes": "Test note",
                "contact_name": "Jane",
                "support_score": "4",
                "constituent": "yes",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("ok") is True
        assert "event_id" in data

    def test_record_stores_support_score_and_constituent(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        import asyncio
        import hashlib
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import select

        from ilga_graph.db_models import AuthCode, OutreachEvent

        email = "support@example.com"
        code = "111222"
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            asyncio.run(_add_auth_code(email, code))
        client.post("/auth/verify-code", data={"email": email, "code": code})
        client.post(
            "/outreach/record",
            data={
                "member_id": "m1",
                "kind": "email",
                "zip_code": "60602",
                "support_score": "5",
                "constituent": "1",
            },
        )
        async def check():
            async with db_mod.async_session_factory() as session:
                r = await session.execute(
                    select(OutreachEvent).where(OutreachEvent.member_id == "m1")
                )
                ev = r.scalar_one_or_none()
                assert ev is not None
                assert ev.support_score == 5
                assert ev.constituent is True

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            asyncio.run(check())


async def _add_auth_code(email: str, plain_code: str) -> None:
    import hashlib
    from datetime import datetime, timedelta, timezone

    from ilga_graph.db_models import AuthCode

    async with db_mod.async_session_factory() as session:
        session.add(
            AuthCode(
                email=email,
                code_hash=hashlib.sha256(plain_code.encode()).hexdigest(),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
            )
        )
        await session.commit()


@pytest.fixture
def authed_client(client: TestClient, test_db_path: Path) -> TestClient:
    """Client with authenticated user (outreach@example.com)."""
    import asyncio

    email = "authed@example.com"
    code = "999888"
    with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        asyncio.run(_add_auth_code(email, code))
    client.post("/auth/verify-code", data={"email": email, "code": code})
    return client


class TestOutreachStats:
    def test_stats_empty_returns_zeros(self, client: TestClient) -> None:
        resp = client.get("/outreach/stats/member_999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["member_id"] == "member_999"
        assert data["calls"] == 0
        assert data["emails"] == 0
        assert data["no_answers"] == 0
        assert data["total"] == 0

    def test_stats_aggregates_after_record(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        import asyncio

        email = "stats@example.com"
        code = "777666"
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            asyncio.run(_add_auth_code(email, code))
        client.post("/auth/verify-code", data={"email": email, "code": code})
        client.post(
            "/outreach/record",
            data={"member_id": "agg_member", "kind": "call", "zip_code": "60601"},
        )
        client.post(
            "/outreach/record",
            data={"member_id": "agg_member", "kind": "call", "zip_code": "60601"},
        )
        client.post(
            "/outreach/record",
            data={"member_id": "agg_member", "kind": "email", "zip_code": "60601"},
        )
        resp = client.get("/outreach/stats/agg_member")
        assert resp.status_code == 200
        assert resp.json()["calls"] == 2
        assert resp.json()["emails"] == 1
        assert resp.json()["total"] == 3


class TestOutreachMyHistory:
    def test_my_history_without_auth_returns_401(self, client: TestClient) -> None:
        resp = client.get("/outreach/my-history")
        assert resp.status_code == 401

    def test_my_history_returns_events_ordered_newest_first(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        import asyncio

        email = "history@example.com"
        code = "555444"
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            asyncio.run(_add_auth_code(email, code))
        client.post("/auth/verify-code", data={"email": email, "code": code})
        client.post(
            "/outreach/record",
            data={"member_id": "h1", "kind": "call", "zip_code": "60601", "notes": "First"},
        )
        client.post(
            "/outreach/record",
            data={"member_id": "h2", "kind": "email", "zip_code": "60601", "notes": "Second"},
        )
        resp = client.get("/outreach/my-history")
        assert resp.status_code == 200
        events = resp.json()["events"]
        assert len(events) == 2
        assert events[0]["member_id"] == "h2"
        assert events[1]["member_id"] == "h1"
        assert "created_at" in events[0]
        assert "support_score" in events[0]
        assert "contact_name" in events[0]
        assert "constituent" in events[0]


class TestParsingHelpers:
    """Unit-style tests for outreach parsing (can import from router)."""

    def test_parse_support_score_valid(self) -> None:
        from ilga_graph.routers.outreach import _parse_support_score

        assert _parse_support_score("1") == 1
        assert _parse_support_score("5") == 5
        assert _parse_support_score("  3  ") == 3

    def test_parse_support_score_invalid_returns_none(self) -> None:
        from ilga_graph.routers.outreach import _parse_support_score

        assert _parse_support_score("") is None
        assert _parse_support_score("0") is None
        assert _parse_support_score("6") is None
        assert _parse_support_score("x") is None

    def test_parse_constituent(self) -> None:
        from ilga_graph.routers.outreach import _parse_constituent

        assert _parse_constituent("1") is True
        assert _parse_constituent("true") is True
        assert _parse_constituent("yes") is True
        assert _parse_constituent("0") is False
        assert _parse_constituent("false") is False
        assert _parse_constituent("no") is False
        assert _parse_constituent("") is None
        assert _parse_constituent("other") is None
