"""Tests for lobbyist / money-intel email signup (lead capture)."""

from __future__ import annotations

import importlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

import ilga_graph.config as cfg_mod
import ilga_graph.db as db_mod
import ilga_graph.dependencies as deps_mod
from ilga_graph.db_models import MoneyIntelLead
from ilga_graph.money_leads import (
    MONEY_LEAD_ROLES,
    normalize_email,
    normalize_optional_text,
    normalize_roles,
)
from ilga_graph.routers import admin as admin_router_mod
from ilga_graph.routers import auth as auth_router_mod
from ilga_graph.routers import money as money_router_mod
from ilga_graph.security import CSRF_COOKIE_NAME, generate_csrf_token
from tests.async_helpers import run_async

_ADMIN_EMAIL = "admin@example.com"


def _make_test_app(db_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db_mod.init_db()
        yield

    app = FastAPI(title="Test Money Signup", lifespan=lifespan)

    @app.middleware("http")
    async def _csrf_cookie_middleware(request, call_next):
        token = generate_csrf_token()
        request.state.csrf_token = token
        request.state.user = getattr(request.state, "user", None)
        response = await call_next(request)
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=token,
            max_age=3600,
            path="/",
            httponly=False,
            samesite="strict",
            secure=False,
        )
        return response

    app.include_router(auth_router_mod.router)
    app.include_router(money_router_mod.router, prefix="/intelligence")
    app.include_router(admin_router_mod.router)
    return app


def _data_with_csrf(client: TestClient, data: dict) -> dict:
    out = dict(data)
    out.setdefault("csrf_token", client.cookies.get(CSRF_COOKIE_NAME, ""))
    return out


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
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_money_signup.db"


@pytest.fixture
def client(test_db_path: Path) -> TestClient:
    env = {
        "ILGA_DB_PATH": str(test_db_path),
        "ILGA_AUTH_SECRET": "test-secret-for-pytest",
        "ILGA_PROFILE": "dev",
        "ILGA_ADMIN_EMAILS": _ADMIN_EMAIL,
        "ILGA_RATE_LIMIT_SUBSCRIBE_EMAIL_PER_HOUR": "100",
    }
    with patch.dict(os.environ, env, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        importlib.reload(deps_mod)
        importlib.reload(auth_router_mod)
        importlib.reload(admin_router_mod)
        importlib.reload(money_router_mod)
        app = _make_test_app(test_db_path)
        with TestClient(app, raise_server_exceptions=True) as c:
            c.get("/auth/me")
            yield c


@pytest.fixture
def admin_client(client: TestClient, test_db_path: Path) -> TestClient:
    code = "111222"
    with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        run_async(_add_auth_code(_ADMIN_EMAIL, code))
    client.post(
        "/auth/verify-code",
        data=_data_with_csrf(client, {"email": _ADMIN_EMAIL, "code": code}),
    )
    return client


class TestNormalizeEmail:
    def test_success_strips_and_lowercases(self) -> None:
        assert normalize_email("  Ada@Example.COM ") == "ada@example.com"

    def test_failure_rejects_empty_and_invalid(self) -> None:
        assert normalize_email("") is None
        assert normalize_email("not-an-email") is None
        assert normalize_email("a" * 321) is None


class TestNormalizeOptionalText:
    def test_success_trims(self) -> None:
        assert normalize_optional_text("  Acme LLP  ", 200) == "Acme LLP"

    def test_failure_empty_or_too_long(self) -> None:
        assert normalize_optional_text("   ", 200) is None
        assert normalize_optional_text("x" * 201, 200) is None


class TestNormalizeRoles:
    def test_success_allowlisted_and_deduped(self) -> None:
        assert normalize_roles(["nonprofit", "lobbyist", "lobbyist"]) == "lobbyist,nonprofit"
        assert set(MONEY_LEAD_ROLES) == {"lobbyist", "lawyer", "nonprofit"}

    def test_failure_unknown_or_empty(self) -> None:
        assert normalize_roles([]) is None
        assert normalize_roles(["influencer"]) is None


class TestMoneyPages:
    def test_money_landing_returns_200_with_form(self, client: TestClient) -> None:
        resp = client.get("/intelligence/money", headers={"Accept": "text/html"})
        assert resp.status_code == 200
        body = resp.text
        assert "Follow the money" in body or "follow-the-money" in body.lower()
        assert 'name="email"' in body
        assert "lobbyist" in body.lower()
        assert "Moneyball" in body
        assert "SOS" not in body
        assert "expenditure" not in body.lower()
        assert "/intelligence/member/3268" in body
        assert "SB0341" in body

    def test_signup_page_returns_200_with_form(self, client: TestClient) -> None:
        resp = client.get("/intelligence/money/signup", headers={"Accept": "text/html"})
        assert resp.status_code == 200
        assert 'name="email"' in resp.text
        assert 'id="money-signup-wrap"' in resp.text


class TestMoneySignupPost:
    def test_success_creates_lead(self, client: TestClient, test_db_path: Path) -> None:
        resp = client.post(
            "/intelligence/money/signup",
            data=_data_with_csrf(
                client,
                {
                    "email": "lobbyist@firm.com",
                    "name": "Pat Lobbyist",
                    "org": "Example Firm",
                    "role": ["lobbyist", "lawyer"],
                },
            ),
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/intelligence/money/signup?status=ok"

        async def _check() -> None:
            async with db_mod.async_session_factory() as session:
                result = await session.execute(
                    select(MoneyIntelLead).where(MoneyIntelLead.email == "lobbyist@firm.com")
                )
                lead = result.scalar_one()
                assert lead.name == "Pat Lobbyist"
                assert lead.org == "Example Firm"
                assert lead.role == "lawyer,lobbyist"

        run_async(_check())

    def test_success_htmx_returns_status_fragment(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        resp = client.post(
            "/intelligence/money/signup",
            data=_data_with_csrf(client, {"email": "htmx@firm.com"}),
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 200
        assert "You're on the list" in resp.text
        assert 'role="status"' in resp.text

    def test_already_subscribed_distinct_state(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        data = _data_with_csrf(client, {"email": "repeat@firm.com", "name": "First"})
        first = client.post("/intelligence/money/signup", data=data, follow_redirects=False)
        assert first.status_code == 303
        assert "status=ok" in first.headers.get("location", "")

        again = client.post(
            "/intelligence/money/signup",
            data=_data_with_csrf(client, {"email": "repeat@firm.com", "org": "Later LLP"}),
            follow_redirects=False,
        )
        assert again.status_code == 303
        assert again.headers.get("location") == "/intelligence/money/signup?status=already"

        htmx = client.post(
            "/intelligence/money/signup",
            data=_data_with_csrf(client, {"email": "repeat@firm.com"}),
            headers={"HX-Request": "true"},
        )
        assert htmx.status_code == 200
        assert "already" in htmx.text.lower()

        async def _check() -> None:
            async with db_mod.async_session_factory() as session:
                result = await session.execute(
                    select(MoneyIntelLead).where(MoneyIntelLead.email == "repeat@firm.com")
                )
                leads = list(result.scalars().all())
                assert len(leads) == 1
                assert leads[0].org == "Later LLP"

        run_async(_check())

    def test_failure_invalid_email(self, client: TestClient) -> None:
        resp = client.post(
            "/intelligence/money/signup",
            data=_data_with_csrf(client, {"email": "not-an-email"}),
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/intelligence/money/signup?status=invalid"

        htmx = client.post(
            "/intelligence/money/signup",
            data=_data_with_csrf(client, {"email": "nope"}),
            headers={"HX-Request": "true"},
        )
        assert htmx.status_code == 400
        assert "valid email" in htmx.text.lower()
        assert 'role="alert"' in htmx.text

    def test_failure_csrf(self, client: TestClient) -> None:
        resp = client.post(
            "/intelligence/money/signup",
            data={"email": "ok@firm.com", "csrf_token": "bad"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "status=csrf" in resp.headers.get("location", "")

        htmx = client.post(
            "/intelligence/money/signup",
            data={"email": "ok@firm.com", "csrf_token": "bad"},
            headers={"HX-Request": "true"},
        )
        assert htmx.status_code == 403
        assert "security" in htmx.text.lower()

    def test_failure_rate_limit(self, client: TestClient) -> None:
        with patch(
            "ilga_graph.routers.money.rate_limit_money_lead",
            return_value=False,
        ):
            resp = client.post(
                "/intelligence/money/signup",
                data=_data_with_csrf(client, {"email": "rate@firm.com"}),
                follow_redirects=False,
            )
            assert resp.status_code == 303
            assert "status=rate" in resp.headers.get("location", "")

            htmx = client.post(
                "/intelligence/money/signup",
                data=_data_with_csrf(client, {"email": "rate@firm.com"}),
                headers={"HX-Request": "true"},
            )
            assert htmx.status_code == 429
            assert "too many" in htmx.text.lower()


class TestMoneyLeadExport:
    def test_csv_requires_admin(self, client: TestClient) -> None:
        resp = client.get("/admin/money-leads.csv", follow_redirects=False)
        assert resp.status_code in (301, 302, 401)

    def test_csv_success_lists_leads(self, admin_client: TestClient, test_db_path: Path) -> None:
        admin_client.post(
            "/intelligence/money/signup",
            data=_data_with_csrf(
                admin_client,
                {"email": "export@firm.com", "name": "Export Me", "role": ["nonprofit"]},
            ),
            follow_redirects=False,
        )
        resp = admin_client.get("/admin/money-leads.csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")
        assert "export@firm.com" in resp.text
        assert "Export Me" in resp.text
        assert "nonprofit" in resp.text

    def test_admin_page_lists_leads(self, admin_client: TestClient) -> None:
        admin_client.post(
            "/intelligence/money/signup",
            data=_data_with_csrf(admin_client, {"email": "listed@firm.com"}),
            follow_redirects=False,
        )
        resp = admin_client.get("/admin/money-leads")
        assert resp.status_code == 200
        assert "listed@firm.com" in resp.text
        assert "/admin/money-leads.csv" in resp.text
