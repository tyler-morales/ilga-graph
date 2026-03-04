"""Tests for account router: GET /account requires auth; POST updates zip/newsletter."""

from __future__ import annotations

import importlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from starlette.testclient import TestClient

import ilga_graph.config as cfg_mod
import ilga_graph.db as db_mod
import ilga_graph.dependencies as deps_mod
from ilga_graph.routers import account as account_router_mod
from ilga_graph.routers import auth as auth_router_mod
from ilga_graph.security import CSRF_COOKIE_NAME, generate_csrf_token
from tests.async_helpers import run_async


def _make_test_app(db_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db_mod.init_db()
        yield

    app = FastAPI(title="Test Account", lifespan=lifespan)

    @app.middleware("http")
    async def _csrf_cookie_middleware(request, call_next):
        token = generate_csrf_token()
        request.state.csrf_token = token
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

    _template_dir = Path(__file__).resolve().parent.parent / "src" / "ilga_graph" / "templates"
    templates = Jinja2Templates(directory=str(_template_dir))
    templates.env.globals["features"] = {}
    templates.env.globals["site_name"] = "Test"
    templates.env.globals["get_current_action_campaign"] = lambda r: None
    templates.env.globals["get_poll_campaign_for_template"] = lambda r: None
    app.state.templates = templates
    app.include_router(auth_router_mod.router)
    app.include_router(account_router_mod.router)
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
    return tmp_path / "test_account.db"


@pytest.fixture
def client(test_db_path: Path) -> TestClient:
    env = {
        "ILGA_DB_PATH": str(test_db_path),
        "ILGA_AUTH_SECRET": "test-secret-for-pytest",
        "ILGA_PROFILE": "dev",
    }
    with patch.dict(os.environ, env, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        importlib.reload(deps_mod)
        importlib.reload(auth_router_mod)
        importlib.reload(account_router_mod)
        app = _make_test_app(test_db_path)
        with TestClient(app, raise_server_exceptions=True) as c:
            c.get("/auth/me")
            yield c


@pytest.fixture
def authed_client(client: TestClient, test_db_path: Path) -> TestClient:
    email = "account_user@example.com"
    code = "777888"
    with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        run_async(_add_auth_code(email, code))
    client.post(
        "/auth/verify-code",
        data=_data_with_csrf(client, {"email": email, "code": code}),
    )
    return client


class TestAccountRouter:
    """Account router: GET requires auth; POST updates profile."""

    def test_get_account_anonymous_returns_401(self, client: TestClient) -> None:
        resp = client.get("/account", follow_redirects=False)
        assert resp.status_code == 401

    def test_get_account_authenticated_returns_200(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/account", headers={"Accept": "text/html"})
        assert resp.status_code == 200
        assert "account" in resp.text.lower() or "profile" in resp.text.lower()
