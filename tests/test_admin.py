"""Tests for admin router: privileged routes require admin auth; unauthed redirect to login."""

from __future__ import annotations

import importlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

import ilga_graph.config as cfg_mod
import ilga_graph.db as db_mod
import ilga_graph.dependencies as deps_mod
from ilga_graph.routers import admin as admin_router_mod
from ilga_graph.routers import auth as auth_router_mod
from ilga_graph.security import CSRF_COOKIE_NAME, generate_csrf_token
from tests.async_helpers import run_async

_ADMIN_EMAIL = "admin@example.com"


def _make_test_app(db_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db_mod.init_db()
        yield

    app = FastAPI(title="Test Admin", lifespan=lifespan)

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

    app.include_router(auth_router_mod.router)
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
    return tmp_path / "test_admin.db"


@pytest.fixture
def client(test_db_path: Path) -> TestClient:
    env = {
        "ILGA_DB_PATH": str(test_db_path),
        "ILGA_AUTH_SECRET": "test-secret-for-pytest",
        "ILGA_PROFILE": "dev",
        "ILGA_ADMIN_EMAILS": _ADMIN_EMAIL,
    }
    with patch.dict(os.environ, env, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        importlib.reload(deps_mod)
        importlib.reload(auth_router_mod)
        importlib.reload(admin_router_mod)
        app = _make_test_app(test_db_path)
        with TestClient(app, raise_server_exceptions=True) as c:
            c.get("/auth/me")
            yield c


@pytest.fixture
def admin_client(client: TestClient, test_db_path: Path) -> TestClient:
    """Client authenticated as admin user."""
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


class TestAdminAuth:
    """Admin routes require admin user; unauthed redirect to /admin/login."""

    def test_get_admin_unauth_returns_401_or_redirects_to_login(self, client: TestClient) -> None:
        resp = client.get("/admin", follow_redirects=False)
        # Minimal app: dependency raises 401; full app would redirect to /admin/login
        assert resp.status_code in (301, 302, 401)
        if resp.status_code in (301, 302):
            assert "admin/login" in resp.headers.get("location", "")

    def test_get_admin_login_returns_200(self, client: TestClient) -> None:
        resp = client.get("/admin/login")
        assert resp.status_code == 200
        assert "login" in resp.text.lower() or "admin" in resp.text.lower()

    def test_get_admin_dashboard_authed_returns_200(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/admin", follow_redirects=False)
        assert resp.status_code == 200
