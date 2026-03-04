"""Tests for updates router: public page, subscribe/unsubscribe, admin compose/send."""

from __future__ import annotations

import importlib
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient
from sqlalchemy import select

import ilga_graph.config as cfg_mod
import ilga_graph.db as db_mod
import ilga_graph.dependencies as deps_mod
from ilga_graph.db_models import User
from ilga_graph.routers import account as account_router_mod
from ilga_graph.routers import admin as admin_router_mod
from ilga_graph.routers import advocacy as advocacy_router_mod
from ilga_graph.routers import auth as auth_router_mod
from ilga_graph.routers import stories as stories_router_mod
from ilga_graph.routers import updates as updates_router_mod
from ilga_graph.security import (
    CSRF_COOKIE_NAME,
    CSRF_MAX_AGE_SECONDS,
    generate_csrf_token,
)
from tests.async_helpers import run_async


def _make_test_app(db_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.send_jobs = {}
        await db_mod.init_db()
        yield

    app = FastAPI(title="Test Updates", lifespan=lifespan)

    @app.middleware("http")
    async def _csrf_cookie_middleware(request, call_next):
        token = generate_csrf_token()
        request.state.csrf_token = token
        response = await call_next(request)
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=token,
            max_age=CSRF_MAX_AGE_SECONDS,
            path="/",
            httponly=False,
            samesite="strict",
            secure=False,
        )
        return response

    @app.middleware("http")
    async def _user_for_html_middleware(request: Request, call_next) -> Response:
        """Set request.state.user for HTML so templates hide subscribe when already subscribed."""
        accept = request.headers.get("accept") or ""
        if "text/html" in accept:
            try:
                async with db_mod.async_session_factory() as db:
                    session_cookie = request.cookies.get(cfg_mod.AUTH_COOKIE_NAME)
                    user_id = (
                        deps_mod.decode_session_token(session_cookie) if session_cookie else None
                    )
                    if user_id is not None:
                        result = await db.execute(select(User).where(User.id == user_id))
                        user = result.scalar_one_or_none()
                    else:
                        user = None
                    request.state.user = user  # type: ignore[attr-defined]
            except Exception:
                request.state.user = None  # type: ignore[attr-defined]
        return await call_next(request)

    _template_dir = Path(__file__).resolve().parent.parent / "src" / "ilga_graph" / "templates"
    templates = Jinja2Templates(directory=str(_template_dir))
    templates.env.globals["features"] = {}
    templates.env.globals["site_name"] = "Test"
    templates.env.globals["meta_description"] = ""
    templates.env.globals["strategic_five_points"] = []
    templates.env.globals["app_base_url"] = "http://testserver"
    templates.env.globals["og_image_url"] = ""
    templates.env.globals["primary_color"] = "#c2410c"
    templates.env.globals["show_beta_banner"] = False
    templates.env.globals["footer_last_updated"] = None
    templates.env.globals["get_current_action_campaign"] = lambda r: None
    templates.env.globals["get_poll_campaign_for_template"] = lambda r: None
    app.state.templates = templates
    app.include_router(auth_router_mod.router)
    app.include_router(account_router_mod.router)
    app.include_router(advocacy_router_mod.router, prefix="/advocacy")
    app.include_router(updates_router_mod.router)
    app.include_router(admin_router_mod.router)
    app.include_router(stories_router_mod.router)
    return app


def _data_with_csrf(client: TestClient, data: dict) -> dict:
    out = dict(data)
    out.setdefault("csrf_token", client.cookies.get(CSRF_COOKIE_NAME, ""))
    return out


# Valid kei_status + kei_impact_slug for POST /updates/kei-status (benefit slugs only).
POLL_SUBMIT_DATA = {"kei_status": "would_want", "kei_impact_slug": "support_cause"}


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
    return tmp_path / "test_updates.db"


@pytest.fixture
def client(test_db_path: Path) -> TestClient:
    env = {
        "ILGA_DB_PATH": str(test_db_path),
        "ILGA_AUTH_SECRET": "test-secret-for-pytest",
        "ILGA_PROFILE": "dev",
        "ILGA_ADMIN_EMAILS": "admin@example.com",
        "ILGA_RATE_LIMIT_VERIFY_CODE_PER_15MIN": "100",
        "ILGA_RATE_LIMIT_REQUEST_CODE_PER_15MIN": "100",
    }
    with patch.dict(os.environ, env, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        importlib.reload(deps_mod)
        importlib.reload(advocacy_router_mod)
        importlib.reload(updates_router_mod)
        importlib.reload(auth_router_mod)
        importlib.reload(account_router_mod)
        importlib.reload(stories_router_mod)
        app = _make_test_app(test_db_path)
        with TestClient(app, raise_server_exceptions=True) as c:
            c.get("/auth/me")
            yield c


@pytest.fixture
def authed_client(client: TestClient, test_db_path: Path) -> TestClient:
    """Client with authenticated user (subscriber@example.com)."""
    email = "subscriber@example.com"
    code = "123456"
    with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        run_async(_add_auth_code(email, code))
    client.post(
        "/auth/verify-code",
        data=_data_with_csrf(client, {"email": email, "code": code}),
    )
    return client


@pytest.fixture
def admin_client(client: TestClient, test_db_path: Path) -> TestClient:
    """Client with authenticated admin (admin@example.com)."""
    email = "admin@example.com"
    code = "654321"
    with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        run_async(_add_auth_code(email, code))
    client.post(
        "/auth/verify-code",
        data=_data_with_csrf(client, {"email": email, "code": code}),
    )
    return client


class TestUpdatesPage:
    """GET /updates returns 200 and campaign status."""

    def test_updates_page_returns_200(self, client: TestClient) -> None:
        resp = client.get("/updates")
        assert resp.status_code == 200
        assert b"Where we are" in resp.content or b"Updates" in resp.content

    def test_updates_page_shows_progress_checklist(self, client: TestClient) -> None:
        """Where we are section includes progress checklist with achieved and pending steps."""
        resp = client.get("/updates")
        assert resp.status_code == 200
        assert b"progress-checklist" in resp.content
        assert b"progress-checklist__item--achieved" in resp.content
        assert b"progress-checklist__item--pending" in resp.content

    def test_updates_page_lists_sent_updates(self, client: TestClient, test_db_path: Path) -> None:
        from datetime import datetime, timezone

        from ilga_graph.db_models import Update

        async def add_sent():
            async with db_mod.async_session_factory() as session:
                u = Update(
                    title="Test Update",
                    body_plain="Body text",
                    sent_at=datetime.now(timezone.utc),
                    sent_count=5,
                )
                session.add(u)
                await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(add_sent())
        resp = client.get("/updates")
        assert resp.status_code == 200
        assert b"Test Update" in resp.content

    def test_updates_page_shows_update_type_in_sidebar_and_article(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """Sent update type label appears in sidebar and type phrase in main article."""
        from datetime import datetime, timezone

        from ilga_graph.db_models import Update

        async def add_sent_update():
            async with db_mod.async_session_factory() as session:
                session.add(
                    Update(
                        title="Committee vote this week",
                        body_plain="Please call your rep.",
                        body_html="<p>Please call your rep.</p>",
                        update_type="major",
                        sent_at=datetime.now(timezone.utc),
                        sent_count=10,
                    )
                )
                await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(add_sent_update())

        resp = client.get("/updates")
        assert resp.status_code == 200
        content = resp.content.decode("utf-8", errors="replace")
        assert "Major" in content
        assert "Major update" in content
        assert "Committee vote this week" in content

    def test_updates_page_sidebar_has_anchor_links(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """Sidebar links are in-page TOC anchors #update-{id}."""
        from datetime import datetime, timezone

        from ilga_graph.db_models import Update

        async def add_sent():
            async with db_mod.async_session_factory() as session:
                u = Update(
                    title="Sidebar Link Test",
                    body_plain="Body",
                    sent_at=datetime.now(timezone.utc),
                    sent_count=1,
                )
                session.add(u)
                await session.commit()
                return u.id

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            uid = run_async(add_sent())
        resp = client.get("/updates")
        assert resp.status_code == 200
        raw = resp.content
        assert f'href="#update-{uid}"'.encode() in raw

    def test_updates_page_no_view_in_new_page_link(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """Public page does not show 'View this update in a new page' link."""
        from datetime import datetime, timezone

        from ilga_graph.db_models import Update

        async def add_sent():
            async with db_mod.async_session_factory() as session:
                u = Update(
                    title="No Link Test",
                    body_plain="Body",
                    sent_at=datetime.now(timezone.utc),
                    sent_count=1,
                )
                session.add(u)
                await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(add_sent())
        resp = client.get("/updates")
        assert resp.status_code == 200
        assert b"View this update in a new page" not in resp.content


class TestUpdateDetailPage:
    """GET /updates/{id} for sent update returns 200; unsent or missing returns 404."""

    def test_update_detail_sent_redirects_to_updates_anchor(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """GET /updates/{id} redirects to /updates#update-{id} for backward compatibility."""
        from datetime import datetime, timezone

        from ilga_graph.db_models import Update

        async def add_sent():
            async with db_mod.async_session_factory() as session:
                u = Update(
                    title="Detail Page Test",
                    body_plain="Email body **content**",
                    body_html="<p>Email body <strong>content</strong></p>",
                    sent_at=datetime.now(timezone.utc),
                    sent_count=3,
                )
                session.add(u)
                await session.commit()
                return u.id

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            uid = run_async(add_sent())
        resp = client.get(f"/updates/{uid}", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers.get("location") == f"/updates#update-{uid}"
        # Follow redirect; content is on the single updates page
        resp2 = client.get(f"/updates/{uid}", follow_redirects=True)
        assert resp2.status_code == 200
        assert b"Detail Page Test" in resp2.content
        assert b"content" in resp2.content

    def test_update_detail_unsent_returns_404(self, client: TestClient, test_db_path: Path) -> None:
        from ilga_graph.db_models import Update

        async def add_draft():
            async with db_mod.async_session_factory() as session:
                u = Update(title="Draft Only", body_plain="Not sent", sent_at=None, sent_count=0)
                session.add(u)
                await session.commit()
                return u.id

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            uid = run_async(add_draft())
        resp = client.get(f"/updates/{uid}")
        assert resp.status_code == 404

    def test_update_detail_missing_returns_404(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        resp = client.get("/updates/99999")
        assert resp.status_code == 404


class TestSubscribeUnsubscribe:
    """Subscription toggle and tokenized unsubscribe."""

    def test_subscribe_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/updates/subscribe")
        assert resp.status_code == 401

    def test_subscribe_sets_wants_updates(self, client: TestClient, test_db_path: Path) -> None:
        email = "sub@example.com"
        code = "111222"
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            run_async(_add_auth_code(email, code))
        client.post(
            "/auth/verify-code",
            data=_data_with_csrf(client, {"email": email, "code": code}),
        )

        # Ensure user exists and set wants_updates False first
        async def set_unsubscribed():
            from sqlalchemy import select

            from ilga_graph.db_models import User

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(User).where(User.email == email))
                u = r.scalar_one_or_none()
                if u:
                    u.wants_updates = False
                    await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(set_unsubscribed())

        resp = client.post("/updates/subscribe", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/updates"

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import User

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(User).where(User.email == email))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.wants_updates is True

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_unsubscribe_with_valid_token_sets_wants_updates_false(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        from ilga_graph.routers.updates import _create_unsubscribe_token

        async def add_user():
            from ilga_graph.db_models import User

            async with db_mod.async_session_factory() as session:
                u = User(email="unsub@example.com", wants_updates=True)
                session.add(u)
                await session.commit()
                return u.id

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(deps_mod)
            importlib.reload(updates_router_mod)
            uid = run_async(add_user())

        token = _create_unsubscribe_token(uid)
        resp = client.get(f"/updates/unsubscribe?token={token}")
        assert resp.status_code == 200
        assert b"unsubscribed" in resp.content.lower()

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import User

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(User).where(User.email == "unsub@example.com"))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.wants_updates is False

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_unsubscribe_invalid_token_shows_message(self, client: TestClient) -> None:
        resp = client.get("/updates/unsubscribe?token=invalid")
        assert resp.status_code == 200
        assert b"Invalid" in resp.content or b"expired" in resp.content.lower()


class TestAccountPage:
    """GET /account and POST /account: profile page and update zip/newsletter."""

    def test_get_account_anonymous_returns_401(self, client: TestClient) -> None:
        """Unauthenticated GET /account returns 401 (main app redirects to home via handler)."""
        resp = client.get(
            "/account",
            headers={"Accept": "text/html"},
            follow_redirects=False,
        )
        assert resp.status_code == 401

    def test_get_account_authenticated_returns_200(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/account", headers={"Accept": "text/html"})
        assert resp.status_code == 200
        assert b"Account" in resp.content
        assert b"subscriber@example.com" in resp.content

    def test_post_account_requires_csrf(self, authed_client: TestClient) -> None:
        resp = authed_client.post(
            "/account",
            data={"zip_code": "", "wants_updates": "1"},
            headers={"Accept": "text/html"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "error=csrf" in resp.headers.get("location", "")

    def test_post_account_saves_wants_updates(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        data = _data_with_csrf(authed_client, {"zip_code": "", "wants_updates": ""})
        resp = authed_client.post(
            "/account",
            data=data,
            headers={"Accept": "text/html"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "saved=1" in resp.headers.get("location", "")

        async def check_unsubscribed():
            async with db_mod.async_session_factory() as session:
                r = await session.execute(
                    select(User).where(User.email == "subscriber@example.com")
                )
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.wants_updates is False

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check_unsubscribed())


class TestSubscribeComponentVisibility:
    """Subscribed users (wants_updates=True) have subscribe components hidden site-wide."""

    def test_footer_subscribe_shown_when_anonymous(self, client: TestClient) -> None:
        """GET /updates without auth shows footer subscribe form."""
        resp = client.get("/updates", headers={"Accept": "text/html"})
        assert resp.status_code == 200
        assert b"footer-email-subscribe-wrap" in resp.content

    def test_footer_subscribe_hidden_when_subscribed(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        """GET /updates as subscribed user does not show footer subscribe form."""
        resp = authed_client.get("/updates", headers={"Accept": "text/html"})
        assert resp.status_code == 200
        assert b"footer-email-subscribe-wrap" not in resp.content

    def test_footer_subscribe_shown_when_logged_in_but_unsubscribed(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        """GET /updates as logged-in user with wants_updates=False shows footer subscribe form."""

        async def set_unsubscribed():
            async with db_mod.async_session_factory() as session:
                r = await session.execute(
                    select(User).where(User.email == "subscriber@example.com")
                )
                u = r.scalar_one_or_none()
                if u:
                    u.wants_updates = False
                    await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(set_unsubscribed())

        resp = authed_client.get("/updates", headers={"Accept": "text/html"})
        assert resp.status_code == 200
        assert b"footer-email-subscribe-wrap" in resp.content


class TestSubscribeUnsubscribeEmail:
    """Public subscribe-email endpoint (no auth)."""

    def test_subscribe_email_creates_user_and_redirects(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """subscribe-email (no auth) creates user with wants_updates=True and redirects."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(deps_mod)
            importlib.reload(updates_router_mod)
        resp = client.post(
            "/updates/subscribe-email",
            data=_data_with_csrf(client, {"email": "  NewSub@Example.com  "}),
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/updates?subscribed=1"

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import User

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(User).where(User.email == "newsub@example.com"))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.wants_updates is True

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_subscribe_email_invalid_returns_400_with_htmx(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """POST /updates/subscribe-email invalid email returns 400 and HTML fragment for HTMX."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        resp = client.post(
            "/updates/subscribe-email",
            data=_data_with_csrf(client, {"email": "not-an-email"}),
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 400
        assert b"valid" in resp.content.lower() or b"email" in resp.content.lower()

    def test_kei_status_authenticated_sets_status(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        """POST /updates/kei-status with auth sets user.kei_status (Turnstile required for all)."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        with patch.object(
            updates_router_mod, "_verify_turnstile", new_callable=AsyncMock, return_value=True
        ):
            resp = authed_client.post(
                "/updates/kei-status",
                data=_data_with_csrf(authed_client, POLL_SUBMIT_DATA),
                headers={"HX-Request": "true"},
            )
        assert resp.status_code == 200
        assert b"Thanks" in resp.content or b"community" in resp.content

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import KeiPollResponse, User

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(User))
                users = list(r.scalars().all())
                assert len(users) >= 1
                u = next((x for x in users if x.email == "subscriber@example.com"), None)
                assert u is not None
                assert u.kei_status == "would_want"
                assert u.kei_impact_slug == "support_cause"
                pr = await session.execute(select(KeiPollResponse))
                responses = list(pr.scalars().all())
                assert len(responses) == 1
                assert responses[0].kei_status == "would_want"
                assert responses[0].user_id == u.id

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_kei_status_anonymous_does_not_persist(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """POST kei-status without auth: no user create/update; returns 200 with results UI."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        with patch.object(
            updates_router_mod, "_verify_turnstile", new_callable=AsyncMock, return_value=True
        ):
            resp = client.post(
                "/updates/kei-status",
                data=_data_with_csrf(
                    client, {"kei_status": "registered", "kei_impact_slug": "support_cause"}
                ),
                headers={"HX-Request": "true"},
            )
        assert resp.status_code == 200
        assert b"change your answer" in resp.content or b"Need to change" in resp.content

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import KeiPollResponse, User

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(User))
                users = list(r.scalars().all())
                assert not any(u.email == "kei.poll@example.com" for u in users)
                pr = await session.execute(select(KeiPollResponse))
                responses = list(pr.scalars().all())
                assert len(responses) == 1
                assert responses[0].kei_status == "registered"
                assert responses[0].user_id is None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_kei_status_anonymous_sets_voted_cookie(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """POST kei-status without auth sets kei_poll_voted cookie; next visit shows results."""
        from ilga_graph.routers.updates import KEI_POLL_VOTED_COOKIE

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        with patch.object(
            updates_router_mod, "_verify_turnstile", new_callable=AsyncMock, return_value=True
        ):
            resp = client.post(
                "/updates/kei-status",
                data=_data_with_csrf(
                    client, {"kei_status": "registered", "kei_impact_slug": "support_cause"}
                ),
                headers={"HX-Request": "true"},
            )
        assert resp.status_code == 200
        assert KEI_POLL_VOTED_COOKIE in resp.cookies
        assert resp.cookies[KEI_POLL_VOTED_COOKIE] == "1"

    def test_updates_page_shows_poll_results_when_cookie_set(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """GET /updates?prompt=kei with kei_poll_voted cookie shows results, not form."""
        from ilga_graph.routers.updates import KEI_POLL_VOTED_COOKIE

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        client.cookies.set(KEI_POLL_VOTED_COOKIE, "1")
        resp = client.get("/updates", params={"prompt": "kei"})
        assert resp.status_code == 200
        html = resp.text
        assert "Results" in html
        assert "updates-kei-poll-wrap" in html
        assert "Need to change your answer" in html

    def test_updates_page_shows_poll_form_when_no_vote(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """GET /updates?prompt=kei without cookie or user kei_status shows poll form."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        resp = client.get("/updates", params={"prompt": "kei"})
        assert resp.status_code == 200
        html = resp.text
        assert "Do you have a kei vehicle" in html or "kei" in html.lower()
        assert "Submit" in html
        assert 'name="kei_status"' in html

    def test_kei_status_invalid_slug_returns_400(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """POST /updates/kei-status with invalid kei_status returns 400."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        resp = client.post(
            "/updates/kei-status",
            data=_data_with_csrf(
                client,
                {"kei_status": "invalid_slug", "kei_impact_slug": "support_cause"},
            ),
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 400

    def test_kei_status_missing_impact_returns_400(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """POST /updates/kei-status without kei_impact_slug returns 400."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        with patch.object(
            updates_router_mod, "_verify_turnstile", new_callable=AsyncMock, return_value=True
        ):
            resp = client.post(
                "/updates/kei-status",
                data=_data_with_csrf(client, {"kei_status": "would_want"}),
                headers={"HX-Request": "true"},
            )
        assert resp.status_code == 400
        assert b"affect" in resp.content.lower() or b"choose" in resp.content.lower()

    def test_kei_status_invalid_impact_returns_400(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """POST /updates/kei-status with invalid kei_impact_slug returns 400."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        with patch.object(
            updates_router_mod, "_verify_turnstile", new_callable=AsyncMock, return_value=True
        ):
            resp = client.post(
                "/updates/kei-status",
                data=_data_with_csrf(
                    client,
                    {"kei_status": "would_want", "kei_impact_slug": "invalid_impact"},
                ),
                headers={"HX-Request": "true"},
            )
        assert resp.status_code == 400

    def test_kei_poll_form_get_returns_form_html(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """GET /updates/kei-poll-form returns poll form partial for change-answer flow."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        resp = client.get("/updates/kei-poll-form?poll_id=footer-kei-poll")
        assert resp.status_code == 200
        html = resp.text
        assert "footer-kei-poll-wrap" in html
        assert "Quick question" in html or "kei" in html.lower()
        assert 'name="kei_status"' in html

    def test_kei_status_results_only_verified_users(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """GET /updates/kei-status-results returns counts from PollResponse
        (Turnstile-verified only)."""
        with patch.object(
            updates_router_mod, "_verify_turnstile", new_callable=AsyncMock, return_value=True
        ):
            resp = client.post(
                "/updates/kei-status",
                data=_data_with_csrf(
                    client,
                    {"kei_status": "registered", "kei_impact_slug": "support_cause"},
                ),
                headers={"HX-Request": "true"},
            )
        assert resp.status_code == 200
        resp = client.get("/updates/kei-status-results")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_responses" in data and "by_status" in data
        for slug in ("registered", "would_want", "would_not_want", "revoked", "denied"):
            assert slug in data["by_status"]
        assert isinstance(data["total_responses"], int) and data["total_responses"] >= 0


class TestPollStandalonePage:
    """GET /poll and POST kei-status with poll_id=standalone-kei-poll (shareable poll-only page)."""

    def test_poll_page_shows_form_when_no_vote(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """GET /poll without cookie shows poll form with standalone-kei-poll."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        resp = client.get("/poll")
        assert resp.status_code == 200
        html = resp.text
        assert "standalone-kei-poll" in html
        assert "Do you have a kei" in html or "kei" in html.lower()
        assert 'name="kei_status"' in html
        assert "poll-standalone" in html

    def test_poll_page_shows_results_when_cookie_or_submitted(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """GET /poll with cookie or ?submitted=1 shows results and link to home."""
        from ilga_graph.kei_poll_context import (
            KEI_POLL_VOTED_COOKIE,
        )

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        client.cookies.set(KEI_POLL_VOTED_COOKIE, "1")
        resp = client.get("/poll")
        assert resp.status_code == 200
        html = resp.text
        assert 'href="/"' in html
        assert "Results" in html or "results" in html.lower()

        resp2 = client.get("/poll", params={"submitted": "1"})
        assert resp2.status_code == 200
        assert 'href="/"' in resp2.text

    def test_poll_results_cta_owner_shows_branch_ctas(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """Poll standalone with choice=registered shows Start outreach CTA; no Share your story."""
        from ilga_graph.kei_poll_context import (
            KEI_POLL_CHOICE_COOKIE,
            KEI_POLL_VOTED_COOKIE,
        )

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        client.cookies.set(KEI_POLL_VOTED_COOKIE, "1")
        client.cookies.set(KEI_POLL_CHOICE_COOKIE, "registered")
        resp = client.get("/poll")
        assert resp.status_code == 200
        html = resp.text
        assert "Start outreach" in html
        assert "/advocacy" in html
        # Share CTA block hidden on standalone; dialog may still be in layout
        assert "wyc-branch-content__share" not in html

    def test_poll_results_cta_non_owner_shows_branch_ctas(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """Poll standalone choice=would_want shows branch CTAs (Learn issue + outreach)."""
        from ilga_graph.kei_poll_context import (
            KEI_POLL_CHOICE_COOKIE,
            KEI_POLL_VOTED_COOKIE,
        )

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        client.cookies.set(KEI_POLL_VOTED_COOKIE, "1")
        client.cookies.set(KEI_POLL_CHOICE_COOKIE, "would_want")
        resp = client.get("/poll")
        assert resp.status_code == 200
        html = resp.text
        assert "Learn about the issue" in html
        assert "Start outreach" in html or "/advocacy" in html
        assert "wyc-branch-content__after-change-cta" in html

    def test_poll_state_sync_after_standalone_vote(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """After standalone poll submit, updates page shows results (cookie shared)."""
        from ilga_graph.kei_poll_context import KEI_POLL_VOTED_COOKIE

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        with patch.object(
            updates_router_mod, "_verify_turnstile", new_callable=AsyncMock, return_value=True
        ):
            client.post(
                "/updates/kei-status",
                data=_data_with_csrf(
                    client,
                    {
                        "kei_status": "would_want",
                        "kei_impact_slug": "support_cause",
                        "poll_id": "standalone-kei-poll",
                        "zip_code": "60001",
                    },
                ),
                follow_redirects=True,
            )
        assert KEI_POLL_VOTED_COOKIE in client.cookies
        resp = client.get("/updates", params={"prompt": "kei"})
        assert resp.status_code == 200
        html = resp.text
        assert "Results" in html
        assert "Learn about the issue" in html or "Start outreach" in html

    def test_kei_status_htmx_returns_branch_cta_in_fragment(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """POST kei-status (HTMX) non-home poll returns success fragment with branch primary CTA."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        with patch.object(
            updates_router_mod, "_verify_turnstile", new_callable=AsyncMock, return_value=True
        ):
            resp = client.post(
                "/updates/kei-status",
                data=_data_with_csrf(
                    client,
                    {
                        "kei_status": "would_want",
                        "kei_impact_slug": "support_cause",
                        "poll_id": "updates-kei-poll",
                        "zip_code": "60001",
                    },
                ),
                headers={"HX-Request": "true"},
            )
        assert resp.status_code == 200
        html = resp.text
        assert "Learn about the issue" in html
        assert "/the-issue" in html

    def test_kei_status_standalone_redirects_to_poll(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """POST kei-status with poll_id=standalone-kei-poll redirects to /poll?submitted=1."""
        from ilga_graph.routers.updates import (
            KEI_POLL_CHOICE_COOKIE,
            KEI_POLL_VOTED_COOKIE,
        )

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        with patch.object(
            updates_router_mod, "_verify_turnstile", new_callable=AsyncMock, return_value=True
        ):
            resp = client.post(
                "/updates/kei-status",
                data=_data_with_csrf(
                    client,
                    {
                        "kei_status": "would_want",
                        "kei_impact_slug": "support_cause",
                        "poll_id": "standalone-kei-poll",
                        "zip_code": "60001",
                    },
                ),
                follow_redirects=False,
            )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/poll?submitted=1"
        assert resp.cookies.get(KEI_POLL_VOTED_COOKIE) == "1"
        assert resp.cookies.get(KEI_POLL_CHOICE_COOKIE) == "would_want"


class TestAdvocacyPersonalizePoll:
    """POST /advocacy/personalize-poll: persist kei_status and kei_impact_slug (drawer flow)."""

    # Use slug in KEI_IMPACT_OPTIONS["would_want"] (advocacy drawer), not main poll benefit set.
    _PERSONALIZE_DATA = {"kei_status": "would_want", "kei_impact_slug": "other"}

    def test_personalize_poll_requires_csrf(self, client: TestClient, test_db_path: Path) -> None:
        """POST without valid CSRF returns 403."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(advocacy_router_mod)
        resp = client.post(
            "/advocacy/personalize-poll",
            data={"kei_status": "would_want", "kei_impact_slug": "other"},
        )
        assert resp.status_code == 403

    def test_personalize_poll_authenticated_sets_user_only(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        """POST with auth: kei_status and kei_impact_slug set on user; no poll
        response rows (counted only after Turnstile at /updates/kei-status)."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(advocacy_router_mod)
        resp = authed_client.post(
            "/advocacy/personalize-poll",
            data=_data_with_csrf(authed_client, self._PERSONALIZE_DATA),
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        async def check():
            async with db_mod.async_session_factory() as session:
                q = select(User).where(User.email == "subscriber@example.com")
                r = await session.execute(q)
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.kei_status == "would_want"
                assert u.kei_impact_slug == "other"

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_personalize_poll_anonymous_sets_cookies(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """POST without auth: 200 and voted/choice/impact cookies set."""
        from ilga_graph.kei_poll_context import (
            KEI_POLL_CHOICE_COOKIE,
            KEI_POLL_VOTED_COOKIE,
        )
        from ilga_graph.routers.advocacy import KEI_IMPACT_SLUG_COOKIE

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(advocacy_router_mod)
        resp = client.post(
            "/advocacy/personalize-poll",
            data=_data_with_csrf(client, self._PERSONALIZE_DATA),
        )
        assert resp.status_code == 200
        assert resp.cookies.get(KEI_POLL_VOTED_COOKIE) == "1"
        assert resp.cookies.get(KEI_POLL_CHOICE_COOKIE) == "would_want"
        assert resp.cookies.get(KEI_IMPACT_SLUG_COOKIE) == "other"

    def test_personalize_poll_invalid_impact_does_not_persist_impact(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        """Valid kei_status but invalid kei_impact_slug: kei_status persisted, impact not."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(advocacy_router_mod)
        resp = authed_client.post(
            "/advocacy/personalize-poll",
            data=_data_with_csrf(
                authed_client,
                {"kei_status": "would_want", "kei_impact_slug": "invalid_impact"},
            ),
        )
        assert resp.status_code == 200

        async def check():
            async with db_mod.async_session_factory() as session:
                q = select(User).where(User.email == "subscriber@example.com")
                r = await session.execute(q)
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.kei_status == "would_want"
                assert u.kei_impact_slug is None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())


class TestAdminGate:
    """Admin routes require admin email."""

    def test_admin_updates_without_auth_returns_401(self, client: TestClient) -> None:
        resp = client.get("/admin/updates")
        assert resp.status_code == 401

    def test_admin_updates_non_admin_returns_403(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/admin/updates")
        assert resp.status_code == 403

    def test_admin_updates_with_admin_returns_200(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/admin/updates")
        assert resp.status_code == 200
        assert b"Compose" in resp.content or b"draft" in resp.content.lower()

    def test_admin_poll_returns_200(self, admin_client: TestClient) -> None:
        """GET /admin/poll redirects to polls list or kei results; final page shows poll content."""
        resp = admin_client.get("/admin/poll")
        assert resp.status_code in (200, 302)
        content = resp.content if resp.status_code == 200 else b""
        if resp.status_code == 302:
            content = admin_client.get(resp.headers["Location"]).content
        assert b"poll" in content.lower()
        assert (
            b"Verified" in content
            or b"verified" in content
            or b"Polls" in content
            or b"Create poll" in content
        )


# Minimal JPEG for story image upload (same as TestUpdateImage).
_MINI_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19"
    b"\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444"
    b"\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01"
    b"\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfe\x02\x1e\xf3\xcf\xff\xd9"
)


async def _set_user_kei_status(db_path: Path, email: str, kei_status: str) -> None:
    """Set kei_status for user by email (for story/statement tests)."""
    from sqlalchemy import update

    from ilga_graph.db_models import User

    async with db_mod.async_session_factory() as session:
        await session.execute(update(User).where(User.email == email).values(kei_status=kei_status))
        await session.commit()


class TestCommunityStories:
    """Community story submit and admin review."""

    def test_post_without_auth_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/community-stories",
            data=_data_with_csrf(
                client, {"name": "A", "location": "B", "story": "C", "consent": "on"}
            ),
            files={"image": ("x.jpg", io.BytesIO(_MINI_JPEG), "image/jpeg")},
        )
        assert resp.status_code == 401

    def test_post_without_csrf_returns_403(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        """POST without valid CSRF token is rejected."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_set_user_kei_status(test_db_path, "subscriber@example.com", "registered"))
        resp = authed_client.post(
            "/community-stories",
            data={"name": "X", "location": "Y", "story": "Z", "consent": "on"},  # no csrf_token
            files={"image": ("p.jpg", io.BytesIO(_MINI_JPEG), "image/jpeg")},
        )
        assert resp.status_code == 403

    def test_post_without_consent_returns_400(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_set_user_kei_status(test_db_path, "subscriber@example.com", "registered"))
        resp = authed_client.post(
            "/community-stories",
            data=_data_with_csrf(
                authed_client, {"name": "Jane", "location": "Chicago", "story": "My story."}
            ),
            files={"image": ("p.jpg", io.BytesIO(_MINI_JPEG), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert b"consent" in resp.content.lower() or b"agree" in resp.content.lower()

    def test_post_valid_creates_pending_story(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_set_user_kei_status(test_db_path, "subscriber@example.com", "registered"))
        resp = authed_client.post(
            "/community-stories",
            data=_data_with_csrf(
                authed_client,
                {
                    "name": "Jane Doe",
                    "location": "Chicago, IL",
                    "story": "I have a kei truck.",
                    "consent": "on",
                },
            ),
            files={"image": ("photo.jpg", io.BytesIO(_MINI_JPEG), "image/jpeg")},
        )
        assert resp.status_code == 200

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import CommunityStory

            async with db_mod.async_session_factory() as session:
                r = await session.execute(
                    select(CommunityStory).where(CommunityStory.name == "Jane Doe")
                )
                row = r.scalar_one_or_none()
                assert row is not None
                assert row.status == "pending"
                assert "stories/" in row.image_path

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_admin_stories_list_returns_200(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/admin/stories")
        assert resp.status_code == 200
        assert b"pending" in resp.content.lower() or b"stories" in resp.content.lower()

    def test_admin_story_review_approve_redirects_and_updates(
        self, client: TestClient, authed_client: TestClient, test_db_path: Path
    ) -> None:
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_set_user_kei_status(test_db_path, "subscriber@example.com", "registered"))
        authed_client.post(
            "/community-stories",
            data=_data_with_csrf(
                authed_client,
                {"name": "Approve Me", "location": "IL", "story": "Story.", "consent": "on"},
            ),
            files={"image": ("p.jpg", io.BytesIO(_MINI_JPEG), "image/jpeg")},
        )

        async def get_story_id():
            from sqlalchemy import select

            from ilga_graph.db_models import CommunityStory

            async with db_mod.async_session_factory() as session:
                r = await session.execute(
                    select(CommunityStory).where(CommunityStory.name == "Approve Me")
                )
                row = r.scalar_one_or_none()
                return row.id if row else None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            story_id = run_async(get_story_id())
        assert story_id is not None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_add_auth_code("admin@example.com", "654321"))
        client.post(
            "/auth/verify-code",
            data=_data_with_csrf(client, {"email": "admin@example.com", "code": "654321"}),
        )
        with patch.object(stories_router_mod, "send_story_review_email", new_callable=AsyncMock):
            resp = client.post(
                f"/admin/stories/{story_id}/review",
                data=_data_with_csrf(client, {"action": "approve"}),
                follow_redirects=False,
            )
        assert resp.status_code == 303
        assert "flash=approved" in resp.headers.get("location", "")

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import CommunityStory

            async with db_mod.async_session_factory() as session:
                r = await session.execute(
                    select(CommunityStory).where(CommunityStory.id == story_id)
                )
                row = r.scalar_one_or_none()
                assert row is not None
                assert row.status == "approved"

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_admin_story_review_already_reviewed_redirects_with_flash(
        self, client: TestClient, authed_client: TestClient, test_db_path: Path
    ) -> None:
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_set_user_kei_status(test_db_path, "subscriber@example.com", "registered"))
        authed_client.post(
            "/community-stories",
            data=_data_with_csrf(
                authed_client,
                {"name": "Already Reviewed", "location": "IL", "story": "Story.", "consent": "on"},
            ),
            files={"image": ("p.jpg", io.BytesIO(_MINI_JPEG), "image/jpeg")},
        )

        async def get_story_id():
            from sqlalchemy import select

            from ilga_graph.db_models import CommunityStory

            async with db_mod.async_session_factory() as session:
                r = await session.execute(
                    select(CommunityStory).where(CommunityStory.name == "Already Reviewed")
                )
                row = r.scalar_one_or_none()
                return row.id if row else None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            story_id = run_async(get_story_id())
        assert story_id is not None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_add_auth_code("admin@example.com", "654321"))
        client.post(
            "/auth/verify-code",
            data=_data_with_csrf(client, {"email": "admin@example.com", "code": "654321"}),
        )
        with patch.object(stories_router_mod, "send_story_review_email", new_callable=AsyncMock):
            client.post(
                f"/admin/stories/{story_id}/review",
                data=_data_with_csrf(client, {"action": "approve"}),
            )
        resp = client.post(
            f"/admin/stories/{story_id}/review",
            data=_data_with_csrf(client, {"action": "approve"}),
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "already_reviewed" in resp.headers.get("location", "")


class TestCommunityStatements:
    """Interest statement submit (non-owners) and admin review."""

    def test_post_with_owner_kei_status_returns_403(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_set_user_kei_status(test_db_path, "subscriber@example.com", "registered"))
        resp = authed_client.post(
            "/community-statements",
            data=_data_with_csrf(
                authed_client,
                {
                    "name": "Jane",
                    "location": "Chicago",
                    "statement": "I would buy one.",
                    "consent": "on",
                },
            ),
        )
        assert resp.status_code == 403

    def test_post_with_non_owner_creates_pending_statement(
        self, authed_client: TestClient, test_db_path: Path
    ) -> None:
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_set_user_kei_status(test_db_path, "subscriber@example.com", "would_want"))
        resp = authed_client.post(
            "/community-statements",
            data=_data_with_csrf(
                authed_client,
                {
                    "name": "Jordan",
                    "location": "Peoria, IL",
                    "statement": "I would buy a kei truck if it were legal.",
                    "consent": "on",
                },
            ),
        )
        assert resp.status_code == 200

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import KeiInterestStatement

            async with db_mod.async_session_factory() as session:
                r = await session.execute(
                    select(KeiInterestStatement).where(KeiInterestStatement.name == "Jordan")
                )
                row = r.scalar_one_or_none()
                assert row is not None
                assert row.status == "pending"

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_admin_statements_list_returns_200(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/admin/statements")
        assert resp.status_code == 200
        assert b"statement" in resp.content.lower() or b"pending" in resp.content.lower()

    def test_admin_statement_review_approve_redirects(
        self, client: TestClient, authed_client: TestClient, test_db_path: Path
    ) -> None:
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_set_user_kei_status(test_db_path, "subscriber@example.com", "would_want"))
        authed_client.post(
            "/community-statements",
            data=_data_with_csrf(
                authed_client,
                {
                    "name": "Stmt Approve",
                    "location": "IL",
                    "statement": "I want one.",
                    "consent": "on",
                },
            ),
        )

        async def get_stmt_id():
            from sqlalchemy import select

            from ilga_graph.db_models import KeiInterestStatement

            async with db_mod.async_session_factory() as session:
                r = await session.execute(
                    select(KeiInterestStatement).where(KeiInterestStatement.name == "Stmt Approve")
                )
                row = r.scalar_one_or_none()
                return row.id if row else None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            stmt_id = run_async(get_stmt_id())
        assert stmt_id is not None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(_add_auth_code("admin@example.com", "654321"))
        client.post(
            "/auth/verify-code",
            data=_data_with_csrf(client, {"email": "admin@example.com", "code": "654321"}),
        )
        with patch.object(
            stories_router_mod, "send_statement_review_email", new_callable=AsyncMock
        ):
            resp = client.post(
                f"/admin/statements/{stmt_id}/review",
                data=_data_with_csrf(client, {"action": "approve"}),
                follow_redirects=False,
            )
        assert resp.status_code == 303
        assert "flash=approved" in resp.headers.get("location", "")


class TestAdminPollsEdgeCases:
    """Poll create/update/results edge cases."""

    def test_create_poll_duplicate_slug_returns_400(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        """Create a poll, then create another with same slug → 400 and error message."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
        admin_client.post(
            "/admin/polls",
            data={
                "title": "First Poll",
                "slug": "dup",
                "placement": "",
                "option_slug": ["a"],
                "option_label": ["Option A"],
            },
            follow_redirects=False,
        )
        resp = admin_client.post(
            "/admin/polls",
            data={
                "title": "Second Poll",
                "slug": "dup",
                "placement": "",
                "option_slug": ["x"],
                "option_label": ["Option X"],
            },
            follow_redirects=False,
        )
        assert resp.status_code == 400
        assert b"already exists" in resp.content.lower() or b"slug" in resp.content.lower()

    def test_create_poll_no_options_returns_400(self, admin_client: TestClient) -> None:
        resp = admin_client.post(
            "/admin/polls",
            data={"title": "No Options", "slug": "noopts", "placement": ""},
            follow_redirects=False,
        )
        assert resp.status_code == 400
        assert b"at least one option" in resp.content.lower() or b"option" in resp.content.lower()

    def test_poll_results_invalid_id_redirects_to_list(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        resp = admin_client.get("/admin/polls/99999/results", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers.get("location", "").rstrip("/").endswith("/admin/polls")


class TestAdminCreateAndSend:
    """Create draft and send (mocking email)."""

    def test_create_draft_redirects_and_stores(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        resp = admin_client.post(
            "/admin/updates",
            data={"title": "February update", "body_plain": "Hello everyone."},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "admin/updates" in resp.headers.get("location", "")

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import Update

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.title == "February update"))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.sent_at is None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_create_draft_with_update_type_major_stores_type(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        resp = admin_client.post(
            "/admin/updates",
            data={
                "title": "Major news",
                "body_plain": "Bill passed committee.",
                "update_type": "major",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import Update

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.title == "Major news"))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.update_type == "major"

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_create_draft_invalid_update_type_falls_back_to_other(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        resp = admin_client.post(
            "/admin/updates",
            data={
                "title": "No type",
                "body_plain": "Body.",
                "update_type": "invalid",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import Update

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.title == "No type"))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.update_type == "other"

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_send_update_sets_sent_at_and_count(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        import re
        import time

        from ilga_graph.db_models import Update, User

        async def setup():
            async with db_mod.async_session_factory() as session:
                session.add(User(email="recipient@example.com", wants_updates=True))
                session.add(
                    Update(
                        title="Blast",
                        body_plain="Content",
                        body_html="<p>Content</p>",
                    )
                )
                await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(setup())

        with patch(
            "ilga_graph.routers.updates.send_email",
            new_callable=AsyncMock,
            return_value=True,
        ):

            async def get_update_id():
                from sqlalchemy import select

                async with db_mod.async_session_factory() as session:
                    r = await session.execute(select(Update).where(Update.title == "Blast"))
                    u = r.scalar_one_or_none()
                    return u.id if u else None

            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(db_mod)
                uid = run_async(get_update_id())
            assert uid is not None

            resp = admin_client.post(f"/admin/updates/{uid}/send", follow_redirects=False)
        assert resp.status_code == 303
        location = resp.headers.get("location", "")
        assert "send/status" in location and "job=" in location
        match = re.search(r"job=([a-f0-9-]+)", location)
        assert match, location
        job_id = match.group(1)
        for _ in range(30):
            status_resp = admin_client.get(f"/admin/updates/send/status/{job_id}")
            if status_resp.status_code != 200:
                break
            data = status_resp.json()
            if data.get("done"):
                break
            time.sleep(0.1)

        async def check():
            from sqlalchemy import select

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.id == uid))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.sent_at is not None
                assert u.sent_count >= 1  # admin + recipient(s) with wants_updates=True

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_send_update_dev_fallback_when_no_subscribers(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        """In dev, when no user has wants_updates=True, send goes to all users for testing."""
        import re
        import time

        from ilga_graph.db_models import Update, User

        async def setup():
            from sqlalchemy import update

            async with db_mod.async_session_factory() as session:
                await session.execute(
                    update(User)
                    .where(User.email == "admin@example.com")
                    .values(wants_updates=False)
                )
                session.add(User(email="dev1@example.com", wants_updates=False))
                session.add(User(email="dev2@example.com", wants_updates=False))
                session.add(
                    Update(
                        title="Dev blast",
                        body_plain="Test body",
                        body_html="<p>Test body</p>",
                    )
                )
                await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(setup())

        with (
            patch("ilga_graph.config.DEV_MODE", True),
            patch(
                "ilga_graph.routers.updates.send_email",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):

            async def get_update_id():
                from sqlalchemy import select

                async with db_mod.async_session_factory() as session:
                    r = await session.execute(select(Update).where(Update.title == "Dev blast"))
                    u = r.scalar_one_or_none()
                    return u.id if u else None

            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(db_mod)
                uid = run_async(get_update_id())
            assert uid is not None

            resp = admin_client.post(f"/admin/updates/{uid}/send", follow_redirects=False)
        assert resp.status_code == 303
        location = resp.headers.get("location", "")
        match = re.search(r"job=([a-f0-9-]+)", location)
        assert match, location
        job_id = match.group(1)
        for _ in range(30):
            status_resp = admin_client.get(f"/admin/updates/send/status/{job_id}")
            if status_resp.status_code != 200:
                break
            data = status_resp.json()
            if data.get("done"):
                break
            time.sleep(0.1)

        async def check():
            from sqlalchemy import select

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.id == uid))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.sent_at is not None
                assert u.sent_count == 3  # dev fallback: admin + dev1 + dev2

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_recipients_page_returns_200_and_lists_subscribers(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        from ilga_graph.db_models import Update, User

        async def setup():
            async with db_mod.async_session_factory() as session:
                session.add(User(email="a@example.com", wants_updates=True))
                session.add(
                    Update(title="Recipients test", body_plain="Body", body_html="<p>Body</p>")
                )
                await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(setup())

        async def get_uid():
            from sqlalchemy import select

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.title == "Recipients test"))
                u = r.scalar_one_or_none()
                return u.id if u else None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            uid = run_async(get_uid())
        assert uid is not None
        resp = admin_client.get(f"/admin/updates/{uid}/recipients")
        assert resp.status_code == 200
        assert b"Choose recipients" in resp.content
        assert b"a@example.com" in resp.content
        assert b"recipient_ids" in resp.content

    def test_send_status_json_404_for_unknown_job(
        self,
        admin_client: TestClient,
    ) -> None:
        resp = admin_client.get("/admin/updates/send/status/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_send_with_zero_recipients_redirects_to_recipients(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        from ilga_graph.db_models import Update, User

        async def setup():
            async with db_mod.async_session_factory() as session:
                session.add(User(email="only@example.com", wants_updates=True))
                session.add(Update(title="Zero test", body_plain="Body", body_html="<p>Body</p>"))
                await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(setup())

        async def get_uid():
            from sqlalchemy import select

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.title == "Zero test"))
                u = r.scalar_one_or_none()
                return u.id if u else None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            uid = run_async(get_uid())
        assert uid is not None
        resp = admin_client.post(
            f"/admin/updates/{uid}/send",
            data={"from_recipients_page": "1"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "recipients" in resp.headers.get("location", "")
        assert "no_recipients" in resp.headers.get("location", "")


class TestUpdateImage:
    """Optional image on update: admin upload, display on page and in email."""

    # Minimal valid JPEG (1x1 pixel).
    _MINI_JPEG = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19"
        b"\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444"
        b"\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01"
        b"\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfe\x02\x1e\xf3\xcf\xff\xd9"
    )

    def test_create_draft_with_image_sets_image_path(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        """POST with optional image stores file and sets update.image_path."""
        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            importlib.reload(updates_router_mod)
        resp = admin_client.post(
            "/admin/updates",
            data={"title": "With image", "body_plain": "Body with image."},
            files={"image": ("photo.jpg", io.BytesIO(self._MINI_JPEG), "image/jpeg")},
            follow_redirects=False,
        )
        assert resp.status_code == 303

        async def check():
            from sqlalchemy import select

            from ilga_graph.db_models import Update

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.title == "With image"))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.image_path is not None
                assert "updates/" in u.image_path
                assert u.image_path.endswith(".jpg")

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())

    def test_public_updates_page_shows_image_when_image_path_set(
        self, client: TestClient, test_db_path: Path
    ) -> None:
        """Sent update with image_path shows img on public page."""
        from datetime import datetime, timezone

        from ilga_graph.db_models import Update

        async def add_sent_with_image():
            async with db_mod.async_session_factory() as session:
                u = Update(
                    title="Update with photo",
                    body_plain="See below.",
                    body_html="<p>See below.</p>",
                    image_path="updates/99_test123.jpg",
                    sent_at=datetime.now(timezone.utc),
                    sent_count=1,
                )
                session.add(u)
                await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(add_sent_with_image())
        resp = client.get("/updates")
        assert resp.status_code == 200
        assert b"Update with photo" in resp.content
        assert b"/static/updates/99_test123.jpg" in resp.content
        assert b"Update image" in resp.content

    def test_render_update_email_html_includes_image_when_image_url_given(self) -> None:
        """_render_update_email_html outputs img tag when image_url is passed."""
        from ilga_graph.routers.updates import _render_update_email_html

        html = _render_update_email_html(
            title="Test",
            body_html="<p>Body</p>",
            unsub_url="https://example.com/unsub",
            image_url="https://example.com/static/updates/1_abc.jpg",
        )
        assert "https://example.com/static/updates/1_abc.jpg" in html
        assert "<img" in html
        assert "Update image" in html

    def test_render_update_email_html_no_image_when_image_url_none(self) -> None:
        """When image_url is None, no img tag is rendered (e.g. when APP_BASE_URL empty)."""
        from ilga_graph.routers.updates import _render_update_email_html

        html = _render_update_email_html(
            title="Test",
            body_html="<p>Body</p>",
            unsub_url="https://example.com/unsub",
            image_url=None,
        )
        assert "<img" not in html
        assert "Body</p>" in html


class TestWelcomeEmailUnsubscribe:
    """Welcome email includes unsubscribe link when unsub_url is provided."""

    def test_welcome_email_includes_unsubscribe_link_when_unsub_url_provided(self) -> None:
        """send_welcome_email(..., unsub_url=...) produces body with Unsubscribe link."""
        from ilga_graph.email_utils import send_welcome_email

        unsub_url = "https://example.com/updates/unsubscribe?token=abc123"
        with patch(
            "ilga_graph.email_utils.send_email",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_send:
            run_async(send_welcome_email("user@example.com", unsub_url=unsub_url))
        mock_send.assert_called_once()
        call = mock_send.call_args
        assert call[0][0] == "user@example.com"
        plain = call[0][2]
        html = call[0][3]
        assert "Unsubscribe" in plain
        assert unsub_url in plain
        assert "Unsubscribe" in html
        assert unsub_url in html

    def test_welcome_email_no_unsubscribe_when_unsub_url_omitted(self) -> None:
        """send_welcome_email without unsub_url does not include Unsubscribe in body."""
        from ilga_graph.email_utils import send_welcome_email

        with patch(
            "ilga_graph.email_utils.send_email",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_send:
            run_async(send_welcome_email("user@example.com"))
        mock_send.assert_called_once()
        plain = mock_send.call_args[0][2]
        html = mock_send.call_args[0][3]
        assert "Unsubscribe" not in plain
        assert "Unsubscribe" not in html


class TestEmailRobustness:
    """Edge cases: invalid recipients, blank user email, partial send still commits."""

    def test_send_email_invalid_recipient_returns_false(self) -> None:
        """Empty or invalid 'to' returns False without raising."""
        from ilga_graph.email_utils import send_email

        async def run() -> None:
            assert await send_email("", "Sub", "Plain", "<p>P</p>") is False
            assert await send_email("  ", "Sub", "Plain", "<p>P</p>") is False
            assert await send_email("no-at-sign", "Sub", "Plain", "<p>P</p>") is False

        run_async(run())

    def test_user_with_blank_email_skipped_in_send(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        """When sending to a mix of valid and blank emails, only valid recipients get the email."""
        import re
        import time

        from sqlalchemy import select

        from ilga_graph.db_models import Update, User

        async def setup() -> int:
            async with db_mod.async_session_factory() as session:
                session.add(User(email="good@example.com", wants_updates=True))
                session.add(User(email=" ", wants_updates=True))  # blank → skipped
                session.add(
                    Update(
                        title="Blank email test",
                        body_plain="Body",
                        body_html="<p>Body</p>",
                    )
                )
                await session.commit()
                r = await session.execute(select(Update).where(Update.title == "Blank email test"))
                upd = r.scalar_one_or_none()
                assert upd is not None
                return upd.id

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            update_id = run_async(setup())

        with patch(
            "ilga_graph.routers.updates.send_email",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_send:
            resp = admin_client.post(
                f"/admin/updates/{update_id}/send",
                follow_redirects=False,
            )
        assert resp.status_code == 303
        location = resp.headers.get("location", "")
        match = re.search(r"job=([a-f0-9-]+)", location)
        assert match
        job_id = match.group(1)
        for _ in range(30):
            status_resp = admin_client.get(f"/admin/updates/send/status/{job_id}")
            assert status_resp.status_code == 200
            if status_resp.json().get("done"):
                break
            time.sleep(0.1)
        # good@example.com and admin@example.com get the email; user with " " is skipped
        assert mock_send.call_count == 2
        recipients = {mock_send.call_args_list[i][0][0] for i in range(2)}
        assert recipients == {"admin@example.com", "good@example.com"}

    def test_partial_send_still_commits_sent_at_and_count(
        self, admin_client: TestClient, test_db_path: Path
    ) -> None:
        """Partial send still commits sent_at/sent_count for successful recipients."""
        import re
        import time

        from sqlalchemy import select

        from ilga_graph.db_models import Update, User

        async def setup() -> int:
            async with db_mod.async_session_factory() as session:
                session.add(User(email="first@example.com", wants_updates=True))
                session.add(User(email="second@example.com", wants_updates=True))
                session.add(
                    Update(
                        title="Partial send test",
                        body_plain="Body",
                        body_html="<p>Body</p>",
                    )
                )
                await session.commit()
                r = await session.execute(select(Update).where(Update.title == "Partial send test"))
                upd = r.scalar_one_or_none()
                assert upd is not None
                return upd.id

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            update_id = run_async(setup())

        call_count = 0

        async def send_email_raise_second(to: str, *args: object, **kwargs: object) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("SMTP failure")
            return True

        with patch(
            "ilga_graph.routers.updates.send_email",
            side_effect=send_email_raise_second,
        ):
            resp = admin_client.post(
                f"/admin/updates/{update_id}/send",
                follow_redirects=False,
            )
        assert resp.status_code == 303
        match = re.search(r"job=([a-f0-9-]+)", resp.headers.get("location", ""))
        assert match
        job_id = match.group(1)
        for _ in range(30):
            status_resp = admin_client.get(f"/admin/updates/send/status/{job_id}")
            assert status_resp.status_code == 200
            data = status_resp.json()
            if data.get("done"):
                # admin + first + second = 3; one fails (first@), two succeed
                assert data.get("sent") == 2
                assert data.get("failed") == 1
                break
            time.sleep(0.1)

        async def check() -> None:
            from sqlalchemy import select

            from ilga_graph.db_models import Update

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.id == update_id))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.sent_at is not None
                assert u.sent_count == 2

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            run_async(check())
