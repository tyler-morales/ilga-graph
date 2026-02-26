"""Tests for updates router: public page, subscribe/unsubscribe, admin compose/send."""

from __future__ import annotations

import asyncio
import importlib
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ilga_graph.config as cfg_mod
import ilga_graph.db as db_mod
import ilga_graph.dependencies as deps_mod
from ilga_graph.routers import auth as auth_router_mod
from ilga_graph.routers import updates as updates_router_mod
from ilga_graph.security import (
    CSRF_COOKIE_NAME,
    CSRF_MAX_AGE_SECONDS,
    generate_csrf_token,
)


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

    app.include_router(auth_router_mod.router)
    app.include_router(updates_router_mod.router)
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
        importlib.reload(updates_router_mod)
        importlib.reload(auth_router_mod)
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
        asyncio.run(_add_auth_code(email, code))
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
        asyncio.run(_add_auth_code(email, code))
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

    def test_updates_page_shows_campaign_timeline(self, client: TestClient) -> None:
        """Where we are section includes campaign timeline with achieved and pending steps."""
        resp = client.get("/updates")
        assert resp.status_code == 200
        assert b"campaign-timeline" in resp.content
        assert b"campaign-timeline__item--achieved" in resp.content
        assert b"campaign-timeline__item--pending" in resp.content

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
            asyncio.run(add_sent())
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
            asyncio.run(add_sent_update())

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
            uid = asyncio.run(add_sent())
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
            asyncio.run(add_sent())
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
            uid = asyncio.run(add_sent())
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
            uid = asyncio.run(add_draft())
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
            asyncio.run(_add_auth_code(email, code))
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
            asyncio.run(set_unsubscribed())

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
            asyncio.run(check())

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
            uid = asyncio.run(add_user())

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
            asyncio.run(check())

    def test_unsubscribe_invalid_token_shows_message(self, client: TestClient) -> None:
        resp = client.get("/updates/unsubscribe?token=invalid")
        assert resp.status_code == 200
        assert b"Invalid" in resp.content or b"expired" in resp.content.lower()


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
            asyncio.run(check())

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
            asyncio.run(check())

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
            asyncio.run(check())

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
            asyncio.run(setup())

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
                uid = asyncio.run(get_update_id())
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
            asyncio.run(check())

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
            asyncio.run(setup())

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
                uid = asyncio.run(get_update_id())
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
            asyncio.run(check())

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
            asyncio.run(setup())

        async def get_uid():
            from sqlalchemy import select

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.title == "Recipients test"))
                u = r.scalar_one_or_none()
                return u.id if u else None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            uid = asyncio.run(get_uid())
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
            asyncio.run(setup())

        async def get_uid():
            from sqlalchemy import select

            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(Update).where(Update.title == "Zero test"))
                u = r.scalar_one_or_none()
                return u.id if u else None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(db_mod)
            uid = asyncio.run(get_uid())
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
            asyncio.run(check())

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
            asyncio.run(add_sent_with_image())
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


class TestEmailRobustness:
    """Edge cases: invalid recipients, blank user email, partial send still commits."""

    def test_send_email_invalid_recipient_returns_false(self) -> None:
        """Empty or invalid 'to' returns False without raising."""
        import asyncio

        from ilga_graph.email_utils import send_email

        async def run() -> None:
            assert await send_email("", "Sub", "Plain", "<p>P</p>") is False
            assert await send_email("  ", "Sub", "Plain", "<p>P</p>") is False
            assert await send_email("no-at-sign", "Sub", "Plain", "<p>P</p>") is False

        asyncio.run(run())

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
            update_id = asyncio.run(setup())

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
            update_id = asyncio.run(setup())

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
            asyncio.run(check())
