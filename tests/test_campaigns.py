"""Tests for campaigns: helpers, visibility, outreach recording with campaign_id."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ilga_graph.campaign_helpers import (
    campaign_outreach_count,
    deactivate_other_campaigns,
    get_active_campaign,
    is_campaign_visible_to_zip,
)
from ilga_graph.zip_crosswalk import ZipDistrictInfo


def _run(coro):
    return asyncio.run(coro)


class TestIsCampaignVisibleToZip:
    """is_campaign_visible_to_zip: all vs by_district, unknown zip."""

    def test_target_type_all_always_visible(self) -> None:
        campaign = MagicMock()
        campaign.target_type = "all"
        campaign.target_district_ids = None
        zip_to_district = {"60601": ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")}
        assert is_campaign_visible_to_zip(campaign, "60601", zip_to_district) is True
        assert is_campaign_visible_to_zip(campaign, None, zip_to_district) is True
        assert is_campaign_visible_to_zip(campaign, "", {}) is True

    def test_target_type_by_district_no_zip_hidden(self) -> None:
        campaign = MagicMock()
        campaign.target_type = "by_district"
        campaign.target_district_ids = json.dumps(["il_house:9"])
        zip_to_district = {"60601": ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")}
        assert is_campaign_visible_to_zip(campaign, None, zip_to_district) is False
        assert is_campaign_visible_to_zip(campaign, "", zip_to_district) is False

    def test_target_type_by_district_unknown_zip_hidden(self) -> None:
        campaign = MagicMock()
        campaign.target_type = "by_district"
        campaign.target_district_ids = json.dumps(["il_house:9"])
        zip_to_district = {"60601": ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")}
        assert is_campaign_visible_to_zip(campaign, "99999", zip_to_district) is False

    def test_target_type_by_district_matching_zip_visible(self) -> None:
        campaign = MagicMock()
        campaign.target_type = "by_district"
        campaign.target_district_ids = json.dumps(["il_house:9", "il_senate:5"])
        zip_to_district = {"60601": ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")}
        assert is_campaign_visible_to_zip(campaign, "60601", zip_to_district) is True

    def test_target_type_by_district_no_match_hidden(self) -> None:
        campaign = MagicMock()
        campaign.target_type = "by_district"
        campaign.target_district_ids = json.dumps(["il_house:99", "il_senate:88"])
        zip_to_district = {"60601": ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")}
        assert is_campaign_visible_to_zip(campaign, "60601", zip_to_district) is False

    def test_target_type_by_district_empty_target_ids_visible(self) -> None:
        campaign = MagicMock()
        campaign.target_type = "by_district"
        campaign.target_district_ids = None
        zip_to_district = {"60601": ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")}
        assert is_campaign_visible_to_zip(campaign, "60601", zip_to_district) is True


class TestCampaignHelpersWithDb:
    """get_active_campaign, campaign_outreach_count, deactivate_other_campaigns with temp DB."""

    @pytest.fixture
    def test_db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_campaigns.db"

    async def _init_and_get_session(self, db_path: Path):
        import ilga_graph.config as cfg_mod
        import ilga_graph.db as db_mod

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            await db_mod.init_db()
            async with db_mod.async_session_factory() as session:
                yield session

    def test_get_active_campaign_returns_none_when_no_campaigns(self, test_db_path: Path) -> None:
        async def run():
            async for session in self._init_and_get_session(test_db_path):
                out = await get_active_campaign(session)
                assert out is None

        _run(run())

    def test_get_active_campaign_returns_only_active(self, test_db_path: Path) -> None:
        import ilga_graph.config as cfg_mod
        import ilga_graph.db as db_mod
        from ilga_graph.db_models import Campaign

        async def run():
            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(cfg_mod)
                importlib.reload(db_mod)
                await db_mod.init_db()
                async with db_mod.async_session_factory() as session:
                    session.add(
                        Campaign(
                            title="Old",
                            message="M",
                            ask="A",
                            target_type="all",
                            is_active=False,
                        )
                    )
                    session.add(
                        Campaign(
                            title="Active",
                            message="M2",
                            ask="A2",
                            target_type="all",
                            is_active=True,
                        )
                    )
                    await session.commit()
                async with db_mod.async_session_factory() as session:
                    active = await get_active_campaign(session)
                    assert active is not None
                    assert active.title == "Active"

        _run(run())

    def test_campaign_outreach_count(self, test_db_path: Path) -> None:
        import ilga_graph.config as cfg_mod
        import ilga_graph.db as db_mod
        from ilga_graph.db_models import Campaign, OutreachEvent

        async def run():
            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(cfg_mod)
                importlib.reload(db_mod)
                await db_mod.init_db()
                async with db_mod.async_session_factory() as session:
                    c = Campaign(
                        title="C1",
                        message="M",
                        ask="A",
                        target_type="all",
                        is_active=True,
                    )
                    session.add(c)
                    await session.flush()
                    session.add(
                        OutreachEvent(
                            member_id="m1",
                            kind="call",
                            campaign_id=c.id,
                        )
                    )
                    session.add(
                        OutreachEvent(
                            member_id="m2",
                            kind="email",
                            campaign_id=c.id,
                        )
                    )
                    await session.commit()
                    cid = c.id
                async with db_mod.async_session_factory() as session:
                    n = await campaign_outreach_count(session, cid)
                    assert n == 2

        _run(run())

    def test_deactivate_other_campaigns(self, test_db_path: Path) -> None:
        from sqlalchemy import select

        import ilga_graph.config as cfg_mod
        import ilga_graph.db as db_mod
        from ilga_graph.db_models import Campaign

        async def run():
            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(cfg_mod)
                importlib.reload(db_mod)
                await db_mod.init_db()
                async with db_mod.async_session_factory() as session:
                    c1 = Campaign(
                        title="C1", message="M", ask="A", target_type="all", is_active=True
                    )
                    c2 = Campaign(
                        title="C2", message="M", ask="A", target_type="all", is_active=True
                    )
                    session.add_all([c1, c2])
                    await session.flush()
                    await session.commit()
                    c1_id, c2_id = c1.id, c2.id
                async with db_mod.async_session_factory() as session:
                    await deactivate_other_campaigns(session, c1_id)
                    await session.commit()
                async with db_mod.async_session_factory() as session:
                    r1 = await session.execute(select(Campaign).where(Campaign.id == c1_id))
                    r2 = await session.execute(select(Campaign).where(Campaign.id == c2_id))
                    assert r1.scalar_one().is_active is True
                    assert r2.scalar_one().is_active is False

        _run(run())


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_campaigns_ilga.db"


@pytest.fixture
def client(test_db_path: Path):
    """Minimal FastAPI app with auth + outreach for campaign_id recording tests."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import ilga_graph.config as cfg_mod
    import ilga_graph.db as db_mod
    import ilga_graph.dependencies as deps_mod
    from ilga_graph.routers import auth as auth_router_mod
    from ilga_graph.routers import outreach as outreach_router_mod
    from ilga_graph.security import (
        CSRF_COOKIE_NAME,
        CSRF_MAX_AGE_SECONDS,
        generate_csrf_token,
    )

    @asynccontextmanager
    async def lifespan(app):
        await db_mod.init_db()
        yield

    app = FastAPI(title="Test Campaigns", lifespan=lifespan)

    @app.middleware("http")
    async def _csrf_mw(request, call_next):
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
    app.include_router(outreach_router_mod.router)

    env = {
        "ILGA_DB_PATH": str(test_db_path),
        "ILGA_AUTH_SECRET": "test-secret",
        "ILGA_PROFILE": "dev",
    }
    with patch.dict(os.environ, env, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        importlib.reload(deps_mod)
        importlib.reload(auth_router_mod)
        importlib.reload(outreach_router_mod)
        with TestClient(app, raise_server_exceptions=True) as c:
            c.get("/auth/me")
            yield c


def _data_with_csrf(client, data: dict) -> dict:
    from ilga_graph.security import CSRF_COOKIE_NAME

    out = dict(data)
    out.setdefault("csrf_token", client.cookies.get(CSRF_COOKIE_NAME, ""))
    return out


class TestOutreachRecordCampaignId:
    """POST /outreach/record with campaign_id stores it when valid."""

    def test_record_with_valid_campaign_id_stores_it(self, client, test_db_path: Path) -> None:
        from sqlalchemy import select

        import ilga_graph.config as cfg_mod
        import ilga_graph.db as db_mod
        from ilga_graph.db_models import AuthCode, Campaign, OutreachEvent

        email = "campaign@example.com"
        code = "123456"

        async def setup():
            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(cfg_mod)
                importlib.reload(db_mod)
                import hashlib
                from datetime import datetime, timedelta, timezone

                async with db_mod.async_session_factory() as session:
                    session.add(
                        AuthCode(
                            email=email,
                            code_hash=hashlib.sha256(code.encode()).hexdigest(),
                            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
                        )
                    )
                    c = Campaign(
                        title="Test Campaign",
                        message="Msg",
                        ask="Ask",
                        target_type="all",
                        is_active=True,
                    )
                    session.add(c)
                    await session.flush()
                    campaign_id = c.id
                    await session.commit()
                    return campaign_id

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            campaign_id = _run(setup())

        with patch(
            "ilga_graph.routers.auth.rate_limit_verify_code",
            return_value=True,
        ):
            client.post(
                "/auth/verify-code",
                data=_data_with_csrf(client, {"email": email, "code": code}),
            )
        with patch(
            "ilga_graph.routers.outreach.find_member_by_id",
            return_value=object(),
        ):
            resp = client.post(
                "/outreach/record",
                data=_data_with_csrf(
                    client,
                    {
                        "member_id": "m1",
                        "kind": "call",
                        "zip_code": "60601",
                        "campaign_id": str(campaign_id),
                    },
                ),
            )
        assert resp.status_code == 200
        assert resp.json().get("ok") is True

        async def check():
            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(cfg_mod)
                importlib.reload(db_mod)
                async with db_mod.async_session_factory() as session:
                    r = await session.execute(
                        select(OutreachEvent).where(OutreachEvent.member_id == "m1")
                    )
                    ev = r.scalar_one_or_none()
                    assert ev is not None
                    assert ev.campaign_id == campaign_id

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            _run(check())

    def test_record_without_campaign_id_stores_null(self, client, test_db_path: Path) -> None:
        from sqlalchemy import select

        import ilga_graph.config as cfg_mod
        import ilga_graph.db as db_mod
        from ilga_graph.db_models import AuthCode, OutreachEvent

        email = "nocamp@example.com"
        code = "654321"

        async def setup():
            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(cfg_mod)
                importlib.reload(db_mod)
                import hashlib
                from datetime import datetime, timedelta, timezone

                async with db_mod.async_session_factory() as session:
                    session.add(
                        AuthCode(
                            email=email,
                            code_hash=hashlib.sha256(code.encode()).hexdigest(),
                            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
                        )
                    )
                    await session.commit()

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            _run(setup())

        with patch(
            "ilga_graph.routers.auth.rate_limit_verify_code",
            return_value=True,
        ):
            client.post(
                "/auth/verify-code",
                data=_data_with_csrf(client, {"email": email, "code": code}),
            )
        with patch(
            "ilga_graph.routers.outreach.find_member_by_id",
            return_value=object(),
        ):
            resp = client.post(
                "/outreach/record",
                data=_data_with_csrf(
                    client,
                    {
                        "member_id": "m2",
                        "kind": "email",
                        "zip_code": "60602",
                    },
                ),
            )
        assert resp.status_code == 200

        async def check():
            with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
                importlib.reload(cfg_mod)
                importlib.reload(db_mod)
                async with db_mod.async_session_factory() as session:
                    r = await session.execute(
                        select(OutreachEvent).where(OutreachEvent.member_id == "m2")
                    )
                    ev = r.scalar_one_or_none()
                    assert ev is not None
                    assert ev.campaign_id is None

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            _run(check())
