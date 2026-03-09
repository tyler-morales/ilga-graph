"""Tests for advocacy page call-hours banner (outside 9am–5pm notice)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ilga_graph.constants import ADV_CALL_PREF_COOKIE
from ilga_graph.zip_crosswalk import ZipDistrictInfo


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


# ZIP that must be in state for advocacy to return results (and thus user_call_pref in context).
TEST_ZIP = "60007"
# Banner id; also in JS (getElementById). "Absent" tests check for this HTML attribute.
BANNER_ID_ATTR = 'id="advocacy-call-hours-banner"'
BANNER_SNIPPET = "Outside typical office hours"


def test_advocacy_call_hours_banner_present_when_call_pref_yes(client: TestClient) -> None:
    """With cookie adv_call_pref=yes and a valid ZIP, response includes the call-hours banner."""
    from ilga_graph import app_state

    # Ensure this ZIP is in state so we get results and user_call_pref in context.
    with patch.dict(
        app_state.state.zip_to_district,
        {TEST_ZIP: ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")},
        clear=False,
    ):
        resp = client.get(
            "/advocacy/",
            params={"zip": TEST_ZIP},
            cookies={ADV_CALL_PREF_COOKIE: "yes"},
            headers={"Accept": "text/html"},
        )
    assert resp.status_code == 200
    assert BANNER_ID_ATTR in resp.text
    assert BANNER_SNIPPET in resp.text


def test_advocacy_call_hours_banner_present_when_call_pref_call_only(client: TestClient) -> None:
    """With cookie adv_call_pref=call_only, response includes the call-hours banner."""
    from ilga_graph import app_state

    with patch.dict(
        app_state.state.zip_to_district,
        {TEST_ZIP: ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")},
        clear=False,
    ):
        resp = client.get(
            "/advocacy/",
            params={"zip": TEST_ZIP},
            cookies={ADV_CALL_PREF_COOKIE: "call_only"},
            headers={"Accept": "text/html"},
        )
    assert resp.status_code == 200
    assert BANNER_ID_ATTR in resp.text


def test_advocacy_call_hours_banner_present_when_call_pref_elevator(client: TestClient) -> None:
    """With cookie adv_call_pref=elevator, response includes the call-hours banner."""
    from ilga_graph import app_state

    with patch.dict(
        app_state.state.zip_to_district,
        {TEST_ZIP: ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")},
        clear=False,
    ):
        resp = client.get(
            "/advocacy/",
            params={"zip": TEST_ZIP},
            cookies={ADV_CALL_PREF_COOKIE: "elevator"},
            headers={"Accept": "text/html"},
        )
    assert resp.status_code == 200
    assert BANNER_ID_ATTR in resp.text


def test_advocacy_call_hours_banner_absent_when_email_only(client: TestClient) -> None:
    """With cookie adv_call_pref=no, response does not include the call-hours banner."""
    from ilga_graph import app_state

    with patch.dict(
        app_state.state.zip_to_district,
        {TEST_ZIP: ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")},
        clear=False,
    ):
        resp = client.get(
            "/advocacy/",
            params={"zip": TEST_ZIP},
            cookies={ADV_CALL_PREF_COOKIE: "no"},
            headers={"Accept": "text/html"},
        )
    assert resp.status_code == 200
    # Banner div must not be present (id appears in JS, so check for the HTML attribute).
    assert BANNER_ID_ATTR not in resp.text


def test_advocacy_call_hours_banner_absent_when_no_pref_cookie(client: TestClient) -> None:
    """With no preference cookie, response does not include the call-hours banner."""
    from ilga_graph import app_state

    with patch.dict(
        app_state.state.zip_to_district,
        {TEST_ZIP: ZipDistrictInfo(il_house="9", il_senate="5", us_house="4")},
        clear=False,
    ):
        resp = client.get(
            "/advocacy/",
            params={"zip": TEST_ZIP},
            headers={"Accept": "text/html"},
        )
    assert resp.status_code == 200
    assert BANNER_ID_ATTR not in resp.text
