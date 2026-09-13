"""Buyer money portal: /money, /money/demo, member/bill views, sample vs statewide."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ilga_graph.intelligence_helpers import is_sample_scale_finance
from tests.test_intelligence_money import _money_test_client


@pytest.fixture
def client() -> TestClient:
    yield from _money_test_client()


def test_portal_uses_existing_sample_scale_helper() -> None:
    assert is_sample_scale_finance({"members_matched": 15, "receipts_indexed": 140}) is True
    assert is_sample_scale_finance({"members_matched": 160, "receipts_indexed": 40_000}) is False


def test_landing_returns_200_and_is_its_own_product(client: TestClient) -> None:
    resp = client.get("/money", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "Illinois Influence" in body
    assert "decision under a deadline" in body
    assert "Open the demo" in body
    assert "Join the waitlist" in body
    assert "not earmarked" in body.lower()
    assert "Moneyball" in body
    assert "lobbyist-registration" in body
    assert "Land of Kei" not in body
    assert "kei vehicle" not in body.lower()
    assert "advocacy-form" not in body
    assert "intelligence-dashboard" not in body
    assert "message-marquee" not in body
    assert "/static/css/money-portal" in body
    assert "SOS" not in body


def test_demo_returns_200_with_live_kpis(client: TestClient) -> None:
    resp = client.get("/money/demo", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "2025-01-01" in body
    assert "not earmarked" in body.lower()
    assert "Watch" in body
    assert "Ask" in body
    assert "Flag" in body
    assert "sample-scale" in body.lower()
    assert "Harmon" in body or "3268" in body
    assert "/money/member/" in body


def test_demo_bill_query_shows_not_earmarked_context(client: TestClient) -> None:
    resp = client.get("/money/demo?bill=SB0341", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "SB0341" in body
    assert "not earmarked" in body.lower()
    assert "Harmon" in body or "Don Harmon" in body
    assert "/money/bill/SB0341" in body


def test_demo_member_query_redirects_to_member_page(client: TestClient) -> None:
    resp = client.get("/money/demo?member=3268", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers.get("location") == "/money/member/3268"


def test_demo_sample_banner_when_sample_scale(client: TestClient) -> None:
    resp = client.get("/money/demo", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "sample-scale" in resp.text.lower()


def test_demo_statewide_copy_omits_sample_banner(client: TestClient) -> None:
    statewide = {
        "source": "https://downloads.elections.il.gov",
        "window_start": "2025-01-01",
        "window_end": "2026-09-13",
        "generated_at": "2026-09-13T00:00:00+00:00",
        "legislative_committees": 200,
        "accepted": 180,
        "review": 5,
        "match_rate": 0.9,
        "match_rate_pct": 90.0,
        "members_matched": 160,
        "receipts_indexed": 40_000,
        "is_sample_scale": False,
    }
    with (
        patch(
            "ilga_graph.routers.money_portal.campaign_finance_summary_view",
            return_value=statewide,
        ),
        patch("ilga_graph.routers.money_portal.is_sample_scale_finance", return_value=False),
    ):
        resp = client.get("/money/demo", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "sample-scale" not in resp.text.lower()
    assert "40000" in resp.text.replace(",", "")
    assert "160" in resp.text


def test_member_page_reuses_trail_for_harmon(client: TestClient) -> None:
    resp = client.get("/money/member/3268", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "Harmon" in body
    assert "Campaign finance" in body
    assert "not Moneyball" in body or "Moneyball" in body
    assert "Citizens for Harmon" in body or "Friends of Don Harmon" in body or "Harmon" in body
    assert "Land of Kei" not in body
    assert "/money/demo" in body


def test_bill_page_reuses_context_for_sb0341(client: TestClient) -> None:
    resp = client.get("/money/bill/SB0341", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "SB0341" in body
    assert "not earmarked" in body.lower()
    assert "Harmon" in body or "Don Harmon" in body
    assert "Land of Kei" not in body


def test_intelligence_engine_still_200(client: TestClient) -> None:
    resp = client.get("/intelligence/money", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "Follow the money" in resp.text
    assert "Bill money context" in resp.text
    assert "money-portal" not in resp.text


def test_legacy_signup_redirects_on_full_app(client: TestClient) -> None:
    resp = client.get("/intelligence/money/signup", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers.get("location") == "/money/signup"
