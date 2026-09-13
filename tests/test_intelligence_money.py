"""Intelligence campaign-finance UI: /intelligence/money, member trail, bill context."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ilga_graph.campaign_finance.ingest import build_index
from ilga_graph.campaign_finance.match import match_committees_to_members
from ilga_graph.campaign_finance.parse import parse_sbe_dir
from ilga_graph.intelligence_helpers import (
    bill_money_context_view,
    campaign_finance_summary_view,
    contributor_matches_member,
    format_ilga_action_text,
    is_sample_scale_finance,
    member_glance_narrative,
    member_money_trail_view,
    top_funded_member_rows,
)
from ilga_graph.models import Bill, Member

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sbe"


def _member(
    member_id: str,
    name: str,
    chamber: str,
    district: str,
    party: str = "Democrat",
) -> Member:
    return Member(
        id=member_id,
        name=name,
        member_url=f"https://www.ilga.gov/{chamber}/Members/Details/{member_id}",
        chamber=chamber,
        party=party,
        district=district,
        bio_text="",
    )


@pytest.fixture()
def sitting_members() -> list[Member]:
    return [
        _member("3264", "Kimberly A. Lightford", "Senate", "4"),
        _member("3268", "Don Harmon", "Senate", "39"),
        _member("9990", "No Match Person", "Senate", "99"),
    ]


@pytest.fixture()
def finance_index(sitting_members: list[Member]):
    parsed = parse_sbe_dir(FIXTURE_DIR)
    report = match_committees_to_members(parsed, sitting_members)
    return build_index(parsed, report, window_start="2025-01-01")


def _money_test_client() -> Iterator[TestClient]:
    """Yield a client with lifespan, then restore leaked AppState fields.

    TestClient context runs startup, which assigns ``state.zip_to_district``.
    Later poll tests treat any 5-digit ZIP as valid only when that map is empty;
    leaking the mock crosswalk makes ZIP 60001 fail (not in mocks/dev).
    """
    from ilga_graph.app_state import state as app_state

    prior_zip = dict(app_state.zip_to_district)
    prior_cf = app_state.campaign_finance
    with patch.dict(os.environ, {"ILGA_PROFILE": "dev", "ILGA_API_KEY": ""}, clear=False):
        import importlib

        import ilga_graph.config as _cfg_mod
        import ilga_graph.main as _main_mod

        importlib.reload(_cfg_mod)
        importlib.reload(_main_mod)
        try:
            with TestClient(_main_mod.app, raise_server_exceptions=False) as test_client:
                from ilga_graph.campaign_finance.load import (
                    attach_index,
                    load_campaign_finance_index,
                )

                if app_state.campaign_finance is None:
                    attach_index(app_state, load_campaign_finance_index(Path("mocks/dev")))
                yield test_client
        finally:
            app_state.zip_to_district = prior_zip
            app_state.campaign_finance = prior_cf


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield from _money_test_client()


def test_money_client_restores_zip_to_district() -> None:
    """Lifespan must not leave zip_to_district populated for later modules."""
    from ilga_graph.app_state import state as app_state
    from ilga_graph.routers.updates import _normalize_poll_zip

    prior = dict(app_state.zip_to_district)
    loaded_during = None
    for _client in _money_test_client():
        loaded_during = dict(app_state.zip_to_district)
        assert loaded_during, "dev lifespan should load the ZIP crosswalk"
        assert "60001" not in loaded_during
    assert app_state.zip_to_district == prior
    assert _normalize_poll_zip("60001") == "60001"
    assert loaded_during is not None


def test_campaign_finance_summary_view_success(finance_index) -> None:
    view = campaign_finance_summary_view(finance_index)
    assert view is not None
    assert view["window_start"] == "2025-01-01"
    assert view["members_matched"] > 0
    assert view["receipts_indexed"] > 0
    assert view["match_rate_pct"] > 0
    assert view["sample_scale"] is True


def test_campaign_finance_summary_view_none_when_index_missing() -> None:
    assert campaign_finance_summary_view(None) is None


def test_member_money_trail_view_success(finance_index, sitting_members: list[Member]) -> None:
    lightford = next(m for m in sitting_members if m.id == "3264")
    view = member_money_trail_view(finance_index, lightford, limit=5)
    assert view is not None
    assert view["member_id"] == "3264"
    assert view["receipt_count"] > 0
    assert view["total_received"] > 0
    assert view["committees"][0]["sbe_committee_id"] == "13872"
    assert view["top_donors"]
    assert view["recent_receipts"]


def test_member_money_trail_view_marks_self_transfer(
    finance_index, sitting_members: list[Member]
) -> None:
    harmon = next(m for m in sitting_members if m.id == "3268")
    view = member_money_trail_view(finance_index, harmon, limit=25)
    assert view is not None
    self_receipts = [r for r in view["recent_receipts"] if r["is_self"]]
    self_donors = [d for d in view["top_donors"] if d["is_self"]]
    assert self_receipts or self_donors


def test_member_money_trail_view_does_not_mark_unrelated_donors(
    finance_index, sitting_members: list[Member]
) -> None:
    lightford = next(m for m in sitting_members if m.id == "3264")
    view = member_money_trail_view(finance_index, lightford, limit=25)
    assert view is not None
    assert any(not r["is_self"] for r in view["recent_receipts"])


def test_member_money_trail_view_none_when_index_missing(
    sitting_members: list[Member],
) -> None:
    assert member_money_trail_view(None, sitting_members[0]) is None


def test_bill_money_context_view_success(finance_index, sitting_members: list[Member]) -> None:
    bill = Bill(
        bill_number="SB0001",
        leg_id="1",
        description="Test bill",
        chamber="S",
        last_action="First Reading",
        last_action_date="1/1/2025",
        primary_sponsor="Kimberly A. Lightford",
        sponsor_ids=["3264", "3268"],
    )
    members_by_id = {m.id: m for m in sitting_members}
    view = bill_money_context_view(finance_index, bill, members_by_id, limit=5)
    assert view is not None
    assert view["bill_number"] == "SB0001"
    assert "not earmarked" in view["match_notes"].lower()
    assert view["total_received_across_sponsors"] > 0
    assert {s["member_id"] for s in view["sponsor_trails"]} >= {"3264", "3268"}


def test_bill_money_context_view_none_when_index_missing(
    sitting_members: list[Member],
) -> None:
    bill = Bill(
        bill_number="SB0001",
        leg_id="1",
        description="Test bill",
        chamber="S",
        last_action="First Reading",
        last_action_date="1/1/2025",
        primary_sponsor="Kimberly A. Lightford",
        sponsor_ids=["3264"],
    )
    assert bill_money_context_view(None, bill, {m.id: m for m in sitting_members}) is None


def test_top_funded_member_rows_ranks_by_receipts(
    finance_index, sitting_members: list[Member]
) -> None:
    rows = top_funded_member_rows(finance_index, {m.id: m for m in sitting_members}, limit=10)
    assert rows
    assert rows[0]["total_received"] >= rows[-1]["total_received"]
    assert {r["member_id"] for r in rows} <= {"3264", "3268"}
    assert all(r["receipt_count"] > 0 and r["total_received"] > 0 for r in rows)


def test_money_page_returns_200(client: TestClient) -> None:
    resp = client.get("/intelligence/money", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "Follow the money" in body
    assert "campaign finance" in body.lower()
    assert "Moneyball" in body


def test_money_page_shows_dev_fixture_summary(client: TestClient) -> None:
    resp = client.get("/intelligence/money", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "2025-01-01" in body
    assert "not earmarked" in body.lower()
    assert "Lightford" in body or "Harmon" in body
    assert "Sample-scale" in body
    assert "dev fixture" not in body.lower()
    assert "CIVIL LAW-TECH" in body
    assert "statewide disclosure" in body.lower()
    assert 'name="email"' in body
    assert "Get updates as this grows" in body


def test_money_page_bill_query_shows_context(client: TestClient) -> None:
    resp = client.get("/intelligence/money?bill=SB0341", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "SB0341" in body
    assert "not earmarked" in body.lower()
    assert "Don Harmon" in body or "Harmon" in body


def test_money_page_empty_state_when_index_missing(client: TestClient) -> None:
    from ilga_graph.app_state import state as app_state

    with patch.object(app_state, "campaign_finance", None):
        resp = client.get("/intelligence/money", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "not loaded" in resp.text.lower() or "No campaign-finance" in resp.text


def test_member_page_includes_money_trail(client: TestClient) -> None:
    resp = client.get("/intelligence/member/3268", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "Campaign finance" in body
    assert "Moneyball" in body
    assert "Citizens for Harmon" in body or "Harmon" in body
    assert "Top donors" in body or "top donor" in body.lower()


def test_member_page_money_empty_when_index_missing(client: TestClient) -> None:
    from ilga_graph.app_state import state as app_state

    with patch.object(app_state, "campaign_finance", None):
        resp = client.get("/intelligence/member/3268", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "Campaign finance" in resp.text
    assert "not loaded" in resp.text.lower() or "No campaign-finance" in resp.text


def test_bill_page_includes_money_context(client: TestClient) -> None:
    resp = client.get("/intelligence/bill/SB0341", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "SB0341" in body
    assert "not earmarked" in body.lower()
    assert "Campaign finance" in body


def test_intelligence_summary_links_to_money(client: TestClient) -> None:
    resp = client.get("/intelligence/", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "/intelligence/money" in resp.text
    assert "Follow the money" in resp.text
    assert "Sample-scale" in resp.text
    assert "dev fixture" not in resp.text.lower()


def test_is_sample_scale_finance_success() -> None:
    assert is_sample_scale_finance(15, 140) is True


def test_is_sample_scale_finance_failure_statewide() -> None:
    assert is_sample_scale_finance(177, 20_000) is False


def test_format_ilga_action_text_success() -> None:
    assert (
        format_ilga_action_text("Filed with Secretary bySen. Don Harmon")
        == "Filed with Secretary by Sen. Don Harmon"
    )
    assert format_ilga_action_text("Referred toAssignments") == "Referred to Assignments"
    assert (
        format_ilga_action_text("Do PassExecutive;  011-000-000")
        == "Do Pass Executive; 011-000-000"
    )


def test_format_ilga_action_text_failure_noop() -> None:
    assert format_ilga_action_text("") == ""
    assert format_ilga_action_text("First Reading") == "First Reading"
    assert format_ilga_action_text("Referred to Assignments") == "Referred to Assignments"


def test_contributor_matches_member_success() -> None:
    assert contributor_matches_member("Donald F Harmon", "Don Harmon") is True


def test_contributor_matches_member_failure() -> None:
    assert contributor_matches_member("JPMorganChase", "Don Harmon") is False
    assert contributor_matches_member("Don Johnson", "Don Harmon") is False
    assert contributor_matches_member("", "Don Harmon") is False


def test_member_glance_narrative_success() -> None:
    text = member_glance_narrative(
        "Don Harmon",
        rank_overall=5,
        influence_label="Low",
        laws_passed=3,
        effectiveness_rate=0.4,
    )
    assert text is not None
    assert "Don Harmon ranks #5" in text
    assert "has passed 3 laws" in text
    assert "They" not in text


def test_member_glance_narrative_failure_empty() -> None:
    assert member_glance_narrative("Don Harmon") is None


def test_member_page_demo_grammar_and_self_transfer(client: TestClient) -> None:
    resp = client.get("/intelligence/member/3268", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "They have passed" not in body
    assert "Influence#" not in body
    assert "influence#" not in body
    assert "Self-transfer" in body


def test_bill_page_demo_action_spacing_and_drivers(client: TestClient) -> None:
    resp = client.get("/intelligence/bill/SB0341", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    assert "Calculating prediction drivers" not in body
    assert "Prediction drivers are not available" in body
    assert "bySen." not in body
    assert "toAssignments" not in body
    assert "Do PassExecutive" not in body
    assert "by Sen." in body or "to Assignments" in body or "Do Pass Executive" in body


def test_bill_explanation_fragment_finished_empty(client: TestClient) -> None:
    resp = client.get("/api/bills/SB0341/explanation", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "Calculating prediction drivers" not in resp.text
    assert "Prediction drivers are not available" in resp.text
    assert "pip install" not in resp.text
