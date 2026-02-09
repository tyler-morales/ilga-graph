"""Tests for the witness-slip export parser and influence-layer analytics."""

from __future__ import annotations

import pytest

from ilga_graph.analytics import controversial_score, lobbyist_alignment
from ilga_graph.models import Bill, Member, WitnessSlip
from ilga_graph.scrapers.witness_slips import (
    _extract_leg_doc_ids,
    _parse_export_text,
)

# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_EXPORT = (
    "Legislation|Name|Firm|Representation|Position|Committee|ScheduledDateTime\n"
    "HB1075|Cecilia Tian|Recurrent Energy||Proponent|Executive|2025-05-31 17:00\n"
    "HB1075|Joseph Hus|Naperville Environment and Sustainability Task Force||"
    "Proponent|Executive|2025-05-31 17:00\n"
    "HB1075|Paul Makarewicz|AES Clean Energy|AES Clean Energy|Proponent|Executive|"
    "2025-05-31 17:00\n"
    "HB1075|Gowri Magati|Lake County Indians Association|Lake County Indians Association|"
    "Proponent|State Government Administration|2025-03-05 14:00\n"
    "HB1075|David Schwartz|Self|Self|Proponent|State Government Administration|"
    "2025-02-19 14:30\n"
    "HB1075|Maura Freeman|Illinois Association of Park Districts|"
    "Illinois Association of Park Districts|Opponent|Executive|2025-05-31 17:00\n"
)

SAMPLE_WITNESS_SLIPS_PAGE = (
    "<html><body>\n"
    '<a href="/Legislation/BillStatus/WitnessSlips?LegDocId=196535&DocNum=1075&'
    'DocTypeID=HB&LegID=156769&GAID=18&SessionID=114">HB1075</a>\n'
    '<a href="/Legislation/BillStatus/WitnessSlips?LegDocId=204772&DocNum=1075&'
    'DocTypeID=HB&LegID=156769&GAID=18&SessionID=114">Senate Amendment 001</a>\n'
    '<a href="/Legislation/BillStatus/VoteHistory?GAID=18&DocNum=1075">Votes</a>\n'
    "</body></html>\n"
)


# ── _parse_export_text ────────────────────────────────────────────────────────


class TestParseExportText:
    def test_parses_all_rows(self) -> None:
        slips = _parse_export_text(SAMPLE_EXPORT)
        assert len(slips) == 6

    def test_field_mapping(self) -> None:
        slips = _parse_export_text(SAMPLE_EXPORT)
        paul = [s for s in slips if s.name == "Paul Makarewicz"][0]
        assert paul.bill_number == "HB1075"
        assert paul.organization == "AES Clean Energy"
        assert paul.representing == "AES Clean Energy"
        assert paul.position == "Proponent"
        assert paul.hearing_committee == "Executive"
        assert paul.hearing_date == "2025-05-31 17:00"
        assert paul.testimony_type == "Record of Appearance Only"

    def test_empty_representing(self) -> None:
        slips = _parse_export_text(SAMPLE_EXPORT)
        cecilia = [s for s in slips if s.name == "Cecilia Tian"][0]
        assert cecilia.representing == ""

    def test_opponent(self) -> None:
        slips = _parse_export_text(SAMPLE_EXPORT)
        maura = [s for s in slips if s.name == "Maura Freeman"][0]
        assert maura.position == "Opponent"
        assert maura.organization == "Illinois Association of Park Districts"

    def test_empty_input(self) -> None:
        assert _parse_export_text("") == []

    def test_header_only(self) -> None:
        header = "Legislation|Name|Firm|Representation|Position|Committee|ScheduledDateTime\n"
        assert _parse_export_text(header) == []

    def test_malformed_line_skipped(self) -> None:
        bad = (
            "Legislation|Name|Firm|Representation|Position|Committee|ScheduledDateTime\n"
            "HB1075|Only Two Fields\n"
        )
        slips = _parse_export_text(bad)
        assert len(slips) == 0


# ── _extract_leg_doc_ids ──────────────────────────────────────────────────────


class TestExtractLegDocIds:
    def test_extracts_ids(self) -> None:
        ids = _extract_leg_doc_ids(SAMPLE_WITNESS_SLIPS_PAGE, "https://www.ilga.gov/")
        assert len(ids) == 2
        assert ids[0] == ("196535", "HB1075")
        assert ids[1] == ("204772", "Senate Amendment 001")

    def test_ignores_non_witness_links(self) -> None:
        ids = _extract_leg_doc_ids(SAMPLE_WITNESS_SLIPS_PAGE, "https://www.ilga.gov/")
        labels = [label for _, label in ids]
        assert "Votes" not in labels

    def test_deduplicates(self) -> None:
        html = """\
        <html><body>
        <a href="/WitnessSlips?LegDocId=100">Bill</a>
        <a href="/WitnessSlips?LegDocId=100">Bill</a>
        </body></html>
        """
        ids = _extract_leg_doc_ids(html, "https://www.ilga.gov/")
        assert len(ids) == 1

    def test_empty_html(self) -> None:
        assert _extract_leg_doc_ids("<html></html>", "https://www.ilga.gov/") == []


# ── lobbyist_alignment ────────────────────────────────────────────────────────


def _make_member_with_bills(bill_numbers: list[str]) -> Member:
    """Helper: create a member whose sponsored_bills match given bill numbers."""
    bills = [
        Bill(
            bill_number=bn, leg_id=str(i), description="test",
            chamber="H", last_action="In Progress",
            last_action_date="1/1/2025", primary_sponsor="Test Sponsor",
        )
        for i, bn in enumerate(bill_numbers)
    ]
    return Member(
        id="9000", name="Test Sponsor",
        member_url="http://example.com", chamber="House",
        party="Democrat", district="1", bio_text="test",
        sponsored_bills=bills, co_sponsor_bills=[],
    )


class TestLobbyistAlignment:
    def test_counts_proponent_orgs(self) -> None:
        member = _make_member_with_bills(["HB1075"])
        slips = _parse_export_text(SAMPLE_EXPORT)
        result = lobbyist_alignment(slips, member)
        # 4 orgs filed as proponents (Self excluded because it has org="Self")
        # Recurrent Energy, Naperville..., AES Clean Energy,
        # Lake County Indians Association, Self (org="Self")
        assert "AES Clean Energy" in result
        assert result["AES Clean Energy"] == 1
        assert "Recurrent Energy" in result
        # Opponent should NOT be included
        assert "Illinois Association of Park Districts" not in result

    def test_empty_org_excluded(self) -> None:
        member = _make_member_with_bills(["HB9999"])
        slip = WitnessSlip(
            name="Nobody", organization="", representing="",
            position="Proponent", hearing_committee="Test",
            hearing_date="2025-01-01", bill_number="HB9999",
        )
        result = lobbyist_alignment([slip], member)
        assert result == {}

    def test_no_matching_bills(self) -> None:
        member = _make_member_with_bills(["SB9999"])
        slips = _parse_export_text(SAMPLE_EXPORT)
        result = lobbyist_alignment(slips, member)
        assert result == {}

    def test_normalises_bill_numbers(self) -> None:
        """HB0100 on the member should match HB100 on the slip."""
        member = _make_member_with_bills(["HB0100"])
        slip = WitnessSlip(
            name="Test", organization="BigCorp", representing="BigCorp",
            position="Proponent", hearing_committee="Finance",
            hearing_date="2025-01-01", bill_number="HB100",
        )
        result = lobbyist_alignment([slip], member)
        assert result == {"BigCorp": 1}

    def test_sorted_descending(self) -> None:
        member = _make_member_with_bills(["HB1"])
        slips = [
            WitnessSlip(
                name="A", organization="OrgA", representing="",
                position="Proponent", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB1",
            ),
            WitnessSlip(
                name="B", organization="OrgB", representing="",
                position="Proponent", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB1",
            ),
            WitnessSlip(
                name="B2", organization="OrgB", representing="",
                position="Proponent", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB1",
            ),
        ]
        result = lobbyist_alignment(slips, member)
        keys = list(result.keys())
        assert keys[0] == "OrgB"
        assert result["OrgB"] == 2
        assert result["OrgA"] == 1


# ── controversial_score ───────────────────────────────────────────────────────


class TestControversialScore:
    def test_hb1075_score(self) -> None:
        slips = _parse_export_text(SAMPLE_EXPORT)
        score = controversial_score(slips, "HB1075")
        # 5 proponents, 1 opponent → 1 / (5 + 1) = 0.1667
        assert score == pytest.approx(0.1667, abs=0.001)

    def test_all_opponents(self) -> None:
        slips = [
            WitnessSlip(
                name="A", organization="X", representing="",
                position="Opponent", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB9",
            ),
            WitnessSlip(
                name="B", organization="Y", representing="",
                position="Opponent", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB9",
            ),
        ]
        assert controversial_score(slips, "HB9") == 1.0

    def test_all_proponents(self) -> None:
        slips = [
            WitnessSlip(
                name="A", organization="X", representing="",
                position="Proponent", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB9",
            ),
        ]
        assert controversial_score(slips, "HB9") == 0.0

    def test_no_matching_bill(self) -> None:
        slips = _parse_export_text(SAMPLE_EXPORT)
        assert controversial_score(slips, "SB9999") == 0.0

    def test_no_slips(self) -> None:
        assert controversial_score([], "HB1") == 0.0

    def test_normalises_bill_number(self) -> None:
        slips = [
            WitnessSlip(
                name="A", organization="X", representing="",
                position="Proponent", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB100",
            ),
            WitnessSlip(
                name="B", organization="Y", representing="",
                position="Opponent", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB100",
            ),
        ]
        # Query with zero-padded form
        assert controversial_score(slips, "HB0100") == 0.5

    def test_ignores_no_position(self) -> None:
        slips = [
            WitnessSlip(
                name="A", organization="X", representing="",
                position="Proponent", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB5",
            ),
            WitnessSlip(
                name="B", organization="Y", representing="",
                position="No Position", hearing_committee="C",
                hearing_date="2025-01-01", bill_number="HB5",
            ),
        ]
        # Only 1 proponent, 0 opponents → 0.0
        assert controversial_score(slips, "HB5") == 0.0
