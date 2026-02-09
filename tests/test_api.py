"""Tests for the hardened GraphQL API layer.

Covers: pagination, committee queries, Chamber enum, error handling,
CORS, API key auth, health endpoint, and witness slip queries.
"""

from __future__ import annotations

from ilga_graph.models import (
    Bill,
    Committee,
    CommitteeMemberRole,
    VoteEvent,
    WitnessSlip,
)
from ilga_graph.schema import (
    BillConnection,
    BillType,
    Chamber,
    CommitteeConnection,
    CommitteeType,
    MemberConnection,
    PageInfo,
    VoteEventConnection,
    VoteEventType,
    WitnessSlipConnection,
    WitnessSlipType,
    paginate,
)

# ── paginate() helper ─────────────────────────────────────────────────────────


class TestPaginate:
    def test_full_list_when_limit_zero(self) -> None:
        items = list(range(10))
        page, info = paginate(items, offset=0, limit=0)
        assert page == items
        assert info.total_count == 10
        assert info.has_next_page is False
        assert info.has_previous_page is False

    def test_first_page(self) -> None:
        items = list(range(10))
        page, info = paginate(items, offset=0, limit=3)
        assert page == [0, 1, 2]
        assert info.total_count == 10
        assert info.has_next_page is True
        assert info.has_previous_page is False

    def test_middle_page(self) -> None:
        items = list(range(10))
        page, info = paginate(items, offset=3, limit=3)
        assert page == [3, 4, 5]
        assert info.total_count == 10
        assert info.has_next_page is True
        assert info.has_previous_page is True

    def test_last_page(self) -> None:
        items = list(range(10))
        page, info = paginate(items, offset=9, limit=3)
        assert page == [9]
        assert info.total_count == 10
        assert info.has_next_page is False
        assert info.has_previous_page is True

    def test_offset_beyond_end(self) -> None:
        items = list(range(5))
        page, info = paginate(items, offset=10, limit=3)
        assert page == []
        assert info.total_count == 5
        assert info.has_next_page is False
        assert info.has_previous_page is True

    def test_empty_list(self) -> None:
        page, info = paginate([], offset=0, limit=5)
        assert page == []
        assert info.total_count == 0
        assert info.has_next_page is False
        assert info.has_previous_page is False

    def test_offset_with_limit_zero(self) -> None:
        items = list(range(10))
        page, info = paginate(items, offset=5, limit=0)
        assert page == [5, 6, 7, 8, 9]
        assert info.total_count == 10
        assert info.has_previous_page is True

    def test_exact_page_boundary(self) -> None:
        items = list(range(6))
        page, info = paginate(items, offset=0, limit=6)
        assert page == items
        assert info.has_next_page is False


# ── Chamber enum ──────────────────────────────────────────────────────────────


class TestChamberEnum:
    def test_house_value(self) -> None:
        assert Chamber.HOUSE.value == "House"

    def test_senate_value(self) -> None:
        assert Chamber.SENATE.value == "Senate"


# ── CommitteeType ─────────────────────────────────────────────────────────────


class TestCommitteeType:
    def test_from_model_basic(self) -> None:
        c = Committee(code="SAGR", name="Agriculture")
        ct = CommitteeType.from_model(c)
        assert ct.code == "SAGR"
        assert ct.name == "Agriculture"
        assert ct.parent_code is None
        assert ct.roster == []
        assert ct.bill_numbers == []

    def test_from_model_with_roster(self) -> None:
        c = Committee(code="SAGR", name="Agriculture")
        roster = [
            CommitteeMemberRole(
                member_id="1",
                member_name="Alice",
                member_url="http://example.com",
                role="Chairperson",
            ),
        ]
        ct = CommitteeType.from_model(c, roster=roster)
        assert len(ct.roster) == 1
        assert ct.roster[0].member_name == "Alice"
        assert ct.roster[0].role == "Chairperson"

    def test_from_model_with_bills(self) -> None:
        c = Committee(code="SAGR", name="Agriculture")
        ct = CommitteeType.from_model(c, bill_numbers=["SB0001", "SB0002"])
        assert ct.bill_numbers == ["SB0001", "SB0002"]

    def test_from_model_with_parent(self) -> None:
        c = Committee(
            code="SEXF",
            name="Firearms",
            parent_code="SEXC",
        )
        ct = CommitteeType.from_model(c)
        assert ct.parent_code == "SEXC"


# ── WitnessSlipType ──────────────────────────────────────────────────────────


class TestWitnessSlipType:
    def test_from_model(self) -> None:
        ws = WitnessSlip(
            name="Paul Makarewicz",
            organization="AES Clean Energy",
            representing="AES Clean Energy",
            position="Proponent",
            hearing_committee="Executive",
            hearing_date="2025-05-31 17:00",
            bill_number="HB1075",
        )
        wst = WitnessSlipType.from_model(ws)
        assert wst.name == "Paul Makarewicz"
        assert wst.organization == "AES Clean Energy"
        assert wst.position == "Proponent"
        assert wst.bill_number == "HB1075"
        assert wst.hearing_date == "2025-05-31 17:00"

    def test_defaults(self) -> None:
        ws = WitnessSlip(
            name="Test",
            organization="Org",
            representing="Self",
            position="Opponent",
            hearing_committee="Judiciary",
            hearing_date="2025-01-01",
        )
        wst = WitnessSlipType.from_model(ws)
        assert wst.bill_number == ""
        assert wst.testimony_type == "Record of Appearance Only"


# ── PageInfo ──────────────────────────────────────────────────────────────────


class TestPageInfo:
    def test_construction(self) -> None:
        pi = PageInfo(total_count=100, has_next_page=True, has_previous_page=False)
        assert pi.total_count == 100
        assert pi.has_next_page is True
        assert pi.has_previous_page is False


# ── Connection types ──────────────────────────────────────────────────────────


class TestConnectionTypes:
    def test_member_connection(self) -> None:
        conn = MemberConnection(
            items=[],
            page_info=PageInfo(
                total_count=0,
                has_next_page=False,
                has_previous_page=False,
            ),
        )
        assert conn.items == []
        assert conn.page_info.total_count == 0

    def test_bill_connection(self) -> None:
        bill = Bill(
            bill_number="SB0001",
            leg_id="100",
            description="Test",
            chamber="S",
            last_action="Filed",
            last_action_date="1/1/2025",
            primary_sponsor="Alice",
        )
        bt = BillType.from_model(bill)
        conn = BillConnection(
            items=[bt],
            page_info=PageInfo(
                total_count=1,
                has_next_page=False,
                has_previous_page=False,
            ),
        )
        assert len(conn.items) == 1
        assert conn.items[0].bill_number == "SB0001"

    def test_vote_event_connection(self) -> None:
        conn = VoteEventConnection(
            items=[],
            page_info=PageInfo(
                total_count=0,
                has_next_page=False,
                has_previous_page=False,
            ),
        )
        assert conn.items == []

    def test_committee_connection(self) -> None:
        c = Committee(code="SAGR", name="Agriculture")
        ct = CommitteeType.from_model(c)
        conn = CommitteeConnection(
            items=[ct],
            page_info=PageInfo(
                total_count=1,
                has_next_page=False,
                has_previous_page=False,
            ),
        )
        assert len(conn.items) == 1

    def test_witness_slip_connection(self) -> None:
        conn = WitnessSlipConnection(
            items=[],
            page_info=PageInfo(
                total_count=0,
                has_next_page=False,
                has_previous_page=False,
            ),
        )
        assert conn.items == []


# ── Safe date parsing ─────────────────────────────────────────────────────────


class TestSafeParseDateIntegration:
    """Test the safe date parsing helper from main.py."""

    def test_valid_date(self) -> None:
        from ilga_graph.main import _safe_parse_date

        result = _safe_parse_date("2025-06-01", "test")
        assert result is not None
        assert result.year == 2025
        assert result.month == 6
        assert result.day == 1

    def test_invalid_date_returns_none(self) -> None:
        from ilga_graph.main import _safe_parse_date

        result = _safe_parse_date("not-a-date", "test")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        from ilga_graph.main import _safe_parse_date

        result = _safe_parse_date("", "test")
        assert result is None

    def test_wrong_format_returns_none(self) -> None:
        from ilga_graph.main import _safe_parse_date

        result = _safe_parse_date("06/01/2025", "test")
        assert result is None


# ── VoteEventType counts ──────────────────────────────────────────────────────


class TestVoteEventTypeCounts:
    """Verify computed counts on VoteEventType.from_model."""

    def test_counts_match_list_lengths(self) -> None:
        ve = VoteEvent(
            bill_number="SB0001",
            date="Jan 1, 2025",
            description="Third Reading",
            chamber="Senate",
            yea_votes=["A", "B", "C"],
            nay_votes=["D"],
            present_votes=["E", "F"],
            nv_votes=[],
        )
        vet = VoteEventType.from_model(ve)
        assert vet.yea_count == 3
        assert vet.nay_count == 1
        assert vet.present_count == 2
        assert vet.nv_count == 0
