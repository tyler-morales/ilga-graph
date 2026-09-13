"""SBE campaign-finance ingest, member matching, and GraphQL money queries."""

from __future__ import annotations

from pathlib import Path

import pytest

from ilga_graph.campaign_finance.ingest import build_index, ingest_from_dir
from ilga_graph.campaign_finance.match import match_committees_to_members
from ilga_graph.campaign_finance.parse import parse_sbe_dir
from ilga_graph.campaign_finance.service import bill_money_context, member_money_trail
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
        _member("3265", "Dave Syverson", "Senate", "35", "Republican"),
        _member("3268", "Don Harmon", "Senate", "39"),
        _member("3312", "Neil Anderson", "Senate", "47", "Republican"),
        _member("3298", 'Emanuel "Chris" Welch', "House", "7"),
        _member("3281", "Sue Rezin", "Senate", "38", "Republican"),
        _member("3269", "Mattie Hunter", "Senate", "3"),
        _member("3270", "Linda Holmes", "Senate", "42"),
        _member("9990", "No Match Person", "Senate", "99"),
    ]


def test_parse_sbe_fixture_uses_official_headers() -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    assert len(parsed.committees) == 16
    assert len(parsed.candidates) >= 16
    assert len(parsed.links) >= 16
    assert len(parsed.receipts) == 160
    lightford = next(c for c in parsed.committees if c.committee_id == "13872")
    assert lightford.name == "Citizens for Lightford"
    assert lightford.type_of_committee == "Candidate"
    assert lightford.status == "A"
    assert all(not r.archived for r in parsed.receipts)
    assert all(r.amount > 0 for r in parsed.receipts)


def test_match_links_committees_to_member_ids(sitting_members: list[Member]) -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    report = match_committees_to_members(parsed, sitting_members)
    by_member = {m.member_id: m for m in report.accepted}
    assert "3264" in by_member
    assert by_member["3264"].committee_id == "13872"
    assert by_member["3264"].method in {"office_district_name", "name_chamber"}
    assert by_member["3264"].confidence >= 0.8
    assert "3268" in by_member
    assert by_member["3268"].committee_id == "16283"
    assert "3312" in by_member
    assert by_member["3312"].committee_id == "25649"
    assert "3298" in by_member
    assert by_member["3298"].committee_id == "19328"
    assert report.match_rate > 0
    assert all(m.member_id != "9990" for m in report.accepted)


def test_match_unmatched_path_for_review(sitting_members: list[Member]) -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    report = match_committees_to_members(parsed, sitting_members)
    unmatched_ids = {u.committee_id for u in report.unmatched}
    # Regan Deering (88) and several sitting members not in this member list
    assert unmatched_ids
    assert all(u.status == "review" for u in report.unmatched)


def test_member_money_trail_aggregates_real_receipts(
    sitting_members: list[Member],
) -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    report = match_committees_to_members(parsed, sitting_members)
    index = build_index(parsed, report, window_start="2025-01-01")
    lightford = next(m for m in sitting_members if m.id == "3264")
    trail = member_money_trail(index, lightford, limit=5)
    assert trail is not None
    assert trail.member_id == "3264"
    assert trail.receipt_count > 0
    assert trail.total_received > 0
    assert trail.committees[0].sbe_committee_id == "13872"
    assert trail.top_donors
    assert trail.recent_receipts
    missing = member_money_trail(index, sitting_members[-1], limit=5)
    assert missing is not None
    assert missing.receipt_count == 0
    assert missing.total_received == 0.0


def test_bill_money_context_joins_sponsors_to_money(
    sitting_members: list[Member],
) -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    report = match_committees_to_members(parsed, sitting_members)
    index = build_index(parsed, report, window_start="2025-01-01")
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
    ctx = bill_money_context(index, bill, members_by_id, vote_events=[], limit=5)
    assert ctx is not None
    assert ctx.bill_number == "SB0001"
    ids = {s.member_id for s in ctx.sponsor_trails}
    assert {"3264", "3268"} <= ids
    assert ctx.total_received_across_sponsors > 0
    assert ctx.top_donors_across_sponsors


def test_ingest_from_dir_writes_sqlite_and_index(
    sitting_members: list[Member], tmp_path: Path
) -> None:
    db_path = tmp_path / "sbe.db"
    index_path = tmp_path / "index.json"
    result = ingest_from_dir(
        FIXTURE_DIR,
        sitting_members,
        db_path=db_path,
        index_path=index_path,
        window_start="2025-01-01",
    )
    assert db_path.exists()
    assert index_path.exists()
    assert result.receipts_stored > 0
    assert result.members_matched >= 4
    assert result.match_rate > 0
    unmatched_path = tmp_path / "unmatched.json"
    assert unmatched_path.exists()


def test_graphql_member_money_trail_and_bill_context(
    sitting_members: list[Member],
) -> None:
    from ilga_graph.app_state import state
    from ilga_graph.campaign_finance.load import attach_index
    from ilga_graph.main import schema

    parsed = parse_sbe_dir(FIXTURE_DIR)
    report = match_committees_to_members(parsed, sitting_members)
    index = build_index(parsed, report, window_start="2025-01-01")
    snapshot = {
        "members": state.members,
        "member_lookup": state.member_lookup,
        "member_lookup_by_id": state.member_lookup_by_id,
        "bills": state.bills,
        "bill_lookup": state.bill_lookup,
        "vote_lookup": state.vote_lookup,
        "campaign_finance": state.campaign_finance,
    }
    state.members = sitting_members
    state.member_lookup = {m.name: m for m in sitting_members}
    state.member_lookup_by_id = {m.id: m for m in sitting_members}
    state.bills = [
        Bill(
            bill_number="SB0001",
            leg_id="1",
            description="Test bill",
            chamber="S",
            last_action="First Reading",
            last_action_date="1/1/2025",
            primary_sponsor="Don Harmon",
            sponsor_ids=["3268", "3264"],
        )
    ]
    state.bill_lookup = {state.bills[0].bill_number: state.bills[0]}
    state.vote_lookup = {}
    attach_index(state, index)

    try:
        trail = schema.execute_sync(
            """
            query ($id: String!) {
              memberMoneyTrail(memberId: $id, limit: 5) {
                memberId
                memberName
                receiptCount
                totalReceived
                committees { sbeCommitteeId name matchMethod }
                topDonors { name totalAmount receiptCount }
              }
            }
            """,
            variable_values={"id": "3268"},
        )
        assert trail.errors is None
        data = trail.data["memberMoneyTrail"]
        assert data["memberId"] == "3268"
        assert data["receiptCount"] > 0
        assert data["totalReceived"] > 0

        empty = schema.execute_sync(
            """
            query {
              memberMoneyTrail(memberId: "missing") { memberId receiptCount }
            }
            """
        )
        assert empty.errors is None
        assert empty.data["memberMoneyTrail"] is None

        bill_q = schema.execute_sync(
            """
            query {
              billMoneyContext(billNumber: "SB0001", limit: 5) {
                billNumber
                totalReceivedAcrossSponsors
                sponsorTrails { memberId totalReceived }
                topDonorsAcrossSponsors { name totalAmount }
              }
            }
            """
        )
        assert bill_q.errors is None
        bdata = bill_q.data["billMoneyContext"]
        assert bdata["billNumber"] == "SB0001"
        assert bdata["totalReceivedAcrossSponsors"] > 0
        assert len(bdata["sponsorTrails"]) >= 2
    finally:
        for key, value in snapshot.items():
            setattr(state, key, value)
