"""SBE committee→member match hardening and unmatched review."""

from __future__ import annotations

import json
import os
from pathlib import Path
from subprocess import run

from ilga_graph.campaign_finance.match import (
    format_unmatched_review,
    match_committees_to_members,
    unmatched_review_rows,
)
from ilga_graph.campaign_finance.parse import (
    ParsedSbeData,
    SbeCandidateRow,
    SbeCommitteeRow,
    SbeLinkRow,
    parse_sbe_dir,
)
from ilga_graph.campaign_finance.store import write_unmatched
from ilga_graph.models import Member

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sbe"
MOCK_MEMBERS = Path(__file__).resolve().parent.parent / "mocks" / "dev" / "members.json"


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


def _load_mock_members() -> list[Member]:
    raw = json.loads(MOCK_MEMBERS.read_text(encoding="utf-8"))
    return [
        Member(
            id=row["id"],
            name=row["name"],
            member_url=row.get("member_url", ""),
            chamber=row["chamber"],
            party=row["party"],
            district=row["district"],
            bio_text=row.get("bio_text", ""),
        )
        for row in raw
    ]


def _parsed(
    *,
    committee_id: str = "1",
    committee_name: str = "Friends of Test",
    candidate_id: str = "c1",
    last_name: str = "Stadelman",
    first_name: str = "Steven",
    office: str = "State Senator",
    district_type: str = "Senate",
    district: str = "99",
) -> ParsedSbeData:
    return ParsedSbeData(
        committees=[
            SbeCommitteeRow(
                committee_id=committee_id,
                name=committee_name,
                type_of_committee="Candidate",
                status="A",
                party="",
                purpose="",
                city="",
                state="",
                refer_name="",
            )
        ],
        candidates=[
            SbeCandidateRow(
                candidate_id=candidate_id,
                last_name=last_name,
                first_name=first_name,
                office=office,
                district_type=district_type,
                district=district,
                party="",
            )
        ],
        links=[SbeLinkRow(committee_id=committee_id, candidate_id=candidate_id)],
    )


def test_fixture_match_rate_against_mock_roster() -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    report = match_committees_to_members(parsed, _load_mock_members())
    accepted_ids = {m.committee_id for m in report.accepted}
    unmatched_ids = {m.committee_id for m in report.unmatched}
    assert report.legislative_committees == 16
    assert len(report.accepted) == 15
    assert unmatched_ids == {"39451"}
    assert "8081" in accepted_ids
    assert report.match_rate == 15 / 16
    deering = next(u for u in report.unmatched if u.committee_id == "39451")
    assert deering.status == "review"
    assert deering.unmatched_reason == "no_sitting_member"
    assert "Deering" in deering.notes
    assert deering.near_misses == []
    assert deering.gold_stub["committee_id"] == "39451"
    assert deering.gold_stub["member_id"] == ""


def test_nickname_alias_matches_stale_district() -> None:
    parsed = _parsed(
        committee_name="Stadelman for State Senate",
        first_name="Steven",
        last_name="Stadelman",
        district="99",
    )
    members = [_member("3296", "Steve Stadelman", "Senate", "34")]
    report = match_committees_to_members(parsed, members)
    assert [m.member_id for m in report.accepted] == ["3296"]
    assert report.accepted[0].method == "name_chamber"
    assert report.accepted[0].confidence == 0.8


def test_nickname_alias_does_not_invent_ambiguous_link() -> None:
    parsed = _parsed(
        first_name="Steven",
        last_name="Stadelman",
        district="99",
    )
    members = [
        _member("3296", "Steve Stadelman", "Senate", "34"),
        _member("4001", "Steven Stadelman", "Senate", "12"),
    ]
    report = match_committees_to_members(parsed, members)
    assert report.accepted == []
    assert report.unmatched[0].unmatched_reason == "ambiguous"
    assert {hit["member_id"] for hit in report.unmatched[0].near_misses} == {
        "3296",
        "4001",
    }


def test_suffix_and_trailing_period_on_real_fixture_rows() -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    members = [
        _member("3276", "Emil Jones, III", "Senate", "14"),
        _member("3294", "Julie A. Morrison", "Senate", "29"),
    ]
    report = match_committees_to_members(parsed, members)
    by_member = {m.member_id: m for m in report.accepted}
    assert by_member["3276"].committee_id == "21361"
    assert by_member["3276"].method == "office_district_name"
    assert by_member["3276"].confidence == 0.95
    assert by_member["3294"].committee_id == "23762"
    assert by_member["3294"].confidence == 0.95


def test_district_change_matches_unique_name_chamber() -> None:
    parsed = _parsed(
        committee_id="39451",
        committee_name="Regan for Illinois",
        first_name="Regan",
        last_name="Deering",
        office="State Representative",
        district_type="Representative",
        district="88",
    )
    members = [_member("3999", "Regan Deering", "House", "95", "Republican")]
    report = match_committees_to_members(parsed, members)
    assert report.accepted[0].member_id == "3999"
    assert report.accepted[0].method == "name_chamber"
    assert report.accepted[0].confidence == 0.8


def test_gold_override_wins_when_rules_cannot(tmp_path: Path) -> None:
    garbled = _parsed(
        committee_id="99901",
        committee_name="Citizens for Lightford",
        first_name="Kim",
        last_name="L-Ford",
        district="4",
    )
    sitting = [_member("3264", "Kimberly A. Lightford", "Senate", "4")]
    without = match_committees_to_members(garbled, sitting)
    assert without.accepted == []
    assert without.unmatched[0].committee_id == "99901"

    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps([{"committee_id": "99901", "member_id": "3264"}]),
        encoding="utf-8",
    )
    with_gold = match_committees_to_members(garbled, sitting, gold_path=gold_path)
    assert with_gold.accepted[0].member_id == "3264"
    assert with_gold.accepted[0].method == "gold"
    assert with_gold.accepted[0].confidence == 1.0


def test_unmatched_review_rows_are_actionable(tmp_path: Path) -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    report = match_committees_to_members(parsed, _load_mock_members())
    rows = unmatched_review_rows(report)
    assert len(rows) == 1
    row = rows[0]
    assert row["committee_id"] == "39451"
    assert row["committee_name"] == "Regan for Illinois"
    assert row["candidate_name"] == "Regan Deering"
    assert row["unmatched_reason"] == "no_sitting_member"
    assert "docs/canonical/sbe_committee_member_gold.json" in row["how_to_promote"]
    assert row["gold_stub"]["member_id"] == ""
    assert row["gold_stub"]["committee_id"] == "39451"
    text = format_unmatched_review(rows)
    assert "39451" in text
    assert "no_sitting_member" in text
    assert "gold" in text.lower()

    out = tmp_path / "unmatched.json"
    write_unmatched(report, out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved[0]["unmatched_reason"] == "no_sitting_member"
    assert saved[0]["how_to_promote"]
    assert "gold_stub" in saved[0]


def test_review_cli_reads_unmatched_json(tmp_path: Path) -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    report = match_committees_to_members(parsed, _load_mock_members())
    unmatched_path = tmp_path / "unmatched.json"
    write_unmatched(report, unmatched_path)
    result = run(
        [
            "python3",
            "scripts/review_sbe_unmatched.py",
            "--unmatched-json",
            str(unmatched_path),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert "39451" in result.stdout
    assert "no_sitting_member" in result.stdout
    assert "Regan for Illinois" in result.stdout


def test_unmatched_prefers_sitting_chamber_candidate_row() -> None:
    parsed = parse_sbe_dir(FIXTURE_DIR)
    members = [_member("9990", "No Match Person", "Senate", "99")]
    report = match_committees_to_members(parsed, members)
    rose = next(u for u in report.unmatched if u.committee_id == "16473")
    assert rose.office == "State Senator"
    assert rose.district == "51"
    considered = {c["office"] for c in rose.candidates_considered}
    assert "State Senator" in considered
    assert "State Representative" in considered
