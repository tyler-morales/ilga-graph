from __future__ import annotations

from datetime import datetime

from ilga_graph.main import _member_career_start, _parse_bill_date
from ilga_graph.models import CareerRange, Member


class TestParseBillDate:
    def test_valid_date(self) -> None:
        result = _parse_bill_date("6/2/2025")
        assert result == datetime(2025, 6, 2)

    def test_valid_date_double_digit(self) -> None:
        result = _parse_bill_date("12/15/2025")
        assert result == datetime(2025, 12, 15)

    def test_invalid_date_returns_max(self) -> None:
        result = _parse_bill_date("invalid")
        assert result == datetime.max

    def test_empty_string_returns_max(self) -> None:
        result = _parse_bill_date("")
        assert result == datetime.max

    def test_wrong_format_returns_max(self) -> None:
        result = _parse_bill_date("2025-06-02")
        assert result == datetime.max


class TestMemberCareerStart:
    def test_single_range(self) -> None:
        member = Member(
            id="1", name="T", member_url="", chamber="Senate",
            party="D", district="1", bio_text="",
            career_ranges=[CareerRange(start_year=2015)],
        )
        assert _member_career_start(member) == 2015

    def test_multiple_ranges_returns_min(self) -> None:
        member = Member(
            id="1", name="T", member_url="", chamber="Senate",
            party="D", district="1", bio_text="",
            career_ranges=[
                CareerRange(start_year=2020),
                CareerRange(start_year=2010, end_year=2014),
            ],
        )
        assert _member_career_start(member) == 2010

    def test_no_ranges_returns_9999(self) -> None:
        member = Member(
            id="1", name="T", member_url="", chamber="Senate",
            party="D", district="1", bio_text="",
            career_ranges=[],
        )
        assert _member_career_start(member) == 9999
