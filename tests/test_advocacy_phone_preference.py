"""Tests for district-first phone selection in advocacy (drawer and cards)."""

from __future__ import annotations

from ilga_graph.advocacy_helpers import get_preferred_phone_for_member
from ilga_graph.models import Member, Office


def _member_with_offices(offices: list[Office]) -> Member:
    return Member(
        id="1",
        name="Test Member",
        party="Democrat",
        district="1",
        chamber="Senate",
        member_url="",
        photo_url="",
        offices=offices,
        committees=[],
        career_ranges=[],
        sponsored_bills=[],
        co_sponsor_bills=[],
        co_sponsor_bill_ids=[],
        bio_text="",
        role="",
        career_timeline_text="",
        associated_members=None,
        email=None,
        roles=[],
    )


class TestGetPreferredPhoneForMember:
    def test_prefers_district_when_both_have_phone(self) -> None:
        offices = [
            Office("Springfield Office", "Capitol", phone="(217) 782-5957", fax=None),
            Office("District Office", "District", phone="(708) 632-4500", fax=None),
        ]
        member = _member_with_offices(offices)
        assert get_preferred_phone_for_member(member) == "(708) 632-4500"

    def test_uses_springfield_when_only_springfield_has_phone(self) -> None:
        offices = [
            Office("Springfield Office", "Capitol", phone="(217) 782-5957", fax=None),
            Office("District Office", "District", phone=None, fax=None),
        ]
        member = _member_with_offices(offices)
        assert get_preferred_phone_for_member(member) == "(217) 782-5957"

    def test_uses_district_when_only_district_has_phone(self) -> None:
        offices = [
            Office("Springfield Office", "Capitol", phone=None, fax=None),
            Office("District Office", "District", phone="(815) 987-7555", fax=None),
        ]
        member = _member_with_offices(offices)
        assert get_preferred_phone_for_member(member) == "(815) 987-7555"

    def test_returns_none_when_no_office_has_phone(self) -> None:
        offices = [
            Office("Springfield Office", "Capitol", phone=None, fax=None),
            Office("District Office", "District", phone=None, fax=None),
        ]
        member = _member_with_offices(offices)
        assert get_preferred_phone_for_member(member) is None

    def test_returns_none_when_member_is_none(self) -> None:
        assert get_preferred_phone_for_member(None) is None

    def test_returns_none_when_no_offices(self) -> None:
        member = _member_with_offices([])
        assert get_preferred_phone_for_member(member) is None
