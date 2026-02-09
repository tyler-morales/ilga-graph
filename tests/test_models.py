from __future__ import annotations

from ilga_graph.models import Bill, CareerRange, Committee, CommitteeMemberRole, Member, Office, WitnessSlip


class TestBill:
    def test_construction(self, sample_bill: Bill) -> None:
        assert sample_bill.bill_number == "SB1527"
        assert sample_bill.leg_id == "160173"
        assert sample_bill.chamber == "S"
        assert sample_bill.last_action_date == "6/2/2025"
        assert sample_bill.primary_sponsor == "Sue Rezin"

    def test_house_bill(self, sample_bill_house: Bill) -> None:
        assert sample_bill_house.bill_number == "HB0054"
        assert sample_bill_house.chamber == "H"


class TestCareerRange:
    def test_open_ended(self, sample_career_range: CareerRange) -> None:
        assert sample_career_range.start_year == 2015
        assert sample_career_range.end_year is None
        assert sample_career_range.chamber == "Senate"

    def test_closed_range(self, sample_career_range_ended: CareerRange) -> None:
        assert sample_career_range_ended.start_year == 2010
        assert sample_career_range_ended.end_year == 2014
        assert sample_career_range_ended.chamber == "House"

    def test_defaults(self) -> None:
        cr = CareerRange(start_year=2020)
        assert cr.end_year is None
        assert cr.chamber is None


class TestMember:
    def test_construction(self, sample_member: Member) -> None:
        assert sample_member.id == "3312"
        assert sample_member.name == "Neil Anderson"
        assert sample_member.party == "Republican"
        assert sample_member.district == "47"
        assert len(sample_member.career_ranges) == 1
        assert len(sample_member.offices) == 1
        assert len(sample_member.sponsored_bills) == 1
        assert len(sample_member.co_sponsor_bills) == 0
        assert len(sample_member.bills) == 1  # property: sponsored + co_sponsor

    def test_defaults(self) -> None:
        m = Member(
            id="1",
            name="Test",
            member_url="http://example.com",
            chamber="Senate",
            party="Democrat",
            district="1",
            bio_text="Bio",
        )
        assert m.role == ""
        assert m.career_timeline_text == ""
        assert m.career_ranges == []
        assert m.committees == []
        assert m.associated_members is None
        assert m.email is None
        assert m.offices == []
        assert m.sponsored_bills == []
        assert m.co_sponsor_bills == []
        assert m.bills == []  # property: sponsored + co_sponsor
        assert m.sponsored_bill_ids == []
        assert m.co_sponsor_bill_ids == []

    def test_no_career_ranges(self, sample_member_no_career: Member) -> None:
        assert sample_member_no_career.career_ranges == []

    def test_house_member(self, sample_member_house: Member) -> None:
        assert sample_member_house.chamber == "House"
        assert sample_member_house.id == "4501"
        assert sample_member_house.name == "Regan Deering"
        assert len(sample_member_house.sponsored_bills) == 1
        assert sample_member_house.sponsored_bills[0].chamber == "H"


class TestOffice:
    def test_construction(self, sample_office: Office) -> None:
        assert sample_office.name == "Springfield Office"
        assert "Capitol Building" in sample_office.address
        assert sample_office.phone == "(217) 782-5957"
        assert sample_office.fax is None


class TestCommittee:
    def test_construction(self, sample_committee: Committee) -> None:
        assert sample_committee.code == "SAGR"
        assert sample_committee.name == "Agriculture"
        assert sample_committee.parent_code is None

    def test_with_parent(self) -> None:
        c = Committee(code="SEXC-FIR", name="Firearms", parent_code="SEXC")
        assert c.parent_code == "SEXC"


class TestWitnessSlip:
    def test_construction(self, sample_witness_slip: WitnessSlip) -> None:
        assert sample_witness_slip.name == "Paul Makarewicz"
        assert sample_witness_slip.organization == "AES Clean Energy"
        assert sample_witness_slip.representing == "AES Clean Energy"
        assert sample_witness_slip.position == "Proponent"
        assert sample_witness_slip.hearing_committee == "Executive"
        assert sample_witness_slip.hearing_date == "2025-05-31 17:00"
        assert sample_witness_slip.bill_number == "HB1075"
        assert sample_witness_slip.testimony_type == "Record of Appearance Only"

    def test_defaults(self) -> None:
        slip = WitnessSlip(
            name="Test",
            organization="Org",
            representing="Self",
            position="Opponent",
            hearing_committee="Judiciary",
            hearing_date="2025-01-01 09:00",
        )
        assert slip.testimony_type == "Record of Appearance Only"
        assert slip.bill_number == ""


class TestCommitteeMemberRole:
    def test_frozen(self) -> None:
        role = CommitteeMemberRole(
            member_id="3312",
            member_name="Neil Anderson",
            member_url="https://example.com",
            role="Member",
        )
        assert role.member_id == "3312"
        assert role.role == "Member"
