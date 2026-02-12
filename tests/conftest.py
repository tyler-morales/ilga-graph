from __future__ import annotations

import pytest

from ilga_graph.models import Bill, CareerRange, Committee, Member, Office, WitnessSlip

# ── Bill fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_bill() -> Bill:
    return Bill(
        bill_number="SB1527",
        leg_id="160173",
        description=(
            "Amends the Illinois Power Agency Act to repeal the nuclear moratorium provision"
        ),
        chamber="S",
        last_action="Rule 3-9(a) / Re-referred to Assignments",
        last_action_date="6/2/2025",
        primary_sponsor="Sue Rezin",
    )


@pytest.fixture
def sample_bill_house() -> Bill:
    return Bill(
        bill_number="HB0054",
        leg_id="155719",
        description="Designates the third week of May as Illinois Soil Health Week each year",
        chamber="H",
        last_action="Referred to Assignments",
        last_action_date="4/8/2025",
        primary_sponsor="Regan Deering",
    )


@pytest.fixture
def sample_bill_passed() -> Bill:
    """A substantive HB that passed."""
    return Bill(
        bill_number="HB0100",
        leg_id="200001",
        description=(
            "Amends the Environmental Protection Act to establish "
            "clean energy standards for utilities"
        ),
        chamber="H",
        last_action="Public Act 104-0001",
        last_action_date="7/1/2025",
        primary_sponsor="Alice Smith",
    )


@pytest.fixture
def sample_bill_sb_passed() -> Bill:
    """A substantive SB that passed."""
    return Bill(
        bill_number="SB0200",
        leg_id="200002",
        description=(
            "Amends the School Code to reform teacher certification "
            "and training requirements statewide"
        ),
        chamber="S",
        last_action="Signed by Governor",
        last_action_date="8/15/2025",
        primary_sponsor="Alice Smith",
    )


@pytest.fixture
def sample_bill_stuck() -> Bill:
    """A substantive HB that is stuck."""
    return Bill(
        bill_number="HB0300",
        leg_id="200003",
        description=(
            "Amends the Illinois Income Tax Act to restructure "
            "individual and corporate tax brackets"
        ),
        chamber="H",
        last_action="Referred to Assignments",
        last_action_date="3/1/2025",
        primary_sponsor="Alice Smith",
    )


@pytest.fixture
def sample_resolution_hr() -> Bill:
    """A house resolution (ceremonial)."""
    return Bill(
        bill_number="HR0010",
        leg_id="300001",
        description="HONORING FIREFIGHTERS",
        chamber="H",
        last_action="Resolution Adopted",
        last_action_date="5/1/2025",
        primary_sponsor="Alice Smith",
    )


@pytest.fixture
def sample_resolution_sr() -> Bill:
    """A senate resolution (ceremonial)."""
    return Bill(
        bill_number="SR0020",
        leg_id="300002",
        description="RECOGNIZING TEACHERS",
        chamber="S",
        last_action="Resolution Adopted",
        last_action_date="5/15/2025",
        primary_sponsor="Alice Smith",
    )


@pytest.fixture
def sample_resolution_sjr() -> Bill:
    """A senate joint resolution (ceremonial)."""
    return Bill(
        bill_number="SJR0005",
        leg_id="300003",
        description="JOINT SESSION RESOLUTION",
        chamber="S",
        last_action="Resolution Adopted",
        last_action_date="6/1/2025",
        primary_sponsor="Alice Smith",
    )


@pytest.fixture
def sample_resolution_hjr() -> Bill:
    """A house joint resolution (ceremonial)."""
    return Bill(
        bill_number="HJR0008",
        leg_id="300004",
        description="MEMORIAL DAY",
        chamber="H",
        last_action="Resolution Adopted",
        last_action_date="5/30/2025",
        primary_sponsor="Alice Smith",
    )


# ── Career / Office / Committee fixtures ──────────────────────────────────────


@pytest.fixture
def sample_career_range() -> CareerRange:
    return CareerRange(start_year=2015, end_year=None, chamber="Senate")


@pytest.fixture
def sample_career_range_ended() -> CareerRange:
    return CareerRange(start_year=2010, end_year=2014, chamber="House")


@pytest.fixture
def sample_office() -> Office:
    return Office(
        name="Springfield Office",
        address="208 A Capitol Building\nSpringfield, IL 62706",
        phone="(217) 782-5957",
    )


@pytest.fixture
def sample_committee() -> Committee:
    return Committee(code="SAGR", name="Agriculture", parent_code=None)


@pytest.fixture
def sample_committee_house() -> Committee:
    return Committee(code="HAGR", name="Agriculture & Conservation", parent_code=None)


# ── Member fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_member(
    sample_bill: Bill, sample_career_range: CareerRange, sample_office: Office
) -> Member:
    return Member(
        id="3312",
        name="Neil Anderson",
        member_url="https://www.ilga.gov/Senate/Members/Details/3312",
        chamber="Senate",
        party="Republican",
        district="47",
        bio_text="State Senator Neil Anderson, a professional firefighter.",
        role="Republican Caucus Chair",
        career_timeline_text="2015 - Present",
        career_ranges=[sample_career_range],
        committees=["SAGR", "SEXC"],
        offices=[sample_office],
        sponsored_bills=[sample_bill],
        co_sponsor_bills=[],
        sponsored_bill_ids=[sample_bill.leg_id],
        co_sponsor_bill_ids=[],
    )


@pytest.fixture
def sample_member_no_career() -> Member:
    return Member(
        id="9999",
        name="Test Member",
        member_url="https://www.ilga.gov/Senate/Members/Details/9999",
        chamber="Senate",
        party="Democrat",
        district="1",
        bio_text="A test member.",
        career_ranges=[],
    )


@pytest.fixture
def sample_member_house(
    sample_bill_house: Bill,
    sample_career_range_ended: CareerRange,
    sample_office: Office,
) -> Member:
    return Member(
        id="4501",
        name="Regan Deering",
        member_url="https://www.ilga.gov/House/Members/Details/4501",
        chamber="House",
        party="Republican",
        district="95",
        bio_text="State Representative Regan Deering serves the 95th district.",
        role="",
        career_timeline_text="2023 - Present",
        career_ranges=[CareerRange(start_year=2023, end_year=None, chamber="House")],
        committees=["HAGR", "HTRN"],
        offices=[sample_office],
        sponsored_bills=[sample_bill_house],
        co_sponsor_bills=[],
        sponsored_bill_ids=[sample_bill_house.leg_id],
        co_sponsor_bill_ids=[],
    )


# ── Rich fixtures for Phase 3 analytics tests ────────────────────────────────


@pytest.fixture
def mixed_bill_member(
    sample_bill_passed: Bill,
    sample_bill_sb_passed: Bill,
    sample_bill_stuck: Bill,
    sample_resolution_hr: Bill,
    sample_resolution_sjr: Bill,
) -> Member:
    """A Democrat member with 3 laws (2 passed, 1 stuck) and 2 resolutions."""
    all_bills = [
        sample_bill_passed,
        sample_bill_sb_passed,
        sample_bill_stuck,
        sample_resolution_hr,
        sample_resolution_sjr,
    ]
    return Member(
        id="5001",
        name="Alice Smith",
        member_url="https://www.ilga.gov/House/Members/Details/5001",
        chamber="House",
        party="Democrat",
        district="10",
        bio_text="Representative Alice Smith.",
        sponsored_bills=all_bills,
        co_sponsor_bills=[],
        sponsored_bill_ids=[b.leg_id for b in all_bills],
        co_sponsor_bill_ids=[],
    )


@pytest.fixture
def cosponsor_republican() -> Member:
    """A Republican who co-sponsors Alice's HB0100 (cross-party)."""
    hb100 = Bill(
        bill_number="HB0100",
        leg_id="200001",
        description=(
            "Amends the Environmental Protection Act to establish "
            "clean energy standards for utilities"
        ),
        chamber="H",
        last_action="Public Act 104-0001",
        last_action_date="7/1/2025",
        primary_sponsor="Alice Smith",
    )
    return Member(
        id="5002",
        name="Bob Jones",
        member_url="https://www.ilga.gov/House/Members/Details/5002",
        chamber="House",
        party="Republican",
        district="11",
        bio_text="Representative Bob Jones.",
        sponsored_bills=[],
        co_sponsor_bills=[hb100],
        sponsored_bill_ids=[],
        co_sponsor_bill_ids=[hb100.leg_id],
    )


@pytest.fixture
def cosponsor_democrat() -> Member:
    """A Democrat who co-sponsors Alice's HB0100 and SB0200 (same-party)."""
    hb100 = Bill(
        bill_number="HB0100",
        leg_id="200001",
        description=(
            "Amends the Environmental Protection Act to establish "
            "clean energy standards for utilities"
        ),
        chamber="H",
        last_action="Public Act 104-0001",
        last_action_date="7/1/2025",
        primary_sponsor="Alice Smith",
    )
    sb200 = Bill(
        bill_number="SB0200",
        leg_id="200002",
        description=(
            "Amends the School Code to reform teacher certification "
            "and training requirements statewide"
        ),
        chamber="S",
        last_action="Signed by Governor",
        last_action_date="8/15/2025",
        primary_sponsor="Alice Smith",
    )
    return Member(
        id="5003",
        name="Carol Davis",
        member_url="https://www.ilga.gov/House/Members/Details/5003",
        chamber="House",
        party="Democrat",
        district="12",
        bio_text="Representative Carol Davis.",
        sponsored_bills=[],
        co_sponsor_bills=[hb100, sb200],
        sponsored_bill_ids=[],
        co_sponsor_bill_ids=[hb100.leg_id, sb200.leg_id],
    )


@pytest.fixture
def member_no_bills() -> Member:
    """A member with no bills at all (edge case)."""
    return Member(
        id="5004",
        name="Empty Earl",
        member_url="https://www.ilga.gov/House/Members/Details/5004",
        chamber="House",
        party="Republican",
        district="99",
        bio_text="Representative Earl.",
        sponsored_bills=[],
        co_sponsor_bills=[],
    )


@pytest.fixture
def member_resolutions_only(
    sample_resolution_hr: Bill,
    sample_resolution_sr: Bill,
) -> Member:
    """A member whose sponsored bills are all resolutions."""
    return Member(
        id="5005",
        name="Res Rita",
        member_url="https://www.ilga.gov/Senate/Members/Details/5005",
        chamber="Senate",
        party="Democrat",
        district="5",
        bio_text="Senator Rita.",
        sponsored_bills=[sample_resolution_hr, sample_resolution_sr],
        co_sponsor_bills=[],
        sponsored_bill_ids=[sample_resolution_hr.leg_id, sample_resolution_sr.leg_id],
        co_sponsor_bill_ids=[],
    )


# ── WitnessSlip fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_witness_slip() -> WitnessSlip:
    return WitnessSlip(
        name="Paul Makarewicz",
        organization="AES Clean Energy",
        representing="AES Clean Energy",
        position="Proponent",
        hearing_committee="Executive",
        hearing_date="2025-05-31 17:00",
        bill_number="HB1075",
    )
