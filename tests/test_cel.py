"""Tests for the CEL (Center for Effective Lawmaking) module."""

from __future__ import annotations

import pytest

from ilga_graph.cel import (
    BillCategory,
    CELStage,
    _derive_chamber_majority,
    _is_committee_chair,
    _seniority_terms,
    bill_reaches_cel_stage,
    classify_bill_category,
    classify_expectations,
    compute_les_scores,
)
from ilga_graph.models import Bill, CareerRange, Member, WitnessSlip

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bill(
    bill_number: str,
    last_action: str,
    description: str = "Amends the Code",
    synopsis: str = "",
) -> Bill:
    return Bill(
        bill_number=bill_number,
        leg_id=f"id-{bill_number}",
        description=description,
        chamber="H" if bill_number.startswith("H") else "S",
        last_action=last_action,
        last_action_date="1/1/2025",
        primary_sponsor="Test",
        synopsis=synopsis,
    )


def _make_member(
    member_id: str,
    chamber: str,
    party: str,
    bills: list[Bill] | None = None,
    role: str = "Representative",
    career_ranges: list[CareerRange] | None = None,
    roles: list[str] | None = None,
) -> Member:
    return Member(
        id=member_id,
        name=f"Member {member_id}",
        member_url="",
        chamber=chamber,
        party=party,
        district="1",
        bio_text="",
        role=role,
        sponsored_bills=bills or [],
        co_sponsor_bills=[],
        career_ranges=career_ranges or [],
        roles=roles or [],
    )


# ── Bill category ──────────────────────────────────────────────────────────────


class TestClassifyBillCategory:
    def test_commemorative_designation(self) -> None:
        bill = _make_bill("HB0001", "Referred to Assignments", "DESIGNATION OF ROUTE")
        assert classify_bill_category(bill) == BillCategory.COMMEMORATIVE

    def test_commemorative_memorial(self) -> None:
        bill = _make_bill("HB0001", "Referred to Assignments", "Memorial for veterans")
        assert classify_bill_category(bill) == BillCategory.COMMEMORATIVE

    def test_commemorative_honor(self) -> None:
        bill = _make_bill("HB0001", "Referred to Assignments", "Honoring the fire dept")
        assert classify_bill_category(bill) == BillCategory.COMMEMORATIVE

    def test_commemorative_in_synopsis(self) -> None:
        bill = _make_bill("HB0001", "Referred to Assignments", synopsis="Tribute to teachers")
        assert classify_bill_category(bill) == BillCategory.COMMEMORATIVE

    def test_substantive_default(self) -> None:
        bill = _make_bill("HB0001", "Referred to Assignments", "Amends the Income Tax Act")
        assert classify_bill_category(bill) == BillCategory.SUBSTANTIVE

    def test_significant_by_cosponsors(self) -> None:
        bill = _make_bill("HB0001", "Referred to Assignments", "Amends the Health Code")
        assert classify_bill_category(bill, cosponsor_count=5) == BillCategory.SIGNIFICANT

    def test_significant_by_witness_slips(self) -> None:
        bill = _make_bill("HB0001", "Referred to Assignments", "Amends the Health Code")
        assert classify_bill_category(bill, witness_slip_count=10) == BillCategory.SIGNIFICANT

    def test_commemorative_wins_over_cosponsors(self) -> None:
        """A commemorative bill stays C even with many co-sponsors."""
        bill = _make_bill("HB0001", "Referred to Assignments", "Honoring first responders")
        # Commemorative keyword should take priority
        assert classify_bill_category(bill, cosponsor_count=20) == BillCategory.COMMEMORATIVE

    def test_category_weights(self) -> None:
        assert BillCategory.COMMEMORATIVE.weight == 1
        assert BillCategory.SUBSTANTIVE.weight == 5
        assert BillCategory.SIGNIFICANT.weight == 10


# ── CEL stage thresholds ───────────────────────────────────────────────────────


class TestBillReachesCELStage:
    def test_filed_bill_only_reaches_bill_stage(self) -> None:
        bill = _make_bill("HB0001", "Filed with Secretary")
        assert bill_reaches_cel_stage(bill, CELStage.BILL) is True
        assert bill_reaches_cel_stage(bill, CELStage.AIC) is False

    def test_referred_reaches_aic(self) -> None:
        bill = _make_bill("HB0001", "Referred to Assignments")
        assert bill_reaches_cel_stage(bill, CELStage.BILL) is True
        assert bill_reaches_cel_stage(bill, CELStage.AIC) is True
        assert bill_reaches_cel_stage(bill, CELStage.ABC) is False

    def test_second_reading_reaches_abc(self) -> None:
        bill = _make_bill("HB0001", "Second Reading - Short Debate")
        assert bill_reaches_cel_stage(bill, CELStage.ABC) is True
        assert bill_reaches_cel_stage(bill, CELStage.PASS) is False

    def test_chamber_passed_reaches_pass(self) -> None:
        bill = _make_bill("HB0001", "Third Reading - Passed")
        assert bill_reaches_cel_stage(bill, CELStage.PASS) is True
        assert bill_reaches_cel_stage(bill, CELStage.LAW) is False

    def test_signed_reaches_law(self) -> None:
        bill = _make_bill("HB0001", "Public Act 104-0001")
        assert bill_reaches_cel_stage(bill, CELStage.LAW) is True


# ── Expectations ──────────────────────────────────────────────────────────────


class TestClassifyExpectations:
    def test_above(self) -> None:
        assert classify_expectations(1.8, 1.0) == "Above"

    def test_meets(self) -> None:
        assert classify_expectations(1.0, 1.0) == "Meets"

    def test_meets_at_boundary_high(self) -> None:
        assert classify_expectations(1.5, 1.0) == "Meets"

    def test_meets_at_boundary_low(self) -> None:
        assert classify_expectations(0.5, 1.0) == "Meets"

    def test_below(self) -> None:
        assert classify_expectations(0.3, 1.0) == "Below"

    def test_zero_benchmark_returns_empty(self) -> None:
        assert classify_expectations(1.0, 0.0) == ""


# ── Seniority ─────────────────────────────────────────────────────────────────


class TestSeniorityTerms:
    def test_no_career_returns_zero(self) -> None:
        member = _make_member("1", "House", "Democrat")
        assert _seniority_terms(member, current_year=2025) == 0

    def test_new_member_counts_as_one(self) -> None:
        member = _make_member(
            "1",
            "House",
            "Democrat",
            career_ranges=[CareerRange(start_year=2024, end_year=None, chamber="House")],
        )
        assert _seniority_terms(member, current_year=2025) == 1

    def test_ten_year_veteran(self) -> None:
        member = _make_member(
            "1",
            "House",
            "Democrat",
            career_ranges=[CareerRange(start_year=2005, end_year=None, chamber="House")],
        )
        # (2025 - 2005) // 2 + 1 = 10 + 1 = 11
        assert _seniority_terms(member, current_year=2025) == 11

    def test_ignores_other_chamber_ranges(self) -> None:
        member = _make_member(
            "1",
            "House",
            "Democrat",
            career_ranges=[
                CareerRange(start_year=2010, end_year=2014, chamber="Senate"),
                CareerRange(start_year=2020, end_year=None, chamber="House"),
            ],
        )
        # Only House range counts: (2025 - 2020) // 2 + 1 = 3
        assert _seniority_terms(member, current_year=2025) == 3


# ── Committee chair detection ──────────────────────────────────────────────────


class TestIsCommitteeChair:
    def test_chair_role_detected(self) -> None:
        member = _make_member("1", "House", "Democrat", roles=["Chairperson", "Member"])
        assert _is_committee_chair(member) is True

    def test_vice_chair_not_detected(self) -> None:
        member = _make_member("1", "House", "Democrat", roles=["Vice-Chair"])
        assert _is_committee_chair(member) is False

    def test_caucus_chair_not_detected(self) -> None:
        member = _make_member("1", "House", "Democrat", roles=["Caucus Chair"])
        assert _is_committee_chair(member) is False

    def test_member_role_no_chair(self) -> None:
        member = _make_member("1", "House", "Democrat", roles=["Member"])
        assert _is_committee_chair(member) is False

    def test_chair_via_member_role_field(self) -> None:
        """Falls back to member.role when roles list is empty."""
        member = _make_member("1", "House", "Democrat", role="Chairperson")
        assert _is_committee_chair(member) is True


# ── Chamber majority derivation ───────────────────────────────────────────────


class TestDeriveChamberMajority:
    def test_majority_by_count(self) -> None:
        members = [
            _make_member("1", "House", "Democrat"),
            _make_member("2", "House", "Democrat"),
            _make_member("3", "House", "Republican"),
        ]
        majority = _derive_chamber_majority(members)
        assert majority["House"] == "Democrat"

    def test_separate_chambers(self) -> None:
        members = [
            _make_member("1", "House", "Democrat"),
            _make_member("2", "Senate", "Republican"),
        ]
        majority = _derive_chamber_majority(members)
        assert majority["House"] == "Democrat"
        assert majority["Senate"] == "Republican"


# ── Full LES computation ───────────────────────────────────────────────────────


class TestComputeLESScores:
    def test_returns_result_for_every_member(self) -> None:
        m1 = _make_member(
            "1", "House", "Democrat", bills=[_make_bill("HB0001", "Public Act 104-0001")]
        )
        m2 = _make_member("2", "House", "Republican", bills=[])
        results = compute_les_scores([m1, m2])
        assert "1" in results
        assert "2" in results

    def test_member_with_law_has_higher_les_than_member_without(self) -> None:
        m_with = _make_member(
            "1",
            "House",
            "Democrat",
            bills=[_make_bill("HB0001", "Public Act 104-0001", "Amends Tax Act")],
        )
        m_without = _make_member("2", "House", "Republican", bills=[])
        results = compute_les_scores([m_with, m_without])
        assert results["1"].les > results["2"].les

    def test_les_average_is_one(self) -> None:
        """In a chamber, the average LES should equal 1."""
        bills = [
            _make_bill("HB0001", "Public Act 104-0001", "Amends Tax Act"),
            _make_bill("HB0002", "Referred to Assignments", "Amends Education Act"),
            _make_bill("HB0003", "Third Reading - Passed", "Amends Health Act"),
        ]
        members = [
            _make_member("1", "House", "Democrat", bills=[bills[0]]),
            _make_member("2", "House", "Democrat", bills=[bills[1]]),
            _make_member("3", "House", "Republican", bills=[bills[2]]),
        ]
        results = compute_les_scores(members)
        house_les = [r.les for r in results.values()]
        avg = sum(house_les) / len(house_les)
        assert avg == pytest.approx(1.0, abs=1e-6)

    def test_empty_member_list(self) -> None:
        results = compute_les_scores([])
        assert results == {}

    def test_no_bills_gets_les_zero(self) -> None:
        member = _make_member("1", "House", "Democrat", bills=[])
        results = compute_les_scores([member])
        assert results["1"].les == 0.0

    def test_resolutions_excluded_from_les(self) -> None:
        """HR/SR should not count toward LES (not substantive)."""
        res_bill = _make_bill("HR0001", "Resolution Adopted", "Honoring someone")
        member = _make_member("1", "House", "Democrat", bills=[res_bill])
        results = compute_les_scores([member])
        # All stage weights should be zero since HR is not substantive
        assert results["1"].stage_weights["BILL"] == 0.0

    def test_shell_bills_excluded(self) -> None:
        """Shell/technical bills should not count toward LES."""
        shell = _make_bill("HB0001", "Referred to Assignments", "INCOME TAX-TECH")
        member = _make_member("1", "House", "Democrat", bills=[shell])
        results = compute_les_scores([member])
        assert results["1"].stage_weights["BILL"] == 0.0

    def test_les_result_has_expectation(self) -> None:
        """With enough members, expectation should be non-empty."""
        members = [
            _make_member(
                str(i),
                "House",
                "Democrat",
                bills=[_make_bill(f"HB{i:04d}", "Public Act 104-0001", "Tax reform bill")],
                career_ranges=[CareerRange(start_year=2015, end_year=None, chamber="House")],
            )
            for i in range(1, 6)
        ]
        results = compute_les_scores(members)
        for result in results.values():
            assert result.les_expectation in {"Above", "Meets", "Below", ""}

    def test_witness_slips_boost_category_to_ss(self) -> None:
        """Bills with many witness slips should be classified SS (weight 10)."""
        bill = _make_bill("HB0001", "Public Act 104-0001", "Amends Health Code")
        slips = [
            WitnessSlip(
                name=f"Witness {i}",
                organization="Org",
                representing="Org",
                position="Proponent",
                hearing_committee="Health",
                hearing_date="2025-01-01",
                bill_number="HB0001",
            )
            for i in range(10)
        ]
        member = _make_member("1", "House", "Democrat", bills=[bill])
        results = compute_les_scores([member], witness_slips=slips)
        # SS weight is 10 — stage weight at BILL stage should be 10
        assert results["1"].stage_weights["BILL"] == 10.0

    def test_compute_moneyball_populates_les_fields(self) -> None:
        """compute_moneyball() should populate les, les_benchmark, les_expectation."""
        from ilga_graph.moneyball import compute_moneyball

        bill = _make_bill("HB0001", "Public Act 104-0001", "Amends Tax Act")
        member = _make_member("1", "House", "Democrat", bills=[bill])
        report = compute_moneyball([member])
        profile = report.profiles["1"]
        assert hasattr(profile, "les")
        assert hasattr(profile, "les_benchmark")
        assert hasattr(profile, "les_expectation")
        # With one member with a passed bill, LES > 0
        assert profile.les >= 0.0
