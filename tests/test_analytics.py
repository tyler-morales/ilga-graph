from __future__ import annotations

from ilga_graph.analytics import (
    BillStatus,
    classify_bill_status,
    compute_all_scorecards,
    compute_scorecard,
    is_substantive,
)
from ilga_graph.models import Bill, Member


# ── is_substantive ────────────────────────────────────────────────────────────


class TestIsSubstantive:
    def test_hb_is_substantive(self) -> None:
        assert is_substantive("HB0100") is True

    def test_sb_is_substantive(self) -> None:
        assert is_substantive("SB1527") is True

    def test_hr_is_not_substantive(self) -> None:
        assert is_substantive("HR0010") is False

    def test_sr_is_not_substantive(self) -> None:
        assert is_substantive("SR0020") is False

    def test_hjr_is_not_substantive(self) -> None:
        assert is_substantive("HJR0008") is False

    def test_sjr_is_not_substantive(self) -> None:
        assert is_substantive("SJR0005") is False

    def test_case_insensitive(self) -> None:
        assert is_substantive("hb0001") is True
        assert is_substantive("sb0001") is True
        assert is_substantive("hr0001") is False
        assert is_substantive("sjr0001") is False

    def test_unknown_prefix_is_not_substantive(self) -> None:
        assert is_substantive("XB0001") is False


# ── compute_scorecard (single member, backward compat) ───────────────────────


class TestComputeScorecard:
    def test_basic_scorecard(self, sample_member: Member) -> None:
        sc = compute_scorecard(sample_member)
        assert sc.primary_bill_count == 1
        assert sc.heat_score == 1
        assert sc.law_heat_score == 1  # SB1527 is substantive
        assert sc.resolutions_count == 0

    def test_mixed_bills(self, mixed_bill_member: Member) -> None:
        sc = compute_scorecard(mixed_bill_member)
        assert sc.primary_bill_count == 5
        assert sc.law_heat_score == 3  # HB0100, SB0200, HB0300
        assert sc.resolutions_count == 2  # HR0010, SJR0005
        assert sc.law_success_rate > 0  # 2/3 passed

    def test_no_bills(self, member_no_bills: Member) -> None:
        sc = compute_scorecard(member_no_bills)
        assert sc.primary_bill_count == 0
        assert sc.law_heat_score == 0
        assert sc.law_success_rate == 0.0
        assert sc.magnet_score == 0.0
        assert sc.bridge_score == 0.0
        assert sc.resolutions_count == 0

    def test_resolutions_only(self, member_resolutions_only: Member) -> None:
        sc = compute_scorecard(member_resolutions_only)
        assert sc.law_heat_score == 0
        assert sc.resolutions_count == 2
        assert sc.law_success_rate == 0.0

    def test_magnet_and_bridge_default_zero_for_single(
        self, sample_member: Member,
    ) -> None:
        """Single-member compute_scorecard cannot calculate network metrics."""
        sc = compute_scorecard(sample_member)
        assert sc.magnet_score == 0.0
        assert sc.bridge_score == 0.0


# ── compute_all_scorecards (batch, full Legislative DNA) ─────────────────────


class TestComputeAllScorecards:
    def test_basic_keys(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
        cosponsor_democrat: Member,
    ) -> None:
        members = [mixed_bill_member, cosponsor_republican, cosponsor_democrat]
        result = compute_all_scorecards(members)
        assert mixed_bill_member.id in result
        assert cosponsor_republican.id in result
        assert cosponsor_democrat.id in result

    def test_law_counts(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
    ) -> None:
        members = [mixed_bill_member, cosponsor_republican]
        sc = compute_all_scorecards(members)[mixed_bill_member.id]
        assert sc.law_heat_score == 3
        assert sc.resolutions_count == 2
        assert sc.primary_bill_count == 5

    def test_law_success_rate(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
    ) -> None:
        members = [mixed_bill_member, cosponsor_republican]
        sc = compute_all_scorecards(members)[mixed_bill_member.id]
        # 2 passed laws out of 3
        assert sc.law_success_rate == round(2 / 3, 4)

    def test_passed_counts_add_up(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
    ) -> None:
        """Total passed = laws passed + resolutions passed; success rate = law_passed / law_heat."""
        members = [mixed_bill_member, cosponsor_republican]
        sc = compute_all_scorecards(members)[mixed_bill_member.id]
        assert sc.law_passed_count == 2
        assert sc.resolutions_passed_count == 2
        assert sc.passed_count == sc.law_passed_count + sc.resolutions_passed_count
        assert sc.law_success_rate == round(sc.law_passed_count / sc.law_heat_score, 4)

    def test_magnet_score(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
        cosponsor_democrat: Member,
    ) -> None:
        """
        HB0100 has 2 co-sponsors (Bob + Carol).
        SB0200 has 1 co-sponsor (Carol).
        HB0300 has 0 co-sponsors.
        Total co-sponsors on laws = 3, total laws = 3.
        magnet_score = 3/3 = 1.0
        """
        members = [mixed_bill_member, cosponsor_republican, cosponsor_democrat]
        sc = compute_all_scorecards(members)[mixed_bill_member.id]
        assert sc.magnet_score == 1.0

    def test_bridge_score(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
        cosponsor_democrat: Member,
    ) -> None:
        """
        Alice (Democrat) has 3 laws:
          - HB0100: co-sponsors include Bob (Republican) -> bridged
          - SB0200: co-sponsors include Carol (Democrat) -> NOT bridged
          - HB0300: no co-sponsors -> NOT bridged
        bridge_score = 1/3 ≈ 0.3333
        """
        members = [mixed_bill_member, cosponsor_republican, cosponsor_democrat]
        sc = compute_all_scorecards(members)[mixed_bill_member.id]
        assert sc.bridge_score == round(1 / 3, 4)

    def test_zero_laws_no_division_error(
        self,
        member_no_bills: Member,
        cosponsor_republican: Member,
    ) -> None:
        members = [member_no_bills, cosponsor_republican]
        sc = compute_all_scorecards(members)[member_no_bills.id]
        assert sc.law_heat_score == 0
        assert sc.magnet_score == 0.0
        assert sc.bridge_score == 0.0
        assert sc.law_success_rate == 0.0

    def test_resolutions_only_member(
        self,
        member_resolutions_only: Member,
        cosponsor_republican: Member,
    ) -> None:
        members = [member_resolutions_only, cosponsor_republican]
        sc = compute_all_scorecards(members)[member_resolutions_only.id]
        assert sc.law_heat_score == 0
        assert sc.resolutions_count == 2
        assert sc.magnet_score == 0.0
        assert sc.bridge_score == 0.0

    def test_cosponsor_scorecard_has_zero_primary(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
    ) -> None:
        """Co-sponsor-only members have 0 primary bills."""
        members = [mixed_bill_member, cosponsor_republican]
        sc = compute_all_scorecards(members)[cosponsor_republican.id]
        assert sc.primary_bill_count == 0
        assert sc.law_heat_score == 0
        assert sc.magnet_score == 0.0

    def test_member_lookup_built_internally(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
    ) -> None:
        """compute_all_scorecards works without explicit member_lookup."""
        members = [mixed_bill_member, cosponsor_republican]
        result = compute_all_scorecards(members)
        assert len(result) == 2


# ── classify_bill_status (existing, verify still works) ──────────────────────


class TestClassifyBillStatus:
    def test_passed(self) -> None:
        assert classify_bill_status("Public Act 104-0001") == BillStatus.PASSED

    def test_signed(self) -> None:
        assert classify_bill_status("Signed by Governor") == BillStatus.PASSED

    def test_vetoed(self) -> None:
        assert classify_bill_status("Total Veto") == BillStatus.VETOED

    def test_stuck(self) -> None:
        assert classify_bill_status("Rule 3-9(a) / Re-referred to Assignments") == BillStatus.STUCK

    def test_in_progress(self) -> None:
        assert classify_bill_status("Second Reading") == BillStatus.IN_PROGRESS
