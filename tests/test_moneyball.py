"""Tests for the Moneyball analytics engine."""

from __future__ import annotations

import pytest

from ilga_graph.analytics import (
    PipelineStage,
    classify_pipeline_stage,
    compute_all_scorecards,
    pipeline_depth,
)
from ilga_graph.models import Bill, Member
from ilga_graph.moneyball import (
    MoneyballProfile,
    MoneyballWeights,
    _assign_badges,
    _build_cosponsor_edges,
    avg_pipeline_depth,
    compute_moneyball,
    degree_centrality,
    is_leadership,
)

# ── Pipeline stage classification ─────────────────────────────────────────────


class TestClassifyPipelineStage:
    def test_signed_public_act(self) -> None:
        assert classify_pipeline_stage("Public Act 104-0001") == PipelineStage.SIGNED

    def test_signed_by_governor(self) -> None:
        assert classify_pipeline_stage("Signed by Governor") == PipelineStage.SIGNED

    def test_crossed_both_houses(self) -> None:
        assert classify_pipeline_stage("Adopted Both Houses") == PipelineStage.CROSSED

    def test_sent_to_governor(self) -> None:
        assert classify_pipeline_stage("Sent to the Governor") == PipelineStage.CROSSED

    def test_chamber_passed(self) -> None:
        assert classify_pipeline_stage("Third Reading - Passed") == PipelineStage.CHAMBER_PASSED

    def test_second_reading(self) -> None:
        assert (
            classify_pipeline_stage("Second Reading - Standard Debate")
            == PipelineStage.SECOND_READING
        )

    def test_placed_on_calendar(self) -> None:
        assert (
            classify_pipeline_stage("Placed on Calendar Order of 2nd Reading")
            == PipelineStage.SECOND_READING
        )

    def test_committee_do_pass(self) -> None:
        assert classify_pipeline_stage("Do Pass / Short Debate") == PipelineStage.COMMITTEE_PASSED

    def test_referred_to_assignments(self) -> None:
        assert classify_pipeline_stage("Referred to Assignments") == PipelineStage.COMMITTEE

    def test_re_referred(self) -> None:
        assert classify_pipeline_stage("Re-referred to Assignments") == PipelineStage.COMMITTEE

    def test_filed_default(self) -> None:
        assert classify_pipeline_stage("Filed with Secretary") == PipelineStage.FILED

    def test_unknown_action(self) -> None:
        assert classify_pipeline_stage("Something unprecedented happened") == PipelineStage.FILED


class TestPipelineDepth:
    def test_signed_is_6(self) -> None:
        assert pipeline_depth("Public Act 104-0001") == 6

    def test_filed_is_0(self) -> None:
        assert pipeline_depth("Filed with Secretary") == 0

    def test_committee_is_1(self) -> None:
        assert pipeline_depth("Referred to Assignments") == 1


# ── Leadership detection ──────────────────────────────────────────────────────


class TestIsLeadership:
    def _make_member(self, role: str) -> Member:
        return Member(
            id="test",
            name="Test",
            member_url="",
            chamber="House",
            party="Democrat",
            district="1",
            bio_text="",
            role=role,
        )

    def test_speaker(self) -> None:
        assert is_leadership(self._make_member("Speaker of the House")) is True

    def test_minority_leader(self) -> None:
        assert is_leadership(self._make_member("Minority Leader")) is True

    def test_majority_whip(self) -> None:
        assert is_leadership(self._make_member("Majority Whip")) is True

    def test_caucus_chair(self) -> None:
        assert is_leadership(self._make_member("Republican Caucus Chair")) is True

    def test_representative_not_leadership(self) -> None:
        assert is_leadership(self._make_member("Representative")) is False

    def test_senator_not_leadership(self) -> None:
        assert is_leadership(self._make_member("Senator")) is False

    def test_empty_role(self) -> None:
        assert is_leadership(self._make_member("")) is False

    def test_president_of_senate(self) -> None:
        assert is_leadership(self._make_member("President of the Senate")) is True

    def test_assistant_majority_leader(self) -> None:
        assert is_leadership(self._make_member("Assistant Majority Leader")) is True


# ── Co-sponsor network ────────────────────────────────────────────────────────


class TestBuildCosponsorEdges:
    def test_shared_bill_creates_edge(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
    ) -> None:
        adjacency = _build_cosponsor_edges([mixed_bill_member, cosponsor_republican])
        assert cosponsor_republican.id in adjacency[mixed_bill_member.id]
        assert mixed_bill_member.id in adjacency[cosponsor_republican.id]

    def test_no_shared_bills_no_edge(
        self,
        mixed_bill_member: Member,
        member_no_bills: Member,
    ) -> None:
        adjacency = _build_cosponsor_edges([mixed_bill_member, member_no_bills])
        assert member_no_bills.id not in adjacency[mixed_bill_member.id]

    def test_multiple_cosponsors(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
        cosponsor_democrat: Member,
    ) -> None:
        members = [mixed_bill_member, cosponsor_republican, cosponsor_democrat]
        adjacency = _build_cosponsor_edges(members)
        # Alice connects to both Bob and Carol
        assert len(adjacency[mixed_bill_member.id]) == 2
        # Bob and Carol also share HB0100, so they connect to each other
        assert cosponsor_democrat.id in adjacency[cosponsor_republican.id]


class TestDegreeCentrality:
    def test_triangle_graph(self) -> None:
        adj = {"A": {"B", "C"}, "B": {"A", "C"}, "C": {"A", "B"}}
        dc = degree_centrality(adj)
        assert dc["A"] == pytest.approx(1.0)
        assert dc["B"] == pytest.approx(1.0)

    def test_star_graph(self) -> None:
        adj = {"hub": {"a", "b", "c"}, "a": {"hub"}, "b": {"hub"}, "c": {"hub"}}
        dc = degree_centrality(adj)
        assert dc["hub"] == pytest.approx(1.0)
        assert dc["a"] == pytest.approx(1 / 3)

    def test_isolate(self) -> None:
        adj = {"A": {"B"}, "B": {"A"}}
        adj["C"] = set()  # isolate
        dc = degree_centrality(adj)
        assert dc["C"] == 0.0

    def test_single_node(self) -> None:
        dc = degree_centrality({"A": set()})
        assert dc["A"] == 0.0


# ── Average pipeline depth ────────────────────────────────────────────────────


class TestAvgPipelineDepth:
    def test_mixed_bills(
        self,
        sample_bill_passed: Bill,
        sample_bill_stuck: Bill,
        sample_resolution_hr: Bill,
    ) -> None:
        # Only HB/SB count: HB0100 (Public Act = 6), HB0300 (Referred = 1)
        bills = [sample_bill_passed, sample_bill_stuck, sample_resolution_hr]
        depth = avg_pipeline_depth(bills)
        assert depth == pytest.approx((6 + 1) / 2)

    def test_no_substantive_bills(self, sample_resolution_hr: Bill) -> None:
        assert avg_pipeline_depth([sample_resolution_hr]) == 0.0

    def test_empty_list(self) -> None:
        assert avg_pipeline_depth([]) == 0.0


# ── Full Moneyball compute ────────────────────────────────────────────────────


class TestComputeMoneyball:
    def test_profiles_created(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
        cosponsor_democrat: Member,
    ) -> None:
        members = [mixed_bill_member, cosponsor_republican, cosponsor_democrat]
        report = compute_moneyball(members)
        assert mixed_bill_member.id in report.profiles
        assert cosponsor_republican.id in report.profiles
        assert cosponsor_democrat.id in report.profiles

    def test_mvp_house_identified(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
        cosponsor_democrat: Member,
    ) -> None:
        members = [mixed_bill_member, cosponsor_republican, cosponsor_democrat]
        report = compute_moneyball(members)
        # Alice should be MVP since she's non-leadership and has the most laws
        assert report.mvp_house_non_leadership is not None

    def test_leadership_detected(self) -> None:
        leader = Member(
            id="L1",
            name="Leader",
            member_url="",
            chamber="House",
            party="Democrat",
            district="1",
            bio_text="",
            role="Speaker of the House",
            sponsored_bills=[],
            co_sponsor_bills=[],
        )
        regular = Member(
            id="R1",
            name="Regular",
            member_url="",
            chamber="House",
            party="Democrat",
            district="2",
            bio_text="",
            role="Representative",
            sponsored_bills=[],
            co_sponsor_bills=[],
        )
        report = compute_moneyball([leader, regular])
        assert report.profiles["L1"].is_leadership is True
        assert report.profiles["R1"].is_leadership is False

    def test_rankings_sorted_by_score(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
        member_no_bills: Member,
    ) -> None:
        members = [mixed_bill_member, cosponsor_republican, member_no_bills]
        report = compute_moneyball(members)
        # Alice (with laws) should rank above members with no laws
        scores = [report.profiles[mid].moneyball_score for mid in report.rankings_overall]
        assert scores == sorted(scores, reverse=True)

    def test_effectiveness_rate_matches_scorecard(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
    ) -> None:
        members = [mixed_bill_member, cosponsor_republican]
        scorecards = compute_all_scorecards(members)
        report = compute_moneyball(members, scorecards=scorecards)
        mb = report.profiles[mixed_bill_member.id]
        sc = scorecards[mixed_bill_member.id]
        assert mb.effectiveness_rate == sc.law_success_rate

    def test_custom_weights(
        self,
        mixed_bill_member: Member,
        cosponsor_republican: Member,
    ) -> None:
        members = [mixed_bill_member, cosponsor_republican]
        # All weight on effectiveness
        w = MoneyballWeights(effectiveness=1.0, pipeline=0, magnet=0, bridge=0, centrality=0)
        report = compute_moneyball(members, weights=w)
        mb = report.profiles[mixed_bill_member.id]
        # Score should be effectiveness_rate * 100
        expected = round(mb.effectiveness_rate * 100, 2)
        assert mb.moneyball_score == expected

    def test_zero_bills_no_division_error(
        self,
        member_no_bills: Member,
    ) -> None:
        report = compute_moneyball([member_no_bills])
        mb = report.profiles[member_no_bills.id]
        assert mb.moneyball_score == 0.0
        assert mb.pipeline_depth_avg == 0.0
        assert mb.network_centrality == 0.0

    def test_chamber_rankings_separate(self) -> None:
        house_member = Member(
            id="H1",
            name="House Rep",
            member_url="",
            chamber="House",
            party="Democrat",
            district="1",
            bio_text="",
            sponsored_bills=[],
            co_sponsor_bills=[],
        )
        senate_member = Member(
            id="S1",
            name="Senator",
            member_url="",
            chamber="Senate",
            party="Republican",
            district="1",
            bio_text="",
            sponsored_bills=[],
            co_sponsor_bills=[],
        )
        report = compute_moneyball([house_member, senate_member])
        assert "H1" in report.rankings_house
        assert "H1" not in report.rankings_senate
        assert "S1" in report.rankings_senate
        assert "S1" not in report.rankings_house


# ── Badges ────────────────────────────────────────────────────────────────────


class TestBadges:
    def _make_profile(self, **kwargs) -> MoneyballProfile:
        defaults = dict(
            member_id="test",
            member_name="Test",
            chamber="House",
            party="Democrat",
            district="1",
            role="Representative",
            is_leadership=False,
            laws_filed=10,
            laws_passed=3,
            effectiveness_rate=0.3,
            magnet_score=5.0,
            bridge_score=0.1,
            resolutions_filed=2,
            resolutions_passed=2,
            pipeline_depth_avg=2.0,
            pipeline_depth_normalized=0.33,
            network_centrality=0.3,
            unique_collaborators=10,
            total_primary_bills=12,
            total_passed=5,
        )
        defaults.update(kwargs)
        return MoneyballProfile(**defaults)

    def test_closer_badge(self) -> None:
        p = self._make_profile(effectiveness_rate=0.30)
        badges = _assign_badges(p)
        assert "Closer" in badges

    def test_no_closer_low_effectiveness(self) -> None:
        p = self._make_profile(effectiveness_rate=0.10)
        badges = _assign_badges(p)
        assert "Closer" not in badges

    def test_coalition_builder(self) -> None:
        p = self._make_profile(magnet_score=12.0)
        badges = _assign_badges(p)
        assert "Coalition Builder" in badges

    def test_bipartisan_bridge(self) -> None:
        p = self._make_profile(bridge_score=0.25)
        badges = _assign_badges(p)
        assert "Bipartisan Bridge" in badges

    def test_pipeline_driver(self) -> None:
        p = self._make_profile(pipeline_depth_avg=4.5, laws_filed=5)
        badges = _assign_badges(p)
        assert "Pipeline Driver" in badges

    def test_network_hub(self) -> None:
        p = self._make_profile(network_centrality=0.6)
        badges = _assign_badges(p)
        assert "Network Hub" in badges

    def test_wide_tent(self) -> None:
        p = self._make_profile(unique_collaborators=25)
        badges = _assign_badges(p)
        assert "Wide Tent" in badges

    def test_hidden_gem(self) -> None:
        p = self._make_profile(
            is_leadership=False,
            laws_passed=3,
            effectiveness_rate=0.20,
            magnet_score=3.0,
        )
        badges = _assign_badges(p)
        assert "Hidden Gem" in badges

    def test_hidden_gem_not_for_leadership(self) -> None:
        p = self._make_profile(
            is_leadership=True,
            laws_passed=3,
            effectiveness_rate=0.20,
            magnet_score=3.0,
        )
        badges = _assign_badges(p)
        assert "Hidden Gem" not in badges

    def test_ceremonial_focus(self) -> None:
        p = self._make_profile(laws_filed=3, resolutions_filed=8)
        badges = _assign_badges(p)
        assert "Ceremonial Focus" in badges
