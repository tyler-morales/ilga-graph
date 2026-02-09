from __future__ import annotations

from pathlib import Path

from ilga_graph.analytics import MemberScorecard
from ilga_graph.exporter import ObsidianExporter
from ilga_graph.models import Bill, CareerRange, Member


class TestRenderBill:
    def test_frontmatter_contains_iso_date(self, sample_bill: Bill) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_bill(sample_bill)
        assert "last_action_date_iso: 2025-06-02" in result

    def test_frontmatter_contains_original_date(self, sample_bill: Bill) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_bill(sample_bill)
        assert 'last_action_date: "6/2/2025"' in result

    def test_frontmatter_contains_bill_number(self, sample_bill: Bill) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_bill(sample_bill)
        assert "bill_number: SB1527" in result

    def test_senate_chamber_tag(self, sample_bill: Bill) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_bill(sample_bill)
        assert "chamber/senate" in result

    def test_house_chamber_tag(self, sample_bill_house: Bill) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_bill(sample_bill_house)
        assert "chamber/house" in result

    def test_sponsor_wikilink(self, sample_bill: Bill) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_bill(sample_bill)
        assert "[[Sue Rezin]]" in result

    def test_cosponsors_rendered(self, sample_bill: Bill) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_bill(sample_bill, cosponsors=["Alice", "Bob"])
        assert "[[Alice]]" in result
        assert "[[Bob]]" in result

    def test_no_cosponsors(self, sample_bill: Bill) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_bill(sample_bill, cosponsors=[])
        assert "- None" in result

    def test_invalid_date_produces_empty_iso(self) -> None:
        bill = Bill(
            bill_number="SB0001",
            leg_id="1",
            description="Test",
            chamber="S",
            last_action="Test",
            last_action_date="invalid-date",
            primary_sponsor="Test",
        )
        exporter = ObsidianExporter()
        result = exporter._render_bill(bill)
        assert "last_action_date_iso: \n" in result


class TestRenderMember:
    def test_frontmatter_contains_career_start_year(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "career_start_year: 2015" in result

    def test_frontmatter_no_career_ranges(self, sample_member_no_career: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member_no_career)
        assert "career_start_year: \n" in result

    def test_frontmatter_contains_party(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "party: Republican" in result

    def test_primary_legislation_links(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "[[SB1527]]" in result

    def test_house_member_chamber_tag(self, sample_member_house: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member_house)
        assert "chamber/house" in result
        assert "chamber: House" in result

    def test_house_member_primary_legislation(self, sample_member_house: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member_house)
        assert "[[HB0054]]" in result

    # ── Phase 3: new frontmatter fields ──

    def test_frontmatter_contains_magnet_score(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "magnet_score:" in result

    def test_frontmatter_contains_bridge_score(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "bridge_score:" in result

    def test_frontmatter_contains_law_success_rate(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "law_success_rate:" in result

    def test_frontmatter_contains_aligned_fields(self, sample_member: Member) -> None:
        """Frontmatter keys must match scorecard terminology."""
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "bills_introduced:" in result
        assert "laws_passed:" in result
        assert "resolutions_filed:" in result
        assert "resolutions_passed:" in result
        assert "resolution_pass_rate:" in result
        assert "total_primary_bills:" in result
        assert "total_passed:" in result
        assert "overall_pass_rate:" in result
        # Old names should be gone
        assert "heat_score:" not in result
        assert "effectiveness_score:" not in result
        assert "bills_total:" not in result
        assert "bills_passed:" not in result

    def test_scorecard_has_lawmaking_section(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "### Lawmaking (HB/SB)" in result

    def test_scorecard_has_resolutions_section(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "### Resolutions (HR/SR/HJR/SJR)" in result

    def test_scorecard_has_overall_section(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_member(sample_member)
        assert "### Overall" in result


class TestScorecardBadges:
    """Test context interpretation badges in the scorecard."""

    def test_coalition_builder_badge(self) -> None:
        sc = MemberScorecard(
            primary_bill_count=5, passed_count=3, vetoed_count=0,
            stuck_count=1, in_progress_count=1, success_rate=0.6,
            heat_score=5, effectiveness_score=3.0,
            law_heat_score=5, law_success_rate=0.6,
            magnet_score=12.0, bridge_score=0.1,
            resolutions_count=0, resolution_pass_rate=0.0,
        )
        exporter = ObsidianExporter()
        result = exporter._render_scorecard(sc)
        assert "Coalition Builder" in result

    def test_bipartisan_bridge_badge(self) -> None:
        sc = MemberScorecard(
            primary_bill_count=5, passed_count=3, vetoed_count=0,
            stuck_count=1, in_progress_count=1, success_rate=0.6,
            heat_score=5, effectiveness_score=3.0,
            law_heat_score=5, law_success_rate=0.6,
            magnet_score=2.0, bridge_score=0.25,
            resolutions_count=0, resolution_pass_rate=0.0,
        )
        exporter = ObsidianExporter()
        result = exporter._render_scorecard(sc)
        assert "Bipartisan Bridge" in result

    def test_ceremonial_focus_badge(self) -> None:
        sc = MemberScorecard(
            primary_bill_count=5, passed_count=2, vetoed_count=0,
            stuck_count=0, in_progress_count=0, success_rate=0.4,
            heat_score=5, effectiveness_score=2.0,
            law_heat_score=1, law_success_rate=1.0,
            magnet_score=0.0, bridge_score=0.0,
            resolutions_count=4, resolution_pass_rate=0.25,
        )
        exporter = ObsidianExporter()
        result = exporter._render_scorecard(sc)
        assert "Ceremonial Focus" in result

    def test_no_badges_when_thresholds_not_met(self) -> None:
        sc = MemberScorecard(
            primary_bill_count=2, passed_count=1, vetoed_count=0,
            stuck_count=0, in_progress_count=1, success_rate=0.5,
            heat_score=2, effectiveness_score=1.0,
            law_heat_score=2, law_success_rate=0.5,
            magnet_score=3.0, bridge_score=0.1,
            resolutions_count=0, resolution_pass_rate=0.0,
        )
        exporter = ObsidianExporter()
        result = exporter._render_scorecard(sc)
        assert "Coalition Builder" not in result
        assert "Bipartisan Bridge" not in result
        assert "Ceremonial Focus" not in result

    def test_lawmaking_passed_matches_success_rate_formula(self) -> None:
        """Success Rate = Passed / Bills Introduced."""
        sc = MemberScorecard(
            primary_bill_count=55, passed_count=9, vetoed_count=0,
            stuck_count=10, in_progress_count=36, success_rate=round(9 / 55, 4),
            heat_score=55, effectiveness_score=9.0,
            law_heat_score=50, law_passed_count=5, law_success_rate=0.10,
            magnet_score=1.5, bridge_score=0.14,
            resolutions_count=5, resolutions_passed_count=4,
            resolution_pass_rate=0.8,
        )
        exporter = ObsidianExporter()
        result = exporter._render_scorecard(sc)
        assert "| Bills Introduced | 50 |" in result
        assert "| Passed | 5 |" in result
        assert "| Success Rate | 10.0% |" in result
        assert "| Resolutions Filed | 5 |" in result
        assert "| Pass Rate | 80.0% |" in result
        assert "| Total Passed | 9 |" in result
        assert "| Overall Pass Rate | 16.4% |" in result
        # Formula column present
        assert "| Formula |" in result


class TestSafeFilename:
    def test_normal_name(self) -> None:
        exporter = ObsidianExporter()
        assert exporter._safe_filename("Neil Anderson") == "Neil Anderson"

    def test_name_with_special_chars(self) -> None:
        exporter = ObsidianExporter()
        result = exporter._safe_filename("Li Arellano, Jr.")
        assert "Li Arellano, Jr." == result

    def test_name_with_disallowed_chars(self) -> None:
        exporter = ObsidianExporter()
        result = exporter._safe_filename("Test/Name")
        assert "/" not in result


class TestRenderCareerRanges:
    def test_with_ranges(self, sample_member: Member) -> None:
        exporter = ObsidianExporter()
        result = exporter._render_career_ranges(sample_member)
        assert "2015 - Present" in result
        assert "(Senate)" in result

    def test_with_ended_range(self) -> None:
        member = Member(
            id="1", name="T", member_url="", chamber="Senate",
            party="D", district="1", bio_text="",
            career_ranges=[CareerRange(start_year=2010, end_year=2014)],
        )
        exporter = ObsidianExporter()
        result = exporter._render_career_ranges(member)
        assert "2010 - 2014" in result

    def test_no_ranges_uses_text(self) -> None:
        member = Member(
            id="1", name="T", member_url="", chamber="Senate",
            party="D", district="1", bio_text="",
            career_timeline_text="2020-",
        )
        exporter = ObsidianExporter()
        result = exporter._render_career_ranges(member)
        assert result == "2020-"

    def test_no_ranges_no_text(self) -> None:
        member = Member(
            id="1", name="T", member_url="", chamber="Senate",
            party="D", district="1", bio_text="",
        )
        exporter = ObsidianExporter()
        result = exporter._render_career_ranges(member)
        assert result == "None"


class TestWriteBaseFiles:
    def test_generates_bills_base(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter(vault_root=tmp_path)
        exporter._write_base_files()
        bills_base = tmp_path / "Bills by Date.base"
        assert bills_base.exists()
        content = bills_base.read_text()
        assert 'file.inFolder("Bills")' in content
        assert "last_action_date_iso" in content

    def test_generates_members_base(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter(vault_root=tmp_path)
        exporter._write_base_files()
        members_base = tmp_path / "Members by Career.base"
        assert members_base.exists()
        content = members_base.read_text()
        assert 'file.inFolder("Members")' in content
        assert "career_start_year" in content

    def test_heat_base_includes_dna_columns(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter(vault_root=tmp_path)
        exporter._write_base_files()
        heat_base = tmp_path / "Members by Heat Score.base"
        assert heat_base.exists()
        content = heat_base.read_text()
        assert "magnet_score" in content
        assert "bridge_score" in content
        assert "law_success_rate" in content
        assert "total_primary_bills" in content
        assert "total_passed" in content
