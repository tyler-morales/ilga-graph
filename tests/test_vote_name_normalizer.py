"""Tests for the vote-name normalizer module."""

from __future__ import annotations

from ilga_graph.models import Member, VoteEvent
from ilga_graph.vote_name_normalizer import (
    _build_variant_map,
    _display_fallback,
    _norm_key,
    _parse_member_name,
    _parse_vote_name,
    _resolve_name,
    normalize_vote_events,
)

# ── Helper to build minimal Member objects ───────────────────────────────────


def _member(name: str, chamber: str = "Senate", mid: str = "") -> Member:
    return Member(
        id=mid or name,
        name=name,
        member_url="",
        chamber=chamber,
        party="D",
        district="1",
        bio_text="",
    )


# ── _norm_key tests ──────────────────────────────────────────────────────────


class TestNormKey:
    def test_lowercase_and_strip_periods(self) -> None:
        assert _norm_key("Murphy, Laura M.") == "murphy, laura m"

    def test_preserves_comma_no_space(self) -> None:
        assert _norm_key("Davis,Jed") == "davis,jed"

    def test_collapses_whitespace(self) -> None:
        assert _norm_key("Costa  Howard") == "costa howard"

    def test_strips_outer_whitespace(self) -> None:
        assert _norm_key("  Murphy ") == "murphy"


# ── _parse_member_name tests ────────────────────────────────────────────────


class TestParseMemberName:
    def test_simple_two_part(self) -> None:
        assert _parse_member_name("Neil Anderson") == ("Neil", "Anderson", "")

    def test_with_middle_initial(self) -> None:
        assert _parse_member_name("Laura M. Murphy") == ("Laura M.", "Murphy", "")

    def test_with_jr_suffix(self) -> None:
        assert _parse_member_name("Elgie R. Sims, Jr.") == ("Elgie R.", "Sims", "Jr.")

    def test_with_jr_suffix_no_period(self) -> None:
        assert _parse_member_name("Marcus C. Evans, Jr") == ("Marcus C.", "Evans", "Jr")

    def test_quoted_nickname(self) -> None:
        first, last, suffix = _parse_member_name('Emanuel "Chris" Welch')
        assert last == "Welch"
        assert suffix == ""
        assert "Chris" in first

    def test_single_word(self) -> None:
        assert _parse_member_name("Cher") == ("", "Cher", "")


# ── _parse_vote_name tests ──────────────────────────────────────────────────


class TestParseVoteName:
    def test_last_only(self) -> None:
        assert _parse_vote_name("Murphy") == ("Murphy", "", "")

    def test_comma_spaced(self) -> None:
        assert _parse_vote_name("Murphy, Laura M") == ("Murphy", "Laura M", "")

    def test_comma_no_space(self) -> None:
        assert _parse_vote_name("Davis,Jed") == ("Davis", "Jed", "")

    def test_suffix_iii(self) -> None:
        assert _parse_vote_name("Harris III, Napoleon") == ("Harris", "Napoleon", "III")

    def test_suffix_jr(self) -> None:
        assert _parse_vote_name("Sims Jr., Elgie R") == ("Sims", "Elgie R", "Jr.")


# ── _build_variant_map tests ────────────────────────────────────────────────


class TestBuildVariantMap:
    def test_unique_last_name_resolves(self) -> None:
        members = [_member("Laura M. Murphy", "Senate")]
        vmap = _build_variant_map(members)
        # Bare last name should resolve
        assert vmap[("Senate", "murphy")] == "Laura M. Murphy"
        # "Murphy, Laura M" (committee PDF format)
        assert vmap[("Senate", "murphy, laura m")] == "Laura M. Murphy"
        # "Murphy,Laura M" (floor duplicate format — unlikely but tested)
        assert vmap[("Senate", "murphy,laura m")] == "Laura M. Murphy"

    def test_duplicate_last_name_not_bare(self) -> None:
        members = [
            _member("Jed Davis", "House", "d1"),
            _member("Lisa Davis", "House", "d2"),
        ]
        vmap = _build_variant_map(members)
        # Bare "davis" should NOT be in the map (ambiguous)
        assert ("House", "davis") not in vmap
        # But specific variants should work
        assert vmap[("House", "davis, jed")] == "Jed Davis"
        assert vmap[("House", "davis,jed")] == "Jed Davis"
        assert vmap[("House", "davis, lisa")] == "Lisa Davis"

    def test_suffix_member_variants(self) -> None:
        members = [_member("Elgie R. Sims, Jr.", "Senate")]
        vmap = _build_variant_map(members)
        # Canonical name
        assert vmap[("Senate", _norm_key("Elgie R. Sims, Jr."))] == "Elgie R. Sims, Jr."
        # Committee PDF: "Sims Jr., Elgie R"
        assert vmap[("Senate", "sims jr, elgie r")] == "Elgie R. Sims, Jr."

    def test_cross_chamber_isolation(self) -> None:
        members = [
            _member("John Smith", "Senate", "s1"),
            _member("Jane Smith", "House", "h1"),
        ]
        vmap = _build_variant_map(members)
        assert vmap[("Senate", "smith")] == "John Smith"
        assert vmap[("House", "smith")] == "Jane Smith"

    def test_full_canonical_resolves(self) -> None:
        members = [_member("John F. Curran", "Senate")]
        vmap = _build_variant_map(members)
        assert vmap[("Senate", "john f curran")] == "John F. Curran"
        assert vmap[("Senate", "curran, john f")] == "John F. Curran"
        assert vmap[("Senate", "curran")] == "John F. Curran"


# ── _display_fallback tests ─────────────────────────────────────────────────


class TestDisplayFallback:
    def test_bare_last_name(self) -> None:
        assert _display_fallback("Murphy") == "Murphy"

    def test_comma_format_normalized(self) -> None:
        assert _display_fallback("Murphy,Laura M") == "Murphy, Laura M"

    def test_already_spaced_comma(self) -> None:
        assert _display_fallback("Murphy, Laura M") == "Murphy, Laura M"


# ── _resolve_name tests ─────────────────────────────────────────────────────


class TestResolveName:
    def setup_method(self) -> None:
        self.members = [
            _member("Laura M. Murphy", "Senate"),
            _member("Jed Davis", "House", "d1"),
            _member("Lisa Davis", "House", "d2"),
        ]
        self.vmap = _build_variant_map(self.members)

    def test_resolve_bare_last(self) -> None:
        assert _resolve_name("Murphy", "Senate", self.vmap) == "Laura M. Murphy"

    def test_resolve_comma_spaced(self) -> None:
        assert _resolve_name("Murphy, Laura M", "Senate", self.vmap) == "Laura M. Murphy"

    def test_resolve_comma_no_space(self) -> None:
        assert _resolve_name("Davis,Jed", "House", self.vmap) == "Jed Davis"

    def test_ambiguous_bare_falls_back(self) -> None:
        # "Davis" is ambiguous in House; should NOT resolve, just return raw
        result = _resolve_name("Davis", "House", self.vmap)
        assert result == "Davis"

    def test_unknown_with_comma_falls_back(self) -> None:
        result = _resolve_name("Zyx, Abc", "Senate", self.vmap)
        assert result == "Zyx, Abc"

    def test_unknown_bare_falls_back(self) -> None:
        result = _resolve_name("Zyx", "Senate", self.vmap)
        assert result == "Zyx"

    def test_wrong_chamber_falls_back(self) -> None:
        # Murphy is in Senate, not House
        result = _resolve_name("Murphy", "House", self.vmap)
        assert result == "Murphy"


# ── normalize_vote_events integration ────────────────────────────────────────


class TestNormalizeVoteEvents:
    def test_replaces_names_in_place(self) -> None:
        members = {
            "Laura M. Murphy": _member("Laura M. Murphy", "Senate"),
            "Neil Anderson": _member("Neil Anderson", "Senate"),
            "Jed Davis": _member("Jed Davis", "House", "d1"),
            "Lisa Davis": _member("Lisa Davis", "House", "d2"),
        }
        events = [
            VoteEvent(
                bill_number="SB0001",
                date="Jan 1, 2025",
                description="Third Reading",
                chamber="Senate",
                yea_votes=["Murphy", "Anderson"],
                nay_votes=["Anderson, Neil"],
            ),
            VoteEvent(
                bill_number="HB0001",
                date="Jan 1, 2025",
                description="Third Reading",
                chamber="House",
                yea_votes=["Davis,Jed", "Davis,Lisa"],
                nay_votes=[],
            ),
        ]
        normalize_vote_events(events, members)

        # Senate event
        assert events[0].yea_votes == ["Laura M. Murphy", "Neil Anderson"]
        assert events[0].nay_votes == ["Neil Anderson"]

        # House event
        assert events[1].yea_votes == ["Jed Davis", "Lisa Davis"]

    def test_preserves_unknown_names(self) -> None:
        members = {"Laura M. Murphy": _member("Laura M. Murphy", "Senate")}
        events = [
            VoteEvent(
                bill_number="SB0001",
                date="",
                description="",
                chamber="Senate",
                yea_votes=["Zyxabc"],
            ),
        ]
        normalize_vote_events(events, members)
        assert events[0].yea_votes == ["Zyxabc"]

    def test_normalizes_fallback_comma_format(self) -> None:
        """Unknown names with commas get consistent spacing."""
        members: dict[str, Member] = {}
        events = [
            VoteEvent(
                bill_number="SB0001",
                date="",
                description="",
                chamber="Senate",
                yea_votes=["Unknown,Alice"],
            ),
        ]
        normalize_vote_events(events, members)
        assert events[0].yea_votes == ["Unknown, Alice"]

    def test_committee_vote_format(self) -> None:
        """Committee PDF format like 'Curran, John F' resolves."""
        members = {"John F. Curran": _member("John F. Curran", "Senate")}
        events = [
            VoteEvent(
                bill_number="SB0001",
                date="",
                description="Executive",
                chamber="Senate",
                yea_votes=["Curran, John F"],
                vote_type="committee",
            ),
        ]
        normalize_vote_events(events, members)
        assert events[0].yea_votes == ["John F. Curran"]

    def test_suffix_resolution(self) -> None:
        """'Sims Jr., Elgie R' from committee PDF resolves to canonical name."""
        members = {
            "Elgie R. Sims, Jr.": _member("Elgie R. Sims, Jr.", "Senate"),
        }
        events = [
            VoteEvent(
                bill_number="SB0001",
                date="",
                description="Executive",
                chamber="Senate",
                yea_votes=["Sims Jr., Elgie R"],
            ),
        ]
        normalize_vote_events(events, members)
        assert events[0].yea_votes == ["Elgie R. Sims, Jr."]

    def test_floor_bare_sims_resolves(self) -> None:
        """Floor vote bare 'Sims' resolves when unambiguous."""
        members = {
            "Elgie R. Sims, Jr.": _member("Elgie R. Sims, Jr.", "Senate"),
        }
        events = [
            VoteEvent(
                bill_number="SB0001",
                date="",
                description="Third Reading",
                chamber="Senate",
                yea_votes=["Sims"],
                vote_type="floor",
            ),
        ]
        normalize_vote_events(events, members)
        assert events[0].yea_votes == ["Elgie R. Sims, Jr."]
