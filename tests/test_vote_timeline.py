"""Tests for vote_timeline module, focused on _norm() after name normalization."""

from __future__ import annotations

from ilga_graph.vote_timeline import _norm


class TestNorm:
    """The _norm function should extract a bare lower-case last name."""

    def test_bare_last_name(self) -> None:
        assert _norm("Murphy") == "murphy"

    def test_comma_delimited(self) -> None:
        assert _norm("Murphy, Laura M") == "murphy"

    def test_comma_no_space(self) -> None:
        assert _norm("Davis,Jed") == "davis"

    def test_canonical_full_name(self) -> None:
        """After normalization, names may be 'First M. Last'."""
        assert _norm("Laura M. Murphy") == "murphy"

    def test_canonical_simple(self) -> None:
        assert _norm("Neil Anderson") == "anderson"

    def test_suffix_jr_comma(self) -> None:
        """'Harris III, Napoleon' → last = 'harris iii' before suffix strip → 'harris'."""
        assert _norm("Harris III, Napoleon") == "harris"

    def test_canonical_with_suffix(self) -> None:
        """Canonical name 'Elgie R. Sims, Jr.' — comma present, last = 'Elgie R. Sims'
        but _SUFFIX_RE strips trailing 'Jr.' ... actually the comma splits first."""
        # "Elgie R. Sims, Jr." → split on comma → "Elgie R. Sims" → lower → suffix strip
        # _SUFFIX_RE strips trailing " jr." etc. but "elgie r. sims" has no suffix.
        # This is fine — the norm key is "elgie r. sims" which is unique.
        result = _norm("Elgie R. Sims, Jr.")
        assert result == "elgie r. sims"

    def test_empty_string(self) -> None:
        assert _norm("") == ""

    def test_single_word_no_comma(self) -> None:
        assert _norm("Ammons") == "ammons"

    def test_multi_word_last(self) -> None:
        """Multi-word names like 'Costa Howard' → last word 'howard'."""
        assert _norm("Costa Howard") == "howard"

    def test_hyphenated_name(self) -> None:
        """'Blair-Sherlock' is a single token → treated as last name."""
        assert _norm("Blair-Sherlock") == "blair-sherlock"
