"""Tests for inline glossary helpers in content router."""

from ilga_graph.routers.content import apply_inline_glossary, _inline_glossary_terms


def test_apply_inline_glossary_wraps_first_occurrence_only() -> None:
    """First occurrence of each term is wrapped in button+popover; second is plain."""
    terms = [
        {"id": "foo", "term": "foo", "definition": "A foo thing."},
        {"id": "bar", "term": "bar", "definition": "A bar thing."},
    ]
    terms_sorted = sorted(terms, key=lambda d: len(d["term"]), reverse=True)
    blocks = ["First foo and then bar.", "Second foo and bar again."]
    result = apply_inline_glossary(blocks, terms_sorted)
    assert len(result) == 2
    assert result[0].count('class="glossary-inline-term"') == 2
    assert "First " in result[0]
    assert "glossary-inline-def" in result[0]
    assert "A foo thing" in result[0]
    assert "A bar thing" in result[0]
    assert result[1].count("glossary-inline-term") == 0
    assert "Second foo and bar again" in result[1]


def test_apply_inline_glossary_empty_blocks_unchanged() -> None:
    """Empty or whitespace-only blocks are returned unchanged; only matching terms are wrapped."""
    terms = [{"id": "word", "term": "word", "definition": "A word."}]
    blocks = ["", "  ", "One word here."]
    result = apply_inline_glossary(blocks, terms)
    assert result[0] == ""
    assert result[1] == "  "
    assert 'glossary-inline-term' in result[2]
    assert "One " in result[2]
    assert " here." in result[2]
    assert "A word." in result[2]


def test_inline_glossary_terms_merge_and_sort() -> None:
    """Merged terms include KEI + SESSION and are sorted by term length descending."""
    terms = _inline_glossary_terms(include_domain=False)
    assert len(terms) > 0
    term_lens = [len(t["term"]) for t in terms]
    assert term_lens == sorted(term_lens, reverse=True)
    ids = [t["id"] for t in terms]
    assert "kei-vehicle" in ids
    assert "lrb" in ids or "committee-deadline" in ids
