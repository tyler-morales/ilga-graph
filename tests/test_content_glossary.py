"""Tests for inline glossary helpers and timeline waterfall in content router."""

from ilga_graph.routers.content import (
    _faq_session_and_deadlines_with_tooltips,
    _inline_glossary_terms,
    _timeline_phases_with_inline_glossary,
    _timeline_waterfall_data,
    apply_inline_glossary,
)


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
    assert "glossary-inline-term" in result[2]
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


def test_faq_session_and_deadlines_with_tooltips_empty() -> None:
    """With no deadlines, returns faq_session with answer_html and empty deadlines list."""
    faq_session, deadlines = _faq_session_and_deadlines_with_tooltips([])
    assert faq_session["title"] == "FAQ — Session calendar & deadlines"
    assert len(faq_session["items"]) == 1
    assert "answer_html" in faq_session["items"][0]
    assert "glossary-inline-term" in faq_session["items"][0]["answer_html"]
    assert deadlines == []


def test_faq_session_and_deadlines_with_tooltips_inlines_terms() -> None:
    """With deadlines, answer and descriptions get tooltip markup for session terms."""
    session_deadlines = [
        {"date": "2026-01-16", "chamber": "House", "description": "LRB Request Deadline."},
        {"date": "2026-03-27", "chamber": "House", "description": "Committee deadline."},
    ]
    faq_session, deadlines = _faq_session_and_deadlines_with_tooltips(session_deadlines)
    assert len(deadlines) == 2
    desc_html = deadlines[0]["description_html"]
    assert "glossary-inline-term" in desc_html or "LRB" in desc_html
    assert deadlines[0]["date"] == "2026-01-16"
    assert deadlines[1]["chamber"] == "House"
    assert "answer_html" in faq_session["items"][0]


def test_timeline_waterfall_data_empty_phases() -> None:
    """Empty phases returns empty months and phases."""
    out = _timeline_waterfall_data([])
    assert out["months"] == []
    assert out["phases"] == []


def test_timeline_waterfall_data_builds_months_and_columns() -> None:
    """Waterfall has months first-to-last; each phase has start_col/end_col."""
    phases = _timeline_phases_with_inline_glossary()
    out = _timeline_waterfall_data(phases)
    assert len(out["months"]) > 0
    assert out["months"][0]["key"] == "2026-02"
    assert out["months"][-1]["key"] == "2027-08"
    assert len(out["phases"]) == 4
    for p in out["phases"]:
        assert "start_col" in p
        assert "end_col" in p
        assert 0 <= p["start_col"] <= p["end_col"] < len(out["months"])
