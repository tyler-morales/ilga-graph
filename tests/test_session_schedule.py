"""Tests for session_schedule loader (reference/session_schedule.json)."""

from datetime import date

from ilga_graph.session_schedule import (
    get_all_deadlines,
    get_events_by_chamber,
    get_events_by_type,
    load_schedule,
    next_deadline_on_or_after,
    session_label,
)


def test_load_schedule_returns_two_chambers() -> None:
    """Schedule has House and Senate."""
    data = load_schedule()
    assert len(data) == 2
    chambers = {b["chamber"] for b in data}
    assert chambers == {"House", "Senate"}


def test_each_chamber_has_required_keys() -> None:
    """Each chamber block has chamber, session, events."""
    for block in load_schedule():
        assert "chamber" in block
        assert "session" in block
        assert "events" in block
        assert isinstance(block["events"], list)


def test_events_have_required_fields() -> None:
    """Every event has date, type, description."""
    for block in load_schedule():
        for ev in block["events"]:
            assert "date" in ev and isinstance(ev["date"], str)
            assert "type" in ev and isinstance(ev["type"], str)
            assert "description" in ev and isinstance(ev["description"], str)


def test_key_deadline_introduction_present() -> None:
    """Introduction Of House/Senate Bills deadline is in the schedule."""
    deadlines = get_all_deadlines()
    intro_descriptions = [
        ev["description"] for _, ev in deadlines if "Introduction" in ev["description"]
    ]
    assert len(intro_descriptions) >= 1
    assert any("Introduction" in d and "Bills" in d for d in intro_descriptions)


def test_get_events_by_chamber() -> None:
    """get_events_by_chamber returns list of events for that chamber."""
    house = get_events_by_chamber("House")
    senate = get_events_by_chamber("Senate")
    assert len(house) > 0 and len(senate) > 0
    assert all(ev.get("date") for ev in house)


def test_get_events_by_type_deadline() -> None:
    """get_events_by_type('Deadline') matches get_all_deadlines()."""
    by_type = get_events_by_type("Deadline")
    all_dead = get_all_deadlines()
    assert len(by_type) == len(all_dead)


def test_next_deadline_on_or_after() -> None:
    """next_deadline_on_or_after returns first deadline on or after given date."""
    # First deadline in schedule is 2026-01-16 (LRB)
    first = next_deadline_on_or_after("2026-01-01")
    assert first is not None
    assert first["date"] >= "2026-01-01"
    # After adjournment: None or no change
    after_adj = next_deadline_on_or_after("2026-06-01")
    assert after_adj is None or after_adj["date"] >= "2026-06-01"
    # Accept date object
    first_dt = next_deadline_on_or_after(date(2026, 1, 1))
    assert first_dt is not None


def test_session_label() -> None:
    """session_label returns 104th GA Spring 2026."""
    label = session_label()
    assert "104" in label
    assert "2026" in label
