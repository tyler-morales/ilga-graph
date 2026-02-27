"""ILGA session schedule loader — single source of truth for House/Senate dates and deadlines.

Loads ``reference/session_schedule.json`` once (cached) and exposes the schedule and
helpers. All session dates, deadlines, and reminders must be derived from this module.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

# Repo root: from src/ilga_graph/session_schedule.py, parents[2] is project root.
_SCHEDULE_PATH = Path(__file__).resolve().parents[2] / "reference" / "session_schedule.json"


def _validate_event(ev: dict, chamber: str, index: int) -> None:
    """Raise ValueError if event dict is missing required keys or has invalid types."""
    if not isinstance(ev, dict):
        raise ValueError(f"session_schedule: chamber {chamber} event[{index}] must be a dict")
    for key in ("date", "type", "description"):
        if key not in ev:
            raise ValueError(f"session_schedule: chamber {chamber} event[{index}] missing {key!r}")
        if not isinstance(ev[key], str):
            raise ValueError(
                f"session_schedule: chamber {chamber} event[{index}].{key} must be str"
            )


def _validate_schedule(data: list) -> None:
    """Validate top-level list and each chamber block. Raises ValueError on failure."""
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("session_schedule.json must be a non-empty list of chamber objects")
    for block in data:
        if not isinstance(block, dict):
            raise ValueError(
                "session_schedule: each item must be a dict with chamber, session, events"
            )
        for key in ("chamber", "session", "events"):
            if key not in block:
                raise ValueError(f"session_schedule: chamber block missing {key!r}")
        if not isinstance(block["events"], list):
            raise ValueError(f"session_schedule: {block.get('chamber')!r} events must be a list")
        for i, ev in enumerate(block["events"]):
            _validate_event(ev, block["chamber"], i)


@lru_cache(maxsize=1)
def load_schedule() -> list[dict]:
    """Load and cache ``reference/session_schedule.json``. Returns list of chamber dicts."""
    if not _SCHEDULE_PATH.exists():
        raise FileNotFoundError(
            f"Session schedule not found at {_SCHEDULE_PATH}. "
            "Ensure reference/session_schedule.json exists."
        )
    with open(_SCHEDULE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    _validate_schedule(data)
    return data


def get_events_by_chamber(chamber: str) -> list[dict]:
    """Return events for one chamber (e.g. 'House' or 'Senate')."""
    for block in load_schedule():
        if block.get("chamber") == chamber:
            return list(block.get("events", []))
    return []


def get_events_by_type(event_type: str) -> list[tuple[str, dict]]:
    """Return (chamber, event) for all events of the given type (e.g. 'Deadline', 'Session')."""
    out: list[tuple[str, dict]] = []
    for block in load_schedule():
        chamber = block.get("chamber", "")
        for ev in block.get("events", []):
            if ev.get("type") == event_type:
                out.append((chamber, ev))
    return out


def get_all_deadlines() -> list[tuple[str, dict]]:
    """Return (chamber, event) for every Deadline in the schedule."""
    return get_events_by_type("Deadline")


def next_deadline_on_or_after(after_date: str | date) -> dict | None:
    """Return the first deadline (any chamber) on or after the given date, or None.

    Prefers earliest date; if multiple on same date, first chamber order (House then Senate).
    """
    if isinstance(after_date, date):
        after_date = after_date.isoformat()
    earliest: dict | None = None
    earliest_date: str | None = None
    for chamber, ev in get_all_deadlines():
        d = ev.get("date", "")
        if d >= after_date and (earliest_date is None or d < earliest_date):
            earliest_date = d
            earliest = {"chamber": chamber, **ev}
    return earliest


def session_label() -> str:
    """Return session label (e.g. '104th GA - Spring 2026') from first chamber."""
    schedule = load_schedule()
    if schedule:
        return schedule[0].get("session", "")
    return ""
