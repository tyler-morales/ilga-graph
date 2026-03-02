"""Tests for advocacy goal steps: contact preference and member email availability."""

from __future__ import annotations

from ilga_graph.routers.advocacy import _build_district_steps, _visible_steps


def test_email_only_senator_no_email_visible_steps_exclude_senator_email() -> None:
    """Email-only: senator no email, rep has email → visible steps = 1, next is Email Rep."""
    your_legislators = [
        {
            "card": {"id": "sen-28", "email": ""},
            "role_label": "Your Senator",
            "role_class": "senator",
        },
        {
            "card": {"id": "rep-9", "email": "rep@example.com"},
            "role_label": "Your Rep",
            "role_class": "rep",
        },
    ]
    user_called: set[str] = set()
    user_emailed: set[str] = set()

    district_steps = _build_district_steps(your_legislators, user_called, user_emailed)
    # Senator: call only (no email step). Rep: call + email.
    assert len(district_steps) == 3
    assert district_steps[0]["member_id"] == "sen-28" and district_steps[0]["action"] == "call"
    assert district_steps[1]["member_id"] == "rep-9" and district_steps[1]["action"] == "call"
    assert district_steps[2]["member_id"] == "rep-9" and district_steps[2]["action"] == "email"

    visible = _visible_steps(district_steps, "no")
    assert len(visible) == 1
    assert visible[0]["action"] == "email" and visible[0]["role_label"] == "Rep"
    assert visible[0]["member_id"] == "rep-9"

    # First undone step is "Email your Rep", never "Email your Senator".
    goal_next = next((s for s in visible if not s["done"]), None)
    assert goal_next is not None
    assert goal_next["action"] == "email"
    assert goal_next["role_label"] == "Rep"
    assert "Senator" not in goal_next["role_label"]


def test_visible_steps_call_pref_returns_all_steps() -> None:
    """When user allows calls, _visible_steps returns all steps unchanged."""
    steps = [
        {"member_id": "1", "role_label": "Senator", "action": "call", "done": False},
        {"member_id": "1", "role_label": "Senator", "action": "email", "done": False},
    ]
    assert _visible_steps(steps, "yes") == steps
    assert _visible_steps(steps, None) == steps


def test_visible_steps_email_only_filters_out_call() -> None:
    """When user is email-only, _visible_steps returns only email steps."""
    steps = [
        {"member_id": "1", "role_label": "Senator", "action": "call", "done": False},
        {"member_id": "1", "role_label": "Senator", "action": "email", "done": False},
    ]
    visible = _visible_steps(steps, "no")
    assert len(visible) == 1
    assert visible[0]["action"] == "email"


def test_build_district_steps_no_email_omits_email_step() -> None:
    """Members without effective email get only a call step."""
    your_legislators = [
        {"card": {"id": "m1", "email": ""}, "role_label": "Senator", "role_class": "s"},
    ]
    steps = _build_district_steps(your_legislators, set(), set())
    assert len(steps) == 1
    assert steps[0]["action"] == "call"


def test_build_district_steps_has_email_includes_email_step() -> None:
    """Members with effective email get call + email steps."""
    your_legislators = [
        {"card": {"id": "m1", "email": "m1@example.com"}, "role_label": "Rep", "role_class": "r"},
    ]
    steps = _build_district_steps(your_legislators, set(), set())
    assert len(steps) == 2
    assert steps[0]["action"] == "call" and steps[1]["action"] == "email"
