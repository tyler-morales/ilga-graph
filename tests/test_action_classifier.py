"""Tests for action classifier and pipeline stage (current vs highest)."""

from __future__ import annotations

from ilga_graph.ml.action_classifier import (
    bill_outcome_from_actions,
    classify_action_history,
)
from ilga_graph.ml.features import compute_bill_stage

# HB3356-style: passed both chambers then Rule 19(b) re-referred — never sent to governor.
HB3356_STYLE_ACTIONS = [
    "Filed with the Clerk by Rep. Carol Ammons",
    "First Reading",
    "Referred to Rules Committee",
    "Assigned to Health Care Licenses Committee",
    "Do Pass / Short Debate Health Care Licenses Committee; 014-000-000",
    "Second Reading - Short Debate",
    "Third Reading - Short Debate - Passed 107-000-000",
    "Arrive in Senate",
    "First Reading",
    "Referred to Assignments",
    "Assigned to Licensed Activities",
    "Do Pass Licensed Activities; 007-000-000",
    "Second Reading",
    "Third Reading - Passed; 056-000-000",
    "Arrived in House",
    "Placed on Calendar Order of Concurrence Senate Amendment(s) 1",
    "Rule 19(b) / Re-referred to Rules Committee",
]


class TestBillOutcomeRollback:
    """Bills re-referred after passing both chambers must show current stage, not Governor."""

    def test_hb3356_style_current_stage_is_in_committee(self) -> None:
        classified = classify_action_history(HB3356_STYLE_ACTIONS)
        outcome = bill_outcome_from_actions(classified)
        # Highest ever reached was PASSED_BOTH (Senate passed), but last action is Rule 19(b).
        assert outcome["highest_stage"] in ("PASSED_BOTH", "CHAMBER_PASSED", "CROSSED_CHAMBERS")
        assert outcome["current_stage"] == "IN_COMMITTEE"
        assert outcome["lifecycle_status"] == "OPEN"

    def test_compute_bill_stage_uses_current_not_highest(self) -> None:
        stage, progress = compute_bill_stage(HB3356_STYLE_ACTIONS)
        # Must not show PASSED_BOTH or Governor; must show In Committee (or equivalent).
        assert stage == "IN_COMMITTEE"
        assert progress >= 0

    def test_sent_to_governor_unchanged(self) -> None:
        actions = [
            "Third Reading - Passed",
            "Arrive in Senate",
            "Third Reading - Passed",
            "Sent to the Governor",
        ]
        classified = classify_action_history(actions)
        outcome = bill_outcome_from_actions(classified)
        assert outcome["current_stage"] in ("GOVERNOR", "PASSED_BOTH")
        assert outcome["highest_stage"] in ("GOVERNOR", "PASSED_BOTH")
