"""Outreach checkpoint step definitions for call and email flows.

Step slugs are used in outreach_step_events for funnel analytics. Definitions
live here so steps can be added or reordered without schema changes.
"""

from __future__ import annotations

# Call (answered path): user completes script and records call
CALL_ANSWERED_STEPS = (
    "drawer_opened",
    "phone_clicked",
    "staffer_name_captured",
    "office_email_captured",
    "end_call_clicked",
    "interest_selected",
    "call_recorded",
    "wrapup_draft_clicked",
    "wrapup_skipped",
)

# Call (no-answer path): user uses voicemail script
CALL_NO_ANSWER_STEPS = (
    "drawer_opened",
    "voicemail_toggled",
    "end_call_clicked_vm",
    "no_answer_recorded",
)

# Email: 6-step guided flow
EMAIL_STEPS = (
    "drawer_opened",
    "signed_in",
    "subject_confirmed",
    "details_filled",
    "pdf_grabbed",
    "send_clicked",
    "email_recorded",
)

# Why-you-care (WYC): making the case to the base; outcome-focused funnel
WYC_STEPS = (
    "wyc_poll_submitted",
    "wyc_branch_viewed",
    "wyc_clicked_to_advocacy",
    "wyc_clicked_to_the_issue",
    "wyc_share_story_clicked",
    "wyc_change_answer_clicked",
)

# All valid (outreach_type -> allowed slugs)
ALLOWED_STEPS: dict[str, tuple[str, ...]] = {
    "call": CALL_ANSWERED_STEPS + CALL_NO_ANSWER_STEPS,
    "email": EMAIL_STEPS,
    "wyc": WYC_STEPS,
}


def is_valid_step(outreach_type: str, step_slug: str) -> bool:
    """Return True if step_slug is allowed for the given outreach_type."""
    allowed = ALLOWED_STEPS.get(outreach_type)
    if not allowed:
        return False
    return step_slug in allowed
