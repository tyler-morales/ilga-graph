"""Ontology actions (verbs): traceable advocacy and intelligence operations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .links import ObjectLink


class ActionType(str, Enum):
    """Canonical action types for advocacy and intelligence."""

    CALL_REP = "call_rep"
    SEND_EMAIL = "send_email"
    FILE_WITNESS_SLIP = "file_witness_slip"
    NO_ANSWER = "no_answer"
    INITIATE_CAMPAIGN = "initiate_campaign"
    COMPLETE_FUNNEL = "complete_funnel"


class OntologyAction(BaseModel):
    """A single traceable action that changes or records state in the ontology."""

    action_id: str
    action_type: str
    actor_id: str | None = None
    target_links: list[ObjectLink] = []
    timestamp: datetime | None = None
    metadata: dict[str, Any] = {}
    outcome: str | None = None


class ActionResult(BaseModel):
    """Result of executing an ontology action."""

    success: bool
    action_id: str
    message: str = ""
    payload: dict[str, Any] = {}


def outreach_event_to_action(
    event_id: int,
    kind: str,
    member_id: str,
    user_id: int | None,
    created_at: datetime | None = None,
    outcome: str | None = None,
    campaign_id: int | None = None,
) -> OntologyAction:
    """Build an OntologyAction from a recorded outreach event (call/email/no_answer)."""
    action_type = {
        "call": ActionType.CALL_REP.value,
        "email": ActionType.SEND_EMAIL.value,
        "no_answer": ActionType.NO_ANSWER.value,
    }.get((kind or "").strip().lower(), kind or "call_rep")
    target_links = [
        ObjectLink(target_id=member_id, target_type="legislator", link_type="target"),
    ]
    if campaign_id is not None:
        target_links.append(
            ObjectLink(target_id=str(campaign_id), target_type="campaign", link_type="campaign"),
        )
    return OntologyAction(
        action_id=str(event_id),
        action_type=action_type,
        actor_id=str(user_id) if user_id is not None else None,
        target_links=target_links,
        timestamp=created_at,
        outcome=outcome or None,
        metadata={},
    )
