"""Link types and object link model for the Legislative Ontology."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class LinkType(str, Enum):
    """Canonical relationship types between ontology objects."""

    # Legislator links
    SPONSORS = "sponsors"
    CO_SPONSORS = "co_sponsors"
    COMMITTEE_MEMBERSHIP = "committee_membership"
    VOTES_CAST = "votes_cast"
    # Bill links
    SPONSORED_BY = "sponsored_by"
    CO_SPONSORED_BY = "co_sponsored_by"
    IN_COMMITTEE = "in_committee"
    VOTE_EVENTS = "vote_events"
    WITNESS_SLIPS = "witness_slips"
    HEARINGS = "hearings"
    # Committee links
    MEMBERS = "members"
    BILLS = "bills"
    COMMITTEE_HEARINGS = "committee_hearings"
    # Organization links
    SLIPS_FILED = "slips_filed"
    LEGISLATORS_ALIGNED = "legislators_aligned"
    # VoteEvent links
    BILL = "bill"
    YEA_VOTERS = "yea_voters"
    NAY_VOTERS = "nay_voters"
    PRESENT_VOTERS = "present_voters"
    # Hearing links
    COMMITTEE = "committee"
    HEARING_BILLS = "hearing_bills"


class ObjectLink(BaseModel):
    """A typed relationship from one ontology object to another."""

    target_id: str
    target_type: str
    link_type: str
    metadata: dict[str, Any] = {}


def link(
    target_id: str, target_type: str, link_type: str | LinkType, **metadata: Any
) -> ObjectLink:
    """Build an ObjectLink with optional metadata."""
    return ObjectLink(
        target_id=target_id,
        target_type=target_type,
        link_type=link_type.value if isinstance(link_type, LinkType) else link_type,
        metadata=dict(metadata),
    )
