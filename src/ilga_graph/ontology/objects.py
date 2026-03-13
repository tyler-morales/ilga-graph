"""Ontology object definitions (nouns) for the Legislative Ontology."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .links import ObjectLink

# ── Object type constants (for object_type field and registry) ──
OBJECT_TYPE_LEGISLATOR = "legislator"
OBJECT_TYPE_BILL = "bill"
OBJECT_TYPE_COMMITTEE = "committee"
OBJECT_TYPE_ORGANIZATION = "organization"
OBJECT_TYPE_VOTE_EVENT = "vote_event"
OBJECT_TYPE_HEARING = "hearing"
OBJECT_TYPE_WITNESS_SLIP = "witness_slip"


class BaseOntologyObject(BaseModel):
    """Base for all ontology objects. Holds identity and typed links."""

    object_id: str
    object_type: str
    links: list[ObjectLink] = []

    def linked_ids(self, link_type: str) -> list[str]:
        return [lnk.target_id for lnk in self.links if lnk.link_type == link_type]


# ── Legislator ─────────────────────────────────────────────────────────────


class LegislatorObject(BaseOntologyObject):
    """Legislator (Member). Links: sponsors, committee_memberships, votes_cast."""

    object_type: str = OBJECT_TYPE_LEGISLATOR
    name: str = ""
    district: str = ""
    party: str = ""
    chamber: str = ""
    role: str = ""
    photo_url: str = ""
    career_ranges: list[dict[str, Any]] = []
    scorecard: dict[str, Any] | None = None
    moneyball: dict[str, Any] | None = None
    influence: dict[str, Any] | None = None


# ── Bill ───────────────────────────────────────────────────────────────────


class BillObject(BaseOntologyObject):
    """Bill. Links: sponsors, committees, vote_events, witness_slips, hearings."""

    object_type: str = OBJECT_TYPE_BILL
    bill_number: str = ""
    leg_id: str = ""
    description: str = ""
    chamber: str = ""
    status: str = ""
    synopsis: str = ""
    last_action: str = ""
    last_action_date: str = ""
    primary_sponsor: str = ""
    prediction: dict[str, Any] | None = None
    controversy_score: float | None = None


# ── Committee ───────────────────────────────────────────────────────────────


class CommitteeObject(BaseOntologyObject):
    """Committee. Links: members, bills, hearings."""

    object_type: str = OBJECT_TYPE_COMMITTEE
    code: str = ""
    name: str = ""


# ── Organization ────────────────────────────────────────────────────────────


class OrganizationObject(BaseOntologyObject):
    """Organization (from witness slips). Links: slips_filed, legislators_aligned."""

    object_type: str = OBJECT_TYPE_ORGANIZATION
    name: str = ""
    canonical_name: str = ""
    slip_count: int = 0


# ── VoteEvent ───────────────────────────────────────────────────────────────


class VoteEventObject(BaseOntologyObject):
    """Vote event. Links: bill, yea_voters, nay_voters, present_voters."""

    object_type: str = OBJECT_TYPE_VOTE_EVENT
    bill_number: str = ""
    date: str = ""
    description: str = ""
    chamber: str = ""
    yea_count: int = 0
    nay_count: int = 0
    present_count: int = 0
    nv_count: int = 0
    vote_type: str = "floor"


# ── Hearing ─────────────────────────────────────────────────────────────────


class HearingObject(BaseOntologyObject):
    """Hearing. Links: committee, bills."""

    object_type: str = OBJECT_TYPE_HEARING
    date: str = ""
    time: str = ""
    location: str = ""
    committee_name: str = ""
    committee_id: str = ""
    posting_date: str = ""
    status: str = ""
    chamber: str = ""


# ── WitnessSlip (leaf object, referenced by Bill/Organization) ──────────────


class WitnessSlipObject(BaseOntologyObject):
    """Witness slip. Links: bill (optional), organization (by name)."""

    object_type: str = OBJECT_TYPE_WITNESS_SLIP
    name: str = ""
    organization: str = ""
    representing: str = ""
    position: str = ""
    hearing_committee: str = ""
    hearing_date: str = ""
    bill_number: str = ""
