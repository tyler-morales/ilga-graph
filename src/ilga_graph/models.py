from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Office:
    name: str
    address: str
    phone: str | None = None
    fax: str | None = None


@dataclass
class CareerRange:
    start_year: int
    end_year: int | None = None
    chamber: str | None = None


@dataclass
class ActionEntry:
    date: str  # e.g. "1/13/2025"
    chamber: str  # e.g. "Senate" or "House"
    action: str  # e.g. "First Reading"


@dataclass
class Bill:
    bill_number: str  # e.g. "SB0029"
    leg_id: str  # e.g. "157128" -- unique key
    description: str
    chamber: str  # "S" or "H"
    last_action: str
    last_action_date: str
    primary_sponsor: str  # Chief Sponsor name from table
    # New fields from BillStatus page
    synopsis: str = ""
    status_url: str = ""  # BillStatus URL for re-fetching
    sponsor_ids: list[str] = field(default_factory=list)  # member IDs of all sponsors
    house_sponsor_ids: list[str] = field(default_factory=list)  # house sponsors
    action_history: list[ActionEntry] = field(default_factory=list)


@dataclass
class Committee:
    code: str
    name: str
    parent_code: str | None = None
    members_list_url: str | None = None


@dataclass
class Member:
    id: str
    name: str
    member_url: str
    chamber: str
    party: str
    district: str
    bio_text: str
    role: str = ""
    career_timeline_text: str = ""
    career_ranges: list[CareerRange] = field(default_factory=list)
    committees: list[str] = field(default_factory=list)
    associated_members: str | None = None
    email: str | None = None
    offices: list[Office] = field(default_factory=list)
    # Hydrated bill objects (populated after loading from normalized cache)
    sponsored_bills: list[Bill] = field(default_factory=list)
    co_sponsor_bills: list[Bill] = field(default_factory=list)
    # Normalized references (leg_ids) used for cache serialization
    sponsored_bill_ids: list[str] = field(default_factory=list)
    co_sponsor_bill_ids: list[str] = field(default_factory=list)

    @property
    def bills(self) -> list[Bill]:
        """All bills (sponsored + co-sponsored) — convenience property."""
        return self.sponsored_bills + self.co_sponsor_bills


@dataclass
class VoteEvent:
    bill_number: str  # e.g. "SB0852"
    date: str  # e.g. "May 22, 2025"
    description: str  # e.g. "Third Reading"
    chamber: str  # "Senate" or "House"
    yea_votes: list[str] = field(default_factory=list)
    nay_votes: list[str] = field(default_factory=list)
    present_votes: list[str] = field(default_factory=list)
    nv_votes: list[str] = field(default_factory=list)
    pdf_url: str = ""
    vote_type: str = "floor"  # "floor" or "committee"


@dataclass
class WitnessSlip:
    name: str  # Witness Name
    organization: str  # Firm, Business, or Agency
    representing: str
    position: str  # "Proponent" / "Opponent" / "No Position"
    hearing_committee: str
    hearing_date: str  # from ScheduledDateTime column
    testimony_type: str = "Record of Appearance Only"  # not in export data
    bill_number: str = ""  # from Legislation column -- useful for analytics


@dataclass(frozen=True)
class CommitteeMemberRole:
    member_id: str
    member_name: str
    member_url: str
    role: str
