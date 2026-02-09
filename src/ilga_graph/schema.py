from __future__ import annotations

from enum import Enum

import strawberry

from .analytics import MemberScorecard, compute_scorecard
from .models import Bill as BillModel
from .models import CareerRange as CareerRangeModel
from .models import Committee as CommitteeModel
from .models import CommitteeMemberRole as CommitteeMemberRoleModel
from .models import Member as MemberModel
from .models import Office as OfficeModel
from .models import VoteEvent as VoteEventModel
from .models import WitnessSlip as WitnessSlipModel
from .moneyball import MoneyballProfile as MoneyballProfileModel

# ── Enums ─────────────────────────────────────────────────────────────────────


@strawberry.enum
class Chamber(Enum):
    """Legislative chamber."""

    HOUSE = "House"
    SENATE = "Senate"


@strawberry.enum
class BillSortField(Enum):
    LAST_ACTION_DATE = "last_action_date"
    BILL_NUMBER = "bill_number"


@strawberry.enum
class MemberSortField(Enum):
    CAREER_START = "career_start"
    NAME = "name"


@strawberry.enum
class LeaderboardSortField(Enum):
    """Sort fields for analytics-oriented queries (moneyball_leaderboard)."""

    MONEYBALL_SCORE = "moneyball_score"
    EFFECTIVENESS_SCORE = "effectiveness_score"
    PIPELINE_DEPTH = "pipeline_depth"
    NETWORK_CENTRALITY = "network_centrality"
    HEAT_SCORE = "heat_score"
    SUCCESS_RATE = "success_rate"
    MAGNET_SCORE = "magnet_score"
    BRIDGE_SCORE = "bridge_score"


@strawberry.enum
class SortOrder(Enum):
    ASC = "asc"
    DESC = "desc"


# ── Pagination ────────────────────────────────────────────────────────────────


@strawberry.type
class PageInfo:
    """Pagination metadata returned with every paginated query."""

    total_count: int = strawberry.field(
        description="Total number of items matching the query (before pagination).",
    )
    has_next_page: bool = strawberry.field(
        description="True when more items exist beyond the current page.",
    )
    has_previous_page: bool = strawberry.field(
        description="True when items exist before the current page.",
    )


@strawberry.type
class BillType:
    bill_number: str
    leg_id: str
    description: str
    chamber: str
    last_action: str
    last_action_date: str
    primary_sponsor: str

    @classmethod
    def from_model(cls, b: BillModel) -> BillType:
        return cls(
            bill_number=b.bill_number,
            leg_id=b.leg_id,
            description=b.description,
            chamber=b.chamber,
            last_action=b.last_action,
            last_action_date=b.last_action_date,
            primary_sponsor=b.primary_sponsor,
        )


@strawberry.type
class OfficeType:
    name: str
    address: str
    phone: str | None = None
    fax: str | None = None

    @classmethod
    def from_model(cls, m: OfficeModel) -> OfficeType:
        return cls(name=m.name, address=m.address, phone=m.phone, fax=m.fax)


@strawberry.type
class CareerRangeType:
    start_year: int
    end_year: int | None = None
    chamber: str | None = None

    @classmethod
    def from_model(cls, m: CareerRangeModel) -> CareerRangeType:
        return cls(start_year=m.start_year, end_year=m.end_year, chamber=m.chamber)


@strawberry.type
class ScorecardType:
    # ── Original metrics ──
    primary_bill_count: int
    passed_count: int
    vetoed_count: int
    stuck_count: int
    in_progress_count: int
    success_rate: float
    heat_score: int
    effectiveness_score: float
    # ── Phase 3: Legislative DNA ──
    law_heat_score: int = 0
    law_passed_count: int = 0
    law_success_rate: float = 0.0
    magnet_score: float = 0.0
    bridge_score: float = 0.0
    resolutions_count: int = 0
    resolutions_passed_count: int = 0

    @classmethod
    def from_model(cls, sc: MemberScorecard) -> ScorecardType:
        return cls(
            primary_bill_count=sc.primary_bill_count,
            passed_count=sc.passed_count,
            vetoed_count=sc.vetoed_count,
            stuck_count=sc.stuck_count,
            in_progress_count=sc.in_progress_count,
            success_rate=sc.success_rate,
            heat_score=sc.heat_score,
            effectiveness_score=sc.effectiveness_score,
            law_heat_score=sc.law_heat_score,
            law_passed_count=sc.law_passed_count,
            law_success_rate=sc.law_success_rate,
            magnet_score=sc.magnet_score,
            bridge_score=sc.bridge_score,
            resolutions_count=sc.resolutions_count,
            resolutions_passed_count=sc.resolutions_passed_count,
        )


@strawberry.type
class MoneyballProfileType:
    """Moneyball analytics profile for a member."""

    moneyball_score: float
    laws_filed: int
    laws_passed: int
    effectiveness_rate: float
    pipeline_depth_avg: float
    pipeline_depth_normalized: float
    network_centrality: float
    unique_collaborators: int
    is_leadership: bool
    rank_overall: int
    rank_chamber: int
    rank_non_leadership: int
    badges: list[str]

    @classmethod
    def from_model(cls, mb: MoneyballProfileModel) -> MoneyballProfileType:
        return cls(
            moneyball_score=mb.moneyball_score,
            laws_filed=mb.laws_filed,
            laws_passed=mb.laws_passed,
            effectiveness_rate=mb.effectiveness_rate,
            pipeline_depth_avg=mb.pipeline_depth_avg,
            pipeline_depth_normalized=mb.pipeline_depth_normalized,
            network_centrality=mb.network_centrality,
            unique_collaborators=mb.unique_collaborators,
            is_leadership=mb.is_leadership,
            rank_overall=mb.rank_overall,
            rank_chamber=mb.rank_chamber,
            rank_non_leadership=mb.rank_non_leadership,
            badges=list(mb.badges),
        )


@strawberry.type
class VoteEventType:
    bill_number: str
    date: str
    description: str
    chamber: str
    yea_votes: list[str] = strawberry.field(default_factory=list)
    nay_votes: list[str] = strawberry.field(default_factory=list)
    present_votes: list[str] = strawberry.field(default_factory=list)
    nv_votes: list[str] = strawberry.field(default_factory=list)
    yea_count: int = 0
    nay_count: int = 0
    present_count: int = 0
    nv_count: int = 0
    pdf_url: str = ""
    vote_type: str = "floor"

    @classmethod
    def from_model(cls, v: VoteEventModel) -> VoteEventType:
        return cls(
            bill_number=v.bill_number,
            date=v.date,
            description=v.description,
            chamber=v.chamber,
            yea_votes=list(v.yea_votes),
            nay_votes=list(v.nay_votes),
            present_votes=list(v.present_votes),
            nv_votes=list(v.nv_votes),
            yea_count=len(v.yea_votes),
            nay_count=len(v.nay_votes),
            present_count=len(v.present_votes),
            nv_count=len(v.nv_votes),
            pdf_url=v.pdf_url,
            vote_type=v.vote_type,
        )


@strawberry.type
class CommitteeMemberRoleType:
    """A member's role on a committee roster."""

    member_id: str
    member_name: str
    member_url: str
    role: str

    @classmethod
    def from_model(cls, r: CommitteeMemberRoleModel) -> CommitteeMemberRoleType:
        return cls(
            member_id=r.member_id,
            member_name=r.member_name,
            member_url=r.member_url,
            role=r.role,
        )


@strawberry.type
class CommitteeType:
    """A legislative committee."""

    code: str
    name: str
    parent_code: str | None = None
    members_list_url: str | None = None
    roster: list[CommitteeMemberRoleType] = strawberry.field(default_factory=list)
    bill_numbers: list[str] = strawberry.field(default_factory=list)

    @classmethod
    def from_model(
        cls,
        c: CommitteeModel,
        roster: list[CommitteeMemberRoleModel] | None = None,
        bill_numbers: list[str] | None = None,
    ) -> CommitteeType:
        return cls(
            code=c.code,
            name=c.name,
            parent_code=c.parent_code,
            members_list_url=c.members_list_url,
            roster=[CommitteeMemberRoleType.from_model(r) for r in (roster or [])],
            bill_numbers=list(bill_numbers or []),
        )


@strawberry.type
class WitnessSlipType:
    """Public hearing testimony record."""

    name: str
    organization: str
    representing: str
    position: str
    hearing_committee: str
    hearing_date: str
    testimony_type: str = "Record of Appearance Only"
    bill_number: str = ""

    @classmethod
    def from_model(cls, ws: WitnessSlipModel) -> WitnessSlipType:
        return cls(
            name=ws.name,
            organization=ws.organization,
            representing=ws.representing,
            position=ws.position,
            hearing_committee=ws.hearing_committee,
            hearing_date=ws.hearing_date,
            testimony_type=ws.testimony_type,
            bill_number=ws.bill_number,
        )


@strawberry.type
class MemberVoteJourneyType:
    """Tracks one member's vote across every event for a bill in one chamber."""

    member_name: str
    chamber: str
    votes: list[str] = strawberry.field(default_factory=list)
    # Each entry is a vote code per event: "Y", "N", "P", "NV", or "--" (absent from event)
    first_appearance: str = ""
    last_vote: str = ""
    changed: bool = False  # True if member flipped Y↔N across any events
    is_committee_member: bool = False  # True if they appeared in any committee event


@strawberry.type
class BillVoteTimelineType:
    """Full analytical view of a bill's vote lifecycle in one chamber."""

    bill_number: str = ""
    chamber: str = ""
    event_labels: list[str] = strawberry.field(default_factory=list)
    journeys: list[MemberVoteJourneyType] = strawberry.field(default_factory=list)
    # Derived analytics
    committee_to_floor_flips: list[str] = strawberry.field(default_factory=list)
    committee_to_floor_dropoffs: list[str] = strawberry.field(default_factory=list)
    floor_newcomers: list[str] = strawberry.field(default_factory=list)
    consistent_yea: list[str] = strawberry.field(default_factory=list)
    consistent_nay: list[str] = strawberry.field(default_factory=list)


@strawberry.type
class MemberType:
    id: str
    name: str
    member_url: str
    chamber: str
    party: str
    district: str
    bio_text: str
    role: str = ""
    career_timeline_text: str = ""
    career_ranges: list[CareerRangeType] = strawberry.field(default_factory=list)
    committees: list[str] = strawberry.field(default_factory=list)
    associated_members: str | None = None
    email: str | None = None
    offices: list[OfficeType] = strawberry.field(default_factory=list)
    sponsored_bills: list[BillType] = strawberry.field(default_factory=list)
    co_sponsor_bills: list[BillType] = strawberry.field(default_factory=list)
    scorecard: ScorecardType | None = None
    moneyball: MoneyballProfileType | None = None

    @classmethod
    def from_model(
        cls,
        m: MemberModel,
        scorecard: MemberScorecard | None = None,
        moneyball_profile: MoneyballProfileModel | None = None,
    ) -> MemberType:
        sc = scorecard if scorecard is not None else compute_scorecard(m)
        return cls(
            id=m.id,
            name=m.name,
            member_url=m.member_url,
            chamber=m.chamber,
            party=m.party,
            district=m.district,
            bio_text=m.bio_text,
            role=m.role,
            career_timeline_text=m.career_timeline_text,
            career_ranges=[CareerRangeType.from_model(cr) for cr in m.career_ranges],
            committees=list(m.committees),
            associated_members=m.associated_members,
            email=m.email,
            offices=[OfficeType.from_model(o) for o in m.offices],
            sponsored_bills=[BillType.from_model(b) for b in m.sponsored_bills],
            co_sponsor_bills=[BillType.from_model(b) for b in m.co_sponsor_bills],
            scorecard=ScorecardType.from_model(sc),
            moneyball=(
                MoneyballProfileType.from_model(moneyball_profile)
                if moneyball_profile is not None
                else None
            ),
        )


# ── Paginated connection types ────────────────────────────────────────────────


@strawberry.type
class MemberConnection:
    """Paginated list of members."""

    items: list[MemberType]
    page_info: PageInfo


@strawberry.type
class BillConnection:
    """Paginated list of bills."""

    items: list[BillType]
    page_info: PageInfo


@strawberry.type
class VoteEventConnection:
    """Paginated list of vote events."""

    items: list[VoteEventType]
    page_info: PageInfo


@strawberry.type
class CommitteeConnection:
    """Paginated list of committees."""

    items: list[CommitteeType]
    page_info: PageInfo


@strawberry.type
class BillSlipAnalyticsType:
    """Witness-slip analytics for a single bill."""

    bill_number: str = ""
    controversy_score: float = 0.0  # 0–1; opponents/(proponents+opponents)


@strawberry.type
class LobbyistAlignmentEntryType:
    """One organisation's proponent slip count on a member's sponsored bills."""

    organization: str = ""
    proponent_count: int = 0


@strawberry.type
class WitnessSlipSummaryType:
    """Per-bill aggregate counts of witness slips by position."""

    bill_number: str = ""
    total_count: int = 0
    proponent_count: int = 0
    opponent_count: int = 0
    no_position_count: int = 0


@strawberry.type
class WitnessSlipSummaryConnection:
    """Paginated list of witness slip summaries (one per bill)."""

    items: list[WitnessSlipSummaryType]
    page_info: PageInfo


@strawberry.type
class WitnessSlipConnection:
    """Paginated list of witness slips."""

    items: list[WitnessSlipType]
    page_info: PageInfo


def paginate(items: list, offset: int, limit: int) -> tuple[list, PageInfo]:
    """Apply offset/limit pagination and build PageInfo.

    When *limit* is 0 the full list is returned (no cap).
    """
    total = len(items)
    if limit > 0:
        page = items[offset : offset + limit]
    else:
        page = items[offset:]
    has_next = limit > 0 and (offset + limit) < total
    has_prev = offset > 0
    return page, PageInfo(
        total_count=total,
        has_next_page=has_next,
        has_previous_page=has_prev,
    )
