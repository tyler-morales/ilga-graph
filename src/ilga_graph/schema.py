from __future__ import annotations

from enum import Enum

import strawberry

from .analytics import (
    MemberScorecard,
    compute_scorecard,
    controversial_score,
    lobbyist_alignment,
    compute_advancement_analytics,  # Import the new function
)
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


# ----- New Type for Advancement Analytics -----
@strawberry.type
class BillAdvancementAnalyticsType:
    """Aggregated analytics comparing witness slip volume and bill advancement."""
    high_volume_stalled: list[str] = strawberry.field(
        default_factory=list,
        description="Bills with high witness slip volume that did not pass.",
    )
    high_volume_passed: list[str] = strawberry.field(
        default_factory=list,
        description="Bills with high witness slip volume that successfully passed.",
    )
# ----- End New Type -----


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


@strawberry.type
class Query:
    @strawberry.field(description="Look up a single member by exact name.")
    def member(self, name: str) -> MemberType | None:
        model = state.member_lookup.get(name)
        if model is None:
            return None
        return MemberType.from_model(
            model,
            state.scorecards.get(model.id),
            _mb_profile(model.id),
        )

    @strawberry.field(
        description="Paginated list of members with optional sorting and chamber filter.",
    )
    def members(
        self,
        sort_by: MemberSortField | None = None,
        sort_order: SortOrder | None = None,
        chamber: Chamber | None = None,
        offset: int = 0,
        limit: int = 0,
    ) -> MemberConnection:
        result = list(state.members)
        chamber_str = _resolve_chamber(chamber)

        # ── Filtering ──
        if chamber_str is not None:
            result = [m for m in result if m.chamber.lower() == chamber_str.lower()]

        if sort_by is not None:
            reverse = sort_order == SortOrder.DESC
            if sort_by == MemberSortField.CAREER_START:
                result.sort(key=_member_career_start, reverse=reverse)
            elif sort_by == MemberSortField.NAME:
                result.sort(key=lambda m: m.name, reverse=reverse)

        page, page_info = paginate(result, offset, limit)
        return MemberConnection(
            items=[
                MemberType.from_model(m, state.scorecards.get(m.id), _mb_profile(m.id))
                for m in page
            ],
            page_info=page_info,
        )

    @strawberry.field(description="Ranked leaderboard by Moneyball Score or any analytics metric.")
    def moneyball_leaderboard(
        self,
        chamber: Chamber | None = None,
        exclude_leadership: bool = False,
        limit: int = 0,
        offset: int = 0,
        sort_by: LeaderboardSortField | None = None,
        sort_order: SortOrder | None = None,
    ) -> MemberConnection:
        """Returns all members by default (limit=0 means no cap).

        Use ``chamber=HOUSE, excludeLeadership=true, limit=1`` to get the MVP.
        """
        if state.moneyball is None:
            return MemberConnection(
                items=[],
                page_info=PageInfo(total_count=0, has_next_page=False, has_previous_page=False),
            )

        chamber_str = _resolve_chamber(chamber)

        # ── Base ranking (by moneyball_score) ──
        if chamber_str and chamber_str.lower() == "house":
            ids = (
                state.moneyball.rankings_house_non_leadership
                if exclude_leadership
                else state.moneyball.rankings_house
            )
        elif chamber_str and chamber_str.lower() == "senate":
            ids = (
                state.moneyball.rankings_senate_non_leadership
                if exclude_leadership
                else state.moneyball.rankings_senate
            )
        else:
            ids = state.moneyball.rankings_overall

        # Resolve to Member models
        id_set = set(ids)
        members = [m for m in state.members if m.id in id_set]

        # ── Optional re-sort by analytics field ──
        if sort_by is not None:
            scorecards = state.scorecards
            profiles = state.moneyball.profiles
            reverse = sort_order == SortOrder.DESC

            def _sort_key(m: Member) -> float:
                if sort_by == LeaderboardSortField.MONEYBALL_SCORE:
                    return profiles[m.id].moneyball_score if m.id in profiles else 0.0
                if sort_by == LeaderboardSortField.EFFECTIVENESS_SCORE:
                    return scorecards[m.id].effectiveness_score if m.id in scorecards else 0.0
                if sort_by == LeaderboardSortField.PIPELINE_DEPTH:
                    return profiles[m.id].pipeline_depth_avg if m.id in profiles else 0.0
                if sort_by == LeaderboardSortField.NETWORK_CENTRALITY:
                    return profiles[m.id].network_centrality if m.id in profiles else 0.0
                if sort_by == LeaderboardSortField.HEAT_SCORE:
                    return float(scorecards[m.id].heat_score) if m.id in scorecards else 0.0
                if sort_by == LeaderboardSortField.SUCCESS_RATE:
                    return scorecards[m.id].success_rate if m.id in scorecards else 0.0
                if sort_by == LeaderboardSortField.MAGNET_SCORE:
                    return scorecards[m.id].magnet_score if m.id in scorecards else 0.0
                if sort_by == LeaderboardSortField.BRIDGE_SCORE:
                    return scorecards[m.id].bridge_score if m.id in scorecards else 0.0
                return 0.0

            members.sort(key=_sort_key, reverse=reverse)
        else:
            # Preserve the pre-computed ranking order
            rank = {mid: i for i, mid in enumerate(ids)}
            members.sort(key=lambda m: rank.get(m.id, len(ids)))

        page, page_info = paginate(members, offset, limit)
        return MemberConnection(
            items=[
                MemberType.from_model(m, state.scorecards.get(m.id), _mb_profile(m.id))
                for m in page
            ],
            page_info=page_info,
        )

    @strawberry.field(description="All vote events for a specific bill (floor + committee).")
    def votes(self, bill_number: str) -> list[VoteEventType]:
        events = state.vote_lookup.get(bill_number, [])
        return [VoteEventType.from_model(v) for v in events]

    @strawberry.field(
        description=(
            "Full vote timeline for a bill in one chamber,"
            " tracking every member's journey across committee and floor events."
        ),
    )
    def bill_vote_timeline(self, bill_number: str, chamber: Chamber) -> BillVoteTimelineType | None:
        return compute_bill_vote_timeline(state.vote_lookup, bill_number, chamber.value)

    @strawberry.field(
        description="All scraped vote events, optionally filtered by type and chamber.",
    )
    def all_vote_events(
        self,
        vote_type: str | None = None,
        chamber: Chamber | None = None,
        offset: int = 0,
        limit: int = 0,
    ) -> VoteEventConnection:
        result = list(state.vote_events)
        chamber_str = _resolve_chamber(chamber)
        if vote_type is not None:
            result = [v for v in result if v.vote_type == vote_type]
        if chamber_str is not None:
            result = [v for v in result if v.chamber.lower() == chamber_str.lower()]
        page, page_info = paginate(result, offset, limit)
        return VoteEventConnection(
            items=[VoteEventType.from_model(v) for v in page],
            page_info=page_info,
        )

    @strawberry.field(description="Look up a single bill by bill number (e.g. 'SB1527').")
    def bill(self, number: str) -> BillType | None:
        model = state.bill_lookup.get(number)
        return BillType.from_model(model) if model else None

    @strawberry.field(
        description="Paginated list of bills with optional sorting and date-range filtering.",
    )
    def bills(
        self,
        sort_by: BillSortField | None = None,
        sort_order: SortOrder | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        offset: int = 0,
        limit: int = 0,
    ) -> BillConnection:
        result = list(state.bills)

        # ── date filtering (with safe parsing) ──
        if date_from is not None:
            from_dt = _safe_parse_date(date_from, "dateFrom")
            if from_dt is not None:
                result = [b for b in result if _parse_bill_date(b.last_action_date) >= from_dt]
        if date_to is not None:
            to_dt = _safe_parse_date(date_to, "dateTo")
            if to_dt is not None:
                result = [b for b in result if _parse_bill_date(b.last_action_date) <= to_dt]

        # ── sorting ──
        if sort_by is not None:
            reverse = sort_order == SortOrder.DESC
            if sort_by == BillSortField.LAST_ACTION_DATE:
                result.sort(
                    key=lambda b: _parse_bill_date(b.last_action_date),
                    reverse=reverse,
                )
            elif sort_by == BillSortField.BILL_NUMBER:
                result.sort(key=lambda b: b.bill_number, reverse=reverse)

        page, page_info = paginate(result, offset, limit)
        return BillConnection(
            items=[BillType.from_model(b) for b in page],
            page_info=page_info,
        )

    # ── Committee queries ─────────────────────────────────────────────────

    @strawberry.field(description="Look up a single committee by its code (e.g. 'SAGR').")
    def committee(self, code: str) -> CommitteeType | None:
        model = state.committee_lookup.get(code)
        if model is None:
            return None
        return CommitteeType.from_model(
            model,
            roster=state.committee_rosters.get(code),
            bill_numbers=state.committee_bills.get(code),
        )

    @strawberry.field(description="Paginated list of committees.")
    def committees(
        self,
        offset: int = 0,
        limit: int = 0,
    ) -> CommitteeConnection:
        page, page_info = paginate(state.committees, offset, limit)
        return CommitteeConnection(
            items=[
                CommitteeType.from_model(
                    c,
                    roster=state.committee_rosters.get(c.code),
                    bill_numbers=state.committee_bills.get(c.code),
                )
                for c in page
            ],
            page_info=page_info,
        )

    # ── Witness slip queries ──────────────────────────────────────────────

    @strawberry.field(description="Witness slips for a specific bill.")
    def witness_slips(
        self,
        bill_number: str,
        offset: int = 0,
        limit: int = 0,
    ) -> WitnessSlipConnection:
        slips = state.witness_slips_lookup.get(bill_number, [])
        page, page_info = paginate(slips, offset, limit)
        return WitnessSlipConnection(
            items=[WitnessSlipType.from_model(ws) for ws in page],
            page_info=page_info,
        )

    def _witness_slip_summary_for_slips(
        self, bill_number: str, slips: list[WitnessSlip]
    ) -> WitnessSlipSummaryType:
        pro = sum(1 for s in slips if s.position == "Proponent")
        opp = sum(1 for s in slips if s.position == "Opponent")
        no_pos = sum(
            1
            for s in slips
            if s.position and "no position" in s.position.lower()
        )
        return WitnessSlipSummaryType(
            bill_number=bill_number,
            total_count=len(slips),
            proponent_count=pro,
            opponent_count=opp,
            no_position_count=no_pos,
        )

    @strawberry.field(
        description="Per-bill witness slip counts by position (no paging).",
    )
    def witness_slip_summary(self, bill_number: str) -> WitnessSlipSummaryType | None:
        slips = state.witness_slips_lookup.get(bill_number, [])
        if not slips:
            return None
        return self._witness_slip_summary_for_slips(bill_number, slips)

    @strawberry.field(
        description="All bills with witness slips, summarized (sorted by slip volume descending).",
    )
    def witness_slip_summaries(
        self,
        offset: int = 0,
        limit: int = 0,
    ) -> WitnessSlipSummaryConnection:
        all_summaries = [
            self._witness_slip_summary_for_slips(bill_number, slips)
            for bill_number, slips in state.witness_slips_lookup.items()
        ]
        all_summaries.sort(key=lambda s: s.total_count, reverse=True)
        page, page_info = paginate(all_summaries, offset, limit)
        return WitnessSlipSummaryConnection(items=page, page_info=page_info)

    @strawberry.field(
        description="Witness-slip analytics for a bill (controversy score 0–1).",
    )
    def bill_slip_analytics(
        self, bill_number: str
    ) -> BillSlipAnalyticsType | None:
        if not state.witness_slips_lookup.get(bill_number):
            return None
        score = controversial_score(state.witness_slips, bill_number)
        return BillSlipAnalyticsType(
            bill_number=bill_number,
            controversy_score=score,
        )

    @strawberry.field(
        description="Organisations that file as proponents on this member's sponsored bills (sorted by count desc).",
    )
    def member_slip_alignment(
        self, member_name: str
    ) -> list[LobbyistAlignmentEntryType]:
        member = state.member_lookup.get(member_name)
        if member is None:
            return []
        alignment = lobbyist_alignment(state.witness_slips, member)
        return [
            LobbyistAlignmentEntryType(
                organization=org,
                proponent_count=count,
            )
            for org, count in alignment.items()
        ]

    # ----- New Query Field for Advancement Analytics -----
    @strawberry.field(
        description="Analytics categorizing bills by witness slip volume and advancement status.",
    )
    def bill_advancement_analytics_summary(
        self,
        volume_percentile_threshold: float = 0.9,
    ) -> BillAdvancementAnalyticsType:
        analytics_results = compute_advancement_analytics(
            state.bills,
            state.witness_slips,
            volume_percentile_threshold=volume_percentile_threshold,
        )
        return BillAdvancementAnalyticsType(
            high_volume_stalled=analytics_results.get("high_volume_stalled", []),
            high_volume_passed=analytics_results.get("high_volume_passed", []),
        )
    # ----- End New Query Field -----


from strawberry.extensions import QueryDepthLimiter # noqa: E402

schema = strawberry.Schema(
    query=Query,
    extensions=[QueryDepthLimiter(max_depth=10)],
)
graphql_app = GraphQLRouter(schema)

app = FastAPI(title="ILGA Graph", lifespan=lifespan)

# ── CORS middleware ──────────────────────────────────────────────────────────
_cors_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API key authentication middleware ────────────────────────────────────────
@app.middleware("http")
async def _api_key_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
    """Require ``X-API-Key`` header when ``ILGA_API_KEY`` env var is set.

    Skips auth for the health endpoint and for OPTIONS (CORS preflight).
    """
    if API_KEY:
        exempt = {"/health", "/docs", "/openapi.json", "/redoc"}
        if request.url.path not in exempt and request.method != "OPTIONS":
            provided = request.headers.get("X-API-Key", "")
            if provided != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
    return await call_next(request)


# ── Request logging middleware ───────────────────────────────────────────────
@app.middleware("http")
async def _request_logging_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
    """Log every request with method, path, and response time."""
    import time as _t

    t0 = _t.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (_t.perf_counter() - t0) * 1000
    LOGGER.info(
        "%s %s %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ── Health endpoint ──────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    """Service health check with data counts."""
    return {
        "status": "ok",
        "ready": len(state.members) > 0,
        "members": len(state.members),
        "bills": len(state.bills),
        "committees": len(state.committees),
        "vote_events": len(state.vote_events),
    }


app.include_router(graphql_app, prefix="/graphql")
