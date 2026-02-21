"""Strawberry GraphQL Query and ML-specific types. Resolvers use state from app_state."""

from __future__ import annotations

import logging

import strawberry

from .analytics import (
    compute_advancement_analytics,
    controversial_score,
    lobbyist_alignment,
)
from .app_state import state
from .date_parse import parse_bill_date, safe_parse_date
from .models import Member, WitnessSlip
from .schema import (
    BillAdvancementAnalyticsType,
    BillConnection,
    BillSlipAnalyticsType,
    BillSortField,
    BillType,
    BillVoteTimelineType,
    Chamber,
    CommitteeConnection,
    CommitteeType,
    LeaderboardSortField,
    LobbyistAlignmentEntryType,
    MemberConnection,
    MemberSortField,
    MemberType,
    PageInfo,
    SearchConnection,
    SearchEntityType,
    SearchResultType,
    SortOrder,
    VoteEventConnection,
    VoteEventType,
    WitnessSlipConnection,
    WitnessSlipSummaryConnection,
    WitnessSlipSummaryType,
    WitnessSlipType,
    paginate,
)
from .search import EntityType as SearchEntityTypeEnum
from .search import search_all
from .vote_timeline import compute_bill_vote_timeline

# ── Helpers ────────────────────────────────────────────────────────────────


def _member_career_start(member: Member) -> int:
    """Return the earliest career start year, or a large value if unknown."""
    if member.career_ranges:
        return min(cr.start_year for cr in member.career_ranges)
    return 9999


def _mb_profile(member_id: str):
    """Safely get moneyball profile for a member."""
    if state.moneyball is None:
        return None
    return state.moneyball.profiles.get(member_id)


def _resolve_chamber(chamber: Chamber | None) -> str | None:
    """Convert a ``Chamber`` enum value to the string used in data models."""
    if chamber is None:
        return None
    return chamber.value


# ── ML Intelligence GraphQL types ──────────────────────────────────────────


@strawberry.type
class BillPredictionType:
    bill_number: str
    description: str
    sponsor: str
    prob_advance: float
    prob_law: float
    predicted_outcome: str
    predicted_destination: str
    confidence: float
    label_reliable: bool
    chamber_origin: str
    introduction_date: str
    current_stage: str = ""
    stage_progress: float = 0.0
    stage_label: str = ""
    days_since_action: int = 0
    last_action_text: str = ""
    last_action_date: str = ""
    stuck_status: str = ""
    stuck_reason: str = ""
    forecast_score: float = 0.0
    forecast_confidence: str = ""


@strawberry.type
class BillPredictionConnection:
    items: list[BillPredictionType]
    page_info: PageInfo


@strawberry.type
class PredictionFactor:
    feature: str
    impact: str
    raw_impact: float


@strawberry.type
class PredictionExplanation:
    base_value: float
    top_positive_factors: list[PredictionFactor]
    top_negative_factors: list[PredictionFactor]


@strawberry.type
class CoalitionMemberType:
    name: str
    party: str
    chamber: str
    district: str


@strawberry.type
class SignatureBillType:
    bill_number: str
    description: str
    yes_votes: int


@strawberry.type
class CoalitionGroupType:
    coalition_id: int
    name: str
    size: int
    dem_count: int
    rep_count: int
    focus_areas: list[str]
    yes_rate: float
    cohesion: float
    signature_bills: list[SignatureBillType]
    members: list[CoalitionMemberType]


@strawberry.type
class SlipAnomalyType:
    bill_number: str
    description: str
    total_slips: int
    anomaly_score: float
    is_anomaly: bool
    anomaly_reason: str
    top_org_share: float
    org_hhi: float
    position_unanimity: float
    n_proponent: int
    n_opponent: int
    unique_orgs: int


@strawberry.type
class SlipAnomalyConnection:
    items: list[SlipAnomalyType]
    page_info: PageInfo


@strawberry.type
class FeatureImportanceType:
    name: str
    importance: float


@strawberry.type
class ModelQualityType:
    model_selected: str
    trust_overall: str
    strengths: list[str]
    issues: list[str]
    test_roc_auc: float | None
    test_accuracy: float | None
    test_precision_pos: float | None
    test_recall_pos: float | None
    test_f1_pos: float | None
    top_features: list[FeatureImportanceType]
    last_run_date: str


@strawberry.type
class PredictionMissType:
    bill_number: str
    description: str
    predicted: str
    actual: str
    confidence: float


@strawberry.type
class AccuracySnapshotType:
    run_date: str
    snapshot_date: str
    days_elapsed: int
    total_testable: int
    correct: int
    accuracy: float
    precision_advance: float
    recall_advance: float
    f1_advance: float
    model_version: str
    biggest_misses: list[PredictionMissType]


def _bill_score_to_type(s) -> BillPredictionType:
    """Convert a BillScore dataclass to a GraphQL BillPredictionType."""
    return BillPredictionType(
        bill_number=s.bill_number,
        description=s.description,
        sponsor=s.sponsor,
        prob_advance=round(s.prob_advance, 4),
        prob_law=round(getattr(s, "prob_law", 0.0), 4),
        predicted_outcome=s.predicted_outcome,
        predicted_destination=getattr(s, "predicted_destination", "Stuck"),
        confidence=round(s.confidence, 4),
        label_reliable=s.label_reliable,
        chamber_origin=s.chamber_origin,
        introduction_date=s.introduction_date,
        current_stage=s.current_stage,
        stage_progress=round(s.stage_progress, 2),
        stage_label=s.stage_label,
        days_since_action=s.days_since_action,
        last_action_text=s.last_action_text,
        last_action_date=s.last_action_date,
        stuck_status=s.stuck_status,
        stuck_reason=s.stuck_reason,
        forecast_score=round(getattr(s, "forecast_score", 0.0), 4),
        forecast_confidence=getattr(s, "forecast_confidence", ""),
    )


# ── Query ───────────────────────────────────────────────────────────────────


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
        if state.moneyball is None:
            return MemberConnection(
                items=[],
                page_info=PageInfo(total_count=0, has_next_page=False, has_previous_page=False),
            )
        chamber_str = _resolve_chamber(chamber)
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
        id_set = set(ids)
        members = [m for m in state.members if m.id in id_set]
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
        if date_from is not None:
            from_dt = safe_parse_date(date_from, "dateFrom")
            if from_dt is not None:
                result = [b for b in result if parse_bill_date(b.last_action_date) >= from_dt]
        if date_to is not None:
            to_dt = safe_parse_date(date_to, "dateTo")
            if to_dt is not None:
                result = [b for b in result if parse_bill_date(b.last_action_date) <= to_dt]
        if sort_by is not None:
            reverse = sort_order == SortOrder.DESC
            if sort_by == BillSortField.LAST_ACTION_DATE:
                result.sort(
                    key=lambda b: parse_bill_date(b.last_action_date),
                    reverse=reverse,
                )
            elif sort_by == BillSortField.BILL_NUMBER:
                result.sort(key=lambda b: b.bill_number, reverse=reverse)
        page, page_info = paginate(result, offset, limit)
        return BillConnection(
            items=[BillType.from_model(b) for b in page],
            page_info=page_info,
        )

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
    def committees(self, offset: int = 0, limit: int = 0) -> CommitteeConnection:
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
        no_pos = sum(1 for s in slips if s.position and "no position" in s.position.lower())
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
    def bill_slip_analytics(self, bill_number: str) -> BillSlipAnalyticsType | None:
        if not state.witness_slips_lookup.get(bill_number):
            return None
        score = controversial_score(state.witness_slips, bill_number)
        return BillSlipAnalyticsType(
            bill_number=bill_number,
            controversy_score=score,
        )

    @strawberry.field(
        description="Orgs filing as proponents on member's sponsored bills (by count desc).",
    )
    def member_slip_alignment(self, member_name: str) -> list[LobbyistAlignmentEntryType]:
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

    @strawberry.field(
        description=(
            "Unified free-text search across members, bills, and committees. "
            "Returns results ranked by relevance. Use entityTypes to restrict."
        ),
    )
    def search(
        self,
        query: str,
        entity_types: list[SearchEntityType] | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> SearchConnection:
        filter_set: set[SearchEntityTypeEnum] | None = None
        if entity_types:
            _map = {
                SearchEntityType.MEMBER: SearchEntityTypeEnum.MEMBER,
                SearchEntityType.BILL: SearchEntityTypeEnum.BILL,
                SearchEntityType.COMMITTEE: SearchEntityTypeEnum.COMMITTEE,
            }
            filter_set = {_map[et] for et in entity_types}
        all_hits = search_all(
            query=query,
            members=state.members,
            bills=state.bills,
            committees=state.committees,
            entity_types=filter_set,
        )
        page, page_info = paginate(all_hits, offset, limit)
        items: list[SearchResultType] = []
        for hit in page:
            member_type = None
            bill_type = None
            committee_type = None
            if hit.member is not None:
                sc = state.scorecards.get(hit.member.id)
                mb = _mb_profile(hit.member.id)
                member_type = MemberType.from_model(hit.member, sc, mb)
            elif hit.bill is not None:
                bill_type = BillType.from_model(hit.bill)
            elif hit.committee is not None:
                committee_type = CommitteeType.from_model(hit.committee)
            items.append(
                SearchResultType(
                    entity_type=hit.entity_type.value,
                    match_field=hit.match_field,
                    match_snippet=hit.match_snippet,
                    relevance_score=round(hit.relevance_score, 4),
                    member=member_type,
                    bill=bill_type,
                    committee=committee_type,
                )
            )
        return SearchConnection(items=items, page_info=page_info)

    @strawberry.field(description="Bill predictions from the ML pipeline.")
    def bill_predictions(
        self,
        outcome: str | None = None,
        min_confidence: float | None = None,
        reliable_only: bool = False,
        forecasts_only: bool = False,
        stuck_status: str | None = None,
        stage: str | None = None,
        sort_by: str = "prob_advance",
        offset: int = 0,
        limit: int = 50,
    ) -> BillPredictionConnection:
        ml = state.ml
        if not ml or not ml.available:
            empty_page = PageInfo(
                total_count=0,
                has_next_page=False,
                has_previous_page=False,
            )
            return BillPredictionConnection(items=[], page_info=empty_page)
        results = list(ml.bill_scores)
        if outcome:
            results = [s for s in results if s.predicted_outcome == outcome.upper()]
        if min_confidence is not None:
            results = [s for s in results if s.confidence >= min_confidence]
        if reliable_only:
            results = [s for s in results if s.label_reliable]
        if forecasts_only:
            results = [s for s in results if not s.label_reliable]
        if stuck_status:
            results = [s for s in results if s.stuck_status == stuck_status.upper()]
        if stage:
            results = [s for s in results if s.current_stage == stage.upper()]
        if sort_by == "confidence":
            results.sort(key=lambda s: -s.confidence)
        else:
            results.sort(key=lambda s: -s.prob_advance)
        page, page_info = paginate(results, offset, limit)
        return BillPredictionConnection(
            items=[_bill_score_to_type(s) for s in page],
            page_info=page_info,
        )

    @strawberry.field(description="Prediction for a single bill by number.")
    def bill_prediction(self, bill_number: str) -> BillPredictionType | None:
        ml = state.ml
        if not ml or not ml.available:
            return None
        for s in ml.bill_scores:
            if s.bill_number == bill_number:
                return _bill_score_to_type(s)
        return None

    @strawberry.field(
        description="SHAP-based explanation for why a bill received its prediction score.",
    )
    def prediction_explanation(self, bill_id: str) -> PredictionExplanation | None:
        ml = state.ml
        if not ml or not ml.available or ml.explainer is None:
            return None
        row_idx = ml._bill_id_to_row.get(bill_id)
        if row_idx is None:
            return None
        try:
            row = ml.feature_matrix[row_idx]
            result = ml.explainer.explain_prediction(row, ml.feature_names)
            return PredictionExplanation(
                base_value=result["base_value"],
                top_positive_factors=[
                    PredictionFactor(**f) for f in result["top_positive_factors"]
                ],
                top_negative_factors=[
                    PredictionFactor(**f) for f in result["top_negative_factors"]
                ],
            )
        except Exception:
            logging.getLogger(__name__).exception("SHAP explanation failed for %s", bill_id)
            return None

    @strawberry.field(description="Discovered voting coalitions from the ML pipeline.")
    def voting_coalitions(self) -> list[CoalitionGroupType]:
        ml = state.ml
        if not ml or not ml.available:
            return []
        groups: dict[int, list] = {}
        for m in ml.coalitions:
            groups.setdefault(m.coalition_id, []).append(m)
        prof_map = {p.coalition_id: p for p in ml.coalition_profiles}
        result = []
        for cid, members in sorted(groups.items()):
            dem = sum(1 for m in members if m.party == "Democrat")
            rep = sum(1 for m in members if m.party == "Republican")
            prof = prof_map.get(cid)
            result.append(
                CoalitionGroupType(
                    coalition_id=cid,
                    name=prof.name if prof else f"Coalition {cid + 1}",
                    size=len(members),
                    dem_count=dem,
                    rep_count=rep,
                    focus_areas=prof.focus_areas if prof else [],
                    yes_rate=round(prof.yes_rate, 3) if prof else 0.0,
                    cohesion=round(prof.cohesion, 3) if prof else 0.0,
                    signature_bills=[
                        SignatureBillType(
                            bill_number=b.get("bill_number", ""),
                            description=b.get("description", ""),
                            yes_votes=b.get("yes_votes", 0),
                        )
                        for b in (prof.signature_bills if prof else [])[:5]
                    ],
                    members=[
                        CoalitionMemberType(
                            name=m.name,
                            party=m.party,
                            chamber=m.chamber,
                            district=m.district,
                        )
                        for m in members
                    ],
                )
            )
        return result

    @strawberry.field(description="Slip anomalies (astroturfing detection) from the ML pipeline.")
    def slip_anomalies(
        self,
        flagged_only: bool = False,
        min_score: float | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> SlipAnomalyConnection:
        ml = state.ml
        if not ml or not ml.available:
            empty_page = PageInfo(total_count=0, has_next_page=False, has_previous_page=False)
            return SlipAnomalyConnection(items=[], page_info=empty_page)
        results = list(ml.anomalies)
        if flagged_only:
            results = [a for a in results if a.is_anomaly]
        if min_score is not None:
            results = [a for a in results if a.anomaly_score >= min_score]
        results.sort(key=lambda a: -a.anomaly_score)
        page, page_info = paginate(results, offset, limit)
        return SlipAnomalyConnection(
            items=[
                SlipAnomalyType(
                    bill_number=a.bill_number,
                    description=a.description,
                    total_slips=a.total_slips,
                    anomaly_score=round(a.anomaly_score, 4),
                    is_anomaly=a.is_anomaly,
                    anomaly_reason=a.anomaly_reason,
                    top_org_share=round(a.top_org_share, 4),
                    org_hhi=round(a.org_hhi, 4),
                    position_unanimity=round(a.position_unanimity, 4),
                    n_proponent=a.n_proponent,
                    n_opponent=a.n_opponent,
                    unique_orgs=a.unique_orgs,
                )
                for a in page
            ],
            page_info=page_info,
        )

    @strawberry.field(description="Model quality assessment from the ML pipeline.")
    def model_quality(self) -> ModelQualityType | None:
        ml = state.ml
        if not ml or not ml.available or not ml.quality:
            return None
        q = ml.quality
        trust = q.get("trust_assessment", {})
        ts = q.get("test_set_metrics", {})
        return ModelQualityType(
            model_selected=q.get("model_selected", ""),
            trust_overall=trust.get("overall", "UNKNOWN"),
            strengths=trust.get("strengths", []),
            issues=trust.get("issues", []),
            test_roc_auc=ts.get("roc_auc"),
            test_accuracy=ts.get("accuracy"),
            test_precision_pos=ts.get("precision_pos"),
            test_recall_pos=ts.get("recall_pos"),
            test_f1_pos=ts.get("f1_pos"),
            top_features=[
                FeatureImportanceType(name=f["name"], importance=round(f["importance"], 4))
                for f in q.get("top_features", [])[:15]
            ],
            last_run_date=ml.last_run_date,
        )

    @strawberry.field(description="Prediction accuracy history across pipeline runs.")
    def prediction_accuracy(self, limit_runs: int = 20) -> list[AccuracySnapshotType]:
        ml = state.ml
        if not ml or not ml.available:
            return []
        runs = ml.accuracy_history[-limit_runs:]
        return [
            AccuracySnapshotType(
                run_date=r.run_date,
                snapshot_date=r.snapshot_date,
                days_elapsed=r.days_elapsed,
                total_testable=r.total_testable,
                correct=r.correct,
                accuracy=round(r.accuracy, 4),
                precision_advance=round(r.precision_advance, 4),
                recall_advance=round(r.recall_advance, 4),
                f1_advance=round(r.f1_advance, 4),
                model_version=r.model_version,
                biggest_misses=[
                    PredictionMissType(
                        bill_number=m.get("bill_number", ""),
                        description=m.get("description", ""),
                        predicted=m.get("predicted", ""),
                        actual=m.get("actual", ""),
                        confidence=m.get("confidence", 0),
                    )
                    for m in r.biggest_misses[:10]
                ],
            )
            for r in runs
        ]
