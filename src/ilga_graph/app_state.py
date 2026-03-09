"""Global application state populated at startup. Used by routers and GraphQL."""

from .analytics import CommitteeStats, MemberScorecard
from .models import Bill, Committee, CommitteeMemberRole, Member, VoteEvent, WitnessSlip
from .moneyball import MoneyballReport
from .voting_record import VotingSummary
from .zip_crosswalk import ZipDistrictInfo


class AppState:
    """Mutable app state: members, bills, ML, influence, etc. Filled in lifespan."""

    def __init__(self) -> None:
        self.members: list[Member] = []
        self.member_lookup: dict[str, Member] = {}  # name-keyed (vote normalization, schema)
        self.member_lookup_by_id: dict[str, Member] = {}  # id-keyed (influence, graph)
        self.bills: list[Bill] = []
        self.bill_lookup: dict[str, Bill] = {}
        self.bills_lookup: dict[str, Bill] = {}  # leg_id -> Bill (for bill detail action_history)
        self.committees: list[Committee] = []
        self.committee_lookup: dict[str, Committee] = {}
        self.committee_rosters: dict[str, list[CommitteeMemberRole]] = {}
        self.committee_bills: dict[str, list[str]] = {}
        self.committee_stats: dict[str, CommitteeStats] = {}
        self.member_committee_roles: dict[str, list[dict]] = {}
        self.scorecards: dict[str, MemberScorecard] = {}
        self.moneyball: MoneyballReport | None = None
        self.vote_events: list[VoteEvent] = []
        self.vote_lookup: dict[str, list[VoteEvent]] = {}
        self.member_vote_records: dict[str, VotingSummary] = {}
        self.category_bill_sets: dict[str, set[str]] = {}
        self.witness_slips: list[WitnessSlip] = []
        self.witness_slips_lookup: dict[str, list[WitnessSlip]] = {}
        self.zip_to_district: dict[str, ZipDistrictInfo] = {}
        self.ml: object | None = None
        self.cosponsor_adjacency: dict[str, set[str]] = {}
        self.pivotality: dict = {}
        self.sponsor_pull: dict = {}
        self.influence: dict = {}
        self.coalition_influence: list = []
        # username -> followers_count for Legislator Twitter tab (from cache, refreshed by script).
        self.twitter_follower_counts: dict[str, int] = {}


state = AppState()
