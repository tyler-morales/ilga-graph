"""The Moneyball — Analytics & Scoring Engine.

Turn raw stats into "Legislative DNA": identify the most effective legislators
by looking beyond volume to focus on real impact.

Key concepts
------------
- **Effectiveness Score**: Laws Passed / Laws Filed (HB/SB only, no fluff).
- **Magnet Score**: Co-sponsorship network strength (avg co-sponsors per law).
- **Bridge Score**: Cross-party co-sponsorship rate.
- **Pipeline Depth**: Average progression of a member's bills through the
  legislative pipeline (0 = filed, 6 = signed by Governor).
- **Network Centrality**: Graph-theory degree centrality in the co-sponsorship
  network — how connected a member is to *unique* other legislators.
- **Moneyball Score**: Composite metric that blends all of the above to surface
  legislators whose impact exceeds their visibility.

Success metric: *"Can we identify the most effective legislator in the House
who is not in leadership?"*
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .analytics import (
    _PIPELINE_MAX_DEPTH,
    MemberScorecard,
    compute_all_scorecards,
    is_substantive,
    pipeline_depth,
)
from .models import Bill, Member

# ── Leadership detection ─────────────────────────────────────────────────────

# Role titles that indicate formal chamber leadership.  Matched
# case-insensitively against ``member.role``.
_LEADERSHIP_TITLES: tuple[str, ...] = (
    "speaker of the house",
    "president of the senate",
    "minority leader",
    "majority leader",
    "assistant majority leader",
    "assistant minority leader",
    "deputy majority leader",
    "deputy minority leader",
    "republican leader",
    "democratic leader",
    "caucus chair",
    "conference chair",
    "majority caucus chair",
    "minority caucus chair",
    "republican caucus chair",
    "democratic caucus chair",
    "majority caucus whip",
    "minority caucus whip",
    "majority whip",
    "minority whip",
    "chief deputy majority whip",
    "chief deputy minority whip",
    "president pro tempore",
    "speaker pro tempore",
)


def is_leadership(member: Member) -> bool:
    """Return ``True`` if *member* holds a formal leadership position.

    Checks both the ``role`` field and the raw ``career_timeline_text`` for
    leadership language.
    """
    role_lower = (member.role or "").lower().strip()
    if not role_lower:
        return False
    # Exact titles
    for title in _LEADERSHIP_TITLES:
        if title in role_lower:
            return True
    # Catch-all for "leader", "whip", "speaker", "president" in the role
    for keyword in ("leader", "whip", "speaker pro", "president pro"):
        if keyword in role_lower:
            return True
    return False


# ── Network centrality ───────────────────────────────────────────────────────


def _build_cosponsor_edges(
    members: list[Member],
) -> dict[str, set[str]]:
    """Build an undirected co-sponsorship adjacency map: member_id -> {peer_ids}.

    An edge exists between member A and member B if they share at least one
    bill (one as primary sponsor, the other as co-sponsor, OR both as
    co-sponsors on the same bill).
    """
    # bill_number -> set of member_ids that touch the bill
    bill_members: dict[str, set[str]] = {}
    for member in members:
        for bill in member.bills:
            bill_members.setdefault(bill.bill_number, set()).add(member.id)

    # Build adjacency
    adjacency: dict[str, set[str]] = {m.id: set() for m in members}
    for _bn, member_ids in bill_members.items():
        ids_list = list(member_ids)
        for i, a in enumerate(ids_list):
            for b in ids_list[i + 1 :]:
                adjacency[a].add(b)
                adjacency[b].add(a)
    return adjacency


def degree_centrality(adjacency: dict[str, set[str]]) -> dict[str, float]:
    """Compute normalized degree centrality for each node.

    Degree centrality = (number of unique connections) / (N - 1), where N
    is the total number of nodes.  Returns 0.0 for isolates or single-node
    graphs.
    """
    n = len(adjacency)
    if n <= 1:
        return {mid: 0.0 for mid in adjacency}
    max_degree = n - 1
    return {mid: len(peers) / max_degree for mid, peers in adjacency.items()}


# ── Pipeline depth ───────────────────────────────────────────────────────────


def avg_pipeline_depth(bills: list[Bill]) -> float:
    """Return the average pipeline depth across a list of bills.

    Only substantive bills (HB/SB) are considered.  Returns 0.0 if there
    are no substantive bills.
    """
    laws = [b for b in bills if is_substantive(b.bill_number)]
    if not laws:
        return 0.0
    total = sum(pipeline_depth(b.last_action) for b in laws)
    return total / len(laws)


# ── Moneyball Profile ────────────────────────────────────────────────────────


@dataclass
class MoneyballProfile:
    """Full Moneyball analytics profile for a single member."""

    member_id: str
    member_name: str
    chamber: str
    party: str
    district: str
    role: str
    is_leadership: bool

    # ── From existing scorecard ──
    laws_filed: int  # law_heat_score (HB/SB only)
    laws_passed: int  # law_passed_count
    effectiveness_rate: float  # laws_passed / laws_filed
    magnet_score: float  # avg co-sponsors per law
    bridge_score: float  # % of laws with cross-party co-sponsor
    resolutions_filed: int
    resolutions_passed: int

    # ── New Moneyball metrics ──
    pipeline_depth_avg: float  # avg progression (0-6) of primary HB/SB
    pipeline_depth_normalized: float  # pipeline_depth_avg / 6
    network_centrality: float  # degree centrality in co-sponsorship graph
    unique_collaborators: int  # number of unique co-sponsorship peers
    total_primary_bills: int
    total_passed: int

    # ── The composite score ──
    moneyball_score: float = 0.0

    # ── Rank (populated by the ranker) ──
    rank_overall: int = 0
    rank_chamber: int = 0
    rank_non_leadership: int = 0

    # ── Badges ──
    badges: list[str] = field(default_factory=list)


class MoneyballWeights:
    """Tunable weights for the composite Moneyball Score.

    All weights should sum to 1.0 for interpretability, but the engine
    normalizes them regardless.
    """

    def __init__(
        self,
        effectiveness: float = 0.30,
        pipeline: float = 0.20,
        magnet: float = 0.20,
        bridge: float = 0.15,
        centrality: float = 0.15,
    ) -> None:
        self.effectiveness = effectiveness
        self.pipeline = pipeline
        self.magnet = magnet
        self.bridge = bridge
        self.centrality = centrality

    @property
    def total(self) -> float:
        return self.effectiveness + self.pipeline + self.magnet + self.bridge + self.centrality


# ── Composite Moneyball Score ────────────────────────────────────────────────


def _normalize_magnet(magnet: float, max_magnet: float) -> float:
    """Normalize magnet score to 0-1 range using the max across all members."""
    if max_magnet <= 0:
        return 0.0
    return min(magnet / max_magnet, 1.0)


def _compute_moneyball_score(
    profile: MoneyballProfile,
    max_magnet: float,
    weights: MoneyballWeights,
) -> float:
    """Compute the composite Moneyball Score (0.0 – 100.0).

    Components (each normalized to 0-1 before weighting):
    - effectiveness_rate: already 0-1
    - pipeline_depth_normalized: already 0-1
    - magnet_normalized: magnet / max_magnet across cohort
    - bridge_score: already 0-1
    - network_centrality: already 0-1
    """
    w = weights
    total_weight = w.total or 1.0

    raw = (
        w.effectiveness * profile.effectiveness_rate
        + w.pipeline * profile.pipeline_depth_normalized
        + w.magnet * _normalize_magnet(profile.magnet_score, max_magnet)
        + w.bridge * profile.bridge_score
        + w.centrality * profile.network_centrality
    )
    return round((raw / total_weight) * 100, 2)


# ── Badge assignment ─────────────────────────────────────────────────────────


def _assign_badges(profile: MoneyballProfile) -> list[str]:
    """Assign achievement badges based on thresholds."""
    badges: list[str] = []

    if profile.laws_filed > 0 and profile.effectiveness_rate >= 0.25:
        badges.append("Closer")  # High effectiveness rate
    if profile.magnet_score >= 10:
        badges.append("Coalition Builder")  # Strong co-sponsor attraction
    if profile.bridge_score >= 0.20:
        badges.append("Bipartisan Bridge")  # Cross-party reach
    if profile.pipeline_depth_avg >= 4.0 and profile.laws_filed >= 3:
        badges.append("Pipeline Driver")  # Bills go far even if not all pass
    if profile.network_centrality >= 0.5:
        badges.append("Network Hub")  # Highly connected legislator
    if profile.unique_collaborators >= 20:
        badges.append("Wide Tent")  # Works with many different legislators
    if (
        profile.resolutions_filed > 0
        and profile.laws_filed > 0
        and profile.resolutions_filed > profile.laws_filed * 2
    ):
        badges.append("Ceremonial Focus")  # Lots of resolutions relative to laws
    if (
        not profile.is_leadership
        and profile.laws_passed >= 2
        and profile.effectiveness_rate >= 0.15
        and profile.magnet_score >= 2.0
    ):
        badges.append("Hidden Gem")  # Effective without the leadership title

    return badges


# ── The Engine ───────────────────────────────────────────────────────────────


@dataclass
class MoneyballReport:
    """Complete Moneyball analytics output for the full legislature."""

    profiles: dict[str, MoneyballProfile]  # member_id -> profile
    rankings_overall: list[str]  # member_ids sorted by moneyball_score DESC
    rankings_house: list[str]
    rankings_senate: list[str]
    rankings_house_non_leadership: list[str]
    rankings_senate_non_leadership: list[str]
    mvp_house_non_leadership: str | None  # member_id of top non-leadership House member
    mvp_senate_non_leadership: str | None  # member_id of top non-leadership Senate member
    weights_used: MoneyballWeights


def compute_moneyball(
    members: list[Member],
    *,
    scorecards: dict[str, MemberScorecard] | None = None,
    weights: MoneyballWeights | None = None,
) -> MoneyballReport:
    """Run the full Moneyball analytics pipeline.

    Parameters
    ----------
    members:
        All scraped members (both chambers).
    scorecards:
        Pre-computed scorecards from :func:`compute_all_scorecards`.
        Built automatically when not supplied.
    weights:
        Optional tuning knobs for the composite score.

    Returns
    -------
    A :class:`MoneyballReport` with profiles, rankings, and the answer to
    "Who is the most effective non-leadership House member?"
    """
    if weights is None:
        weights = MoneyballWeights()

    # ── Step 1: Scorecards (reuse or compute) ──
    if scorecards is None:
        scorecards = compute_all_scorecards(members)

    # ── Step 2: Co-sponsorship network ──
    adjacency = _build_cosponsor_edges(members)
    centralities = degree_centrality(adjacency)

    # ── Step 3: Build raw profiles ──
    profiles: dict[str, MoneyballProfile] = {}
    for member in members:
        sc = scorecards.get(member.id)
        if sc is None:
            continue

        depth = avg_pipeline_depth(member.sponsored_bills)
        leadership = is_leadership(member)

        profiles[member.id] = MoneyballProfile(
            member_id=member.id,
            member_name=member.name,
            chamber=member.chamber,
            party=member.party,
            district=member.district,
            role=member.role,
            is_leadership=leadership,
            laws_filed=sc.law_heat_score,
            laws_passed=sc.law_passed_count,
            effectiveness_rate=sc.law_success_rate,
            magnet_score=sc.magnet_score,
            bridge_score=sc.bridge_score,
            resolutions_filed=sc.resolutions_count,
            resolutions_passed=sc.resolutions_passed_count,
            pipeline_depth_avg=round(depth, 2),
            pipeline_depth_normalized=(
                round(depth / _PIPELINE_MAX_DEPTH, 4) if _PIPELINE_MAX_DEPTH > 0 else 0.0
            ),
            network_centrality=round(centralities.get(member.id, 0.0), 4),
            unique_collaborators=len(adjacency.get(member.id, set())),
            total_primary_bills=sc.primary_bill_count,
            total_passed=sc.passed_count,
        )

    # ── Step 4: Compute composite scores ──
    max_magnet = max((p.magnet_score for p in profiles.values()), default=0.0)
    for profile in profiles.values():
        profile.moneyball_score = _compute_moneyball_score(
            profile,
            max_magnet,
            weights,
        )

    # ── Step 5: Assign badges ──
    for profile in profiles.values():
        profile.badges = _assign_badges(profile)

    # ── Step 6: Rank ──
    all_sorted = sorted(profiles.values(), key=lambda p: p.moneyball_score, reverse=True)
    for i, p in enumerate(all_sorted, 1):
        p.rank_overall = i

    house = [p for p in all_sorted if p.chamber == "House"]
    senate = [p for p in all_sorted if p.chamber == "Senate"]
    for i, p in enumerate(house, 1):
        p.rank_chamber = i
    for i, p in enumerate(senate, 1):
        p.rank_chamber = i

    house_nl = [p for p in house if not p.is_leadership]
    senate_nl = [p for p in senate if not p.is_leadership]
    for i, p in enumerate(house_nl, 1):
        p.rank_non_leadership = i
    for i, p in enumerate(senate_nl, 1):
        p.rank_non_leadership = i

    return MoneyballReport(
        profiles=profiles,
        rankings_overall=[p.member_id for p in all_sorted],
        rankings_house=[p.member_id for p in house],
        rankings_senate=[p.member_id for p in senate],
        rankings_house_non_leadership=[p.member_id for p in house_nl],
        rankings_senate_non_leadership=[p.member_id for p in senate_nl],
        mvp_house_non_leadership=house_nl[0].member_id if house_nl else None,
        mvp_senate_non_leadership=senate_nl[0].member_id if senate_nl else None,
        weights_used=weights,
    )
