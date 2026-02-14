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
    BillStatus,
    MemberScorecard,
    classify_bill_status,
    compute_all_scorecards,
    is_shell_bill,
    is_substantive,
    pipeline_depth,
)
from .models import Bill, CommitteeMemberRole, Member

# ── Role aggregation ─────────────────────────────────────────────────────────


def populate_member_roles(
    members: list[Member],
    committee_rosters: dict[str, list[CommitteeMemberRole]],
) -> None:
    """Aggregate all role titles into each member's ``roles`` list.

    Sources:
    1. ``member.role`` — the profile-page title (e.g. "Senator", "President
       of the Senate").
    2. Committee roster entries — each ``CommitteeMemberRole.role`` where the
       member appears (e.g. "Chairperson", "Minority Spokesperson").

    Duplicates are removed while preserving order.  The function mutates
    *members* in place.
    """
    # Build member_id -> set of committee roles
    roster_roles: dict[str, list[str]] = {}
    for _code, roles in committee_rosters.items():
        for cmr in roles:
            if cmr.member_id:
                roster_roles.setdefault(cmr.member_id, []).append(cmr.role)

    for member in members:
        seen: set[str] = set()
        aggregated: list[str] = []

        # Profile role first
        if member.role and member.role not in seen:
            seen.add(member.role)
            aggregated.append(member.role)

        # Committee roles
        for r in roster_roles.get(member.id, []):
            if r and r not in seen:
                seen.add(r)
                aggregated.append(r)

        member.roles = aggregated


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


# ── Institutional power scoring ──────────────────────────────────────────────


def compute_institutional_weight(member: Member) -> float:
    """Return an institutional-power bonus (0.0 – 1.0) based on role titles.

    Tiers (highest match wins — no stacking):

    * **1.0** — top chamber leadership: President, Leader, Speaker.
    * **0.5** — committee leadership: Chair (excluding "Caucus Chair"),
      Spokesperson.
    * **0.25** — party management: Whip, Caucus Chair.

    Roles are drawn from ``member.roles`` (aggregated profile + committee
    roster titles).  If ``member.roles`` is empty, falls back to
    ``member.role`` for backward compatibility.
    """
    roles_to_check = member.roles or ([member.role] if member.role else [])

    tier_1 = False  # President / Leader / Speaker
    tier_2 = False  # Chair / Spokesperson
    tier_3 = False  # Whip / Caucus Chair

    for role in roles_to_check:
        rl = role.lower()

        # Tier 1: top chamber leadership
        if "president" in rl or "leader" in rl or "speaker" in rl:
            tier_1 = True
            break  # Can't get higher — short-circuit

        # Tier 3 check first (Caucus Chair) so we can exclude it from Tier 2
        if "caucus chair" in rl or "whip" in rl:
            tier_3 = True
            continue

        # Tier 2: committee chair or spokesperson (but NOT "Caucus Chair",
        # which was already captured above)
        if "chair" in rl or "spokesperson" in rl:
            tier_2 = True
            continue

    if tier_1:
        return 1.0
    if tier_2:
        return 0.5
    if tier_3:
        return 0.25
    return 0.0


# ── Network centrality ───────────────────────────────────────────────────────


def build_cosponsor_edges(
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


def betweenness_centrality(adjacency: dict[str, set[str]]) -> dict[str, float]:
    """Compute normalized betweenness centrality for each node.

    Betweenness centrality measures how often a node lies on shortest paths
    between other nodes.  High betweenness = "bridge" or "connector" who
    links different groups.  This is the key signal for *structural*
    influence: legislators who broker relationships between blocs.

    Uses networkx for the computation; normalizes to 0-1.
    Returns 0.0 for all nodes if the graph has fewer than 3 nodes.
    """
    import networkx as nx

    if len(adjacency) < 3:
        return {mid: 0.0 for mid in adjacency}

    G = nx.Graph()
    for mid, peers in adjacency.items():
        for peer in peers:
            G.add_edge(mid, peer)

    # normalized=True divides by (n-1)(n-2)/2 so values are 0-1
    bc = nx.betweenness_centrality(G, normalized=True)

    # Ensure every member from adjacency is represented (even isolates)
    return {mid: round(bc.get(mid, 0.0), 6) for mid in adjacency}


# ── Pipeline depth ───────────────────────────────────────────────────────────


def avg_pipeline_depth(bills: list[Bill]) -> float:
    """Return the average pipeline depth across a list of bills.

    Only substantive, non-shell bills (HB/SB) are considered.  Returns 0.0
    if there are no qualifying bills.
    """
    laws = [b for b in bills if is_substantive(b.bill_number) and not is_shell_bill(b)]
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
    betweenness: float  # betweenness centrality — bridge/connector influence
    unique_collaborators: int  # number of unique co-sponsorship peers
    total_primary_bills: int
    total_passed: int

    # ── Influence Network metrics (human-readable power picture) ──
    collaborator_republicans: int = 0  # unique co-sponsorship peers who are Republican
    collaborator_democrats: int = 0  # unique co-sponsorship peers who are Democrat
    collaborator_other: int = 0  # unique co-sponsorship peers of other parties
    magnet_vs_chamber: float = 0.0  # magnet_score / chamber avg magnet (e.g. 2.1x)
    cosponsor_passage_rate: float = 0.0  # passage rate of bills this member co-sponsors
    cosponsor_passage_multiplier: float = 0.0  # cosponsor rate / chamber median (e.g. 1.3x)
    chamber_median_cosponsor_rate: float = 0.0  # chamber median for transparency
    passage_rate_vs_caucus: float = 0.0  # member passage rate / caucus avg (e.g. 2.3x)
    caucus_avg_passage_rate: float = 0.0  # party+chamber avg passage rate for transparency

    # ── Institutional power bonus (0.0 – 1.0) ──
    institutional_weight: float = 0.0

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

    Default allocation (v2 — with institutional power bonus):

    * effectiveness  24%  (was 30%)
    * pipeline       16%  (was 20%)
    * magnet         16%  (was 20%)
    * bridge         12%  (was 15%)
    * centrality     12%  (was 15%)
    * institutional  20%  (new)
    """

    def __init__(
        self,
        effectiveness: float = 0.24,
        pipeline: float = 0.16,
        magnet: float = 0.16,
        bridge: float = 0.12,
        centrality: float = 0.12,
        institutional: float = 0.20,
    ) -> None:
        self.effectiveness = effectiveness
        self.pipeline = pipeline
        self.magnet = magnet
        self.bridge = bridge
        self.centrality = centrality
        self.institutional = institutional

    @property
    def total(self) -> float:
        return (
            self.effectiveness
            + self.pipeline
            + self.magnet
            + self.bridge
            + self.centrality
            + self.institutional
        )


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
    - institutional_weight: already 0-1
    """
    w = weights
    total_weight = w.total or 1.0

    raw = (
        w.effectiveness * profile.effectiveness_rate
        + w.pipeline * profile.pipeline_depth_normalized
        + w.magnet * _normalize_magnet(profile.magnet_score, max_magnet)
        + w.bridge * profile.bridge_score
        + w.centrality * profile.network_centrality
        + w.institutional * profile.institutional_weight
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
    if profile.betweenness >= 0.02:
        badges.append("Bridge Connector")  # Key broker between legislative groups
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


# ── Institutional Power Badges ────────────────────────────────────────────────


@dataclass
class PowerBadge:
    """A visual badge indicating institutional power.

    Rendered prominently at the top of each advocacy card so staff
    can immediately see who holds formal power.
    """

    label: str  # e.g. "LEADERSHIP"
    icon: str  # CSS class suffix — "leadership", "chair", "influence"
    explanation: str  # human-readable context shown on click/hover
    css_class: str  # full CSS class — "power-badge-leadership"


def compute_power_badges(
    profile: MoneyballProfile,
    committee_roles: list[dict],
    chamber_size: int,
) -> list[PowerBadge]:
    """Return institutional power badges for a legislator.

    Three badge types (additive — a member can earn all three):

    1. **LEADERSHIP** — formal chamber or party role (institutional_weight >= 0.25).
    2. **COMMITTEE CHAIR** — chairs one or more committees (gatekeeper power).
    3. **TOP 5% INFLUENCE** — rank_chamber in the top 5% of their chamber.
    """
    import math

    badges: list[PowerBadge] = []

    # ── 1. LEADERSHIP ──
    if profile.institutional_weight >= 0.25:
        role_display = profile.role or "Leadership"
        if profile.institutional_weight >= 1.0:
            explanation = (
                f"Top chamber leader \u2014 {role_display}. "
                "Controls floor schedule and party strategy."
            )
        elif profile.institutional_weight >= 0.5:
            explanation = (
                f"Committee leader \u2014 {role_display}. Controls which bills get hearings."
            )
        else:
            explanation = (
                f"Party management \u2014 {role_display}. Coordinates caucus votes and whip counts."
            )
        badges.append(
            PowerBadge(
                label="LEADERSHIP",
                icon="leadership",
                explanation=explanation,
                css_class="power-badge-leadership",
            )
        )

    # ── 2. COMMITTEE CHAIR ──
    chaired: list[str] = []
    for cr in committee_roles:
        role_lower = cr.get("role", "").lower()
        if "chair" in role_lower and "vice" not in role_lower:
            chaired.append(cr.get("name", "Unknown Committee"))

    if chaired:
        if len(chaired) == 1:
            explanation = f"Controls which bills get hearings in {chaired[0]}."
        else:
            names = ", ".join(chaired)
            explanation = (
                f"Chairs {len(chaired)} committees: {names}. "
                "Controls the hearing schedule for each."
            )
        badges.append(
            PowerBadge(
                label="COMMITTEE CHAIR",
                icon="chair",
                explanation=explanation,
                css_class="power-badge-chair",
            )
        )

    # ── 3. TOP 5% INFLUENCE ──
    if chamber_size > 0 and profile.rank_chamber > 0:
        cutoff = max(math.ceil(chamber_size * 0.05), 1)
        if profile.rank_chamber <= cutoff:
            chamber_label = "senators" if profile.chamber == "Senate" else "representatives"
            pct = round((profile.rank_chamber / chamber_size) * 100, 1)
            explanation = (
                f"Ranked #{profile.rank_chamber} of {chamber_size} "
                f"{chamber_label} in overall legislative effectiveness."
            )
            # Show "TOP 1%" vs "TOP 5%" based on actual percentile
            pct_label = f"TOP {max(int(pct), 1)}%" if pct <= 5 else "TOP 5%"
            badges.append(
                PowerBadge(
                    label=pct_label + " INFLUENCE",
                    icon="influence",
                    explanation=explanation,
                    css_class="power-badge-influence",
                )
            )

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
    _member_lookup = {m.id: m for m in members}

    # ── Step 2: Co-sponsorship network ──
    adjacency = build_cosponsor_edges(members)
    centralities = degree_centrality(adjacency)
    betweenness_scores = betweenness_centrality(adjacency)

    # ── Step 3: Build raw profiles ──
    profiles: dict[str, MoneyballProfile] = {}
    for member in members:
        sc = scorecards.get(member.id)
        if sc is None:
            continue

        depth = avg_pipeline_depth(member.sponsored_bills)
        leadership = is_leadership(member)
        inst_weight = compute_institutional_weight(member)

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
            pipeline_depth_normalized=round(depth / _PIPELINE_MAX_DEPTH, 4)
            if _PIPELINE_MAX_DEPTH > 0
            else 0.0,
            network_centrality=round(centralities.get(member.id, 0.0), 4),
            betweenness=round(betweenness_scores.get(member.id, 0.0), 6),
            unique_collaborators=len(adjacency.get(member.id, set())),
            total_primary_bills=sc.primary_bill_count,
            total_passed=sc.passed_count,
            institutional_weight=inst_weight,
        )

    # ── Step 3b: Influence Network metrics ──
    # Party breakdown of collaborators
    for member_id, profile in profiles.items():
        peers = adjacency.get(member_id, set())
        rep_count = 0
        dem_count = 0
        other_count = 0
        for peer_id in peers:
            peer = _member_lookup.get(peer_id)
            if peer is None:
                continue
            party_lower = peer.party.lower()
            if "republican" in party_lower:
                rep_count += 1
            elif "democrat" in party_lower:
                dem_count += 1
            else:
                other_count += 1
        profile.collaborator_republicans = rep_count
        profile.collaborator_democrats = dem_count
        profile.collaborator_other = other_count

    # Chamber average magnet score (for "Nx higher than chamber average")
    chamber_magnets: dict[str, list[float]] = {}
    for p in profiles.values():
        if p.laws_filed > 0:  # only count members who have filed laws
            chamber_magnets.setdefault(p.chamber, []).append(p.magnet_score)
    chamber_avg_magnet: dict[str, float] = {}
    for chamber, magnets in chamber_magnets.items():
        chamber_avg_magnet[chamber] = sum(magnets) / len(magnets) if magnets else 0.0

    for profile in profiles.values():
        avg_mag = chamber_avg_magnet.get(profile.chamber, 0.0)
        if avg_mag > 0 and profile.laws_filed > 0:
            profile.magnet_vs_chamber = round(profile.magnet_score / avg_mag, 1)

    # Caucus average passage rate (party + chamber) — for "Nx higher than caucus"
    caucus_rates: dict[str, list[float]] = {}  # "Senate-Democrat" -> [rates]
    for p in profiles.values():
        if p.laws_filed > 0:
            caucus_key = f"{p.chamber}-{p.party}"
            caucus_rates.setdefault(caucus_key, []).append(p.effectiveness_rate)
    caucus_avg_passage: dict[str, float] = {}
    for key, rates in caucus_rates.items():
        caucus_avg_passage[key] = sum(rates) / len(rates) if rates else 0.0

    for profile in profiles.values():
        if profile.laws_filed > 0:
            caucus_key = f"{profile.chamber}-{profile.party}"
            avg_rate = caucus_avg_passage.get(caucus_key, 0.0)
            profile.caucus_avg_passage_rate = round(avg_rate, 4)
            if avg_rate > 0:
                profile.passage_rate_vs_caucus = round(
                    profile.effectiveness_rate / avg_rate,
                    1,
                )

    # Co-sponsored bill passage rate & peer-normalised baseline
    # For each member: what % of bills they co-sponsor end up passing?
    # Baseline: median co-sponsor passage rate across chamber members.
    # Using chamber-wide bill passage rate would inflate the multiplier
    # because popular/likely-to-pass bills naturally attract more co-sponsors
    # (selection bias).  Comparing against the median co-sponsor rate
    # normalises for this effect and shows genuine "picking winners" ability.
    member_cosponsor_rates: dict[str, tuple[str, float]] = {}  # member_id -> (chamber, rate)
    for member in members:
        profile = profiles.get(member.id)
        if profile is None:
            continue
        cosponsor_bills = [
            b
            for b in member.co_sponsor_bills
            if is_substantive(b.bill_number) and not is_shell_bill(b)
        ]
        if cosponsor_bills:
            cosponsor_passed = sum(
                1
                for b in cosponsor_bills
                if classify_bill_status(b.last_action) == BillStatus.PASSED
            )
            rate = cosponsor_passed / len(cosponsor_bills)
            profile.cosponsor_passage_rate = round(rate, 4)
            member_cosponsor_rates[member.id] = (member.chamber, rate)

    # Compute median co-sponsor passage rate per chamber
    chamber_cosponsor_rates: dict[str, list[float]] = {}
    for _mid, (chamber, rate) in member_cosponsor_rates.items():
        chamber_cosponsor_rates.setdefault(chamber, []).append(rate)

    chamber_median_cosponsor: dict[str, float] = {}
    for chamber, rates in chamber_cosponsor_rates.items():
        sorted_rates = sorted(rates)
        n = len(sorted_rates)
        if n == 0:
            chamber_median_cosponsor[chamber] = 0.0
        elif n % 2 == 1:
            chamber_median_cosponsor[chamber] = sorted_rates[n // 2]
        else:
            chamber_median_cosponsor[chamber] = (
                sorted_rates[n // 2 - 1] + sorted_rates[n // 2]
            ) / 2

    # Compute multiplier vs peer median (not vs raw chamber passage rate)
    for member_id, (chamber, rate) in member_cosponsor_rates.items():
        profile = profiles[member_id]
        median_rate = chamber_median_cosponsor.get(chamber, 0.0)
        profile.chamber_median_cosponsor_rate = round(median_rate, 4)
        if median_rate > 0:
            profile.cosponsor_passage_multiplier = round(
                rate / median_rate,
                1,
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
