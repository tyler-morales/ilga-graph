"""True Influence Engine — unified influence scoring for legislators.

Computes influence signals that go beyond the Moneyball composite:

1. **Vote Pivotality** — how often a member casts the deciding vote on close
   roll calls.  High pivotality = swing voter / real voting power.
2. **Sponsor Pull** — marginal effect of sponsoring/co-sponsoring a bill on
   its probability of advancement (from the ML bill predictor).
3. **Influence Score** — unified 0–100 composite combining Moneyball,
   betweenness centrality, pivotality, and sponsor pull.

All outputs are designed to integrate with the existing MoneyballProfile and
the ML pipeline, and are surfaced via ``/intelligence`` and GraphQL.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from .models import Member, VoteEvent

LOGGER = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VOTE PIVOTALITY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PivotalVote:
    """A single close vote where the member was on the winning side."""

    bill_number: str
    date: str
    description: str
    chamber: str
    margin: int  # |yea - nay|
    total_votes: int
    member_vote: str  # "Y" or "N"
    outcome: str  # "passed" or "failed"


@dataclass
class MemberPivotality:
    """Vote pivotality profile for a single legislator."""

    member_name: str
    member_id: str  # empty if unresolved

    # Core pivotality stats
    close_votes_total: int = 0  # close roll calls this member participated in
    pivotal_winning: int = 0  # close votes where they voted with the winning side
    pivotal_losing: int = 0  # close votes where they voted with the losing side
    pivotal_rate: float = 0.0  # pivotal_winning / close_votes_total

    # Deeper influence signals
    swing_votes: int = 0  # margin-of-1 votes (true tiebreakers)
    floor_pivotal: int = 0  # pivotal on floor votes specifically
    committee_pivotal: int = 0  # pivotal on committee votes

    # For display
    pivotal_votes: list[PivotalVote] = field(default_factory=list)


# ── Close-vote threshold ─────────────────────────────────────────────────────
# A vote is "close" if |yea - nay| <= MARGIN_THRESHOLD.
# Very close legislatures like IL often have tight committee votes.
# We use 5 as default — captures meaningful close calls without noise.
MARGIN_THRESHOLD = 5


def compute_vote_pivotality(
    vote_events: list[VoteEvent],
    member_lookup: dict[str, Member],
    *,
    margin_threshold: int = MARGIN_THRESHOLD,
) -> dict[str, MemberPivotality]:
    """Compute pivotality for every legislator across all vote events.

    For each vote event with |yea - nay| <= margin_threshold:
    - Every member who voted with the winning side is "pivotal-winning"
    - Every member who voted with the losing side is "pivotal-losing"
    - If margin == 1, they get a "swing vote" credit (true tiebreaker)

    Returns ``{member_name: MemberPivotality}``.
    """
    # Build name -> member_id lookup
    name_to_id: dict[str, str] = {}
    for m in member_lookup.values():
        name_to_id[m.name] = m.id

    # Accumulators
    pivotality: dict[str, MemberPivotality] = defaultdict(
        lambda: MemberPivotality(member_name="", member_id="")
    )

    close_vote_count = 0

    for event in vote_events:
        yea_count = len(event.yea_votes)
        nay_count = len(event.nay_votes)
        margin = abs(yea_count - nay_count)
        total_votes = yea_count + nay_count

        # Skip empty or unanimously one-sided votes
        if total_votes < 3:
            continue
        if margin > margin_threshold:
            continue

        close_vote_count += 1

        # Determine outcome
        outcome = "passed" if yea_count > nay_count else "failed"
        winning_side = "Y" if yea_count >= nay_count else "N"

        is_floor = event.vote_type == "floor"

        # Build the PivotalVote record
        pv = PivotalVote(
            bill_number=event.bill_number,
            date=event.date,
            description=event.description,
            chamber=event.chamber,
            margin=margin,
            total_votes=total_votes,
            member_vote="",
            outcome=outcome,
        )

        # Score yea voters
        for name in event.yea_votes:
            p = pivotality[name]
            if not p.member_name:
                p.member_name = name
                p.member_id = name_to_id.get(name, "")
            p.close_votes_total += 1

            if winning_side == "Y":
                p.pivotal_winning += 1
                vote_record = PivotalVote(**{**pv.__dict__, "member_vote": "Y"})
                p.pivotal_votes.append(vote_record)
            else:
                p.pivotal_losing += 1

            if margin <= 1:
                p.swing_votes += 1

            if is_floor:
                p.floor_pivotal += 1 if winning_side == "Y" else 0
            else:
                p.committee_pivotal += 1 if winning_side == "Y" else 0

        # Score nay voters
        for name in event.nay_votes:
            p = pivotality[name]
            if not p.member_name:
                p.member_name = name
                p.member_id = name_to_id.get(name, "")
            p.close_votes_total += 1

            if winning_side == "N":
                p.pivotal_winning += 1
                vote_record = PivotalVote(**{**pv.__dict__, "member_vote": "N"})
                p.pivotal_votes.append(vote_record)
            else:
                p.pivotal_losing += 1

            if margin <= 1:
                p.swing_votes += 1

            if is_floor:
                p.floor_pivotal += 1 if winning_side == "N" else 0
            else:
                p.committee_pivotal += 1 if winning_side == "N" else 0

    # Compute rates
    for p in pivotality.values():
        if p.close_votes_total > 0:
            p.pivotal_rate = round(p.pivotal_winning / p.close_votes_total, 4)
        # Sort pivotal votes by margin (tightest first)
        p.pivotal_votes.sort(key=lambda v: (v.margin, v.date))
        # Keep top 20 for display
        p.pivotal_votes = p.pivotal_votes[:20]

    LOGGER.info(
        "Pivotality: %d close votes (margin <= %d), %d members scored.",
        close_vote_count,
        margin_threshold,
        len(pivotality),
    )

    return dict(pivotality)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPONSOR PULL (marginal influence from bill predictor)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SponsorPull:
    """Measures whether a member's sponsorship/co-sponsorship is associated
    with higher probability of bill advancement."""

    member_id: str
    member_name: str

    # As primary sponsor
    sponsored_bills: int = 0
    sponsored_avg_prob: float = 0.0  # avg P(advance) of bills they sponsor
    chamber_avg_prob: float = 0.0  # avg P(advance) across all bills in chamber
    sponsor_lift: float = 0.0  # sponsored_avg_prob - chamber_avg_prob

    # As co-sponsor
    cosponsored_bills: int = 0
    cosponsored_avg_prob: float = 0.0  # avg P(advance) of bills they co-sponsor
    cosponsor_lift: float = 0.0  # cosponsored_avg_prob - chamber_avg_prob

    # Overall
    pull_score: float = 0.0  # blended lift score (normalized 0-1 later)


def compute_sponsor_pull(
    members: list[Member],
    bill_scores: dict[str, float],
) -> dict[str, SponsorPull]:
    """Compute sponsor pull for each member using ML bill scores.

    For each member:
    - Collect P(advance) for all bills they primary-sponsor
    - Collect P(advance) for all bills they co-sponsor
    - Compare both averages against the chamber average
    - Positive "lift" = their involvement is associated with higher success

    Parameters
    ----------
    members:
        All scraped members.
    bill_scores:
        ``{bill_leg_id: prob_advance}`` from the ML bill predictor.
        If empty (pipeline hasn't run), returns empty dict.

    Returns ``{member_id: SponsorPull}``.
    """
    if not bill_scores:
        LOGGER.info("Sponsor pull: no bill scores available (run ML pipeline).")
        return {}

    # Chamber average P(advance) — baseline
    chamber_probs: dict[str, list[float]] = {}  # "Senate" -> [probs]
    for member in members:
        for bill in member.sponsored_bills:
            prob = bill_scores.get(bill.leg_id)
            if prob is not None:
                chamber_probs.setdefault(member.chamber, []).append(prob)

    chamber_avg: dict[str, float] = {}
    for chamber, probs in chamber_probs.items():
        chamber_avg[chamber] = sum(probs) / len(probs) if probs else 0.0

    pulls: dict[str, SponsorPull] = {}

    for member in members:
        sp = SponsorPull(
            member_id=member.id,
            member_name=member.name,
            chamber_avg_prob=chamber_avg.get(member.chamber, 0.0),
        )

        # Primary sponsor bills
        sponsor_probs = []
        for bill in member.sponsored_bills:
            prob = bill_scores.get(bill.leg_id)
            if prob is not None:
                sponsor_probs.append(prob)
        if sponsor_probs:
            sp.sponsored_bills = len(sponsor_probs)
            sp.sponsored_avg_prob = round(sum(sponsor_probs) / len(sponsor_probs), 4)
            sp.sponsor_lift = round(sp.sponsored_avg_prob - sp.chamber_avg_prob, 4)

        # Co-sponsor bills
        cosponsor_probs = []
        for bill in member.co_sponsor_bills:
            prob = bill_scores.get(bill.leg_id)
            if prob is not None:
                cosponsor_probs.append(prob)
        if cosponsor_probs:
            sp.cosponsored_bills = len(cosponsor_probs)
            sp.cosponsored_avg_prob = round(sum(cosponsor_probs) / len(cosponsor_probs), 4)
            sp.cosponsor_lift = round(sp.cosponsored_avg_prob - sp.chamber_avg_prob, 4)

        # Blended pull: weight sponsor lift 2x since primary sponsorship
        # is a stronger signal than co-sponsorship
        raw_pull = (2 * sp.sponsor_lift + sp.cosponsor_lift) / 3
        sp.pull_score = round(raw_pull, 4)

        pulls[member.id] = sp

    # Log summary
    positive_pull = sum(1 for p in pulls.values() if p.pull_score > 0)
    LOGGER.info(
        "Sponsor pull: %d members scored, %d with positive pull.",
        len(pulls),
        positive_pull,
    )

    return pulls


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UNIFIED INFLUENCE SCORE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class InfluenceProfile:
    """Unified influence profile combining all influence signals."""

    member_id: str
    member_name: str
    chamber: str
    party: str

    # ── Component scores (all 0-1 normalized) ──
    moneyball_normalized: float = 0.0  # moneyball_score / 100
    betweenness_normalized: float = 0.0  # betweenness / max_betweenness
    pivotality_normalized: float = 0.0  # pivotal_rate (already 0-1)
    pull_normalized: float = 0.0  # pull_score normalized to 0-1

    # ── The composite ──
    influence_score: float = 0.0  # 0-100 composite

    # ── Rank ──
    rank_overall: int = 0
    rank_chamber: int = 0

    # ── Human-readable summary ──
    influence_label: str = ""  # "High", "Moderate", "Low"
    influence_signals: list[str] = field(
        default_factory=list
    )  # e.g. ["Gets bills passed", "Bridge between blocs"]


@dataclass
class InfluenceWeights:
    """Tunable weights for the unified influence score.

    Moneyball already captures outcome + network + institutional power,
    so it gets the largest share.  Betweenness captures *structural*
    bridge influence that degree centrality misses.  Pivotality captures
    *voting* influence.  Pull captures *predictive* influence from ML.

    Default allocation:
        moneyball    40%   (outcome + network + institutional already blended)
        betweenness  20%   (bridge/connector influence)
        pivotality   20%   (swing vote / close-call influence)
        pull         20%   (ML-derived: their sponsorship predicts success)
    """

    moneyball: float = 0.40
    betweenness: float = 0.20
    pivotality: float = 0.20
    pull: float = 0.20

    @property
    def total(self) -> float:
        return self.moneyball + self.betweenness + self.pivotality + self.pull


def compute_influence_scores(
    moneyball_profiles: dict,  # member_id -> MoneyballProfile
    pivotality_data: dict[str, MemberPivotality],
    pull_data: dict[str, SponsorPull],
    member_lookup: dict[str, Member],
    *,
    weights: InfluenceWeights | None = None,
) -> dict[str, InfluenceProfile]:
    """Compute unified influence scores for all legislators.

    Normalizes each component to 0-1, then blends with weights.

    Parameters
    ----------
    moneyball_profiles:
        ``{member_id: MoneyballProfile}`` — must have ``moneyball_score``
        and ``betweenness`` fields.
    pivotality_data:
        ``{member_name: MemberPivotality}`` — keyed by name (from vote PDFs).
    pull_data:
        ``{member_id: SponsorPull}`` — keyed by member_id.
    member_lookup:
        ``{member_id: Member}`` for party/chamber info.
    weights:
        Tuning knobs.  Defaults to InfluenceWeights().

    Returns ``{member_id: InfluenceProfile}``.
    """
    if weights is None:
        weights = InfluenceWeights()

    w = weights
    w_total = w.total or 1.0

    # ── Normalize betweenness to 0-1 (relative to max in the cohort) ──
    max_betweenness = max(
        (getattr(p, "betweenness", 0.0) for p in moneyball_profiles.values()),
        default=0.0,
    )

    # ── Build name -> pivotality lookup ──
    name_to_member_id: dict[str, str] = {}
    for m in member_lookup.values():
        name_to_member_id[m.name] = m.id

    pivotality_by_id: dict[str, MemberPivotality] = {}
    for name, piv in pivotality_data.items():
        mid = name_to_member_id.get(name, piv.member_id)
        if mid:
            pivotality_by_id[mid] = piv

    # ── Normalize pull to 0-1 ──
    pull_scores = [p.pull_score for p in pull_data.values() if p.pull_score > 0]
    max_pull = max(pull_scores) if pull_scores else 1.0
    min_pull = min((p.pull_score for p in pull_data.values()), default=0.0)
    pull_range = max_pull - min_pull if max_pull > min_pull else 1.0

    profiles: dict[str, InfluenceProfile] = {}

    for member_id, mb in moneyball_profiles.items():
        member = member_lookup.get(member_id)
        if member is None:
            continue

        ip = InfluenceProfile(
            member_id=member_id,
            member_name=mb.member_name,
            chamber=mb.chamber,
            party=mb.party,
        )

        # 1. Moneyball (already 0-100, normalize to 0-1)
        ip.moneyball_normalized = round(mb.moneyball_score / 100.0, 4)

        # 2. Betweenness (normalize to 0-1 vs max)
        raw_bt = getattr(mb, "betweenness", 0.0)
        ip.betweenness_normalized = round(
            raw_bt / max_betweenness if max_betweenness > 0 else 0.0, 4
        )

        # 3. Pivotality (pivotal_rate is already 0-1)
        piv = pivotality_by_id.get(member_id)
        if piv:
            ip.pivotality_normalized = round(piv.pivotal_rate, 4)

        # 4. Pull (normalize to 0-1)
        sp = pull_data.get(member_id)
        if sp:
            ip.pull_normalized = round(max((sp.pull_score - min_pull) / pull_range, 0.0), 4)

        # ── Composite ──
        raw = (
            w.moneyball * ip.moneyball_normalized
            + w.betweenness * ip.betweenness_normalized
            + w.pivotality * ip.pivotality_normalized
            + w.pull * ip.pull_normalized
        )
        ip.influence_score = round((raw / w_total) * 100, 2)

        # ── Human-readable signals ──
        signals = []
        if ip.moneyball_normalized >= 0.5:
            signals.append("Gets bills passed (high Moneyball)")
        if ip.betweenness_normalized >= 0.3:
            signals.append("Bridges legislative blocs (high betweenness)")
        if ip.pivotality_normalized >= 0.5:
            signals.append("Swing voter on close calls (high pivotality)")
        if piv and piv.swing_votes >= 2:
            signals.append(f"Cast deciding vote {piv.swing_votes}x (margin-of-1)")
        if sp and sp.sponsor_lift > 0.1:
            signals.append("Bills they sponsor outperform predictions (positive pull)")
        ip.influence_signals = signals

        # ── Label ──
        if ip.influence_score >= 60:
            ip.influence_label = "High"
        elif ip.influence_score >= 30:
            ip.influence_label = "Moderate"
        else:
            ip.influence_label = "Low"

        profiles[member_id] = ip

    # ── Rank ──
    all_sorted = sorted(profiles.values(), key=lambda p: p.influence_score, reverse=True)
    for i, p in enumerate(all_sorted, 1):
        p.rank_overall = i

    house = [p for p in all_sorted if p.chamber == "House"]
    senate = [p for p in all_sorted if p.chamber == "Senate"]
    for i, p in enumerate(house, 1):
        p.rank_chamber = i
    for i, p in enumerate(senate, 1):
        p.rank_chamber = i

    # Log distribution
    high = sum(1 for p in profiles.values() if p.influence_label == "High")
    mod = sum(1 for p in profiles.values() if p.influence_label == "Moderate")
    low = sum(1 for p in profiles.values() if p.influence_label == "Low")
    LOGGER.info(
        "Influence scores: %d members — %d High, %d Moderate, %d Low.",
        len(profiles),
        high,
        mod,
        low,
    )

    return profiles


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COALITION INFLUENCE ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CoalitionInfluence:
    """Influence metadata for a single coalition / voting bloc."""

    coalition_id: int
    coalition_name: str

    # Top influencers in this bloc (by influence_score)
    top_influencer_id: str = ""
    top_influencer_name: str = ""
    top_influencer_score: float = 0.0
    top_influencer_label: str = ""

    # Bridge members: bloc members with highest betweenness
    # (they connect THIS bloc to other blocs)
    bridge_member_id: str = ""
    bridge_member_name: str = ""
    bridge_member_betweenness: float = 0.0

    # Bloc-level influence stats
    avg_influence: float = 0.0
    max_influence: float = 0.0
    high_influence_count: int = 0  # members labeled "High"
    total_members: int = 0


def enrich_coalitions_with_influence(
    coalition_members: list[dict],
    influence_profiles: dict[str, InfluenceProfile],
    moneyball_profiles: dict | None = None,
) -> list[CoalitionInfluence]:
    """Annotate each coalition with its top influencer and bridge member.

    Parameters
    ----------
    coalition_members:
        List of dicts with at least ``member_id``, ``coalition_id``,
        ``coalition_name``.  Typically from ``ml_loader.coalitions``
        or ``coalitions.parquet``.
    influence_profiles:
        ``{member_id: InfluenceProfile}`` from ``compute_influence_scores()``.
    moneyball_profiles:
        ``{member_id: MoneyballProfile}`` — used for betweenness scores.
        Optional; if not provided, bridge member detection uses influence
        betweenness_normalized.

    Returns a list of ``CoalitionInfluence`` (one per coalition).
    """
    # Group members by coalition_id
    blocs: dict[int, list[dict]] = {}
    for m in coalition_members:
        cid = m.get("coalition_id", -1)
        if isinstance(cid, int) and cid >= 0:
            blocs.setdefault(cid, []).append(m)

    results: list[CoalitionInfluence] = []

    for cid, members in sorted(blocs.items()):
        name = members[0].get("coalition_name", "") if members else ""
        ci = CoalitionInfluence(
            coalition_id=cid,
            coalition_name=name,
            total_members=len(members),
        )

        # Collect influence scores for bloc members
        bloc_influence: list[tuple[str, str, float]] = []
        bloc_betweenness: list[tuple[str, str, float]] = []

        for m in members:
            mid = m.get("member_id", "")
            mname = m.get("name", "")
            ip = influence_profiles.get(mid)
            if ip:
                bloc_influence.append((mid, mname, ip.influence_score))
                bloc_betweenness.append((mid, mname, ip.betweenness_normalized))
                if ip.influence_label == "High":
                    ci.high_influence_count += 1
            elif moneyball_profiles and mid in moneyball_profiles:
                # Fallback: use moneyball_score as influence proxy
                mb = moneyball_profiles[mid]
                bloc_influence.append((mid, mname, mb.moneyball_score))
                bt = getattr(mb, "betweenness", 0.0)
                bloc_betweenness.append((mid, mname, bt))

        if bloc_influence:
            scores = [s for _, _, s in bloc_influence]
            ci.avg_influence = round(sum(scores) / len(scores), 2)
            ci.max_influence = round(max(scores), 2)

            # Top influencer
            top = max(bloc_influence, key=lambda x: x[2])
            ci.top_influencer_id = top[0]
            ci.top_influencer_name = top[1]
            ci.top_influencer_score = round(top[2], 2)
            ip = influence_profiles.get(top[0])
            ci.top_influencer_label = ip.influence_label if ip else ""

        if bloc_betweenness:
            # Bridge member (highest betweenness in the bloc)
            bridge = max(bloc_betweenness, key=lambda x: x[2])
            if bridge[2] > 0:
                ci.bridge_member_id = bridge[0]
                ci.bridge_member_name = bridge[1]
                ci.bridge_member_betweenness = round(bridge[2], 4)

        results.append(ci)

    LOGGER.info("Coalition influence: %d blocs enriched.", len(results))
    return results
