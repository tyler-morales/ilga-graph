from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .models import Bill, Committee, CommitteeMemberRole, Member, WitnessSlip

# ── Status classification ────────────────────────────────────────────────────


class BillStatus(str, Enum):
    """High-level legislative outcome buckets.

    Inherits from ``str`` so values compare equal to plain strings
    (e.g. ``BillStatus.PASSED == "passed"``).  Compatible with Python 3.9+.
    """

    PASSED = "passed"
    VETOED = "vetoed"
    STUCK = "stuck"
    IN_PROGRESS = "in_progress"


class PipelineStage(str, Enum):
    """Fine-grained bill progression through the legislative pipeline.

    Each stage has a numeric ``depth`` that represents how far a bill has
    progressed (0 = dead on arrival, 6 = signed into law).  Used by the
    Moneyball engine to compute average pipeline depth per member.
    """

    FILED = "filed"  # depth 0: introduced but never moved
    COMMITTEE = "committee"  # depth 1: assigned/referred to committee
    COMMITTEE_PASSED = "committee_passed"  # depth 2: passed out of committee
    SECOND_READING = "second_reading"  # depth 3: reached 2nd/3rd reading
    CHAMBER_PASSED = "chamber_passed"  # depth 4: passed originating chamber
    CROSSED = "crossed"  # depth 5: passed both chambers
    SIGNED = "signed"  # depth 6: signed by Governor / Public Act

    @property
    def depth(self) -> int:
        return _PIPELINE_DEPTHS[self]


_PIPELINE_DEPTHS: dict[PipelineStage, int] = {
    PipelineStage.FILED: 0,
    PipelineStage.COMMITTEE: 1,
    PipelineStage.COMMITTEE_PASSED: 2,
    PipelineStage.SECOND_READING: 3,
    PipelineStage.CHAMBER_PASSED: 4,
    PipelineStage.CROSSED: 5,
    PipelineStage.SIGNED: 6,
}

_PIPELINE_MAX_DEPTH = max(_PIPELINE_DEPTHS.values())


_PASSED_TOKENS: tuple[str, ...] = (
    "Public Act",
    "Adopted Both Houses",
    "Resolution Adopted",
    "Appointment Confirmed",
    "Signed by Governor",
)

_STUCK_TOKENS: tuple[str, ...] = (
    "Referred to Assignments",
    "Re-referred",
    "Rule 3-9",
    "Rule 19",
)

# Tokens for fine-grained pipeline classification
_SIGNED_TOKENS: tuple[str, ...] = (
    "Public Act",
    "Signed by Governor",
    "Appointment Confirmed",
)
_CROSSED_TOKENS: tuple[str, ...] = (
    "Adopted Both Houses",
    "Passed Both Houses",
    "Sent to the Governor",
    "Resolution Adopted",
)
_CHAMBER_PASSED_TOKENS: tuple[str, ...] = (
    "Third Reading - Passed",
    "Third Reading - Short Debate - Passed",
    "Passed the Senate",
    "Passed the House",
)
_SECOND_READING_TOKENS: tuple[str, ...] = (
    "Second Reading",
    "Third Reading",
    "Placed on Calendar",
    "Short Debate",
)
_COMMITTEE_PASSED_TOKENS: tuple[str, ...] = (
    "Do Pass",
    "Reported Out",
    "Assigned to",
    "Be Adopted",
)
_COMMITTEE_TOKENS: tuple[str, ...] = (
    "Referred to",
    "Assigned to",
    "Re-referred",
)


def classify_bill_status(last_action: str) -> BillStatus:
    """Classify a bill's ``last_action`` into a high-level bucket."""
    if any(token in last_action for token in _PASSED_TOKENS):
        return BillStatus.PASSED
    if "Veto" in last_action:
        return BillStatus.VETOED
    if any(token in last_action for token in _STUCK_TOKENS):
        return BillStatus.STUCK
    return BillStatus.IN_PROGRESS


def classify_pipeline_stage(last_action: str) -> PipelineStage:
    """Classify a bill's ``last_action`` into a fine-grained pipeline stage.

    The stages are checked from *most advanced* to *least advanced* so the
    highest applicable stage wins.
    """
    if any(token in last_action for token in _SIGNED_TOKENS):
        return PipelineStage.SIGNED
    if any(token in last_action for token in _CROSSED_TOKENS):
        return PipelineStage.CROSSED
    if any(token in last_action for token in _CHAMBER_PASSED_TOKENS):
        return PipelineStage.CHAMBER_PASSED
    # "Do Pass" and "Reported Out" indicate committee passage.
    # Check this BEFORE second-reading because actions like
    # "Do Pass / Short Debate" contain both tokens.
    if "Do Pass" in last_action or "Reported Out" in last_action or "Be Adopted" in last_action:
        return PipelineStage.COMMITTEE_PASSED
    if any(token in last_action for token in _SECOND_READING_TOKENS):
        return PipelineStage.SECOND_READING
    if any(token in last_action for token in _COMMITTEE_TOKENS):
        return PipelineStage.COMMITTEE
    return PipelineStage.FILED


def pipeline_depth(last_action: str) -> int:
    """Return the numeric depth (0-6) for a bill's last action."""
    return classify_pipeline_stage(last_action).depth


# ── Bill type classification ─────────────────────────────────────────────────

_SUBSTANTIVE_PREFIXES: tuple[str, ...] = ("HB", "SB")
_RESOLUTION_PREFIXES: tuple[str, ...] = ("HR", "SR", "HJR", "SJR")


def is_substantive(bill_number: str) -> bool:
    """Return ``True`` if *bill_number* is a substantive law (HB/SB).

    Resolutions (HR, SR, HJR, SJR) and any unrecognised prefixes return
    ``False``.
    """
    upper = bill_number.upper()
    # Check SB/HB but avoid matching SJR/HJR by checking substantive first
    # then ensuring it's not a resolution prefix.
    if any(upper.startswith(p) for p in _RESOLUTION_PREFIXES):
        return False
    return any(upper.startswith(p) for p in _SUBSTANTIVE_PREFIXES)


def is_shell_bill(bill: Bill) -> bool:
    """Return ``True`` if *bill* is a procedural shell / placeholder.

    Shell bills are filed as legislative vehicles and were never intended to
    pass on their own merits.  They should be excluded from the effectiveness
    denominator so they don't penalise leadership who file them.

    Detection uses keyword matching on the abbreviated ILGA description:

    - Description contains "Technical" or "Shell" (case-insensitive), **or**
    - Description ends with "-TECH" (ILGA abbreviation for technical bills,
      e.g. "LOCAL GOVERNMENT-TECH", "TRANSPORTATION-TECH").

    Note: The ``bill.description`` field from ILGA index pages is always
    short (≤30 chars).  We do NOT use a length threshold — that would
    incorrectly mark every bill as a shell.
    """
    desc = (bill.description or "").strip()
    desc_upper = desc.upper()
    if "TECHNICAL" in desc_upper or "SHELL" in desc_upper:
        return True
    if desc_upper.endswith("-TECH"):
        return True
    return False


# ── Co-sponsor map ───────────────────────────────────────────────────────────


def _build_cosponsor_map(members: list[Member]) -> dict[str, list[Member]]:
    """Build a mapping of ``bill_number -> list[co-sponsor Member]``.

    A co-sponsor is any member who has the bill in their
    ``co_sponsor_bills`` (disjoint from ``sponsored_bills``).
    """
    cosponsor_map: dict[str, list[Member]] = {}
    for member in members:
        for bill in member.co_sponsor_bills:
            cosponsor_map.setdefault(bill.bill_number, []).append(member)
    return cosponsor_map


# ── Scorecard ────────────────────────────────────────────────────────────────


@dataclass
class MemberScorecard:
    """Computed legislative metrics for a single member."""

    # ── Original metrics (backward-compatible) ──
    primary_bill_count: int
    passed_count: int
    vetoed_count: int
    stuck_count: int
    in_progress_count: int
    success_rate: float  # passed / total, 0.0 when no bills
    heat_score: int  # alias for primary_bill_count
    effectiveness_score: float  # heat_score * success_rate

    # ── Phase 3: Legislative DNA metrics ──
    law_heat_score: int = 0  # count of primary HB/SB only
    law_passed_count: int = (
        0  # count of primary HB/SB that passed (so law_success_rate = this / law_heat_score)
    )
    law_success_rate: float = 0.0  # passage rate of HB/SB only: law_passed_count / law_heat_score
    magnet_score: float = 0.0  # avg co-sponsors per primary law
    bridge_score: float = 0.0  # % of primary laws with cross-party co-sponsor
    resolutions_count: int = 0  # count of HR/SR/SJR/HJR
    resolutions_passed_count: int = (
        0  # count of primary resolutions that passed (passed_count = law_passed_count + this)
    )
    resolution_pass_rate: float = 0.0  # resolutions_passed_count / resolutions_count


def compute_scorecard(member: Member) -> MemberScorecard:
    """Derive a :class:`MemberScorecard` from a member's primary bills.

    This single-member convenience function does **not** compute the
    Phase 3 network metrics (``magnet_score``, ``bridge_score``) because
    those require the full member list.  Use :func:`compute_all_scorecards`
    for the complete Legislative DNA profile.
    """
    bills: list[Bill] = member.sponsored_bills
    total = len(bills)

    counts: dict[BillStatus, int] = {s: 0 for s in BillStatus}
    for bill in bills:
        counts[classify_bill_status(bill.last_action)] += 1

    success_rate = counts[BillStatus.PASSED] / total if total > 0 else 0.0

    # Separate substantive laws from resolutions, excluding shell bills
    # from the effectiveness denominator so procedural placeholders don't
    # penalise leadership.
    laws = [b for b in bills if is_substantive(b.bill_number) and not is_shell_bill(b)]
    resolutions = [b for b in bills if not is_substantive(b.bill_number)]
    law_count = len(laws)

    law_passed = sum(1 for b in laws if classify_bill_status(b.last_action) == BillStatus.PASSED)
    resolutions_passed = sum(
        1 for b in resolutions if classify_bill_status(b.last_action) == BillStatus.PASSED
    )
    law_sr = law_passed / law_count if law_count > 0 else 0.0
    res_count = len(resolutions)
    res_pr = resolutions_passed / res_count if res_count > 0 else 0.0

    return MemberScorecard(
        primary_bill_count=total,
        passed_count=counts[BillStatus.PASSED],
        vetoed_count=counts[BillStatus.VETOED],
        stuck_count=counts[BillStatus.STUCK],
        in_progress_count=counts[BillStatus.IN_PROGRESS],
        success_rate=round(success_rate, 4),
        heat_score=total,
        effectiveness_score=round(total * success_rate, 2),
        law_heat_score=law_count,
        law_passed_count=law_passed,
        law_success_rate=round(law_sr, 4),
        magnet_score=0.0,  # requires full member list
        bridge_score=0.0,  # requires full member list
        resolutions_count=res_count,
        resolutions_passed_count=resolutions_passed,
        resolution_pass_rate=round(res_pr, 4),
    )


def compute_all_scorecards(
    members: list[Member],
    member_lookup: dict[str, Member] | None = None,
) -> dict[str, MemberScorecard]:
    """Compute full Legislative DNA scorecards for every member.

    Parameters
    ----------
    members:
        Complete list of all members (used to derive co-sponsor maps).
    member_lookup:
        Optional ``{member_id: Member}`` dict.  Built automatically from
        *members* when not supplied.

    Returns
    -------
    dict mapping ``member.id`` to its :class:`MemberScorecard`.
    """
    if member_lookup is None:
        member_lookup = {m.id: m for m in members}

    cosponsor_map = _build_cosponsor_map(members)

    scorecards: dict[str, MemberScorecard] = {}
    for member in members:
        bills: list[Bill] = member.sponsored_bills
        total = len(bills)

        # ── Status counts (all primary bills) ──
        counts: dict[BillStatus, int] = {s: 0 for s in BillStatus}
        for bill in bills:
            counts[classify_bill_status(bill.last_action)] += 1

        success_rate = counts[BillStatus.PASSED] / total if total > 0 else 0.0

        # ── Filter: laws vs resolutions (shell bills excluded from denominator) ──
        laws = [b for b in bills if is_substantive(b.bill_number) and not is_shell_bill(b)]
        resolutions = [b for b in bills if not is_substantive(b.bill_number)]
        law_count = len(laws)

        law_passed = sum(
            1 for b in laws if classify_bill_status(b.last_action) == BillStatus.PASSED
        )
        resolutions_passed = sum(
            1 for b in resolutions if classify_bill_status(b.last_action) == BillStatus.PASSED
        )
        law_sr = law_passed / law_count if law_count > 0 else 0.0

        # ── Magnet & Bridge (laws only) ──
        total_cosponsors = 0
        bridged_count = 0
        for law in laws:
            cosponsors = cosponsor_map.get(law.bill_number, [])
            total_cosponsors += len(cosponsors)

            # Bridge: does at least one co-sponsor belong to a different party?
            has_cross_party = False
            for cs in cosponsors:
                # cs is already a Member object from the cosponsor_map
                if cs.party != member.party:
                    has_cross_party = True
                    break
            if has_cross_party:
                bridged_count += 1

        magnet = total_cosponsors / law_count if law_count > 0 else 0.0
        bridge = bridged_count / law_count if law_count > 0 else 0.0
        res_count = len(resolutions)
        res_pr = resolutions_passed / res_count if res_count > 0 else 0.0

        scorecards[member.id] = MemberScorecard(
            primary_bill_count=total,
            passed_count=counts[BillStatus.PASSED],
            vetoed_count=counts[BillStatus.VETOED],
            stuck_count=counts[BillStatus.STUCK],
            in_progress_count=counts[BillStatus.IN_PROGRESS],
            success_rate=round(success_rate, 4),
            heat_score=total,
            effectiveness_score=round(total * success_rate, 2),
            law_heat_score=law_count,
            law_passed_count=law_passed,
            law_success_rate=round(law_sr, 4),
            magnet_score=round(magnet, 2),
            bridge_score=round(bridge, 4),
            resolutions_count=res_count,
            resolutions_passed_count=resolutions_passed,
            resolution_pass_rate=round(res_pr, 4),
        )

    return scorecards


# ── Influence Layer: Witness-Slip Analytics ───────────────────────────────────


def lobbyist_alignment(
    slips: list[WitnessSlip],
    member: Member,
) -> dict[str, int]:
    """Count which organisations consistently file as proponents on a member's bills.

    Parameters
    ----------
    slips:
        Full list of witness slips (may span many bills).
    member:
        The legislator whose sponsored bills we want to analyse.

    Returns
    -------
    dict mapping ``organization -> count`` of proponent slips filed on the
    member's sponsored bills, sorted descending by count.  Organisations with
    empty names are excluded.
    """
    sponsored_bills: set[str] = set()
    for bill in member.sponsored_bills:
        sponsored_bills.add(bill.bill_number)
    # Also match the normalised form (e.g. "HB1075" vs "HB100")
    # The bill_number on WitnessSlip comes from the export and may lack
    # leading zeros, so normalise both sides for comparison.
    sponsored_normalised: set[str] = set()
    for bn in sponsored_bills:
        sponsored_normalised.add(_normalise_bill_number(bn))

    org_counts: dict[str, int] = {}
    for slip in slips:
        if slip.position != "Proponent":
            continue
        if not slip.organization:
            continue
        slip_bn = _normalise_bill_number(slip.bill_number)
        if slip_bn in sponsored_normalised:
            org_counts[slip.organization] = org_counts.get(slip.organization, 0) + 1

    # Sort descending by count
    return dict(sorted(org_counts.items(), key=lambda item: item[1], reverse=True))


def controversial_score(
    slips: list[WitnessSlip],
    bill_number: str,
) -> float:
    """Compute a controversy ratio for a bill based on witness-slip opposition.

    Formula: ``total_opponents / (total_proponents + total_opponents)``

    A score near 1.0 means almost all slips are opponents ("Hot Button").
    Returns 0.0 when there are no proponents or opponents.

    Parameters
    ----------
    slips:
        Full list of witness slips (may span many bills).
    bill_number:
        The bill to compute the score for (e.g. ``"HB1075"``).
    """
    target = _normalise_bill_number(bill_number)
    proponents = 0
    opponents = 0
    for slip in slips:
        if _normalise_bill_number(slip.bill_number) != target:
            continue
        if slip.position == "Proponent":
            proponents += 1
        elif slip.position == "Opponent":
            opponents += 1

    total = proponents + opponents
    if total == 0:
        return 0.0
    return round(opponents / total, 4)


def _normalise_bill_number(bn: str) -> str:
    """Strip leading zeros from the numeric portion of a bill number.

    ``"HB0100"`` and ``"HB100"`` both normalise to ``"HB100"``.
    """
    m = re.match(r"([A-Za-z]+)0*(\d+)", bn)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    return bn.upper()


# ── Advanced Analytics: Slip Volume vs. Advancement ───────────────────────────


def compute_advancement_analytics(
    bills: list[Bill],
    witness_slips: list[WitnessSlip],
    volume_percentile_threshold: float = 0.9,
) -> dict[str, list[str]]:
    """
    Analyze bills by witness slip volume and advancement status.

    Categorizes bills into 'high_volume_stalled' and 'high_volume_passed',
    based on slip volume, controversy score, and pipeline depth.

    'High volume' is defined by the ``volume_percentile_threshold`` (e.g., 0.9
    means bills in the top 10% of slip volume).
    'Stalled' bills are those not classified as PASSED or VETOED.
    'Passed' bills are those classified as PASSED.
    """
    # Group slips by bill number for easier lookup & normalization
    slips_by_bill: dict[str, list[WitnessSlip]] = {}
    for slip in witness_slips:
        normalized_bn = _normalise_bill_number(slip.bill_number)
        slips_by_bill.setdefault(normalized_bn, []).append(slip)

    bill_metrics: list[dict] = []
    for bill in bills:
        normalized_bill_number = _normalise_bill_number(bill.bill_number)
        bill_slips = slips_by_bill.get(normalized_bill_number, [])
        volume = len(bill_slips)

        # Focus on bills with at least one slip for these analytics
        if volume == 0:
            continue

        # Use original bill_number for controversial_score, if it expects it
        # Our internal normalisation is primarily for matching unique bills.
        controversy = controversial_score(witness_slips, bill.bill_number)
        depth = pipeline_depth(bill.last_action)
        status = classify_bill_status(bill.last_action)

        bill_metrics.append(
            {
                "bill_number": bill.bill_number,
                "normalized_bill_number": normalized_bill_number,
                "volume": volume,
                "controversy_score": controversy,
                "pipeline_depth": depth,
                "status": status,
            }
        )

    if not bill_metrics:
        return {"high_volume_stalled": [], "high_volume_passed": []}

    # Determine high volume threshold based on percentile
    sorted_volumes = sorted([m["volume"] for m in bill_metrics], reverse=True)
    if not sorted_volumes:  # Should not happen if bill_metrics is not empty, but for safety
        return {"high_volume_stalled": [], "high_volume_passed": []}

    # Calculate the index for the percentile. E.g., 0.9 percentile means top 10%.
    # If N=100, index = 90, meaning the 91st item (0-indexed) is the threshold value.
    # We need the minimum volume that qualifies for 'high volume'.
    volume_index_for_percentile = int(len(sorted_volumes) * volume_percentile_threshold)

    # Ensure index is within bounds. If threshold is 0.9 and len is 5, index is 4.
    # This picks the 5th element (which then implies top 20%).
    # If N=10 and threshold=0.9, index=9, picks 10th element (top 10%)
    # Make sure index doesn't exceed the last element.
    volume_threshold = sorted_volumes[min(volume_index_for_percentile, len(sorted_volumes) - 1)]

    # Ensure threshold is at least 1 if there are any volumes > 0
    # If the highest volume is 1, and N=5, index=4, threshold=1. If N=100, index=90, threshold=1.
    # This logic captures bills WITH at least one slip.
    if (
        volume_threshold == 0
    ):  # If top volumes are all 0 (shouldn't happen if volume > 0 filter applied)
        volume_threshold = 1  # Ensure at least 1 slip counts as 'some' volume

    # Categorize bills
    high_volume_stalled_bills = []
    high_volume_passed_bills = []

    for metrics in bill_metrics:
        # Check if bill volume meets or exceeds the calculated threshold
        if metrics["volume"] >= volume_threshold:
            if metrics["status"] == BillStatus.PASSED:
                high_volume_passed_bills.append(metrics["bill_number"])
            else:  # IN_PROGRESS, STUCK, VETOED, etc. are considered 'stalled' in this context
                high_volume_stalled_bills.append(metrics["bill_number"])

    return {
        "high_volume_stalled": high_volume_stalled_bills,
        "high_volume_passed": high_volume_passed_bills,
    }


# ── Committee-level analytics ─────────────────────────────────────────────────


def _normalise_committee_bill_number(bn: str) -> str:
    """Normalise a bill number from a committee bills page to match cache format.

    Committee pages may list bills without leading zeros (``SB79`` instead of
    ``SB0079``) and sometimes append amendment suffixes like ``(SCA1)``.  The
    bill cache always stores 4-digit zero-padded numbers (``SB0079``).

    This function strips the suffix and zero-pads the numeric portion to 4
    digits so lookups against the bill cache succeed.
    """
    # Strip amendment / SCA suffixes: "SB228 (SCA1)" -> "SB228"
    cleaned = re.sub(r"\s*\(.*\)\s*$", "", bn).strip()
    m = re.match(r"^([A-Za-z]+)(\d+)$", cleaned)
    if m:
        prefix, num = m.group(1), m.group(2)
        return f"{prefix.upper()}{num.zfill(4)}"
    return cleaned.upper()


@dataclass
class CommitteeStats:
    """Aggregate bill-outcome metrics for a single committee."""

    total_bills: int
    advanced_count: int  # bills that passed out of committee stage
    passed_count: int  # bills that became law (Public Act / signed)
    advancement_rate: float  # advanced_count / total_bills
    pass_rate: float  # passed_count / total_bills


def _build_committee_name_index(
    committees: list[Committee],
) -> dict[str, str]:
    """Build a fuzzy committee-name -> code lookup for action_history matching.

    Action history entries use full names like ``"Executive Committee"`` or
    ``"Judiciary - Civil Committee"`` while our committee data has short names
    like ``"Executive"`` or ``"Judiciary - Civil"``.  This builds a lookup
    that handles both forms plus several abbreviation variants.
    """
    index: dict[str, str] = {}
    for c in committees:
        lower = c.name.lower().strip()
        index[lower] = c.code
        # "Executive" -> also match "Executive Committee"
        index[lower + " committee"] = c.code
        # "Approp- Health and Human" -> also match "Appropriations- Health..."
        if lower.startswith("approp-"):
            expanded = "appropriations-" + lower[len("approp-") :]
            index[expanded] = c.code
            index[expanded + " committee"] = c.code
            # Some action_history uses "Appropriations- " (with space)
            expanded_sp = "appropriations- " + lower[len("approp- ") :].lstrip()
            index[expanded_sp] = c.code
            index[expanded_sp + " committee"] = c.code
    return index


_ASSIGNED_RE = re.compile(r"(?:Assigned to|Referred to)(.+)", re.IGNORECASE)


def _build_full_committee_bills(
    committees: list[Committee],
    committee_bills: dict[str, list[str]],
    all_bills: list[Bill],
) -> dict[str, set[str]]:
    """Merge committee bills from ILGA page data AND bill action_history.

    The ILGA committee bills page only shows bills **currently pending** in
    each committee.  Bills that already passed through are no longer listed.
    To get accurate historical throughput, we also scan every bill's
    ``action_history`` for "Assigned to"/"Referred to" actions and match the
    committee name back to a committee code.

    Returns a mapping of committee code -> set of bill_numbers (normalised).
    """
    name_index = _build_committee_name_index(committees)

    # Seed with the ILGA page data (normalised)
    full: dict[str, set[str]] = {}
    for code, raw_bns in committee_bills.items():
        seen: set[str] = set()
        for raw_bn in raw_bns:
            seen.add(_normalise_committee_bill_number(raw_bn))
        full[code] = seen

    # Augment with action_history scanning
    for bill in all_bills:
        for entry in bill.action_history:
            m = _ASSIGNED_RE.match(entry.action)
            if not m:
                continue
            cname = m.group(1).strip().lower()
            # Skip generic routing (Assignments, Rules Committee)
            if cname in ("assignments", "rules committee"):
                continue
            code = name_index.get(cname)
            if code is not None:
                full.setdefault(code, set()).add(bill.bill_number)

    return full


def compute_committee_stats(
    committees: list[Committee],
    committee_bills: dict[str, list[str]],
    bill_lookup: dict[str, Bill],
) -> dict[str, CommitteeStats]:
    """Compute per-committee bill advancement and passage statistics.

    Merges two data sources for comprehensive coverage:

    1. **ILGA committee bills page** — lists bills currently pending in each
       committee (from ``committee_bills``).
    2. **Bill action_history** — "Assigned to"/"Referred to" actions across
       all bills reveal historical committee assignments, including bills
       that already passed through and are no longer listed on the page.

    For each committee's full bill set, we classify pipeline stage to
    determine:

    - **advanced_count**: bills that progressed past the committee stage
      (pipeline depth >= 2, i.e. COMMITTEE_PASSED or beyond).
    - **passed_count**: bills that became law (status == PASSED).

    Parameters
    ----------
    committees:
        Full list of ``Committee`` objects.
    committee_bills:
        Mapping of committee code -> list of bill numbers (raw from scrape).
    bill_lookup:
        Mapping of bill leg_id -> ``Bill``.

    Returns
    -------
    dict mapping committee code to its :class:`CommitteeStats`.
    """
    # Build a bill_number -> Bill index for fast lookup
    bn_lookup: dict[str, Bill] = {}
    for bill in bill_lookup.values():
        bn_lookup[bill.bill_number] = bill

    # Merge ILGA page data + action_history for full coverage
    full_bills = _build_full_committee_bills(
        committees,
        committee_bills,
        list(bill_lookup.values()),
    )

    stats: dict[str, CommitteeStats] = {}

    for committee in committees:
        bill_numbers = full_bills.get(committee.code, set())
        total = 0
        advanced = 0
        passed = 0

        for bn in bill_numbers:
            bill = bn_lookup.get(bn)
            if bill is None:
                continue
            total += 1
            stage = classify_pipeline_stage(bill.last_action)
            if stage.depth >= PipelineStage.COMMITTEE_PASSED.depth:
                advanced += 1
            status = classify_bill_status(bill.last_action)
            if status == BillStatus.PASSED:
                passed += 1

        stats[committee.code] = CommitteeStats(
            total_bills=total,
            advanced_count=advanced,
            passed_count=passed,
            advancement_rate=round(advanced / total, 4) if total > 0 else 0.0,
            pass_rate=round(passed / total, 4) if total > 0 else 0.0,
        )

    return stats


def build_member_committee_roles(
    committees: list[Committee],
    committee_rosters: dict[str, list[CommitteeMemberRole]],
    committee_stats: dict[str, CommitteeStats],
) -> dict[str, list[dict]]:
    """Build a reverse index: member_id -> list of committee assignment dicts.

    Each dict contains the committee code, name, role, leadership flag, and
    bill advancement statistics — ready for template rendering.

    Roles are sorted with leadership positions first:
    Chair > Vice-Chair > Minority Spokesperson > Member.
    """
    _ROLE_SORT_ORDER = {
        "chair": 0,
        "vice-chair": 1,
        "minority spokesperson": 2,
        "member": 3,
    }

    member_roles: dict[str, list[dict]] = {}
    committee_name_lookup = {c.code: c.name for c in committees}

    for code, roster in committee_rosters.items():
        cname = committee_name_lookup.get(code, code)
        cstats = committee_stats.get(code)

        for cmr in roster:
            role_lower = cmr.role.lower()
            is_leadership = "chair" in role_lower or "spokesperson" in role_lower

            entry = {
                "code": code,
                "name": cname,
                "role": cmr.role,
                "is_leadership": is_leadership,
                "total_bills": cstats.total_bills if cstats else 0,
                "advanced_count": cstats.advanced_count if cstats else 0,
                "advancement_rate_pct": (round(cstats.advancement_rate * 100) if cstats else 0),
                "passed_count": cstats.passed_count if cstats else 0,
                "pass_rate_pct": (round(cstats.pass_rate * 100) if cstats else 0),
            }

            member_roles.setdefault(cmr.member_id, []).append(entry)

    # Sort each member's committees: leadership first, then alphabetical
    for mid, roles in member_roles.items():
        roles.sort(
            key=lambda r: (
                _ROLE_SORT_ORDER.get(r["role"].lower(), 99),
                r["name"],
            )
        )

    return member_roles
