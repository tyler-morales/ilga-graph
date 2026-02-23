"""Center for Effective Lawmaking (CEL) — Legislative Effectiveness Score.

Implements the CEL methodology adapted for the Illinois General Assembly:

* Three bill categories: Commemorative (C, α=1), Substantive (S, β=5),
  Substantive & Significant (SS, γ=10).
* Five cumulative stages: BILL → AIC → ABC → PASS → LAW.
* LES formula: chamber-relative weighted stage fractions, normalized so the
  chamber average = 1.
* Benchmark OLS regression: predicted LES based on seniority, majority-party
  status, and committee-chair status.
* Expectations: Above (LES/Benchmark > 1.5), Meets (0.5–1.5), Below (< 0.5).

References
----------
Bucchianeri, Peter, Craig Volden, and Alan E. Wiseman.
"Legislative Effectiveness in the American States."
*American Political Science Review* 119(1): 21–39.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .analytics import classify_pipeline_stage, is_shell_bill, is_substantive
from .models import Bill, Member, WitnessSlip

# ── Bill category ─────────────────────────────────────────────────────────────

# Weight multipliers per category (CEL: α=1, β=5, γ=10)
_CATEGORY_WEIGHTS: dict[str, int] = {
    "C": 1,
    "S": 5,
    "SS": 10,
}

# Keywords indicating a commemorative bill (case-insensitive match on description/synopsis)
_COMMEMORATIVE_KEYWORDS: tuple[str, ...] = (
    "designat",  # "designate", "designation"
    "memorial",
    "honor",
    "honour",
    "recogniz",  # "recognize", "recognition"
    "congratulat",
    "commemo",  # "commemorate", "commemorating"
    "private relief",
    "renaming",
    "rename",
    "tribute",
    "celebrat",  # "celebrate", "celebrating"
    "proclaim",
)

# Thresholds for Substantive & Significant (proxy definitions for Illinois)
_SS_MIN_COSPONSORS: int = 5  # >= 5 co-sponsors on the bill
_SS_MIN_WITNESS_SLIPS: int = 10  # >= 10 witness slips on the bill


class BillCategory(str, Enum):
    """Three-tier bill classification for CEL-style scoring.

    * **C** (Commemorative, α=1): renaming, memorial, honorary bills.
    * **S** (Substantive, β=5):   default for HB/SB that are not C or SS.
    * **SS** (Substantive & Significant, γ=10): major bills proxied by
      high co-sponsor count or high witness-slip volume.
    """

    COMMEMORATIVE = "C"
    SUBSTANTIVE = "S"
    SIGNIFICANT = "SS"

    @property
    def weight(self) -> int:
        return _CATEGORY_WEIGHTS[self.value]


def _is_commemorative(bill: Bill) -> bool:
    """Return True if a bill appears to be purely commemorative.

    Uses keyword matching on ``description`` and ``synopsis``.
    """
    text = f"{bill.description} {bill.synopsis}".lower()
    return any(kw in text for kw in _COMMEMORATIVE_KEYWORDS)


def classify_bill_category(
    bill: Bill,
    cosponsor_count: int = 0,
    witness_slip_count: int = 0,
) -> BillCategory:
    """Classify *bill* into C / S / SS.

    Parameters
    ----------
    bill:
        The bill to classify.
    cosponsor_count:
        Number of co-sponsors for this bill (from the full member list).
    witness_slip_count:
        Number of witness slips filed for this bill.

    Returns
    -------
    :class:`BillCategory`
    """
    if _is_commemorative(bill):
        return BillCategory.COMMEMORATIVE

    if cosponsor_count >= _SS_MIN_COSPONSORS or witness_slip_count >= _SS_MIN_WITNESS_SLIPS:
        return BillCategory.SIGNIFICANT

    return BillCategory.SUBSTANTIVE


# ── CEL stages ────────────────────────────────────────────────────────────────


class CELStage(str, Enum):
    """Five cumulative CEL stages.

    A bill that reached a later stage counts at *all* earlier stages too
    (i.e. the stages are thresholds, not mutually exclusive buckets).

    Mapping from existing ILGA pipeline depths (0–6):

    * BILL  — depth >= 0  (introduced/filed)
    * AIC   — depth >= 1  (any action in committee)
    * ABC   — depth >= 3  (action beyond committee, i.e. second reading+)
    * PASS  — depth >= 4  (passed chamber of origin)
    * LAW   — depth >= 6  (signed by Governor / Public Act)
    """

    BILL = "BILL"
    AIC = "AIC"
    ABC = "ABC"
    PASS = "PASS"
    LAW = "LAW"


# Minimum pipeline depth required to "count" at each CEL stage
_CEL_STAGE_MIN_DEPTH: dict[CELStage, int] = {
    CELStage.BILL: 0,
    CELStage.AIC: 1,
    CELStage.ABC: 3,
    CELStage.PASS: 4,
    CELStage.LAW: 6,
}

_ALL_CEL_STAGES: tuple[CELStage, ...] = (
    CELStage.BILL,
    CELStage.AIC,
    CELStage.ABC,
    CELStage.PASS,
    CELStage.LAW,
)


def bill_reaches_cel_stage(bill: Bill, stage: CELStage) -> bool:
    """Return True if *bill* has reached (or surpassed) *stage*."""
    depth = classify_pipeline_stage(bill.last_action).depth
    return depth >= _CEL_STAGE_MIN_DEPTH[stage]


# ── LES computation ───────────────────────────────────────────────────────────


def _normalise_bill_number(bn: str) -> str:
    """Strip leading zeros from the numeric portion of a bill number."""
    m = re.match(r"([A-Za-z]+)0*(\d+)", bn)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    return bn.upper()


@dataclass
class MemberLESResult:
    """Per-member CEL Legislative Effectiveness Score and benchmark."""

    member_id: str
    les: float  # Normalized LES (chamber avg = 1)
    les_benchmark: float  # OLS-predicted LES from seniority/majority/chair
    les_expectation: str  # "Above", "Meets", or "Below"
    # Raw stage weights (for transparency / debugging)
    stage_weights: dict[str, float]  # stage name -> weighted bill count


def _bill_cosponsor_counts(members: list[Member]) -> dict[str, int]:
    """Build a bill_number -> co-sponsor count map."""
    counts: dict[str, int] = {}
    for member in members:
        for bill in member.co_sponsor_bills:
            key = _normalise_bill_number(bill.bill_number)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _bill_witness_slip_counts(
    witness_slips: list[WitnessSlip],
) -> dict[str, int]:
    """Build a bill_number -> witness slip count map."""
    counts: dict[str, int] = {}
    for slip in witness_slips:
        key = _normalise_bill_number(slip.bill_number)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _seniority_terms(member: Member, current_year: int = 2025) -> int:
    """Return the number of completed or current terms in the member's current chamber.

    Each 2-year block counts as one term.  Falls back to 0 if no career data.
    """
    chamber = member.chamber  # "House" or "Senate"
    years: list[int] = []
    for cr in member.career_ranges:
        if cr.chamber and cr.chamber.lower() != chamber.lower():
            continue
        years.append(cr.start_year)

    if not years:
        return 0

    first_year = min(years)
    elapsed = current_year - first_year
    # A new member in their first term counts as 1; round up partial terms.
    return max(1, (elapsed // 2) + 1)


def _is_majority_party(member: Member, chamber_majority: dict[str, str]) -> bool:
    """Return True if *member* is in the majority party for their chamber."""
    majority_party = chamber_majority.get(member.chamber, "")
    if not majority_party:
        return False
    return member.party.lower() == majority_party.lower()


def _is_committee_chair(member: Member) -> bool:
    """Return True if *member* chairs at least one committee.

    Checks ``member.roles`` (aggregated from profile + committee rosters).
    """
    roles_to_check = member.roles or ([member.role] if member.role else [])
    for role in roles_to_check:
        rl = role.lower()
        if "chair" in rl and "vice" not in rl and "caucus" not in rl:
            return True
    return False


def _derive_chamber_majority(members: list[Member]) -> dict[str, str]:
    """Derive the majority party in each chamber by simple member count."""
    counts: dict[str, dict[str, int]] = {}
    for member in members:
        ch = member.chamber
        counts.setdefault(ch, {})
        counts[ch][member.party] = counts[ch].get(member.party, 0) + 1

    majority: dict[str, str] = {}
    for chamber, party_counts in counts.items():
        if party_counts:
            majority[chamber] = max(party_counts, key=lambda p: party_counts[p])
    return majority


def _ols_regression(
    X: list[list[float]],
    y: list[float],
) -> list[float]:
    """Simple OLS regression: returns coefficient vector [intercept, *betas].

    Uses the closed-form (X^T X)^{-1} X^T y.  Adds an intercept column.
    Falls back to [mean(y)] if the system is singular or underdetermined.
    """
    n = len(y)
    if n == 0:
        return [0.0]

    # Augment X with intercept column
    Xa = [[1.0] + row for row in X]
    p = len(Xa[0])

    if n < p:
        # Underdetermined — return grand mean as benchmark
        mean_y = sum(y) / n
        return [mean_y] + [0.0] * (p - 1)

    # Compute X^T X  (p x p)
    XtX = [[0.0] * p for _ in range(p)]
    for row in Xa:
        for i in range(p):
            for j in range(p):
                XtX[i][j] += row[i] * row[j]

    # Compute X^T y  (p x 1)
    Xty = [0.0] * p
    for k, row in enumerate(Xa):
        for i in range(p):
            Xty[i] += row[i] * y[k]

    # Invert X^T X using Gaussian elimination with partial pivoting
    try:
        coeffs = _solve_linear(XtX, Xty)
    except ZeroDivisionError:
        mean_y = sum(y) / n
        return [mean_y] + [0.0] * (p - 1)

    return coeffs


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix [A | b]
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Partial pivot
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        if abs(pivot) < 1e-12:
            raise ZeroDivisionError("Singular matrix")

        for row in range(col + 1, n):
            factor = M[row][col] / pivot
            for k in range(col, n + 1):
                M[row][k] -= factor * M[col][k]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


def classify_expectations(les: float, benchmark: float) -> str:
    """Classify a member's effectiveness relative to expectations.

    Returns
    -------
    * ``"Above"``  if LES / Benchmark > 1.5
    * ``"Meets"``  if 0.5 <= LES / Benchmark <= 1.5
    * ``"Below"``  if LES / Benchmark < 0.5
    * ``""``       if benchmark is zero (undefined)
    """
    if benchmark <= 0:
        return ""
    ratio = les / benchmark
    if ratio > 1.5:
        return "Above"
    if ratio < 0.5:
        return "Below"
    return "Meets"


# ── Public API ────────────────────────────────────────────────────────────────


def compute_les_scores(
    members: list[Member],
    witness_slips: list[WitnessSlip] | None = None,
    current_year: int = 2025,
) -> dict[str, MemberLESResult]:
    """Compute CEL-style Legislative Effectiveness Scores for all members.

    Parameters
    ----------
    members:
        Full list of members (both chambers).
    witness_slips:
        Optional witness slips used to proxy Substantive & Significant bills.
    current_year:
        Reference year for computing seniority.

    Returns
    -------
    dict mapping ``member.id`` to :class:`MemberLESResult`.
    """
    if witness_slips is None:
        witness_slips = []

    cosponsor_counts = _bill_cosponsor_counts(members)
    slip_counts = _bill_witness_slip_counts(witness_slips)
    chamber_majority = _derive_chamber_majority(members)

    # ── Step 1: Per-member, per-stage weighted bill counts ──
    # stage_weights[member_id][stage] = sum of category weights for bills
    # introduced by that member that reached at least that stage.
    stage_weights: dict[str, dict[CELStage, float]] = {}

    for member in members:
        sw: dict[CELStage, float] = {s: 0.0 for s in _ALL_CEL_STAGES}

        for bill in member.sponsored_bills:
            if not is_substantive(bill.bill_number):
                continue
            if is_shell_bill(bill):
                continue

            bn = _normalise_bill_number(bill.bill_number)
            category = classify_bill_category(
                bill,
                cosponsor_count=cosponsor_counts.get(bn, 0),
                witness_slip_count=slip_counts.get(bn, 0),
            )
            weight = category.weight

            for stage in _ALL_CEL_STAGES:
                if bill_reaches_cel_stage(bill, stage):
                    sw[stage] += weight

        stage_weights[member.id] = sw

    # ── Step 2: Chamber totals per stage ──
    # Group by chamber so we compare within-chamber only
    chamber_totals: dict[str, dict[CELStage, float]] = {}
    chamber_members: dict[str, list[Member]] = {}
    for member in members:
        ch = member.chamber
        chamber_members.setdefault(ch, []).append(member)
        if ch not in chamber_totals:
            chamber_totals[ch] = {s: 0.0 for s in _ALL_CEL_STAGES}
        for stage in _ALL_CEL_STAGES:
            chamber_totals[ch][stage] += stage_weights[member.id][stage]

    # ── Step 3: Raw LES fractions ──
    raw_les: dict[str, float] = {}
    for member in members:
        ch = member.chamber
        totals = chamber_totals[ch]
        fracs: list[float] = []
        for stage in _ALL_CEL_STAGES:
            total = totals[stage]
            member_w = stage_weights[member.id][stage]
            fracs.append(member_w / total if total > 0 else 0.0)
        # Sum fractions and multiply by N/5 to normalize avg to 1
        raw_les[member.id] = sum(fracs)

    # Normalize per chamber: LES = raw * (N / 5) where N = number of members
    les_scores: dict[str, float] = {}
    for ch, ch_members in chamber_members.items():
        n = len(ch_members)
        scale = n / 5.0 if n > 0 else 1.0
        for m in ch_members:
            les_scores[m.id] = round(raw_les[m.id] * scale, 4)

    # ── Step 4: Benchmark (OLS per chamber) ──
    benchmark_scores: dict[str, float] = {}
    for ch, ch_members in chamber_members.items():
        X: list[list[float]] = []
        y_vals: list[float] = []
        for m in ch_members:
            seniority = float(_seniority_terms(m, current_year))
            majority = 1.0 if _is_majority_party(m, chamber_majority) else 0.0
            chair = 1.0 if _is_committee_chair(m) else 0.0
            X.append([seniority, majority, chair])
            y_vals.append(les_scores[m.id])

        coeffs = _ols_regression(X, y_vals)

        for m, xi in zip(ch_members, X):
            predicted = coeffs[0] + sum(coeffs[j + 1] * xi[j] for j in range(len(xi)))
            benchmark_scores[m.id] = round(max(predicted, 0.0), 4)

    # ── Step 5: Assemble results ──
    results: dict[str, MemberLESResult] = {}
    for member in members:
        les = les_scores.get(member.id, 0.0)
        benchmark = benchmark_scores.get(member.id, 0.0)
        sw_named = {s.value: stage_weights[member.id][s] for s in _ALL_CEL_STAGES}
        results[member.id] = MemberLESResult(
            member_id=member.id,
            les=les,
            les_benchmark=benchmark,
            les_expectation=classify_expectations(les, benchmark),
            stage_weights=sw_named,
        )

    return results
