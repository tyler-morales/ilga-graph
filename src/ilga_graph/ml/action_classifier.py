"""Classify raw ILGA action text into structured categories.

Uses the action_types.json reference to map raw action strings into:
- category_id (e.g., "introduction", "committee_action", "governor")
- category_label (e.g., "Committee Action", "Governor Action")
- outcome_signal (e.g., "positive", "negative_terminal")
- meaning (human-readable explanation)
- is_bill_action (True if about the bill itself, not an amendment)
- progress_stage (pipeline stage this action implies, if any)

The classifier handles ILGA's inconsistent formatting (missing spaces after
"to", vote tallies appended to action text, etc.) and properly distinguishes
between bill-level actions and amendment-level actions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_REFERENCE_PATH = Path(__file__).parent / "action_types.json"


@dataclass(frozen=True)
class ClassifiedAction:
    """Structured classification of a raw action text."""

    raw_text: str
    category_id: str  # e.g. "introduction", "governor"
    category_label: str  # e.g. "Introduction & Filing", "Governor Action"
    outcome_signal: str  # e.g. "positive", "negative_terminal", "neutral"
    meaning: str  # human-readable explanation
    is_bill_action: bool  # True if about the bill itself (not an amendment)
    progress_stage: str | None  # pipeline stage, if applicable
    rule_reference: str | None = None  # Senate Rule citation, e.g. "Rule 3-11"


# ── Amendment prefix detection ───────────────────────────────────────────────

_AMENDMENT_PREFIX_RE = re.compile(
    r"^(House|Senate)\s+(Committee|Floor)\s+Amendment\s+No\.\s+\d+\s*",
    re.IGNORECASE,
)
_CONFERENCE_PREFIX_RE = re.compile(
    r"^Conference Committee Report\s+No\.\s+\d+\s*",
    re.IGNORECASE,
)


def _strip_amendment_prefix(text: str) -> tuple[bool, str]:
    """If action starts with an amendment prefix, strip it and return (True, rest).

    Returns (False, original_text) if not an amendment action.
    """
    m = _AMENDMENT_PREFIX_RE.match(text)
    if m:
        return True, text[m.end() :].strip()
    m = _CONFERENCE_PREFIX_RE.match(text)
    if m:
        return True, text[m.end() :].strip()
    return False, text


# ── Normalize for matching ───────────────────────────────────────────────────


def _normalize_for_match(text: str) -> str:
    """Normalize action text for matching — fix ILGA quirks.

    ILGA often omits spaces: "Assigned toTransportation" instead of
    "Assigned to Transportation".  We normalize by inserting spaces
    after known keywords.
    """
    # Fix missing spaces after "to", "by", "from" before uppercase
    t = re.sub(r"(to|by|from)([A-Z])", r"\1 \2", text)
    # Fix "To[Subcommittee]" — ILGA writes "ToElections" without space
    if re.match(r"^To[A-Z]", t):
        t = "To " + t[2:]
    # Strip vote tallies (e.g., ";058-000-000" or "056-000-000")
    t = re.sub(r"[;,]?\s*\d{3}-\d{3}-\d{3}\s*$", "", t)
    return t.strip()


# ── Reference loader ─────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_reference() -> list[dict]:
    """Load the action_types.json reference file."""
    with open(_REFERENCE_PATH) as f:
        data = json.load(f)
    return data["categories"]


# ── Main classifier ──────────────────────────────────────────────────────────

_FALLBACK = ClassifiedAction(
    raw_text="",
    category_id="other",
    category_label="Other",
    outcome_signal="neutral",
    meaning="Uncategorized action.",
    is_bill_action=True,
    progress_stage=None,
    rule_reference=None,
)


def classify_action(raw_text: str) -> ClassifiedAction:
    """Classify a raw ILGA action text into a structured category.

    Returns a ClassifiedAction with category, meaning, and outcome signal.
    """
    text = raw_text.strip()
    if not text:
        return ClassifiedAction(
            raw_text=raw_text,
            category_id="other",
            category_label="Other",
            outcome_signal="neutral",
            meaning="Empty action text.",
            is_bill_action=True,
            progress_stage=None,
        )

    # Step 1: detect and strip amendment prefix
    is_amendment, base_text = _strip_amendment_prefix(text)

    # If it's an amendment action, classify as "amendment" category
    if is_amendment:
        # Try to sub-classify the base action (e.g., "Tabled", "Adopted")
        sub = _match_action(base_text)
        sub_meaning = sub.meaning if sub else "Amendment sub-action."
        sub_signal = sub.outcome_signal if sub else "neutral"
        return ClassifiedAction(
            raw_text=raw_text,
            category_id="amendment",
            category_label="Amendment Actions",
            outcome_signal=sub_signal if sub_signal != "neutral" else "neutral",
            meaning=f"[Amendment] {sub_meaning}",
            is_bill_action=False,
            progress_stage=None,
        )

    # Step 2: match against known action patterns
    match = _match_action(text)
    if match:
        return match

    # Step 3: fallback heuristics for patterns not in the reference
    tl = text.lower()

    # "Recommends Be Adopted" — committee recommendation on amendments
    if "recommends be adopted" in tl:
        return ClassifiedAction(
            raw_text=raw_text,
            category_id="committee_action",
            category_label="Committee Action",
            outcome_signal="positive",
            meaning="Committee recommends adoption.",
            is_bill_action=True,
            progress_stage="PASSED_COMMITTEE",
            rule_reference="Rule 3-11(a)",
        )

    # "Placed on Calendar Order of Executive Appointments" — appointment scheduling
    if "executive appointments" in tl and "placed on calendar" in tl:
        return ClassifiedAction(
            raw_text=raw_text,
            category_id="appointment",
            category_label="Executive Appointments",
            outcome_signal="neutral",
            meaning="Appointment scheduled on calendar for consideration.",
            is_bill_action=True,
            progress_stage=None,
            rule_reference="Rule 10-1",
        )

    # Deadline patterns: "Rule 2-10 Third Reading/Passage Deadline"
    if tl.startswith("rule 2-10"):
        return ClassifiedAction(
            raw_text=raw_text,
            category_id="deadline",
            category_label="Deadlines & Re-referrals",
            outcome_signal="neutral",
            meaning="Senate deadline established or extended.",
            is_bill_action=True,
            progress_stage=None,
            rule_reference="Rule 2-10",
        )

    # Rule-engine disambiguation: re-referral to Assignments (Rule 3-9)
    try:
        from ilga_graph.ml.rule_engine import is_re_referral_to_assignments

        if is_re_referral_to_assignments(raw_text):
            return ClassifiedAction(
                raw_text=raw_text,
                category_id="deadline",
                category_label="Deadlines & Re-referrals",
                outcome_signal="negative_weak",
                meaning="Re-referred to Assignments — missed deadline or inactivity. NOT terminal.",
                is_bill_action=True,
                progress_stage=None,
                rule_reference="Rule 3-9",
            )
    except Exception:
        pass

    # Rule-engine disambiguation: favorable report (Rule 3-11)
    try:
        from ilga_graph.ml.rule_engine import is_favorable_report as _is_fav

        if _is_fav(raw_text):
            return ClassifiedAction(
                raw_text=raw_text,
                category_id="committee_action",
                category_label="Committee Action",
                outcome_signal="positive",
                meaning="Committee favorable report (Rule 3-11).",
                is_bill_action=True,
                progress_stage="PASSED_COMMITTEE",
                rule_reference="Rule 3-11",
            )
    except Exception:
        pass

    # Catch-all for "Placed on Calendar" variants
    if tl.startswith("placed on calendar") or tl.startswith("placed calendar"):
        return ClassifiedAction(
            raw_text=raw_text,
            category_id="floor_process",
            category_label="Floor Process",
            outcome_signal="positive_weak",
            meaning="Bill placed on a calendar for consideration.",
            is_bill_action=True,
            progress_stage="FLOOR_VOTE",
            rule_reference="Rule 4-4",
        )

    return ClassifiedAction(
        raw_text=raw_text,
        category_id="other",
        category_label="Other",
        outcome_signal="neutral",
        meaning="Uncategorized action.",
        is_bill_action=True,
        progress_stage=None,
        rule_reference=None,
    )


def _match_action(text: str) -> ClassifiedAction | None:
    """Try to match action text against the reference patterns."""
    categories = _load_reference()
    normalized = _normalize_for_match(text)
    lower = normalized.lower()
    raw_lower = text.lower()

    for cat in categories:
        cat_id = cat["id"]
        cat_label = cat["label"]
        cat_stage = cat.get("progress_stage")

        for action_def in cat["actions"]:
            pattern = action_def["pattern"]
            match_type = action_def.get("match_type", "startswith")
            pl = pattern.lower()

            matched = False

            if match_type == "exact":
                matched = lower == pl or raw_lower == pl
            elif match_type == "exact_bill_only":
                # Only match if it's EXACTLY this text (e.g., "Tabled" alone)
                matched = lower == pl or raw_lower == pl
            elif match_type == "startswith":
                matched = lower.startswith(pl) or raw_lower.startswith(pl)
            elif match_type == "startswith_prefix":
                # For amendment prefixes -- these are handled above
                matched = lower.startswith(pl) or raw_lower.startswith(pl)
            elif match_type == "startswith_subcommittee":
                # "To " prefix for subcommittee/subject referrals
                # Avoid matching "Total Veto..." etc.
                # After normalization, "ToElections" becomes "To Elections"
                if lower.startswith("to ") or raw_lower.startswith("to "):
                    rest = lower[3:].strip()
                    # Only match if it's not a known false-positive prefix
                    if (
                        rest
                        and not rest.startswith("total")
                        and not rest.startswith("the ")
                        and "veto" not in rest
                    ):
                        matched = True
            elif match_type == "contains":
                matched = pl in lower or pl in raw_lower
            elif match_type == "amendment_only":
                # This only matches in the context of amendments
                # (handled above in classify_action)
                matched = False

            if matched:
                # rule_reference: prefer action-level, fall back to category-level
                rr = action_def.get("rule_reference") or cat.get("rule_reference")
                return ClassifiedAction(
                    raw_text=text,
                    category_id=cat_id,
                    category_label=cat_label,
                    outcome_signal=action_def.get("outcome_signal", "neutral"),
                    meaning=action_def["meaning"],
                    is_bill_action=True,
                    progress_stage=cat_stage,
                    rule_reference=rr,
                )

    return None


# ── Stage rollback detection ─────────────────────────────────────────────────
# Bills that passed both chambers can be re-referred (e.g. Rule 19(b)) before
# being sent to the governor. We must use current stage, not highest ever.


def _is_stage_rollback(raw_text: str) -> bool:
    """True if this action sends the bill back to committee after it had advanced.

    E.g. Rule 19(b) / Re-referred to Rules Committee after House concurrence
    (HB3356): bill passed both chambers but was re-referred and never sent to
    the governor. We treat this as rolling back to IN_COMMITTEE.
    """
    t = raw_text.strip().lower()
    if t.startswith("rule 19(a)") or t.startswith("rule 19(b)"):
        return True
    if t.startswith("rule 3-9(a)") or t.startswith("rule 3-9(b)"):
        return True
    if "re-referred to rules committee" in t or "re-referred to rules" in t:
        return True
    if "re-referred to assignments" in t:
        return True
    return False


# Stages that imply the bill has passed both chambers (and could be sent to gov).
_STAGES_AT_OR_PAST_BOTH = frozenset({"CROSSED_CHAMBERS", "PASSED_BOTH", "GOVERNOR"})


# ── Convenience: classify a full action history ──────────────────────────────


def classify_action_history(
    actions: list[str],
) -> list[ClassifiedAction]:
    """Classify an entire bill's action history.

    Returns a list of ClassifiedAction in the same order as the input.
    """
    return [classify_action(a) for a in actions]


def bill_outcome_from_actions(
    classified: list[ClassifiedAction],
) -> dict:
    """Derive bill outcome summary from classified actions.

    Actions are processed in chronological order. The returned stage is the
    *current* stage: if the bill passed both chambers but was then re-referred
    (e.g. Rule 19(b) / Re-referred to Rules Committee), current_stage is
    IN_COMMITTEE, not PASSED_BOTH. This fixes bills like HB3356 that never
    reached the governor.

    Returns a dict with:
        lifecycle_status: OPEN | PASSED | VETOED
        highest_stage: the highest pipeline stage ever reached
        current_stage: the stage as of the last action (rollbacks applied)
        terminal_action: the terminal action text (if any)
        positive_signals: count of positive outcome signals
        negative_signals: count of negative outcome signals
        has_veto: whether the bill was vetoed
        has_override_attempt: whether an override was attempted
    """
    lifecycle = "OPEN"
    highest_stage = "FILED"
    current_stage = "FILED"
    stage_order = [
        "FILED",
        "IN_COMMITTEE",
        "PASSED_COMMITTEE",
        "FLOOR_VOTE",
        "CHAMBER_PASSED",
        "CROSSED_CHAMBERS",
        "PASSED_BOTH",
        "GOVERNOR",
        "SIGNED",
    ]
    terminal_action = None
    positive = 0
    negative = 0
    has_veto = False
    has_override = False
    has_signed = False

    for ca in classified:
        if not ca.is_bill_action:
            continue

        # Track signals
        if ca.outcome_signal.startswith("positive"):
            positive += 1
        elif ca.outcome_signal.startswith("negative"):
            negative += 1

        # Rollback: re-referred after passing both chambers → current stage back to committee
        if _is_stage_rollback(ca.raw_text) and current_stage in _STAGES_AT_OR_PAST_BOTH:
            current_stage = "IN_COMMITTEE"

        # Advance highest and current stage when this action has a progress_stage
        if ca.progress_stage and ca.progress_stage in stage_order:
            idx = stage_order.index(ca.progress_stage)
            cur_idx = stage_order.index(highest_stage)
            if idx > cur_idx:
                highest_stage = ca.progress_stage
            cur_idx = stage_order.index(current_stage)
            if idx > cur_idx:
                current_stage = ca.progress_stage

        # Track terminal outcomes
        if ca.outcome_signal == "positive_terminal":
            has_signed = True
            terminal_action = ca.raw_text
        elif ca.outcome_signal == "negative_terminal":
            has_veto = True
            terminal_action = ca.raw_text

        # Override attempts
        if "override" in ca.raw_text.lower() or "veto stands" in ca.raw_text.lower():
            has_override = True

    # Determine lifecycle
    if has_signed:
        lifecycle = "PASSED"
    elif has_veto:
        lifecycle = "VETOED"

    return {
        "lifecycle_status": lifecycle,
        "highest_stage": highest_stage,
        "current_stage": current_stage,
        "terminal_action": terminal_action,
        "positive_signals": positive,
        "negative_signals": negative,
        "has_veto": has_veto,
        "has_override_attempt": has_override,
    }


# ── ETL-friendly: category string for parquet column ─────────────────────────


def action_category_for_etl(raw_text: str) -> str:
    """Return a simple category string suitable for a parquet column.

    This replaces the old _classify_action() in pipeline.py.
    """
    ca = classify_action(raw_text)
    return ca.category_id
