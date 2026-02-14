"""
ILGA Rules Engine — machine-readable glossary loader and rule-based helpers.

Loads ``reference/ilga_rules.json`` once (cached) and exposes functions used by
the action classifier, feature engineering, lifecycle logic, and UI layer.

All definitions are grounded in the 104th GA rules (Senate SR-4 + House Rules).
The glossary is bicameral — most entries carry both ``senate_rule`` and
``house_rule`` fields.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Glossary path — relative to repo root
# ---------------------------------------------------------------------------
_RULES_PATH = Path(__file__).resolve().parents[3] / "reference" / "ilga_rules.json"


# ---------------------------------------------------------------------------
# Loader (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_rules() -> dict:
    """Load and cache ``reference/ilga_rules.json``.  Returns the full dict."""
    if not _RULES_PATH.exists():
        raise FileNotFoundError(
            f"ILGA rules glossary not found at {_RULES_PATH}. "
            "Ensure reference/ilga_rules.json exists."
        )
    with open(_RULES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Committee classification
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_procedural_committees() -> frozenset[str]:
    """Return lowercase names of procedural / service committees.

    Replaces the hardcoded ``_PROCEDURAL_COMMITTEES`` set in ``features.py``.
    Source: Rule 3-4, Rule 3-5.
    """
    rules = load_rules()
    names: set[str] = set()
    for entry in rules["committees"]["procedural"]:
        names.add(entry["name"].lower())
        for alias in entry.get("aliases", []):
            names.add(alias.lower())
    return frozenset(names)


def is_procedural_committee(name: str) -> bool:
    """Return True if *name* is a procedural / service committee."""
    return name.strip().lower() in get_procedural_committees()


# ---------------------------------------------------------------------------
# Favorable / unfavorable report detection  (Rule 3-11)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _favorable_patterns() -> tuple[str, ...]:
    rules = load_rules()
    return tuple(p.lower() for p in rules["outcomes"]["favorable_report"]["patterns"])


@lru_cache(maxsize=1)
def _unfavorable_patterns() -> tuple[str, ...]:
    rules = load_rules()
    return tuple(p.lower() for p in rules["outcomes"]["unfavorable_report"]["patterns"])


def is_favorable_report(action_text: str) -> bool:
    """Return True if *action_text* is a Rule 3-11 favorable committee report.

    Matches: do pass, do pass as amended, be adopted, be adopted as amended,
    recommend do adopt, approved for consideration.
    """
    tl = action_text.strip().lower()
    return any(tl.startswith(p) for p in _favorable_patterns())


def is_unfavorable_report(action_text: str) -> bool:
    """Return True if *action_text* is a Rule 3-11 unfavorable committee report."""
    tl = action_text.strip().lower()
    return any(tl.startswith(p) for p in _unfavorable_patterns())


# ---------------------------------------------------------------------------
# Re-referral detection  (Rule 3-9, House Rule 19)
# ---------------------------------------------------------------------------
_RE_REFERRAL_PATTERNS: tuple[str, ...] = (
    "rule 19(a)",
    "rule 19(b)",
    "rule 3-9(a)",
    "rule 3-9(b)",
    "pursuant to senate rule 3-9",
    "re-referred to rules committee",
    "re-referred to assignments",
)


def is_re_referral_to_assignments(action_text: str) -> bool:
    """Return True if *action_text* is a deadline-triggered re-referral.

    These are **NOT** terminal — the bill goes back to Assignments / Rules
    and *can* be re-referred to a standing committee (Rule 3-9).
    """
    tl = action_text.strip().lower()
    return any(tl.startswith(p) for p in _RE_REFERRAL_PATTERNS)


def is_missed_committee_deadline(action_text: str) -> bool:
    """Rule 19(a) or Rule 3-9(a) — missed committee report deadline."""
    tl = action_text.strip().lower()
    return tl.startswith("rule 19(a)") or tl.startswith("rule 3-9(a)")


def is_missed_floor_deadline(action_text: str) -> bool:
    """Rule 19(b) — passed committee but missed floor vote deadline."""
    tl = action_text.strip().lower()
    return tl.startswith("rule 19(b)")


# ---------------------------------------------------------------------------
# Tabling detection  (Rule 7-10)
# ---------------------------------------------------------------------------
_TABLING_RE = re.compile(r"^(tabled|motion to table|laid on the table)", re.IGNORECASE)


def is_tabled(action_text: str) -> bool:
    """Return True if *action_text* is a tabling action (Rule 7-10).

    Tabling in IL is NOT terminal — bills can be taken from the table
    with a majority or three-fifths vote (Rule 7-11).
    """
    return bool(_TABLING_RE.match(action_text.strip()))


# ---------------------------------------------------------------------------
# Consent calendar detection  (Rule 6-1)
# ---------------------------------------------------------------------------
_CONSENT_PATTERNS: tuple[str, ...] = (
    "consent calendar",
    "resolutions consent calendar",
    "congratulatory consent calendar",
    "agreed resolutions",
)


def is_on_consent_calendar(action_text: str) -> bool:
    """Return True if *action_text* indicates placement on a consent calendar."""
    tl = action_text.strip().lower()
    return any(p in tl for p in _CONSENT_PATTERNS)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
def get_stage_definition(stage: str) -> dict | None:
    """Return the glossary entry for a pipeline *stage* (e.g. ``IN_COMMITTEE``).

    Returns ``None`` if the stage is not in the glossary.
    """
    rules = load_rules()
    return rules["stages"].get(stage)


def get_valid_next_stages(stage: str) -> list[str]:
    """Return the list of valid next stages from the glossary for *stage*."""
    defn = get_stage_definition(stage)
    if defn is None:
        return []
    return defn.get("next_stages", [])


def get_stage_rule(stage: str, chamber: str | None = None) -> str:
    """Return the rule citation string for a pipeline *stage*.

    If *chamber* is ``"senate"`` or ``"house"``, returns the chamber-specific
    rule.  Otherwise returns both joined with ``"; "``.
    """
    defn = get_stage_definition(stage)
    if defn is None:
        return ""
    # v2 glossary uses senate_rule / house_rule; v1 used rule
    if chamber == "senate":
        return defn.get("senate_rule", defn.get("rule", ""))
    if chamber == "house":
        return defn.get("house_rule", defn.get("rule", ""))
    sr = defn.get("senate_rule", "")
    hr = defn.get("house_rule", "")
    if sr and hr:
        return f"{sr}; {hr}"
    return sr or hr or defn.get("rule", "")


# ---------------------------------------------------------------------------
# Vote thresholds
# ---------------------------------------------------------------------------
def get_vote_threshold(threshold_key: str) -> dict | None:
    """Return the vote threshold definition from the glossary.

    *threshold_key*: one of ``simple_majority``, ``majority_of_those_elected``,
    ``three_fifths_of_those_elected``, ``two_thirds_of_those_elected``,
    ``majority_of_those_appointed``.
    """
    rules = load_rules()
    return rules["vote_thresholds"].get(threshold_key)


def votes_required_for_passage(chamber: str = "senate") -> int:
    """Return the number of votes required for bill passage.

    Senate Rule 5-1(f): majority of those elected = 30.
    House Rule 37(f): majority of those elected = 60.
    """
    vt = get_vote_threshold("majority_of_those_elected")
    if vt:
        key = (
            f"{chamber}_votes_required"
            if chamber in ("senate", "house")
            else "senate_votes_required"
        )
        return vt.get(key, vt.get("votes_required", 30))
    return 30 if chamber == "senate" else 60


def votes_required_for_override(chamber: str = "senate") -> int:
    """Return the number of votes required for a veto override.

    Senate Rule 9-6: three-fifths of those elected = 36.
    House Article IX: three-fifths of those elected = 71.
    """
    vt = get_vote_threshold("three_fifths_of_those_elected")
    if vt:
        key = (
            f"{chamber}_votes_required"
            if chamber in ("senate", "house")
            else "senate_votes_required"
        )
        return vt.get(key, vt.get("votes_required", 36))
    return 36 if chamber == "senate" else 71


def votes_required_for_discharge(chamber: str = "senate") -> int:
    """Return the number of votes required to discharge a committee.

    Senate Rule 7-9: three-fifths of those elected = 36.
    House Rule 58: 60 members elected.
    """
    if chamber == "house":
        return 60
    return 36


def chamber_member_count(chamber: str = "senate") -> int:
    """Return total members elected for the chamber.

    Senate: 59 members.  House: 118 members.
    """
    return 59 if chamber == "senate" else 118


# ---------------------------------------------------------------------------
# Rule tooltips for UI
# ---------------------------------------------------------------------------
_TOOLTIP_MAP: dict[str, str] = {
    # Stages (bicameral)
    "FILED": "Senate Rule 5-1(d)/House Rule 37(d): Bill introduced, referred to Assignments/Rules.",
    "IN_COMMITTEE": "Senate Rule 3-8/3-11; House Rule 18/22: Bill assigned to standing committee.",
    "PASSED_COMMITTEE": (
        "Senate Rule 3-12(a)/House Rule 24(a): Favorably reported; on Second Reading."
    ),
    "FLOOR_VOTE": "Senate Rule 5-2; House Rule 38: Third Reading — final floor vote.",
    "CROSSED_CHAMBERS": "Bill passed origin chamber, received by second chamber.",
    "PASSED_BOTH": "Senate Rule 8-1/House Rule 72: Passed both chambers; sent to Governor.",
    "GOVERNOR": "Article IX: Governor may sign, veto, or amend.",
    "SIGNED": "Article IX: Governor signed — bill is law.",
    "VETOED": "Article IX: Governor vetoed. Override requires 3/5 (Senate 36, House 71).",
    # Common actions (bicameral)
    "rule 19(a)": (
        "House Rule 19(a): Missed committee deadline; re-referred to Rules. NOT terminal."
    ),
    "rule 19(b)": "House Rule 19(b): Missed floor deadline; re-referred to Rules. NOT terminal.",
    "rule 3-9(a)": "Senate Rule 3-9(a): Missed deadline; re-referred to Assignments. NOT terminal.",
    "rule 3-9(b)": "Senate Rule 3-9(b): 31 days without session; re-referred to Assignments.",
    "do pass": "Senate Rule 3-11/House Rule 22: Committee recommends passage (favorable report).",
    "do not pass": (
        "Senate Rule 3-12(a)/House Rule 24(a): Committee recommends against; lies on table."
    ),
    "tabled": "Senate Rule 7-10/House Rule 60: Bill tabled. Can be taken from table.",
    "consent calendar": "Senate Rule 6-1(c)/House Rule 42: Non-controversial; no debate permitted.",
}


def get_rule_tooltip(key: str) -> str | None:
    """Return a short rule-citation tooltip for a stage or action keyword.

    *key* can be a stage name (e.g. ``IN_COMMITTEE``) or a lowercase action
    keyword (e.g. ``rule 19(a)``, ``do pass``).  Returns ``None`` if no
    tooltip is defined.
    """
    # Try exact match first, then lowercase prefix match
    tip = _TOOLTIP_MAP.get(key)
    if tip:
        return tip
    kl = key.strip().lower()
    for pattern, tooltip in _TOOLTIP_MAP.items():
        if kl.startswith(pattern):
            return tooltip
    return None


def get_action_rule_tooltip(action_text: str) -> str | None:
    """Return a rule tooltip for a raw ILGA action text string.

    Tries common patterns in priority order.
    """
    tl = action_text.strip().lower()
    # Deadline re-referrals
    if tl.startswith("rule 19(a)"):
        return _TOOLTIP_MAP["rule 19(a)"]
    if tl.startswith("rule 19(b)"):
        return _TOOLTIP_MAP["rule 19(b)"]
    if "rule 3-9(a)" in tl:
        return _TOOLTIP_MAP["rule 3-9(a)"]
    if "rule 3-9(b)" in tl:
        return _TOOLTIP_MAP["rule 3-9(b)"]
    # Committee reports
    if is_favorable_report(action_text):
        return _TOOLTIP_MAP["do pass"]
    if is_unfavorable_report(action_text):
        return _TOOLTIP_MAP["do not pass"]
    # Tabling
    if is_tabled(action_text):
        return _TOOLTIP_MAP["tabled"]
    # Consent calendar
    if is_on_consent_calendar(action_text):
        return _TOOLTIP_MAP["consent calendar"]
    return None
