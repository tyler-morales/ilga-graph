"""Resolve SBE candidate committees (and donors) onto ILGA Member IDs.

Matching is imperfect by design. Accepted matches are those with a clear
office + district + name hit, or a unique name + chamber hit. Everything
else is written to the review list (unmatched).

Per Hardball Ch 3 (docs/hardball-spec/04-ch3-decision-making.md): money
is how access is obtained; we join official receipts onto sitting members
so later UI can show donor → committee → member → vote context. Lobbyist
entity join is out of scope.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import Member
from .parse import ParsedSbeData, SbeCandidateRow, SbeCommitteeRow

_SUFFIX_RE = re.compile(r",?\s+(?:Jr\.?|Sr\.?|III|II|IV)\s*$", re.IGNORECASE)
_NICKNAME_RE = re.compile(r'"([^"]+)"')
_NON_LETTERS = re.compile(r"[^a-z0-9]+")

LEGISLATIVE_OFFICES = {
    "state senator": "Senate",
    "state representative": "House",
}

CANDIDATE_COMMITTEE_TYPES = {"candidate", ""}

# Bidirectional legal-name / nickname pairs. Applied only to first-token
# overlap, never as a standalone match. Keep this short so confidence stays
# honest (no fuzzy string scores).
FIRST_NAME_ALIASES: dict[str, frozenset[str]] = {
    "bill": frozenset({"william", "will", "wm"}),
    "william": frozenset({"bill", "will", "wm"}),
    "will": frozenset({"william", "bill"}),
    "wm": frozenset({"william", "bill"}),
    "bob": frozenset({"robert"}),
    "robert": frozenset({"bob"}),
    "chris": frozenset({"christopher", "christian"}),
    "christopher": frozenset({"chris"}),
    "christian": frozenset({"chris"}),
    "dave": frozenset({"david"}),
    "david": frozenset({"dave"}),
    "liz": frozenset({"elizabeth"}),
    "lisa": frozenset({"elizabeth"}),
    "elizabeth": frozenset({"liz", "lisa"}),
    "mike": frozenset({"michael"}),
    "michael": frozenset({"mike"}),
    "steve": frozenset({"steven"}),
    "steven": frozenset({"steve"}),
    "sue": frozenset({"susan", "suzanne"}),
    "susan": frozenset({"sue"}),
    "suzanne": frozenset({"sue"}),
}

GOLD_PATH_HINT = "docs/canonical/sbe_committee_member_gold.json"


@dataclass
class MemberMatch:
    committee_id: str
    committee_name: str
    member_id: str | None
    member_name: str | None
    method: str
    confidence: float
    status: str  # accepted | review
    candidate_id: str = ""
    candidate_name: str = ""
    candidate_last: str = ""
    office: str = ""
    district: str = ""
    notes: str = ""
    unmatched_reason: str = ""
    near_misses: list[dict[str, str]] = field(default_factory=list)
    candidates_considered: list[dict[str, str]] = field(default_factory=list)
    how_to_promote: str = ""
    gold_stub: dict[str, str] = field(default_factory=dict)


@dataclass
class MatchReport:
    accepted: list[MemberMatch] = field(default_factory=list)
    unmatched: list[MemberMatch] = field(default_factory=list)
    legislative_committees: int = 0

    @property
    def match_rate(self) -> float:
        if self.legislative_committees == 0:
            return 0.0
        return len(self.accepted) / self.legislative_committees


def _norm(text: str) -> str:
    return _NON_LETTERS.sub(" ", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    return [t for t in _norm(text).split() if t]


def _strip_suffix(text: str) -> str:
    return _SUFFIX_RE.sub("", text).strip()


def _nicknames(text: str) -> list[str]:
    return [n.lower() for n in _NICKNAME_RE.findall(text)]


def _expand_first_tokens(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for token in tokens:
        out.update(FIRST_NAME_ALIASES.get(token, ()))
    return out


def _first_tokens(first_name: str) -> set[str]:
    cleaned = _NICKNAME_RE.sub(" ", first_name)
    toks = set(_tokens(_strip_suffix(cleaned)))
    toks.update(_nicknames(first_name))
    # drop lone initials
    return _expand_first_tokens({t for t in toks if len(t) > 1})


def _last_key(last_name: str) -> str:
    return _norm(_strip_suffix(last_name))


def _district_key(raw: str) -> str:
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits.lstrip("0") or digits


def _chamber_from_office(office: str, district_type: str) -> str | None:
    office_key = _norm(office)
    if office_key in LEGISLATIVE_OFFICES:
        return LEGISLATIVE_OFFICES[office_key]
    dtype = _norm(district_type)
    if dtype == "senate":
        return "Senate"
    if dtype in {"representative", "house"}:
        return "House"
    return None


def _is_legislative_candidate(cand: SbeCandidateRow) -> bool:
    return _chamber_from_office(cand.office, cand.district_type) is not None


def _member_index(members: list[Member]) -> dict[tuple[str, str, str], list[Member]]:
    """(chamber_lower, district, last) -> members."""
    index: dict[tuple[str, str, str], list[Member]] = {}
    for member in members:
        _first, last, _suffix = _split_member_name(member.name)
        key = (member.chamber.lower(), _district_key(member.district), _last_key(last))
        index.setdefault(key, []).append(member)
        loose = (member.chamber.lower(), "", _last_key(last))
        index.setdefault(loose, []).append(member)
    return index


def _split_member_name(full_name: str) -> tuple[str, str, str]:
    suffix = ""
    match = _SUFFIX_RE.search(full_name)
    if match:
        suffix = match.group().strip().lstrip(",").strip()
        full_name = full_name[: match.start()].strip()
    parts = full_name.split()
    if len(parts) <= 1:
        return ("", full_name, suffix)
    return (" ".join(parts[:-1]), parts[-1], suffix)


def _first_overlap(member: Member, cand: SbeCandidateRow) -> bool:
    member_first, _last, _suf = _split_member_name(member.name)
    member_toks = _first_tokens(member_first)
    cand_toks = _first_tokens(cand.first_name)
    if not member_toks or not cand_toks:
        return False
    return bool(member_toks & cand_toks)


def load_gold_overrides(path: Path | None) -> dict[str, str]:
    """committee_id -> member_id from a JSON list of objects."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    out: dict[str, str] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("committee_id") or "").strip()
        mid = str(row.get("member_id") or "").strip()
        if cid and mid:
            out[cid] = mid
    return out


def match_committees_to_members(
    parsed: ParsedSbeData,
    members: list[Member],
    *,
    gold_path: Path | None = None,
) -> MatchReport:
    """Match Active Candidate committees that are linked to legislative candidates."""
    members_by_id = {m.id: m for m in members}
    index = _member_index(members)
    gold = load_gold_overrides(gold_path)
    committees = {c.committee_id: c for c in parsed.committees}
    candidates = {c.candidate_id: c for c in parsed.candidates}

    links_by_committee: dict[str, list[SbeCandidateRow]] = {}
    for link in parsed.links:
        cand = candidates.get(link.candidate_id)
        if cand is None or not _is_legislative_candidate(cand):
            continue
        links_by_committee.setdefault(link.committee_id, []).append(cand)

    report = MatchReport()
    seen_committees: set[str] = set()

    for committee_id, cands in links_by_committee.items():
        committee = committees.get(committee_id)
        if committee is None:
            continue
        if committee.type_of_committee.lower() not in CANDIDATE_COMMITTEE_TYPES:
            continue
        # Historical Final/dissolved committees drown the review queue; MVP
        # only auto-matches currently Active candidate committees.
        if committee.status and committee.status.upper() != "A":
            continue
        seen_committees.add(committee_id)
        match = _best_match(committee, cands, members_by_id, index, gold)
        if match.status == "accepted" and match.member_id:
            report.accepted.append(match)
        else:
            report.unmatched.append(match)

    report.legislative_committees = len(seen_committees)
    return report


def _candidate_display(cand: SbeCandidateRow) -> str:
    return f"{cand.first_name} {cand.last_name}".strip()


def _candidate_brief(cand: SbeCandidateRow) -> dict[str, str]:
    return {
        "candidate_id": cand.candidate_id,
        "candidate_name": _candidate_display(cand),
        "office": cand.office,
        "district": _district_key(cand.district),
    }


def _best_match(
    committee: SbeCommitteeRow,
    cands: list[SbeCandidateRow],
    members_by_id: dict[str, Member],
    index: dict[tuple[str, str, str], list[Member]],
    gold: dict[str, str],
) -> MemberMatch:
    considered = [_candidate_brief(c) for c in cands]
    sitting = list(members_by_id.values())
    if committee.committee_id in gold:
        member = members_by_id.get(gold[committee.committee_id])
        if member:
            return MemberMatch(
                committee_id=committee.committee_id,
                committee_name=committee.name,
                member_id=member.id,
                member_name=member.name,
                method="gold",
                confidence=1.0,
                status="accepted",
                notes="canonical gold override",
                candidates_considered=considered,
            )

    scored: list[MemberMatch] = []
    for cand in cands:
        chamber = _chamber_from_office(cand.office, cand.district_type)
        if chamber is None:
            continue
        last = _last_key(cand.last_name)
        district = _district_key(cand.district)
        display = _candidate_display(cand)
        exact_hits = index.get((chamber.lower(), district, last), [])
        if len(exact_hits) == 1 and _first_overlap(exact_hits[0], cand):
            member = exact_hits[0]
            scored.append(
                MemberMatch(
                    committee_id=committee.committee_id,
                    committee_name=committee.name,
                    member_id=member.id,
                    member_name=member.name,
                    method="office_district_name",
                    confidence=0.95,
                    status="accepted",
                    candidate_id=cand.candidate_id,
                    candidate_name=display,
                    candidate_last=cand.last_name,
                    office=cand.office,
                    district=district,
                    candidates_considered=considered,
                )
            )
            continue
        if len(exact_hits) == 1:
            member = exact_hits[0]
            scored.append(
                MemberMatch(
                    committee_id=committee.committee_id,
                    committee_name=committee.name,
                    member_id=member.id,
                    member_name=member.name,
                    method="office_district_name",
                    confidence=0.85,
                    status="accepted",
                    candidate_id=cand.candidate_id,
                    candidate_name=display,
                    candidate_last=cand.last_name,
                    office=cand.office,
                    district=district,
                    notes="district+last; first name not confirmed",
                    candidates_considered=considered,
                )
            )
            continue
        loose_hits = [
            m for m in index.get((chamber.lower(), "", last), []) if _first_overlap(m, cand)
        ]
        unique_ids = {m.id for m in loose_hits}
        if len(unique_ids) == 1:
            member = loose_hits[0]
            scored.append(
                MemberMatch(
                    committee_id=committee.committee_id,
                    committee_name=committee.name,
                    member_id=member.id,
                    member_name=member.name,
                    method="name_chamber",
                    confidence=0.8,
                    status="accepted",
                    candidate_id=cand.candidate_id,
                    candidate_name=display,
                    candidate_last=cand.last_name,
                    office=cand.office,
                    district=district,
                    notes="district missing or mismatched; unique name+chamber",
                    candidates_considered=considered,
                )
            )
            continue
        scored.append(
            MemberMatch(
                committee_id=committee.committee_id,
                committee_name=committee.name,
                member_id=None,
                member_name=None,
                method="unresolved",
                confidence=0.0,
                status="review",
                candidate_id=cand.candidate_id,
                candidate_name=display,
                candidate_last=cand.last_name,
                office=cand.office,
                district=district,
                notes="no unique sitting-member match",
                candidates_considered=considered,
            )
        )

    accepted = [m for m in scored if m.status == "accepted"]
    if accepted:
        accepted.sort(key=lambda m: m.confidence, reverse=True)
        return accepted[0]
    if scored:
        return _explain_unmatched(_pick_unresolved(scored, sitting), cands, sitting)
    return _explain_unmatched(
        MemberMatch(
            committee_id=committee.committee_id,
            committee_name=committee.name,
            member_id=None,
            member_name=None,
            method="unresolved",
            confidence=0.0,
            status="review",
            notes="legislative candidate link present but no usable candidate row",
            unmatched_reason="no_usable_candidate",
            candidates_considered=considered,
        ),
        cands,
        sitting,
    )


def _pick_unresolved(scored: list[MemberMatch], members: list[Member]) -> MemberMatch:
    """Prefer the candidate row whose chamber has a same-last sitting member."""

    def _rank(row: MemberMatch) -> tuple[int, int]:
        last = _last_key(row.candidate_last)
        chamber = _chamber_from_office(row.office, "")
        same_last = [m for m in members if _last_key(_split_member_name(m.name)[1]) == last]
        chamber_hits = sum(1 for m in same_last if chamber and m.chamber == chamber)
        sitting_chamber = 1 if chamber and any(m.chamber == chamber for m in members) else 0
        return (chamber_hits, sitting_chamber)

    return max(scored, key=_rank)


def _near_misses_for(cands: list[SbeCandidateRow], members: list[Member]) -> list[dict[str, str]]:
    last_keys = {_last_key(c.last_name) for c in cands if c.last_name}
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for member in members:
        _first, last, _suffix = _split_member_name(member.name)
        if _last_key(last) not in last_keys:
            continue
        if member.id in seen:
            continue
        seen.add(member.id)
        blockers: list[str] = []
        for cand in cands:
            if _last_key(cand.last_name) != _last_key(last):
                continue
            chamber = _chamber_from_office(cand.office, cand.district_type)
            sbe_district = _district_key(cand.district)
            if chamber and chamber != member.chamber:
                blockers.append(
                    f"SBE {chamber} {sbe_district} vs sitting {member.chamber} {member.district}"
                )
            elif sbe_district and sbe_district != _district_key(member.district):
                blockers.append(f"SBE district {sbe_district} vs sitting {member.district}")
            if not _first_overlap(member, cand):
                blockers.append(
                    f"first names do not overlap (sbe={cand.first_name!r} member={_first!r})"
                )
        hits.append(
            {
                "member_id": member.id,
                "member_name": member.name,
                "chamber": member.chamber,
                "district": member.district,
                "blocker": "; ".join(dict.fromkeys(blockers)) or "not unique under match rules",
            }
        )
    return hits


def _gold_stub(
    committee_id: str, committee_name: str, near_misses: list[dict[str, str]]
) -> dict[str, str]:
    member_id = ""
    if len(near_misses) == 1:
        member_id = near_misses[0]["member_id"]
    return {
        "committee_id": committee_id,
        "member_id": member_id,
        "committee_name": committee_name,
        "notes": (
            "Confirm this sitting member owns the Active Candidate committee, "
            f"then append this object to {GOLD_PATH_HINT}. Leave member_id empty "
            "until confirmed. Do not guess."
        ),
    }


def _how_to_promote(gold_stub: dict[str, str]) -> str:
    if gold_stub.get("member_id"):
        return (
            f"If the near-miss sitting member is correct, append {gold_stub} to "
            f"{GOLD_PATH_HINT} and re-run ingest. Confirm the ILGA member_id first."
        )
    return (
        f"If this Active Candidate committee belongs to a sitting member, set "
        f"member_id on the gold_stub, append it to {GOLD_PATH_HINT}, and re-run "
        "ingest. Do not guess member_id."
    )


def _explain_unmatched(
    match: MemberMatch,
    cands: list[SbeCandidateRow],
    members: list[Member],
) -> MemberMatch:
    near = _near_misses_for(cands, members)
    last_labels = sorted({_strip_suffix(c.last_name) for c in cands if c.last_name})
    last_text = ", ".join(last_labels) or "(missing last name)"
    if not cands:
        reason = "no_usable_candidate"
        notes = "legislative candidate link present but no usable candidate row"
    elif not near:
        reason = "no_sitting_member"
        notes = f"No sitting member with last name {last_text} in the current roster."
    elif len(near) > 1:
        reason = "ambiguous"
        notes = (
            f"Multiple sitting members share last name {last_text}; "
            "no unique office+district or name+chamber hit."
        )
    else:
        reason = "near_miss"
        notes = f"Sitting member near-miss for {last_text}: {near[0]['blocker']}"
    stub = _gold_stub(match.committee_id, match.committee_name, near)
    match.unmatched_reason = reason
    match.notes = notes
    match.near_misses = near
    match.gold_stub = stub
    match.how_to_promote = _how_to_promote(stub)
    if not match.candidates_considered:
        match.candidates_considered = [_candidate_brief(c) for c in cands]
    return match


def unmatched_review_rows(report: MatchReport) -> list[dict[str, Any]]:
    """Actionable unmatched payload for unmatched.json and the review CLI."""
    rows: list[dict[str, Any]] = []
    for match in report.unmatched:
        rows.append(
            {
                "committee_id": match.committee_id,
                "committee_name": match.committee_name,
                "candidate_id": match.candidate_id,
                "candidate_name": match.candidate_name,
                "office": match.office,
                "district": match.district,
                "method": match.method,
                "status": match.status,
                "unmatched_reason": match.unmatched_reason,
                "notes": match.notes,
                "near_misses": match.near_misses,
                "candidates_considered": match.candidates_considered,
                "how_to_promote": match.how_to_promote,
                "gold_stub": match.gold_stub
                or _gold_stub(match.committee_id, match.committee_name, match.near_misses),
            }
        )
    return rows


def format_unmatched_review(rows: list[dict[str, Any]]) -> str:
    """Human-readable unmatched review. No disclosure amounts."""
    if not rows:
        return "No unmatched Active Candidate legislative committees.\n"
    lines = [f"Unmatched committees: {len(rows)}", ""]
    for row in rows:
        lines.append(f"{row.get('committee_id')}  {row.get('committee_name')}")
        lines.append(
            f"  candidate: {row.get('candidate_name')}  {row.get('office')} {row.get('district')}"
        )
        lines.append(f"  reason: {row.get('unmatched_reason')}")
        lines.append(f"  why: {row.get('notes')}")
        near = row.get("near_misses") or []
        if near:
            lines.append("  near misses:")
            for hit in near:
                lines.append(
                    f"    {hit.get('member_id')} {hit.get('member_name')} "
                    f"({hit.get('chamber')} {hit.get('district')}) "
                    f"— {hit.get('blocker')}"
                )
        else:
            lines.append("  near misses: none")
        lines.append(f"  promote: {row.get('how_to_promote')}")
        stub = row.get("gold_stub") or {}
        lines.append(f"  gold stub: {json.dumps(stub, sort_keys=True)}")
        lines.append("")
    return "\n".join(lines)


def resolve_contributor_member_id(
    last_only_name: str,
    first_name: str,
    members: list[Member],
) -> str | None:
    """Best-effort: donor is a sitting member (transfer / self-receipt)."""
    last = _last_key(last_only_name)
    firsts = _first_tokens(first_name) if first_name else set()
    hits: list[Member] = []
    for member in members:
        _mf, ml, _s = _split_member_name(member.name)
        if _last_key(ml) != last:
            continue
        if firsts and not (_first_tokens(_mf) & firsts):
            continue
        hits.append(member)
    if len(hits) == 1:
        return hits[0].id
    return None
