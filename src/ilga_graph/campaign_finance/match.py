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
    office: str = ""
    district: str = ""
    notes: str = ""


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


def _first_tokens(first_name: str) -> set[str]:
    cleaned = _NICKNAME_RE.sub(" ", first_name)
    toks = set(_tokens(_strip_suffix(cleaned)))
    toks.update(_nicknames(first_name))
    # drop lone initials
    return {t for t in toks if len(t) > 1}


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
        first, last, _suffix = _split_member_name(member.name)
        key = (member.chamber.lower(), _district_key(member.district), _last_key(last))
        index.setdefault(key, []).append(member)
        # also index without district for name+chamber fallback
        loose = (member.chamber.lower(), "", _last_key(last))
        index.setdefault(loose, []).append(member)
        # nickname last stays the same; first handled at compare time
        _ = first
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


def _best_match(
    committee: SbeCommitteeRow,
    cands: list[SbeCandidateRow],
    members_by_id: dict[str, Member],
    index: dict[tuple[str, str, str], list[Member]],
    gold: dict[str, str],
) -> MemberMatch:
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
            )

    scored: list[MemberMatch] = []
    for cand in cands:
        chamber = _chamber_from_office(cand.office, cand.district_type)
        if chamber is None:
            continue
        last = _last_key(cand.last_name)
        district = _district_key(cand.district)
        display = f"{cand.first_name} {cand.last_name}".strip()
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
                    office=cand.office,
                    district=district,
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
                    office=cand.office,
                    district=district,
                    notes="district+last; first name not confirmed",
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
                    office=cand.office,
                    district=district,
                    notes="district missing or mismatched; unique name+chamber",
                )
            )
            continue
        if _committee_name_hit(committee.name, cand, members_by_id.values()):
            # handled below via committee-name fallback
            pass
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
                office=cand.office,
                district=district,
                notes="no unique sitting-member match",
            )
        )

    accepted = [m for m in scored if m.status == "accepted"]
    if accepted:
        accepted.sort(key=lambda m: m.confidence, reverse=True)
        return accepted[0]
    if scored:
        return scored[0]
    return MemberMatch(
        committee_id=committee.committee_id,
        committee_name=committee.name,
        member_id=None,
        member_name=None,
        method="unresolved",
        confidence=0.0,
        status="review",
        notes="legislative candidate link present but no usable candidate row",
    )


def _committee_name_hit(
    committee_name: str, cand: SbeCandidateRow, members: list[Member] | None = None
) -> bool:
    name = _norm(committee_name)
    last = _last_key(cand.last_name)
    firsts = _first_tokens(cand.first_name)
    if last and last in name and any(f in name for f in firsts):
        return True
    if members:
        for member in members:
            _first, last_m, _s = _split_member_name(member.name)
            if _last_key(last_m) in name and any(t in name for t in _first_tokens(_first)):
                return True
    return False


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
