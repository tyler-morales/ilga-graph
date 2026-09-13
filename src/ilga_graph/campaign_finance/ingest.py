"""Build the in-memory / on-disk campaign-finance index."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ..models import Member
from .download import SBE_DOWNLOAD_BASE, download_sbe_files
from .match import MatchReport, match_committees_to_members, resolve_contributor_member_id
from .parse import ParsedSbeData, parse_sbe_dir
from .store import today_iso, write_json_index, write_sqlite, write_unmatched

DEFAULT_WINDOW_START = "2025-01-01"


@dataclass
class CampaignFinanceIndex:
    source: str
    window_start: str
    window_end: str
    generated_at: str
    match_stats: dict[str, Any]
    committees_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    matches_by_member: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    matches_by_committee: dict[str, dict[str, Any]] = field(default_factory=dict)
    receipts_by_committee: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    unmatched: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "generated_at": self.generated_at,
            "match_stats": self.match_stats,
            "committees": list(self.committees_by_id.values()),
            "matches": [m for matches in self.matches_by_member.values() for m in matches],
            "receipts": [r for recs in self.receipts_by_committee.values() for r in recs],
            "unmatched": self.unmatched,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CampaignFinanceIndex:
        committees = {c["committee_id"]: c for c in payload.get("committees", [])}
        matches_by_member: dict[str, list[dict[str, Any]]] = {}
        matches_by_committee: dict[str, dict[str, Any]] = {}
        for match in payload.get("matches", []):
            mid = match.get("member_id")
            if mid:
                matches_by_member.setdefault(mid, []).append(match)
            if match.get("committee_id"):
                matches_by_committee[match["committee_id"]] = match
        receipts_by_committee: dict[str, list[dict[str, Any]]] = {}
        for row in payload.get("receipts", []):
            receipts_by_committee.setdefault(row.get("committee_id"), []).append(row)
        return cls(
            source=payload.get("source", ""),
            window_start=payload.get("window_start", ""),
            window_end=payload.get("window_end", ""),
            generated_at=payload.get("generated_at", ""),
            match_stats=payload.get("match_stats") or {},
            committees_by_id=committees,
            matches_by_member=matches_by_member,
            matches_by_committee=matches_by_committee,
            receipts_by_committee=receipts_by_committee,
            unmatched=list(payload.get("unmatched") or []),
        )


@dataclass
class IngestResult:
    window_start: str
    receipts_stored: int
    members_matched: int
    match_rate: float
    index_path: str
    unmatched_path: str


def _receipt_dict(row: Any, contributor_member_id: str | None) -> dict[str, Any]:
    return {
        "receipt_id": row.receipt_id,
        "committee_id": row.committee_id,
        "last_only_name": row.last_only_name,
        "first_name": row.first_name,
        "received_date": row.received_date.isoformat() if row.received_date else "",
        "amount": row.amount,
        "occupation": row.occupation,
        "employer": row.employer,
        "city": row.city,
        "state": row.state,
        "d2_part": row.d2_part,
        "description": row.description,
        "contributor_member_id": contributor_member_id,
    }


def build_index(
    parsed: ParsedSbeData,
    report: MatchReport,
    *,
    window_start: str = DEFAULT_WINDOW_START,
    window_end: str | None = None,
    source: str = SBE_DOWNLOAD_BASE,
    members: list[Member] | None = None,
) -> CampaignFinanceIndex:
    accepted_ids = {m.committee_id for m in report.accepted}
    review_ids = {m.committee_id for m in report.unmatched}
    keep = accepted_ids | review_ids
    committees = {
        c.committee_id: {
            "committee_id": c.committee_id,
            "name": c.name,
            "type_of_committee": c.type_of_committee,
            "status": c.status,
            "party": c.party,
            "purpose": c.purpose,
            "city": c.city,
            "state": c.state,
        }
        for c in parsed.committees
        if c.committee_id in keep
    }
    matches_by_member: dict[str, list[dict[str, Any]]] = {}
    matches_by_committee: dict[str, dict[str, Any]] = {}
    for match in report.accepted:
        payload = asdict(match)
        matches_by_member.setdefault(match.member_id or "", []).append(payload)
        matches_by_committee[match.committee_id] = payload

    receipts_by_committee: dict[str, list[dict[str, Any]]] = {}
    sitting = members or []
    for row in parsed.receipts:
        if row.committee_id not in accepted_ids:
            continue
        donor_id = resolve_contributor_member_id(row.last_only_name, row.first_name, sitting)
        receipts_by_committee.setdefault(row.committee_id, []).append(_receipt_dict(row, donor_id))

    unmatched = [asdict(m) for m in report.unmatched]
    return CampaignFinanceIndex(
        source=source,
        window_start=window_start,
        window_end=window_end or today_iso(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        match_stats={
            "legislative_committees": report.legislative_committees,
            "accepted": len(report.accepted),
            "review": len(report.unmatched),
            "match_rate": round(report.match_rate, 4),
            "members_matched": len(matches_by_member),
            "receipts_indexed": sum(len(v) for v in receipts_by_committee.values()),
        },
        committees_by_id=committees,
        matches_by_member=matches_by_member,
        matches_by_committee=matches_by_committee,
        receipts_by_committee=receipts_by_committee,
        unmatched=unmatched,
    )


def ingest_from_dir(
    directory: Path,
    members: list[Member],
    *,
    db_path: Path,
    index_path: Path,
    window_start: str = DEFAULT_WINDOW_START,
    source_url: str = SBE_DOWNLOAD_BASE,
    gold_path: Path | None = None,
) -> IngestResult:
    since = date.fromisoformat(window_start)
    parsed = parse_sbe_dir(directory, since=since)
    report = match_committees_to_members(parsed, members, gold_path=gold_path)
    index = build_index(
        parsed,
        report,
        window_start=window_start,
        source=source_url,
        members=members,
    )
    keep = set(index.committees_by_id)
    receipts_stored = write_sqlite(
        parsed,
        report,
        db_path,
        window_start=window_start,
        source_url=source_url,
        keep_committee_ids=keep,
    )
    write_json_index(index.to_payload(), index_path)
    unmatched_path = index_path.with_name("unmatched.json")
    write_unmatched(report, unmatched_path)
    return IngestResult(
        window_start=window_start,
        receipts_stored=receipts_stored,
        members_matched=len({m.member_id for m in report.accepted if m.member_id}),
        match_rate=report.match_rate,
        index_path=str(index_path),
        unmatched_path=str(unmatched_path),
    )


def ingest_from_sbe(
    members: list[Member],
    *,
    work_dir: Path,
    db_path: Path,
    index_path: Path,
    window_start: str = DEFAULT_WINDOW_START,
    gold_path: Path | None = None,
    skip_download: bool = False,
) -> IngestResult:
    since = date.fromisoformat(window_start)
    if not skip_download:
        download_sbe_files(work_dir, since=since)
    return ingest_from_dir(
        work_dir,
        members,
        db_path=db_path,
        index_path=index_path,
        window_start=window_start,
        gold_path=gold_path,
    )


def load_index_json(path: Path) -> CampaignFinanceIndex:
    return CampaignFinanceIndex.from_payload(json.loads(path.read_text(encoding="utf-8")))
