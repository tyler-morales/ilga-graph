"""Persist SBE committees, receipts, and member matches to SQLite + JSON."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ..db_models import Base, SbeCommittee, SbeIngestRun, SbeMemberMatch, SbeReceipt
from .match import MatchReport, unmatched_review_rows
from .parse import ParsedSbeData, SbeCommitteeRow, SbeReceiptRow


def _engine(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def ensure_tables(db_path: Path) -> None:
    engine = _engine(db_path)
    Base.metadata.create_all(
        engine,
        tables=[
            SbeCommittee.__table__,
            SbeReceipt.__table__,
            SbeMemberMatch.__table__,
            SbeIngestRun.__table__,
        ],
    )
    engine.dispose()


def _committee_orm(row: SbeCommitteeRow) -> SbeCommittee:
    return SbeCommittee(
        id=row.committee_id,
        name=row.name,
        type_of_committee=row.type_of_committee,
        status=row.status,
        party=row.party,
        purpose=row.purpose,
        city=row.city,
        state=row.state,
        refer_name=row.refer_name,
    )


def _receipt_orm(row: SbeReceiptRow) -> SbeReceipt:
    return SbeReceipt(
        id=row.receipt_id,
        committee_id=row.committee_id,
        received_date=row.received_date.isoformat() if row.received_date else None,
        amount=row.amount,
        last_only_name=row.last_only_name,
        first_name=row.first_name,
        occupation=row.occupation,
        employer=row.employer,
        city=row.city,
        state=row.state,
        d2_part=row.d2_part,
        description=row.description,
        archived=row.archived,
    )


def write_sqlite(
    parsed: ParsedSbeData,
    report: MatchReport,
    db_path: Path,
    *,
    window_start: str,
    source_url: str,
    keep_committee_ids: set[str] | None = None,
) -> int:
    """Replace money-layer tables. Returns receipts stored."""
    ensure_tables(db_path)
    engine = _engine(db_path)
    keep = keep_committee_ids
    receipts = [r for r in parsed.receipts if keep is None or r.committee_id in keep]
    with Session(engine) as session:
        session.query(SbeReceipt).delete()
        session.query(SbeMemberMatch).delete()
        session.query(SbeCommittee).delete()
        for committee in parsed.committees:
            if keep is not None and committee.committee_id not in keep:
                # still store unmatched review committees
                if not any(u.committee_id == committee.committee_id for u in report.unmatched):
                    continue
            session.merge(_committee_orm(committee))
        for receipt in receipts:
            session.merge(_receipt_orm(receipt))
        for match in report.accepted + report.unmatched:
            session.add(
                SbeMemberMatch(
                    committee_id=match.committee_id,
                    member_id=match.member_id,
                    match_method=match.method,
                    confidence=match.confidence,
                    status=match.status,
                    candidate_id=match.candidate_id or None,
                    candidate_name=match.candidate_name or None,
                    office=match.office or None,
                    district=match.district or None,
                    notes=match.notes or None,
                )
            )
        session.add(
            SbeIngestRun(
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                source_url=source_url,
                window_start=window_start,
                committees_loaded=len(parsed.committees),
                receipts_loaded=len(receipts),
                members_matched=len({m.member_id for m in report.accepted if m.member_id}),
                match_rate=report.match_rate,
                notes=f"accepted={len(report.accepted)} review={len(report.unmatched)}",
            )
        )
        session.commit()
    engine.dispose()
    return len(receipts)


def write_json_index(index_payload: dict[str, Any], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")


def write_unmatched(report: MatchReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(unmatched_review_rows(report), indent=2), encoding="utf-8")


def today_iso() -> str:
    return date.today().isoformat()
