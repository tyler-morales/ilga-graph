"""Parse official SBE tab-delimited Campaign Disclosure files."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _parse_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = _read_text(path).splitlines()
    if not lines:
        return [], []
    header = [h.strip() for h in lines[0].split("\t")]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            parts = parts + [""] * (len(header) - len(parts))
        rows.append({h: parts[i].strip() if i < len(parts) else "" for i, h in enumerate(header)})
    return header, rows


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def parse_amount(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return 0.0


def parse_date(value: str) -> date | None:
    if not value:
        return None
    token = value.strip().split()[0]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    return None


@dataclass
class SbeCommitteeRow:
    committee_id: str
    name: str
    type_of_committee: str
    status: str
    party: str
    purpose: str
    city: str
    state: str
    refer_name: str


@dataclass
class SbeCandidateRow:
    candidate_id: str
    last_name: str
    first_name: str
    office: str
    district_type: str
    district: str
    party: str


@dataclass
class SbeLinkRow:
    committee_id: str
    candidate_id: str


@dataclass
class SbeReceiptRow:
    receipt_id: str
    committee_id: str
    last_only_name: str
    first_name: str
    received_date: date | None
    amount: float
    occupation: str
    employer: str
    city: str
    state: str
    d2_part: str
    description: str
    archived: bool


@dataclass
class ParsedSbeData:
    committees: list[SbeCommitteeRow] = field(default_factory=list)
    candidates: list[SbeCandidateRow] = field(default_factory=list)
    links: list[SbeLinkRow] = field(default_factory=list)
    receipts: list[SbeReceiptRow] = field(default_factory=list)


def _committee_from_row(row: dict[str, str]) -> SbeCommitteeRow:
    return SbeCommitteeRow(
        committee_id=row.get("ID", ""),
        name=row.get("Name", ""),
        type_of_committee=row.get("TypeOfCommittee", ""),
        status=row.get("Status", ""),
        party=row.get("PartyAffiliation", ""),
        purpose=row.get("Purpose", ""),
        city=row.get("City", ""),
        state=row.get("State", ""),
        refer_name=row.get("ReferName", ""),
    )


def _candidate_from_row(row: dict[str, str]) -> SbeCandidateRow:
    return SbeCandidateRow(
        candidate_id=row.get("ID", ""),
        last_name=row.get("LastName", ""),
        first_name=row.get("FirstName", ""),
        office=row.get("Office", ""),
        district_type=row.get("DistrictType", ""),
        district=row.get("District", ""),
        party=row.get("PartyAffiliation", ""),
    )


def _receipt_from_row(row: dict[str, str]) -> SbeReceiptRow:
    return SbeReceiptRow(
        receipt_id=row.get("ID", ""),
        committee_id=row.get("CommitteeID", ""),
        last_only_name=row.get("LastOnlyName", ""),
        first_name=row.get("FirstName", ""),
        received_date=parse_date(row.get("RcvDate", "")),
        amount=parse_amount(row.get("Amount", "")),
        occupation=row.get("Occupation", ""),
        employer=row.get("Employer", ""),
        city=row.get("City", ""),
        state=row.get("State", ""),
        d2_part=row.get("D2Part", ""),
        description=row.get("Description", ""),
        archived=parse_bool(row.get("Archived", "")),
    )


def parse_committees(path: Path) -> list[SbeCommitteeRow]:
    _, rows = _parse_tsv(path)
    return [_committee_from_row(r) for r in rows if r.get("ID")]


def parse_candidates(path: Path) -> list[SbeCandidateRow]:
    _, rows = _parse_tsv(path)
    return [_candidate_from_row(r) for r in rows if r.get("ID")]


def parse_links(path: Path) -> list[SbeLinkRow]:
    _, rows = _parse_tsv(path)
    return [
        SbeLinkRow(committee_id=r.get("CommitteeID", ""), candidate_id=r.get("CandidateID", ""))
        for r in rows
        if r.get("CommitteeID") and r.get("CandidateID")
    ]


def parse_receipts(
    path: Path,
    *,
    since: date | None = None,
    skip_archived: bool = True,
) -> list[SbeReceiptRow]:
    _, rows = _parse_tsv(path)
    out: list[SbeReceiptRow] = []
    for row in rows:
        rec = _receipt_from_row(row)
        if not rec.receipt_id or not rec.committee_id:
            continue
        if skip_archived and rec.archived:
            continue
        if since and rec.received_date and rec.received_date < since:
            continue
        out.append(rec)
    return out


def parse_sbe_dir(
    directory: Path,
    *,
    since: date | None = None,
    skip_archived: bool = True,
) -> ParsedSbeData:
    """Parse Committees + Candidates + links + Receipts from one directory."""
    data = ParsedSbeData()
    committees = directory / "Committees.txt"
    candidates = directory / "Candidates.txt"
    links = directory / "CmteCandidateLinks.txt"
    receipts = directory / "Receipts.txt"
    if committees.exists():
        data.committees = parse_committees(committees)
    if candidates.exists():
        data.candidates = parse_candidates(candidates)
    if links.exists():
        data.links = parse_links(links)
    if receipts.exists():
        data.receipts = parse_receipts(receipts, since=since, skip_archived=skip_archived)
    return data
