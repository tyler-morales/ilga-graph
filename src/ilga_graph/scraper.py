from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import CACHE_DIR, MOCK_DEV_DIR
from .models import Bill, CareerRange, Committee, CommitteeMemberRole, Member, Office

LOGGER = logging.getLogger(__name__)

# ── Pre-compiled regex patterns ──────────────────────────────────────────────

_RE_PARTY_LETTER = re.compile(r"\([RDI]\)", re.IGNORECASE)
_RE_PARTY_CAPTURE = re.compile(r"\(([RDI])\)")
_RE_TITLE_PREFIX = re.compile(r"^\s*(sen\.?|senator|rep\.?|representative)\s+", re.IGNORECASE)
_RE_WHITESPACE = re.compile(r"\s+")
_RE_NAME_SUFFIX = re.compile(r"(,?\s*(jr\.?|sr\.?|ii|iii|iv|v))\s*$", re.IGNORECASE)
_RE_NAME_PARTY = re.compile(r"^(?P<name>.+?)\s*\([RDI]\)")
_RE_PARTY_FULL = re.compile(r"\b(Republican|Democrat|Independent)\b", re.IGNORECASE)
_RE_DISTRICT = re.compile(r"\b(\d+)(?:st|nd|rd|th)?\s+District\b", re.IGNORECASE)
_RE_CAREER_RANGE = re.compile(
    r"(?P<start>\d{4})\s*[-\u2013\u2014]\s*(?P<end>\d{4}|Present)"
    r"(?:\s*\((?P<chamber>[^)]+)\))?",
    re.IGNORECASE,
)
_RE_BIO_PANE = re.compile(r"pane-Biography", re.IGNORECASE)
_RE_CONTACT_PANE = re.compile(r"pane-Contact", re.IGNORECASE)
_RE_MEMBER_INFO_COL = re.compile(r"\bmember-info-col\b")
_RE_CARD_BODY = re.compile(r"\bcard-body\b")
_RE_CONTACT_HEADING = re.compile(r"Contact (?:Info|Information)", re.IGNORECASE)
_RE_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_RE_PHONE = re.compile(r"(?:\(\d{3}\)\s*\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b)")
_RE_FAX = re.compile(
    r"((?:\(\d{3}\)\s*\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b))\s*Fax",
    re.IGNORECASE,
)
_RE_ROW_CLASS = re.compile(r"\brow\b")
_RE_COL_SM_4 = re.compile(r"\bcol-sm-4\b")
_RE_COL_SM_8 = re.compile(r"\bcol-sm-8\b")
_RE_COMMITTEE_ROLE = re.compile(
    r"\b(?P<role>Chairperson|Chair|Vice Chair|Member|Minority Spokesperson|Spokesperson)\b"
    r"[,\s]+(?P<committee>[A-Z][A-Za-z0-9\s&\-]+?)(?:\.|;|,|$)",
    re.IGNORECASE,
)
_RE_LEG_ID = re.compile(r"LegID=(\d+)")

# ── Cache & mock directories (from config) ──────────────────────────────────

# ── Constant token sets ──────────────────────────────────────────────────────

_SECTION_STOP_TOKENS: frozenset[str] = frozenset(
    {
        "Biography",
        "Associated Representative:",
        "Associated Representatives:",
        "Associated Senator:",
        "Associated Senators:",
        "Associated Members:",
        "Committees",
        "Bills",
    }
)

_ASSOCIATED_LABELS: frozenset[str] = frozenset(
    {
        "Associated Representative:",
        "Associated Representatives:",
        "Associated Senator:",
        "Associated Senators:",
        "Associated Members:",
    }
)


# ── Module-level utility functions ───────────────────────────────────────────


def _extract_text_lines(soup: BeautifulSoup) -> list[str]:
    return [line for raw in soup.get_text("\n", strip=True).splitlines() if (line := raw.strip())]


def _member_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key in ("MemberID", "memberid", "id"):
        if key in query and query[key]:
            return query[key][0]
    path_parts = [part for part in parsed.path.split("/") if part]
    return path_parts[-1] if path_parts else url


def _collect_section(lines: list[str], start_index: int, stop_tokens: frozenset[str]) -> list[str]:
    collected: list[str] = []
    for line in lines[start_index:]:
        if line in stop_tokens:
            break
        collected.append(line)
    return collected


def _last_name_from_normalized(normalized_name: str) -> str:
    if not normalized_name:
        return ""
    parts = normalized_name.split()
    return parts[-1] if parts else ""


# ── Caching ──────────────────────────────────────────────────────────────────


def load_cache(filename: str, *, seed_fallback: bool = False) -> list[dict] | None:
    path = CACHE_DIR / filename
    if path.exists():
        LOGGER.info("Loading cache from %s", path)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if seed_fallback:
        seed_path = MOCK_DEV_DIR / filename
        if seed_path.exists():
            LOGGER.info("Loading mock data from %s (no cache found)", seed_path)
            with open(seed_path, encoding="utf-8") as f:
                return json.load(f)
    return None


def load_dict_cache(filename: str, *, seed_fallback: bool = False) -> dict | None:
    """Load a dict-shaped JSON cache file (e.g. rosters, committee bills)."""
    path = CACHE_DIR / filename
    if path.exists():
        LOGGER.info("Loading cache from %s", path)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if seed_fallback:
        seed_path = MOCK_DEV_DIR / filename
        if seed_path.exists():
            LOGGER.info("Loading mock data from %s (no cache found)", seed_path)
            with open(seed_path, encoding="utf-8") as f:
                return json.load(f)
    return None


def save_cache(filename: str, data: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved cache to %s (%d entries)", path, len(data))


def save_dict_cache(filename: str, data: dict) -> None:
    """Save a dict-shaped JSON cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved cache to %s (%d keys)", path, len(data))


def cache_age_days(filename: str) -> float | None:
    """Return the age of a cache file in days, or None if it doesn't exist."""
    path = CACHE_DIR / filename
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    return (time.time() - mtime) / 86400


def _warn_stale_cache(filename: str, max_age_days: float = 7.0) -> None:
    """Log a warning if a cache file is older than *max_age_days*."""
    age = cache_age_days(filename)
    if age is not None and age > max_age_days:
        LOGGER.warning(
            "Cache file %s is %.1f days old (threshold: %.0f). "
            "Consider re-scraping with `make scrape --force-refresh`.",
            filename,
            age,
            max_age_days,
        )


def _committee_from_dict(d: dict) -> Committee:
    return Committee(
        code=d["code"],
        name=d["name"],
        parent_code=d.get("parent_code"),
        members_list_url=d.get("members_list_url"),
    )


def _committee_member_role_from_dict(d: dict) -> CommitteeMemberRole:
    return CommitteeMemberRole(
        member_id=d["member_id"],
        member_name=d["member_name"],
        member_url=d["member_url"],
        role=d["role"],
    )


def _bill_from_dict(b: dict) -> Bill:
    from .models import ActionEntry, VoteEvent, WitnessSlip

    action_history = []
    for a in b.get("action_history", []):
        if isinstance(a, dict) and "date" in a and "chamber" in a and "action" in a:
            action_history.append(
                ActionEntry(date=a["date"], chamber=a["chamber"], action=a["action"])
            )

    vote_events = []
    for v in b.get("vote_events", []):
        if isinstance(v, dict):
            vote_events.append(
                VoteEvent(
                    bill_number=v.get("bill_number", ""),
                    date=v.get("date", ""),
                    description=v.get("description", ""),
                    chamber=v.get("chamber", ""),
                    yea_votes=v.get("yea_votes", []),
                    nay_votes=v.get("nay_votes", []),
                    present_votes=v.get("present_votes", []),
                    nv_votes=v.get("nv_votes", []),
                    pdf_url=v.get("pdf_url", ""),
                    vote_type=v.get("vote_type", "floor"),
                )
            )

    witness_slips = []
    for ws in b.get("witness_slips", []):
        if isinstance(ws, dict):
            witness_slips.append(
                WitnessSlip(
                    name=ws.get("name", ""),
                    organization=ws.get("organization", ""),
                    representing=ws.get("representing", ""),
                    position=ws.get("position", ""),
                    hearing_committee=ws.get("hearing_committee", ""),
                    hearing_date=ws.get("hearing_date", ""),
                    testimony_type=ws.get("testimony_type", "Record of Appearance Only"),
                    bill_number=ws.get("bill_number", ""),
                )
            )

    return Bill(
        bill_number=b["bill_number"],
        leg_id=b["leg_id"],
        description=b["description"],
        chamber=b["chamber"],
        last_action=b.get("last_action", ""),
        last_action_date=b.get("last_action_date", ""),
        primary_sponsor=b.get("primary_sponsor", ""),
        synopsis=b.get("synopsis", ""),
        status_url=b.get("status_url", ""),
        sponsor_ids=b.get("sponsor_ids", []),
        house_sponsor_ids=b.get("house_sponsor_ids", []),
        action_history=action_history,
        vote_events=vote_events,
        witness_slips=witness_slips,
    )


def _member_metadata_dict(m: Member) -> dict:
    """Serialize a Member WITHOUT embedded bill objects (normalized format)."""
    return {
        "id": m.id,
        "name": m.name,
        "member_url": m.member_url,
        "chamber": m.chamber,
        "party": m.party,
        "district": m.district,
        "bio_text": m.bio_text,
        "role": m.role,
        "career_timeline_text": m.career_timeline_text,
        "career_ranges": [asdict(cr) for cr in m.career_ranges],
        "committees": m.committees,
        "associated_members": m.associated_members,
        "email": m.email,
        "offices": [asdict(o) for o in m.offices],
        "roles": m.roles,
        "sponsored_bill_ids": m.sponsored_bill_ids,
        "co_sponsor_bill_ids": m.co_sponsor_bill_ids,
    }


# ── Normalized cache ─────────────────────────────────────────────────────────


def save_normalized_cache(
    members: list[Member],
    bills: dict[str, Bill],
) -> None:
    """Save members and bills in normalized format (atomic writes)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Members
    members_data = [_member_metadata_dict(m) for m in members]
    path_m = CACHE_DIR / "members.json"
    tmp_m = path_m.with_suffix(".json.tmp")
    with open(tmp_m, "w", encoding="utf-8") as f:
        json.dump(members_data, f, indent=2, ensure_ascii=False)
    tmp_m.replace(path_m)

    # Bills
    bills_data = {lid: asdict(b) for lid, b in bills.items()}
    path_b = CACHE_DIR / "bills.json"
    tmp_b = path_b.with_suffix(".json.tmp")
    with open(tmp_b, "w", encoding="utf-8") as f:
        json.dump(bills_data, f, indent=2, ensure_ascii=False)
    tmp_b.replace(path_b)

    LOGGER.info(
        "Saved normalized cache: %d members (%s), %d bills (%s)",
        len(members_data),
        path_m,
        len(bills_data),
        path_b,
    )


def load_normalized_cache(
    *,
    seed_fallback: bool = False,
) -> tuple[list[Member], dict[str, Bill]] | None:
    """Load normalized members + bills from cache or seed.

    Returns ``(members, bills_lookup)`` or ``None`` if cache is missing.
    Members will have ``sponsored_bill_ids`` / ``co_sponsor_bill_ids`` set
    but ``sponsored_bills`` / ``co_sponsor_bills`` empty -- call
    :func:`hydrate_members` to populate the full objects.
    """
    members_file = "members.json"
    bills_file = "bills.json"

    members_raw = load_cache(members_file, seed_fallback=seed_fallback)
    bills_raw = load_dict_cache(bills_file, seed_fallback=seed_fallback)

    if members_raw is None or bills_raw is None:
        return None

    bills_lookup: dict[str, Bill] = {lid: _bill_from_dict(bd) for lid, bd in bills_raw.items()}

    members: list[Member] = []
    for d in members_raw:
        m = Member(
            id=d["id"],
            name=d["name"],
            member_url=d["member_url"],
            chamber=d["chamber"],
            party=d["party"],
            district=d["district"],
            bio_text=d["bio_text"],
            role=d.get("role", ""),
            career_timeline_text=d.get("career_timeline_text", ""),
            career_ranges=[
                CareerRange(
                    start_year=cr["start_year"],
                    end_year=cr.get("end_year"),
                    chamber=cr.get("chamber"),
                )
                for cr in d.get("career_ranges", [])
            ],
            committees=d.get("committees", []),
            associated_members=d.get("associated_members"),
            email=d.get("email"),
            offices=[
                Office(
                    name=o["name"],
                    address=o["address"],
                    phone=o.get("phone"),
                    fax=o.get("fax"),
                )
                for o in d.get("offices", [])
            ],
            roles=d.get("roles", []),
            sponsored_bill_ids=d.get("sponsored_bill_ids", []),
            co_sponsor_bill_ids=d.get("co_sponsor_bill_ids", []),
        )
        members.append(m)

    LOGGER.info(
        "Loaded normalized cache: %d members, %d unique bills.",
        len(members),
        len(bills_lookup),
    )
    return members, bills_lookup


def hydrate_members(
    members: list[Member],
    bills_lookup: dict[str, Bill],
) -> list[Member]:
    """Populate ``sponsored_bills`` and ``co_sponsor_bills`` from ID lists."""
    for member in members:
        member.sponsored_bills = [
            bills_lookup[bid] for bid in member.sponsored_bill_ids if bid in bills_lookup
        ]
        member.co_sponsor_bills = [
            bills_lookup[bid] for bid in member.co_sponsor_bill_ids if bid in bills_lookup
        ]
    return members


# ── Bill table parsing ───────────────────────────────────────────────────────


def parse_bill_table(soup: BeautifulSoup) -> list[Bill]:
    table = soup.find("table", class_="table table-striped border")
    if not table:
        return []

    bills: list[Bill] = []
    for row in table.find_all("tr"):
        # Bill # is in a <th scope="row">, not a <td>
        header_cell = row.find("th", scope="row")
        if not header_cell:
            continue
        bill_link = header_cell.find("a", class_="billlist", href=True)
        if not bill_link:
            continue
        href = bill_link["href"]
        leg_id_match = _RE_LEG_ID.search(href)
        if not leg_id_match:
            continue

        # Remaining columns are <td>: Sponsor, Description, Chamber, Last Action, Date
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        leg_id = leg_id_match.group(1)
        bill_number = bill_link.get_text(" ", strip=True)
        chamber = "S" if bill_number.upper().startswith("S") else "H"

        chief_sponsor = cells[0].get_text(" ", strip=True)
        description = cells[1].get_text(" ", strip=True)
        # cells[2] is the Chamber column from the table; we derive chamber from bill_number
        last_action = cells[3].get_text(" ", strip=True)
        last_action_date = cells[4].get_text(" ", strip=True)

        bills.append(
            Bill(
                bill_number=bill_number,
                leg_id=leg_id,
                description=description,
                chamber=chamber,
                last_action=last_action,
                last_action_date=last_action_date,
                primary_sponsor=chief_sponsor,
            )
        )
    return bills


def parse_committee_bill_table(soup: BeautifulSoup) -> list[str]:
    """Parse the bill table on a committee bills page.

    Returns just bill number strings (e.g. ["SB2756", "SB2923"]).
    The committee table uses <td> with <a class="text-nowrap"> for the bill link,
    unlike the member table which uses <th scope="row">.
    """
    table = soup.find("table", class_="table table-striped border")
    if not table:
        return []
    bill_numbers: list[str] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        link = cells[0].find("a", class_="text-nowrap", href=True)
        if link:
            bill_numbers.append(link.get_text(strip=True))
    return bill_numbers


# ── Scraper ──────────────────────────────────────────────────────────────────


@dataclass
class ILGAScraper:
    base_url: str = "https://www.ilga.gov/"
    timeout_seconds: int = 20
    max_workers: int = 3
    request_delay: float = 1.0
    seed_fallback: bool = False
    name_map: dict[str, str] = field(default_factory=dict)
    _session: requests.Session = field(default_factory=requests.Session, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_request_time: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Configure retry adapter for resilient HTTP requests."""
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=5,
            pool_maxsize=5,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ── throttled HTTP ────────────────────────────────────────────────────

    def _throttled_get(self, url: str, **kwargs: object) -> requests.Response:
        """GET with a global per-instance rate limit to avoid IP blocks."""
        with self._lock:
            elapsed = time.time() - self._last_request_time
            wait = max(0.0, self.request_delay - elapsed)
            # Reserve our slot by advancing the timestamp before releasing the lock.
            self._last_request_time = time.time() + wait
        if wait > 0:
            time.sleep(wait)
        kwargs.setdefault("timeout", self.timeout_seconds)
        return self._session.get(url, **kwargs)

    # ── public API ───────────────────────────────────────────────────────

    def fetch_members(self, chamber: str = "Senate", limit: int = 0) -> list[Member]:
        # ── Try normalized cache first (members.json + bills.json) ──
        _warn_stale_cache("members.json")
        normalized = load_normalized_cache(seed_fallback=self.seed_fallback)
        if normalized is not None:
            all_members, bills_lookup = normalized
            # Filter to requested chamber
            members = [m for m in all_members if m.chamber == chamber]
            hydrate_members(members, bills_lookup)
            for m in members:
                self._update_name_map(m)
            LOGGER.info(
                "Loaded %d %s members from normalized cache.",
                len(members),
                chamber,
            )
            return members

        # ── No cache -- scrape from ilga.gov ──
        list_url = urljoin(self.base_url, f"{chamber}/Members/List")
        try:
            resp = self._throttled_get(list_url)
            resp.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.exception("Failed to fetch member list from %s: %s", list_url, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        member_urls = list(self._extract_member_urls(soup, list_url, chamber))
        if limit > 0:
            member_urls = member_urls[:limit]

        total = len(member_urls)
        LOGGER.info("Scraping %d %s members...", total, chamber)
        members: list[Member] = []
        completed = 0
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_url = {
                pool.submit(self.scrape_details, url, chamber): url for url in member_urls
            }
            for future in as_completed(future_to_url):
                completed += 1
                elapsed = time.perf_counter() - t_start
                try:
                    result = future.result()
                    if result is not None:
                        members.append(result)
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        LOGGER.info(
                            "  [%d/%d] Scraped %s (%.0fs elapsed, ~%.0fs remaining)",
                            completed,
                            total,
                            result.name,
                            elapsed,
                            eta,
                        )
                    else:
                        LOGGER.warning(
                            "  [%d/%d] Failed: %s",
                            completed,
                            total,
                            future_to_url[future],
                        )
                except Exception:
                    LOGGER.exception(
                        "  [%d/%d] Unexpected error scraping %s",
                        completed,
                        total,
                        future_to_url[future],
                    )

        LOGGER.info(
            "Mapped %d unique names to %d unique Member IDs.",
            len(self.name_map),
            len(set(self.name_map.values())),
        )

        # Don't save per-chamber; let the caller save normalized after merging both chambers
        return members

    def fetch_member_by_url(self, url: str, chamber: str) -> Member | None:
        return self.scrape_details(url, chamber)

    def export_name_map(self, filename: str = "name_map.json") -> None:
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(self.name_map, handle, indent=2, sort_keys=True)

    def fetch_all_committees(
        self,
    ) -> tuple[
        list[Committee],
        dict[str, list[CommitteeMemberRole]],
        dict[str, list[str]],
    ]:
        """Fetch committees, rosters, and bills for both chambers.

        Uses a single ``committees.json`` where each committee entry includes
        its ``roster`` and ``bill_numbers`` inline.
        """
        # ── Try loading from unified committees.json cache/seed ──
        _warn_stale_cache("committees.json")
        committees_raw = load_cache("committees.json", seed_fallback=self.seed_fallback)

        if committees_raw is not None:
            committees: list[Committee] = []
            rosters: dict[str, list[CommitteeMemberRole]] = {}
            bills: dict[str, list[str]] = {}

            for d in committees_raw:
                committees.append(_committee_from_dict(d))
                code = d["code"]
                # Parse inline roster if present
                roster_raw = d.get("roster", [])
                if roster_raw:
                    rosters[code] = [_committee_member_role_from_dict(r) for r in roster_raw]
                # Parse inline bill_numbers if present
                bill_nums = d.get("bill_numbers", [])
                if bill_nums:
                    bills[code] = bill_nums

            LOGGER.info(
                "\u2705 Loaded %d committees from cache (rosters: %s, bills: %s).",
                len(committees),
                f"{len(rosters)} committees" if rosters else "none",
                f"{len(bills)} committees" if bills else "none",
            )
            return committees, rosters, bills

        # ── No cache -- scrape from ilga.gov ──
        LOGGER.info("\U0001f578\ufe0f Scraping committees...")
        senate_committees = self.fetch_committees_index("Senate")
        house_committees = self.fetch_committees_index("House")
        committees = senate_committees + house_committees

        rosters = self.fetch_committee_rosters(committees)
        bills = self.fetch_committee_bills(committees)

        self._save_unified_committee_cache(committees, rosters, bills)
        LOGGER.info("\u2705 Scraped and cached %d committees.", len(committees))
        return committees, rosters, bills

    def _save_unified_committee_cache(
        self,
        committees: list[Committee],
        rosters: dict[str, list[CommitteeMemberRole]],
        bills: dict[str, list[str]],
    ) -> None:
        """Save all committee data into a single committees.json (atomic write)."""
        data = []
        for c in committees:
            entry = asdict(c)
            entry["roster"] = [asdict(r) for r in rosters.get(c.code, [])]
            entry["bill_numbers"] = bills.get(c.code, [])
            data.append(entry)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / "committees.json"
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
        LOGGER.info("Saved %d committees to %s", len(data), path)

    def fetch_committees_index(self, chamber: str = "Senate") -> list[Committee]:
        cache_filename = f"{chamber.lower()}_committees.json"
        cached = load_cache(cache_filename, seed_fallback=self.seed_fallback)
        if cached is not None:
            committees = [
                Committee(
                    code=d["code"],
                    name=d["name"],
                    parent_code=d.get("parent_code"),
                    members_list_url=d.get("members_list_url"),
                )
                for d in cached
            ]
            LOGGER.info(
                "Loaded %d committees from cache (%s).",
                len(committees),
                cache_filename,
            )
            return committees

        list_url = urljoin(self.base_url, f"{chamber}/Committees")
        try:
            resp = self._throttled_get(list_url)
            resp.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.exception("Failed to fetch committees index from %s: %s", list_url, exc)
            return []

        committees = self._extract_committees_index(
            BeautifulSoup(resp.text, "html.parser"), list_url
        )
        save_cache(
            cache_filename,
            [
                {
                    "code": c.code,
                    "name": c.name,
                    "parent_code": c.parent_code,
                    "members_list_url": c.members_list_url,
                }
                for c in committees
            ],
        )
        return committees

    def fetch_committee_rosters(
        self, committees: Iterable[Committee]
    ) -> dict[str, list[CommitteeMemberRole]]:
        """Scrape rosters for all committees.

        No separate cache; stored in unified ``committees.json``.
        """
        items = [(c.code, c.members_list_url) for c in committees if c.members_list_url]
        rosters: dict[str, list[CommitteeMemberRole]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_code = {
                pool.submit(self._scrape_committee_roster, url): code for code, url in items
            }
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    roster = future.result()
                    if roster:
                        rosters[code] = roster
                except Exception:
                    LOGGER.exception("Unexpected error scraping roster for %s", code)
        return rosters

    def fetch_committee_bills(self, committees: Iterable[Committee]) -> dict[str, list[str]]:
        """Scrape the bills page for each committee.

        No separate cache; stored in unified ``committees.json``.

        Derives the bills URL from each committee's members_list_url by
        replacing '/MembersList/' (or '/Members/') with '/Bills/'.
        Returns a dict mapping committee code -> list of bill number strings.
        """
        items: list[tuple[str, str]] = []
        for c in committees:
            if not c.members_list_url:
                continue
            bills_url = c.members_list_url.replace("/MembersList/", "/Bills/").replace(
                "/Members/", "/Bills/"
            )
            items.append((c.code, bills_url))

        result: dict[str, list[str]] = {}

        def _scrape_one(code: str, url: str) -> tuple[str, list[str]]:
            try:
                resp = self._throttled_get(url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                return code, parse_committee_bill_table(soup)
            except requests.RequestException as exc:
                LOGGER.warning("Failed to fetch committee bills from %s: %s", url, exc)
                return code, []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_code = {pool.submit(_scrape_one, code, url): code for code, url in items}
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    code, bills = future.result()
                    if bills:
                        result[code] = bills
                except Exception:
                    LOGGER.exception(
                        "Unexpected error scraping bills for committee %s",
                        code,
                    )
        return result

    # ── detail scraping ──────────────────────────────────────────────────

    def scrape_details(self, url: str, chamber: str) -> Member | None:
        try:
            resp = self._throttled_get(url)
            resp.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.exception("Failed to fetch member detail from %s: %s", url, exc)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        text_lines = _extract_text_lines(soup)

        name = self._extract_name(soup) or "Unknown"
        party = self._extract_party(soup, text_lines) or "Unknown"
        district = self._extract_district(text_lines) or "Unknown"
        role, career_timeline_text = self._extract_role_timeline(soup, text_lines)
        bio_text = self._extract_bio_text(soup, text_lines)
        associated_members = self._extract_associated_members(soup, text_lines)
        offices, email = self._extract_offices_and_email(soup)
        career_ranges = self._parse_career_ranges(career_timeline_text)

        committees = self._extract_committees_table(soup)
        if not committees:
            committees = self._extract_committees_from_bio(bio_text)

        # ── Pass 1: All bills from the standard profile ──
        all_bills = parse_bill_table(soup)

        # ── Pass 2: Primary-only (sponsored) bills ──
        primary_url = url + ("&" if "?" in url else "?") + "Primary=True"
        sponsored_bills: list[Bill] = []
        try:
            primary_resp = self._throttled_get(primary_url)
            primary_resp.raise_for_status()
            primary_soup = BeautifulSoup(primary_resp.text, "html.parser")
            sponsored_bills = parse_bill_table(primary_soup)
        except requests.RequestException as exc:
            LOGGER.warning("Failed to fetch primary bills from %s: %s", primary_url, exc)

        # ── Derive co-sponsor bills (all bills minus sponsored) ──
        sponsored_leg_ids = {b.leg_id for b in sponsored_bills}
        co_sponsor_bills = [b for b in all_bills if b.leg_id not in sponsored_leg_ids]

        member = Member(
            id=_member_id_from_url(url),
            name=name,
            member_url=url,
            chamber=chamber,
            party=party,
            role=role,
            career_timeline_text=career_timeline_text,
            career_ranges=career_ranges,
            district=district,
            bio_text=bio_text,
            committees=committees,
            associated_members=associated_members,
            email=email,
            offices=offices,
            sponsored_bills=sponsored_bills,
            co_sponsor_bills=co_sponsor_bills,
        )
        self._update_name_map(member)
        return member

    # ── name handling ────────────────────────────────────────────────────

    def normalize_name(self, raw_name: str) -> str:
        if not raw_name:
            return ""
        cleaned = _RE_PARTY_LETTER.sub("", raw_name.strip())
        cleaned = _RE_TITLE_PREFIX.sub("", cleaned)
        cleaned = _RE_WHITESPACE.sub(" ", cleaned)
        cleaned = _RE_NAME_SUFFIX.sub("", cleaned)
        cleaned = cleaned.replace(".", " ").replace(",", " ")
        return _RE_WHITESPACE.sub(" ", cleaned).strip().lower()

    def _display_name_from_raw(self, raw_name: str) -> str:
        if not raw_name:
            return ""
        cleaned = _RE_TITLE_PREFIX.sub("", raw_name.strip())
        cleaned = _RE_PARTY_LETTER.sub("", cleaned)
        if "," in cleaned:
            return self.normalize_name(cleaned.split(",", 1)[0])
        return _last_name_from_normalized(self.normalize_name(cleaned))

    def _add_name_key(self, key: str, member_id: str) -> None:
        if not key:
            return
        existing = self.name_map.get(key)
        if existing and existing != member_id:
            return
        self.name_map[key] = member_id

    def _update_name_map(self, member: Member) -> None:
        if not member.name or not member.id:
            return
        with self._lock:
            normalized_full = self.normalize_name(member.name)
            self._add_name_key(normalized_full, member.id)

            last_name = _last_name_from_normalized(normalized_full)
            if last_name and last_name in self.name_map and self.name_map[last_name] != member.id:
                last_name = ""
            self._add_name_key(last_name, member.id)

            display_name = self._display_name_from_raw(member.name)
            if (
                display_name
                and display_name in self.name_map
                and self.name_map[display_name] != member.id
            ):
                return
            self._add_name_key(display_name, member.id)

    # ── extraction helpers ───────────────────────────────────────────────

    def _extract_member_urls(
        self, soup: BeautifulSoup, base_url: str, chamber: str
    ) -> Iterable[str]:
        chamber_segment = f"/{chamber}/Members/Details/"
        seen_ids: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if chamber_segment not in href:
                continue
            full_url = urljoin(base_url, href)
            mid = _member_id_from_url(full_url)
            if not mid or not mid.isdigit():
                continue
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            yield full_url

    def _extract_name(self, soup: BeautifulSoup) -> str | None:
        matched_header = self._find_name_party_header(soup)
        if matched_header:
            match = _RE_NAME_PARTY.search(matched_header)
            if match:
                return match.group("name").strip()
            return matched_header.split("-")[0].strip()
        heading = soup.find(["h1", "h2"])
        if heading and heading.get_text(strip=True):
            return heading.get_text(" ", strip=True).split("-")[0].strip()
        title = soup.find("title")
        if title and title.get_text(strip=True):
            return title.get_text(strip=True).split("|")[0].strip()
        return None

    def _extract_party(self, soup: BeautifulSoup, text_lines: list[str]) -> str | None:
        matched_header = self._find_name_party_header(soup)
        if matched_header:
            match = _RE_PARTY_CAPTURE.search(matched_header)
            if match:
                return {"R": "Republican", "D": "Democrat", "I": "Independent"}[match.group(1)]
        match = _RE_PARTY_FULL.search(" ".join(text_lines))
        return match.group(1).title() if match else None

    def _extract_district(self, text_lines: list[str]) -> str | None:
        match = _RE_DISTRICT.search(" ".join(text_lines))
        return match.group(1) if match else None

    def _extract_role_timeline(self, soup: BeautifulSoup, text_lines: list[str]) -> tuple[str, str]:
        header = self._find_name_party_header(soup)
        if not header:
            return "", ""
        start_index = None
        for idx, line in enumerate(text_lines):
            if line == header:
                start_index = idx + 1
                break
        if start_index is None:
            return "", ""
        role = ""
        career = ""
        for line in text_lines[start_index:]:
            if not line.strip():
                continue
            if _RE_PARTY_LETTER.search(line):
                continue
            if _RE_DISTRICT.search(line):
                break
            if not role:
                role = line
                continue
            if not career:
                career = line
                continue
        return role, career

    def _parse_career_ranges(self, text: str) -> list[CareerRange]:
        if not text:
            return []
        ranges: list[CareerRange] = []
        current_year = datetime.now().year
        for chunk in (part.strip() for part in text.split(";") if part.strip()):
            match = _RE_CAREER_RANGE.search(chunk)
            if not match:
                continue
            start_year = int(match.group("start"))
            end_raw = match.group("end")
            end_year = current_year if end_raw.lower() == "present" else int(end_raw)
            chamber_raw = match.group("chamber")
            chamber: str | None = None
            if chamber_raw:
                chamber_text = chamber_raw.strip().lower()
                if "house" in chamber_text:
                    chamber = "House"
                elif "senate" in chamber_text:
                    chamber = "Senate"
                else:
                    chamber = chamber_raw.strip()
            ranges.append(CareerRange(start_year=start_year, end_year=end_year, chamber=chamber))
        return ranges

    def _extract_bio_text(self, soup: BeautifulSoup, text_lines: list[str]) -> str:
        bio_section = soup.find(id=_RE_BIO_PANE)
        if bio_section:
            return bio_section.get_text(" ", strip=True)
        if "Biography" in text_lines:
            bio_index = text_lines.index("Biography")
            collected = _collect_section(text_lines, bio_index + 1, _SECTION_STOP_TOKENS)
            return " ".join(collected).strip()
        return " ".join(text_lines)

    def _extract_committees_table(self, soup: BeautifulSoup) -> list[str]:
        committee_pane = soup.find(id="pane-Committees")
        if not committee_pane:
            return []

        table = committee_pane.find("table")
        if not table:
            return []

        header_cells = [cell.get_text(" ", strip=True) for cell in table.find_all("th")]
        if any("Bill" in h for h in header_cells):
            return []

        committees: list[str] = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
            if cells and cells[0]:
                committees.append(cells[0])
        return list(dict.fromkeys(committees))

    def _extract_committees_index(self, soup: BeautifulSoup, base_url: str) -> list[Committee]:
        table = soup.find("table")
        if not table:
            return []

        committees: list[Committee] = []
        current_parent_code: str | None = None

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            name_text = cells[0].get_text(" ", strip=True)
            code_text = cells[1].get_text(" ", strip=True)
            if not code_text or code_text.lower() == "code":
                continue
            members_list_url = self._extract_committee_members_list_url(cells[0], base_url)

            is_subcommittee, cleaned_name = self._parse_committee_name(name_text)
            parent_code = current_parent_code if is_subcommittee else None
            if not is_subcommittee:
                current_parent_code = code_text

            committees.append(
                Committee(
                    code=code_text,
                    name=cleaned_name,
                    parent_code=parent_code,
                    members_list_url=members_list_url,
                )
            )

        return committees

    def _parse_committee_name(self, raw_name: str) -> tuple[bool, str]:
        cleaned = raw_name.strip()
        if cleaned.startswith("-"):
            return True, cleaned.lstrip("-").strip()
        return False, cleaned

    def _extract_committee_members_list_url(self, cell: Tag, base_url: str) -> str | None:
        anchor = cell.find("a", href=True)
        if not anchor:
            return None
        href = anchor["href"]
        if not href:
            return None
        return self._members_list_url(urljoin(base_url, href))

    def _members_list_url(self, committee_url: str) -> str:
        if "/MembersList/" in committee_url:
            return committee_url
        if "/Members/" in committee_url:
            return committee_url.replace("/Members/", "/MembersList/", 1)
        return committee_url

    def _scrape_committee_roster(self, members_list_url: str) -> list[CommitteeMemberRole]:
        try:
            resp = self._throttled_get(members_list_url)
            resp.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.exception(
                "Failed to fetch committee members list from %s: %s",
                members_list_url,
                exc,
            )
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        table = self._find_committee_roster_table(soup)
        if not table:
            return []

        roster: list[CommitteeMemberRole] = []
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            role = cells[0].get_text(" ", strip=True)
            members_cell = cells[1]
            links = members_cell.find_all("a", href=True)
            if not links:
                name_text = members_cell.get_text(" ", strip=True)
                if name_text:
                    roster.append(
                        CommitteeMemberRole(
                            member_id="",
                            member_name=name_text,
                            member_url="",
                            role=role or "Member",
                        )
                    )
                continue
            for link in links:
                member_url = urljoin(members_list_url, link["href"])
                name_text = link.get_text(" ", strip=True)
                if not name_text:
                    name_text = members_cell.get_text(" ", strip=True)
                roster.append(
                    CommitteeMemberRole(
                        member_id=_member_id_from_url(member_url),
                        member_name=name_text,
                        member_url=member_url,
                        role=role or "Member",
                    )
                )
        return roster

    def _find_committee_roster_table(self, soup: BeautifulSoup) -> Tag | None:
        for table in soup.find_all("table"):
            headers = [cell.get_text(" ", strip=True).lower() for cell in table.find_all("th")]
            if not headers:
                continue
            if any("role" in h for h in headers) and any("member" in h for h in headers):
                return table
        return None

    def _find_name_party_header(self, soup: BeautifulSoup) -> str | None:
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
            text = heading.get_text(" ", strip=True)
            if _RE_PARTY_LETTER.search(text):
                return text
        return None

    def _extract_committees_from_bio(self, bio_text: str) -> list[str]:
        return list(
            dict.fromkeys(
                m.group("committee").strip() for m in _RE_COMMITTEE_ROLE.finditer(bio_text)
            )
        )

    def _extract_associated_members(self, soup: BeautifulSoup, text_lines: list[str]) -> str | None:
        for heading in soup.find_all(["h3", "h4", "strong", "b"]):
            text = heading.get_text(" ", strip=True)
            if text in _ASSOCIATED_LABELS:
                parent = heading.find_parent()
                if parent is None:
                    continue
                links = [link.get_text(" ", strip=True) for link in parent.find_all("a")]
                if links:
                    return ", ".join(links)
        for idx, line in enumerate(text_lines):
            if line in _ASSOCIATED_LABELS:
                collected = _collect_section(text_lines, idx + 1, _SECTION_STOP_TOKENS)
                return ", ".join(collected) if collected else None
        return None

    def _extract_offices_and_email(self, soup: BeautifulSoup) -> tuple[list[Office], str | None]:
        container = self._find_contact_container(soup)
        if not container:
            return [], None

        all_text = " ".join(container.stripped_strings)
        email_match = _RE_EMAIL.search(all_text)
        email = email_match.group(0) if email_match else None

        offices = self._extract_offices_from_rows(container)
        if not offices:
            labels = {
                "Springfield Office:",
                "District Office:",
                "Other Contact Info:",
            }
            blocks = self._collect_label_blocks(container, labels)
            offices = [
                office
                for label, lines in blocks.items()
                if (office := self._parse_office_block(label.replace(":", ""), lines))
            ]
        return offices, email

    def _find_contact_container(self, soup: BeautifulSoup) -> Tag | None:
        contact_pane = soup.find(id=_RE_CONTACT_PANE)
        if contact_pane:
            return contact_pane
        member_info = soup.find("div", class_=_RE_MEMBER_INFO_COL)
        if member_info:
            card_body = member_info.find("div", class_=_RE_CARD_BODY)
            if card_body and "Office" in card_body.get_text(" ", strip=True):
                return card_body
        card_body = soup.find("div", class_=_RE_CARD_BODY)
        if card_body and "Office" in card_body.get_text(" ", strip=True):
            return card_body
        contact_heading = soup.find(string=_RE_CONTACT_HEADING)
        if contact_heading:
            parent = contact_heading.parent
            if parent is not None and parent.parent is not None:
                return parent.parent
            if parent is not None:
                return parent
        return None

    def _extract_offices_from_rows(self, container: Tag) -> list[Office]:
        offices: list[Office] = []
        for row in container.find_all("div", class_=_RE_ROW_CLASS):
            label_div = row.find("div", class_=_RE_COL_SM_4)
            value_div = row.find("div", class_=_RE_COL_SM_8)
            if not label_div or not value_div:
                continue
            label_text = label_div.get_text(" ", strip=True)
            if not label_text.endswith("Office:") and "Office" not in label_text:
                continue
            raw_lines = [line.strip() for line in value_div.stripped_strings if line.strip()]
            if not raw_lines:
                continue
            office = self._parse_office_block(label_text.replace(":", ""), raw_lines)
            if office:
                offices.append(office)
        return offices

    def _collect_label_blocks(self, container: Tag, labels: set[str]) -> dict[str, list[str]]:
        blocks: dict[str, list[str]] = {label: [] for label in labels}
        current_label: str | None = None

        for node in container.descendants:
            if isinstance(node, Tag) and node.name in ("b", "strong"):
                label_text = node.get_text(" ", strip=True)
                if label_text in labels:
                    current_label = label_text
                continue

            if not current_label:
                continue

            if isinstance(node, Tag) and node.name == "br":
                blocks[current_label].append("\n")
                continue

            if isinstance(node, NavigableString):
                if node.parent and node.parent.name in ("b", "strong"):
                    continue
                text = str(node)
                if not text or text.strip() in labels:
                    continue
                blocks[current_label].append(text)

        cleaned: dict[str, list[str]] = {}
        for label, parts in blocks.items():
            if not parts:
                continue
            lines = [line.strip() for line in "".join(parts).splitlines() if line.strip()]
            if lines:
                cleaned[label] = lines
        return cleaned

    def _parse_office_block(self, name: str, lines: list[str]) -> Office | None:
        address_lines: list[str] = []
        phones: list[str] = []
        faxes: list[str] = []
        for line in lines:
            faxes.extend(f.strip() for f in _RE_FAX.findall(line))
            phones.extend(p.strip() for p in _RE_PHONE.findall(line))
            stripped = _RE_EMAIL.sub("", line)
            stripped = _RE_FAX.sub("", stripped)
            stripped = _RE_PHONE.sub("", stripped).strip()
            if stripped:
                address_lines.append(stripped)
        fax = sorted(set(faxes))[0] if faxes else None
        deduped_phones = [p for p in sorted(set(phones)) if p != fax] if phones else []
        phone = deduped_phones[0] if deduped_phones else None
        address = "\n".join(address_lines).strip()
        if not any([address, phone, fax]):
            return None
        return Office(name=name, address=address, phone=phone, fax=fax)
