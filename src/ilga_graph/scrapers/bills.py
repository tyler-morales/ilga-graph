"""Bill-first scraper: legislation index + BillStatus pages.

Scrapes the ILGA legislation range pages to build a bill index,
then fetches individual BillStatus pages for full detail (sponsors,
last action, synopsis, action history).
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import BASE_URL, CACHE_DIR, GA_ID, MOCK_DEV_DIR, SESSION_ID
from ..models import ActionEntry, Bill

LOGGER = logging.getLogger(__name__)

BILLS_CACHE_FILE = CACHE_DIR / "bills.json"
METADATA_FILE = CACHE_DIR / "scrape_metadata.json"

_RE_LEG_ID = re.compile(r"LegId=(\d+)", re.IGNORECASE)
_RE_DOC_NUM = re.compile(r"DocNum=(\d+)", re.IGNORECASE)
_RE_DOC_TYPE = re.compile(r"DocTypeID=(\w+)", re.IGNORECASE)
_RE_MEMBER_ID = re.compile(r"/Members/Details/(\d+)", re.IGNORECASE)


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class BillIndexEntry:
    """Lightweight bill entry from a range page (no detail)."""

    bill_number: str  # e.g. "SB0001"
    leg_id: str  # e.g. "157091"
    description: str  # e.g. "$GEN ASSEMBLY-TECH"
    doc_type: str  # "SB" or "HB"
    status_url: str  # full BillStatus URL


# ── Session builder ──────────────────────────────────────────────────────────


def _build_session(request_delay: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ── Range page scraping (bill index) ─────────────────────────────────────────


def _range_url(doc_type: str, num1: int, num2: int) -> str:
    """Build legislation range page URL using config GA_ID/SESSION_ID."""
    return (
        f"{BASE_URL}Legislation/RegularSession/{doc_type}"
        f"?num1={num1:04d}&num2={num2:04d}"
        f"&DocTypeID={doc_type}&GaId={GA_ID}&SessionId={SESSION_ID}"
    )


def _parse_range_page(html: str, doc_type: str) -> list[BillIndexEntry]:
    """Parse a bill range page table into BillIndexEntry objects."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="table")
    if not table:
        return []

    entries: list[BillIndexEntry] = []
    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        link = cells[0].find("a", href=True)
        if not link:
            continue
        href = link["href"]
        leg_id_match = _RE_LEG_ID.search(href)
        if not leg_id_match:
            continue

        bill_number = link.get_text(strip=True).replace("\xa0", "").strip()
        description = cells[1].get_text(strip=True)
        leg_id = leg_id_match.group(1)
        status_url = urljoin(BASE_URL, href)

        entries.append(
            BillIndexEntry(
                bill_number=bill_number,
                leg_id=leg_id,
                description=description,
                doc_type=doc_type,
                status_url=status_url,
            )
        )
    return entries


# Each legislation range page returns up to 100 bills (e.g. num1=1, num2=100).
_RANGE_PAGE_SIZE = 100


def scrape_bill_index(
    doc_type: str = "SB",
    limit: int = 0,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> list[BillIndexEntry]:
    """Scrape the legislation index for a doc type (SB or HB).

    Paginates through range pages (0001-0100, 0101-0200, ...) until we have
    enough entries or a page returns no bills. With ``limit > 0``, returns at
    most *limit* entries. With ``limit == 0``, fetches all range pages until
    empty (full index, ~4800 SB + ~4800 HB per session).
    """
    sess = session or _build_session()
    entries: list[BillIndexEntry] = []
    num1 = 1

    while True:
        num2 = num1 + _RANGE_PAGE_SIZE - 1
        url = _range_url(doc_type, num1, num2)
        LOGGER.info("Fetching bill index: %s (range %d-%d)", doc_type, num1, num2)
        resp = sess.get(url, timeout=timeout)
        resp.raise_for_status()
        time.sleep(request_delay)

        page_entries = _parse_range_page(resp.text, doc_type)
        if not page_entries:
            LOGGER.info("No more %s bills at range %d-%d; stopping.", doc_type, num1, num2)
            break

        entries.extend(page_entries)
        LOGGER.info("Parsed %d %s from range %d-%d (total %s: %d).", len(page_entries), doc_type, num1, num2, doc_type, len(entries))

        if limit > 0 and len(entries) >= limit:
            entries = entries[:limit]
            break
        if len(page_entries) < _RANGE_PAGE_SIZE:
            # Last page was partial; no more pages
            break
        num1 = num2 + 1

    return entries


def scrape_all_bill_indexes(
    sb_limit: int = 100,
    hb_limit: int = 100,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> list[BillIndexEntry]:
    """Scrape bill indexes for both SB and HB, returning combined list."""
    sess = session or _build_session()
    sb = scrape_bill_index(
        "SB", limit=sb_limit, session=sess, timeout=timeout, request_delay=request_delay
    )
    hb = scrape_bill_index(
        "HB", limit=hb_limit, session=sess, timeout=timeout, request_delay=request_delay
    )
    LOGGER.info("Total bill index: %d SB + %d HB = %d", len(sb), len(hb), len(sb) + len(hb))
    return sb + hb


# ── BillStatus page scraping ─────────────────────────────────────────────────


def _parse_last_action(soup: BeautifulSoup) -> tuple[str, str, str]:
    """Extract last action date, chamber, and action text from BillStatus page.

    Returns (last_action_date, chamber, last_action_text).
    """
    # Find the "Last Action" header
    last_action_h5 = soup.find("h5", string=re.compile(r"Last Action", re.IGNORECASE))
    if not last_action_h5:
        return ("", "", "")

    # The action is in a sibling .list-group -> .list-group-item
    list_group = last_action_h5.find_next("div", class_="list-group")
    if not list_group:
        return ("", "", "")

    item = list_group.find("span", class_="list-group-item")
    if not item:
        return ("", "", "")

    # The bold span has "date - chamber:"
    bold_span = item.find("span", class_="fw-bold")
    if not bold_span:
        return ("", "", "")

    bold_text = bold_span.get_text(strip=True)
    # Parse "7/28/2025 - Senate:" or "7/28/2025 -\nSenate:"
    parts = re.split(r"\s*-\s*", bold_text, maxsplit=1)
    date_str = parts[0].strip() if parts else ""
    chamber = parts[1].rstrip(":").strip() if len(parts) > 1 else ""

    # Action text is everything after the bold span
    action_text = item.get_text(strip=True)
    # Remove the bold prefix
    if bold_text and action_text.startswith(bold_text):
        action_text = action_text[len(bold_text) :].strip()
    # Normalize internal whitespace to single space (e.g. "To  Ethics" -> "To Ethics")
    action_text = re.sub(r"\s+", " ", action_text).strip()

    return (date_str, chamber, action_text)


def _parse_sponsors(soup: BeautifulSoup) -> tuple[str, list[str], list[str]]:
    """Extract primary sponsor name and sponsor member IDs from #sponsorDiv.

    Returns (primary_sponsor_name, senate_sponsor_ids, house_sponsor_ids).
    """
    sponsor_div = soup.find("div", id="sponsorDiv")
    if not sponsor_div:
        return ("", [], [])

    senate_ids: list[str] = []
    house_ids: list[str] = []
    primary_name = ""

    # Find all sponsor links
    all_links = sponsor_div.find_all("a", href=_RE_MEMBER_ID)

    # Determine which section each link belongs to by looking for h5 headers
    house_h5 = sponsor_div.find("h5", string=re.compile(r"House Sponsors", re.IGNORECASE))
    house_start_pos = house_h5.sourceline if house_h5 else float("inf")

    for link in all_links:
        mid_match = _RE_MEMBER_ID.search(link["href"])
        if not mid_match:
            continue
        member_id = mid_match.group(1)
        link_pos = link.sourceline if link.sourceline else 0

        if link_pos >= house_start_pos:
            house_ids.append(member_id)
        else:
            senate_ids.append(member_id)

    # Primary sponsor is the first link in the div
    if all_links:
        primary_name = all_links[0].get_text(strip=True)

    return (primary_name, senate_ids, house_ids)


def _parse_synopsis(soup: BeautifulSoup) -> str:
    """Extract the synopsis text from the BillStatus page."""
    synopsis_h5 = soup.find("h5", string=re.compile(r"Synopsis As Introduced", re.IGNORECASE))
    if not synopsis_h5:
        return ""
    list_group = synopsis_h5.find_next("div", class_="list-group")
    if not list_group:
        return ""
    item = list_group.find("span", class_="list-group-item")
    if not item:
        return ""
    return item.get_text(strip=True)


def _parse_action_history(soup: BeautifulSoup) -> list[ActionEntry]:
    """Parse the Actions table from a BillStatus page."""
    actions_h5 = soup.find("h5", string=re.compile(r"^Actions$", re.IGNORECASE))
    if not actions_h5:
        return []
    table = actions_h5.find_next("table", class_="table")
    if not table:
        return []

    entries: list[ActionEntry] = []
    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        date = cells[0].get_text(strip=True)
        chamber = cells[1].get_text(strip=True)
        action = cells[2].get_text(strip=True)
        if date:
            entries.append(ActionEntry(date=date, chamber=chamber, action=action))
    return entries


def scrape_bill_status(
    bill_status_url: str,
    index_entry: BillIndexEntry | None = None,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> Bill | None:
    """Scrape a single BillStatus page and return a full Bill object."""
    sess = session or _build_session()

    try:
        resp = sess.get(bill_status_url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch BillStatus %s: %s", bill_status_url, exc)
        return None

    time.sleep(request_delay)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract bill_number and leg_id from URL if not provided
    bill_number = ""
    leg_id = ""
    doc_type = ""
    if index_entry:
        bill_number = index_entry.bill_number
        leg_id = index_entry.leg_id
        doc_type = index_entry.doc_type
    else:
        lid = _RE_LEG_ID.search(bill_status_url)
        dnum = _RE_DOC_NUM.search(bill_status_url)
        dtype = _RE_DOC_TYPE.search(bill_status_url)
        leg_id = lid.group(1) if lid else ""
        doc_type = dtype.group(1) if dtype else ""
        if dnum and dtype:
            bill_number = f"{dtype.group(1)}{int(dnum.group(1)):04d}"

    if not leg_id:
        LOGGER.warning("No leg_id for BillStatus URL, skipping: %s", bill_status_url)
        return None

    # Title/description from h5
    title_h5 = soup.find("h5", class_="fw-bold")
    description = ""
    if index_entry:
        description = index_entry.description
    elif title_h5:
        description = title_h5.get_text(strip=True)

    # Last action
    last_action_date, _la_chamber, last_action = _parse_last_action(soup)

    # Sponsors
    primary_sponsor, senate_ids, house_ids = _parse_sponsors(soup)

    # Chamber from doc_type
    chamber = "S" if doc_type.upper().startswith("S") else "H"

    # Synopsis
    synopsis = _parse_synopsis(soup)

    # Action history
    action_history = _parse_action_history(soup)

    return Bill(
        bill_number=bill_number,
        leg_id=leg_id,
        description=description,
        chamber=chamber,
        last_action=last_action,
        last_action_date=last_action_date,
        primary_sponsor=primary_sponsor,
        synopsis=synopsis,
        status_url=bill_status_url,
        sponsor_ids=senate_ids,
        house_sponsor_ids=house_ids,
        action_history=action_history,
    )


def scrape_all_bills(
    index: list[BillIndexEntry],
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
    max_workers: int = 3,
    use_cache: bool = True,
    seed_fallback: bool = False,
    checkpoint_interval: int = 50,
) -> dict[str, Bill]:
    """Scrape BillStatus pages for all bills in the index.

    Returns ``{leg_id: Bill}`` dict.

    Parameters
    ----------
    checkpoint_interval:
        Save a checkpoint to disk every *N* completed bills so that progress
        is not lost if the scrape is interrupted mid-run.  Set to 0 to
        disable checkpointing.
    """
    # Try cache first
    if use_cache:
        cached = load_bill_cache(seed_fallback=seed_fallback)
        if cached is not None:
            return cached

    sess = session or _build_session()
    total = len(index)
    LOGGER.info("Scraping %d BillStatus pages...", total)

    bills: dict[str, Bill] = {}
    completed = 0
    last_checkpoint = 0
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_entry = {
            pool.submit(
                scrape_bill_status,
                entry.status_url,
                entry,
                sess,
                timeout,
                request_delay,
            ): entry
            for entry in index
        }
        for future in as_completed(future_to_entry):
            completed += 1
            entry = future_to_entry[future]
            elapsed = time.perf_counter() - t_start
            try:
                bill = future.result()
                if bill:
                    bills[bill.leg_id] = bill
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    if completed % 20 == 0 or completed == total:
                        LOGGER.info(
                            "  [%d/%d] %.0fs elapsed, ~%.0fs remaining",
                            completed,
                            total,
                            elapsed,
                            eta,
                        )
                else:
                    LOGGER.warning("  [%d/%d] Failed: %s", completed, total, entry.bill_number)
            except Exception:
                LOGGER.exception("  [%d/%d] Error scraping %s", completed, total, entry.bill_number)

            # ── Checkpoint: persist progress every N bills ──
            if (
                checkpoint_interval > 0
                and completed - last_checkpoint >= checkpoint_interval
                and bills
            ):
                LOGGER.info(
                    "  Checkpoint: saving %d bills at %d/%d...", len(bills), completed, total
                )
                save_bill_cache(bills)
                last_checkpoint = completed

    elapsed_total = time.perf_counter() - t_start
    LOGGER.info("Bill scraping complete: %d bills in %.1fs", len(bills), elapsed_total)

    # Save final cache
    save_bill_cache(bills)
    save_scrape_metadata(len(bills))

    return bills


# ── Cache helpers ─────────────────────────────────────────────────────────────


def _bill_to_dict(bill: Bill) -> dict:
    """Serialize a Bill to a JSON-safe dict."""
    d = asdict(bill)
    # action_history is a list of ActionEntry dataclasses -- already dicts via asdict
    return d


def _bill_from_dict(d: dict) -> Bill:
    """Deserialize a Bill from a cache dict."""
    action_history = []
    for a in d.get("action_history", []):
        if isinstance(a, dict) and "date" in a and "chamber" in a and "action" in a:
            action_history.append(
                ActionEntry(date=a["date"], chamber=a["chamber"], action=a["action"])
            )
    return Bill(
        bill_number=d["bill_number"],
        leg_id=d["leg_id"],
        description=d["description"],
        chamber=d["chamber"],
        last_action=d.get("last_action", ""),
        last_action_date=d.get("last_action_date", ""),
        primary_sponsor=d.get("primary_sponsor", ""),
        synopsis=d.get("synopsis", ""),
        status_url=d.get("status_url", ""),
        sponsor_ids=d.get("sponsor_ids", []),
        house_sponsor_ids=d.get("house_sponsor_ids", []),
        action_history=action_history,
    )


def save_bill_cache(bills: dict[str, Bill]) -> None:
    """Save bills dict to cache/bills.json."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {lid: _bill_to_dict(b) for lid, b in bills.items()}
    with open(BILLS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved %d bills to %s", len(data), BILLS_CACHE_FILE)


def load_bill_cache(*, seed_fallback: bool = False) -> dict[str, Bill] | None:
    """Load bills from cache/bills.json (or mocks/dev/ if seed_fallback)."""
    path = BILLS_CACHE_FILE
    if not path.exists():
        if seed_fallback:
            seed_path = MOCK_DEV_DIR / "bills.json"
            if seed_path.exists():
                LOGGER.info("Loading bill cache from seed: %s", seed_path)
                path = seed_path
            else:
                return None
        else:
            return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    bills = {lid: _bill_from_dict(d) for lid, d in raw.items()}
    LOGGER.info("Loaded %d bills from cache (%s).", len(bills), path)
    return bills


def save_scrape_metadata(bill_count: int) -> None:
    """Save scrape metadata (timestamps, counts)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            existing = json.load(f)

    existing["last_bill_scrape_at"] = datetime.now(timezone.utc).isoformat()
    existing["bill_index_count"] = bill_count

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    LOGGER.info("Updated scrape metadata: %s", METADATA_FILE)


def load_scrape_metadata() -> dict:
    """Load scrape metadata, or empty dict if missing."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Incremental scraping ─────────────────────────────────────────────────────


def incremental_bill_scrape(
    sb_limit: int = 100,
    hb_limit: int = 100,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
    max_workers: int = 3,
    rescrape_recent_days: int = 30,
) -> dict[str, Bill]:
    """Incremental scrape: fetch index, compare to cache, only scrape changes.

    1. Load existing bill cache
    2. Scrape bill indexes (range pages) to get current leg_ids
    3. Identify new bills (not in cache)
    4. Optionally re-scrape bills with recent last_action_date
    5. Merge results into cache
    """
    sess = session or _build_session()

    # Load existing cache
    existing = load_bill_cache() or {}
    existing_ids = set(existing.keys())

    # Scrape fresh index
    index = scrape_all_bill_indexes(
        sb_limit=sb_limit,
        hb_limit=hb_limit,
        session=sess,
        timeout=timeout,
        request_delay=request_delay,
    )
    fresh_ids = {e.leg_id for e in index}

    # Find new bills
    new_ids = fresh_ids - existing_ids
    LOGGER.info(
        "Incremental: %d in index, %d in cache, %d new.",
        len(fresh_ids),
        len(existing_ids),
        len(new_ids),
    )

    # Also find recently-active bills to re-check
    rescrape_ids: set[str] = set()
    if rescrape_recent_days > 0:
        cutoff = datetime.now()
        for lid, bill in existing.items():
            if lid in fresh_ids and bill.last_action_date:
                try:
                    parsed = datetime.strptime(bill.last_action_date, "%m/%d/%Y")
                    age_days = (cutoff - parsed).days
                    if age_days <= rescrape_recent_days:
                        rescrape_ids.add(lid)
                except ValueError:
                    pass
        LOGGER.info(
            "Incremental: %d existing bills with activity in last %d days to re-check.",
            len(rescrape_ids),
            rescrape_recent_days,
        )

    # Bills to scrape = new + recently active
    to_scrape_ids = new_ids | rescrape_ids
    to_scrape = [e for e in index if e.leg_id in to_scrape_ids]

    if not to_scrape:
        LOGGER.info("Incremental: nothing to scrape, cache is up to date.")
        return existing

    LOGGER.info("Incremental: scraping %d BillStatus pages...", len(to_scrape))

    # Scrape only the delta
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_entry = {
            pool.submit(
                scrape_bill_status,
                entry.status_url,
                entry,
                sess,
                timeout,
                request_delay,
            ): entry
            for entry in to_scrape
        }
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                bill = future.result()
                if bill:
                    existing[bill.leg_id] = bill
            except Exception:
                LOGGER.exception("Error scraping %s", entry.bill_number)

    elapsed = time.perf_counter() - t_start
    LOGGER.info(
        "Incremental scrape complete: %d bills updated in %.1fs. Total: %d.",
        len(to_scrape),
        elapsed,
        len(existing),
    )

    # Save merged cache
    save_bill_cache(existing)
    save_scrape_metadata(len(existing))

    return existing
