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

from ..config import BASE_URL, CACHE_DIR, GA_ID, SESSION_ID
from ..data_source import get_data_dir
from ..models import ActionEntry, Bill, VoteEvent, WitnessSlip
from ..normalize import normalize_chamber, normalize_date, validate_bill_cache
from ._log import fmt_duration, log_phase, log_progress
from .hearings import (
    hearings_to_bill_numbers,
    save_hearings_cache,
    scrape_hearing_schedules,
)

LOGGER = logging.getLogger(__name__)

BILLS_CACHE_FILE = CACHE_DIR / "bills.json"
FULLTEXT_DIR = CACHE_DIR / "fulltext"
METADATA_FILE = CACHE_DIR / "scrape_metadata.json"
BILL_INDEX_CHECKPOINT_FILE = CACHE_DIR / "bill_index_checkpoint.json"

_RE_LEG_ID = re.compile(r"LegId=(\d+)", re.IGNORECASE)
_RE_DOC_NUM = re.compile(r"DocNum=(\d+)", re.IGNORECASE)
_RE_DOC_TYPE = re.compile(r"DocTypeID=(\w+)", re.IGNORECASE)
_RE_MEMBER_ID = re.compile(r"/Members/Details/(\d+)", re.IGNORECASE)


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class BillIndexEntry:
    """Lightweight bill entry from a range page (no detail)."""

    bill_number: str  # e.g. "SB0001", "SR0042", "HJR0003"
    leg_id: str  # e.g. "157091"
    description: str  # e.g. "$GEN ASSEMBLY-TECH"
    doc_type: str  # "SB", "HB", "SR", "HR", "SJR", "HJR", "SJRCA", "HJRCA", "EO", "JSR", "AM"
    status_url: str  # full BillStatus URL


@dataclass
class DocTypeInfo:
    """A document type discovered from the /Legislation index page."""

    doc_type: str  # e.g. "SB", "HB", "SR", "AM"
    label: str  # e.g. "Senate Bills", "Appointment Messages"
    range_urls: list[tuple[int, int, str]]  # [(num1, num2, full_url), ...]


# ── Session builder ──────────────────────────────────────────────────────────


def _build_session(
    request_delay: float = 0.5,
    pool_size: int = 10,
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
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


_RE_SESSION_HREF = re.compile(r"/Legislation/RegularSession/(\w+)\?SessionId=")


def _discover_doc_types(
    session: requests.Session,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> list[DocTypeInfo]:
    """Fetch /Legislation and discover all doc types with their range URLs.

    The ILGA Legislation page lists every document type (SB, HB, SR, HR,
    SJR, HJR, SJRCA, HJRCA, EO, JSR, AM) with clickable range links
    (e.g. "0001 - 0100", "4001 - 4052").  Parsing this page gives us the
    exact ranges so we never need blind pagination or recycled-data guards.
    """
    url = f"{BASE_URL}Legislation"
    LOGGER.info("Discovering doc types from %s ...", url)

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        time.sleep(request_delay)
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch %s: %s", url, exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    doc_types: list[DocTypeInfo] = []

    # Each section header is an <a> like:
    #   <a class="btn btn-light fw-bold p-1"
    #      href="/Legislation/RegularSession/SB?SessionId=114">
    #     Senate Bills
    #   </a>
    # Followed by a sibling <div class="row"> containing the range links.
    for header_link in soup.find_all("a", href=_RE_SESSION_HREF):
        label = header_link.get_text(strip=True)
        href = header_link.get("href", "")

        m = _RE_SESSION_HREF.search(href)
        if not m:
            continue
        doc_type = m.group(1)

        # Walk up to the parent <div class="mt-1 ..."> then find the
        # next sibling <div class="row"> that holds the range links.
        parent = header_link.find_parent("div")
        if not parent:
            continue
        row_div = parent.find_next_sibling("div", class_="row")
        if not row_div:
            continue

        ranges: list[tuple[int, int, str]] = []
        for link in row_div.find_all("a", href=True):
            rh = link["href"]
            n1 = re.search(r"num1=(\d+)", rh)
            n2 = re.search(r"num2=(\d+)", rh)
            if n1 and n2:
                ranges.append(
                    (
                        int(n1.group(1)),
                        int(n2.group(1)),
                        urljoin(BASE_URL, rh),
                    )
                )

        if ranges:
            doc_types.append(DocTypeInfo(doc_type=doc_type, label=label, range_urls=ranges))
            LOGGER.debug(
                "  %s (%s): %d range pages, %d–%d",
                doc_type,
                label,
                len(ranges),
                ranges[0][0],
                ranges[-1][1],
            )

    LOGGER.info(
        "Discovered %d doc types: %s",
        len(doc_types),
        ", ".join(f"{d.doc_type}({len(d.range_urls)}pg)" for d in doc_types),
    )
    return doc_types


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


# ── Index checkpoint helpers ─────────────────────────────────────────────────


def _save_index_checkpoint(index_data: dict[str, list[BillIndexEntry]]) -> None:
    """Save bill index checkpoint to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Convert BillIndexEntry objects to dicts
    serializable = {
        doc_type: [asdict(entry) for entry in entries] for doc_type, entries in index_data.items()
    }

    with open(BILL_INDEX_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def _load_index_checkpoint() -> dict[str, list[BillIndexEntry]]:
    """Load bill index checkpoint from disk, or return empty dict."""
    if not BILL_INDEX_CHECKPOINT_FILE.exists():
        return {}

    try:
        with open(BILL_INDEX_CHECKPOINT_FILE, encoding="utf-8") as f:
            raw = json.load(f)

        # Convert dicts back to BillIndexEntry objects
        return {
            doc_type: [BillIndexEntry(**entry_dict) for entry_dict in entries]
            for doc_type, entries in raw.items()
        }
    except Exception as e:
        LOGGER.warning("Failed to load index checkpoint: %s", e)
        return {}


def _clear_index_checkpoint() -> None:
    """Remove the index checkpoint file."""
    if BILL_INDEX_CHECKPOINT_FILE.exists():
        BILL_INDEX_CHECKPOINT_FILE.unlink()
        LOGGER.info("Cleared bill index checkpoint")


def scrape_bill_index(
    doc_type: str = "SB",
    limit: int = 0,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
    save_checkpoints: bool = True,
    range_urls: list[tuple[int, int, str]] | None = None,
) -> list[BillIndexEntry]:
    """Scrape the legislation index for a single doc type.

    If *range_urls* is provided (from ``_discover_doc_types``), fetches
    those exact URLs — no guessing, no recycled-data issues.  Otherwise
    falls back to blind range pagination starting at ``num1=1`` with a
    recycled-data guard.

    With ``limit > 0``, returns at most *limit* entries.
    With ``limit == 0``, fetches all entries.
    """
    sess = session or _build_session()
    entries: list[BillIndexEntry] = []

    if range_urls:
        # ── Known ranges from /Legislation discovery ──
        total_pages = len(range_urls)
        for page_idx, (num1, num2, url) in enumerate(range_urls, 1):
            LOGGER.info(
                "Fetching %s %d–%d (page %d/%d)",
                doc_type,
                num1,
                num2,
                page_idx,
                total_pages,
            )

            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
            time.sleep(request_delay)

            page_entries = _parse_range_page(resp.text, doc_type)
            if not page_entries:
                LOGGER.info("  (empty page)")
                continue

            entries.extend(page_entries)
            LOGGER.info(
                "  ✓ %d entries: %s to %s (total %s: %d)",
                len(page_entries),
                page_entries[0].bill_number,
                page_entries[-1].bill_number,
                doc_type,
                len(entries),
            )

            if save_checkpoints:
                existing_checkpoint = _load_index_checkpoint()
                existing_checkpoint[doc_type] = entries
                _save_index_checkpoint(existing_checkpoint)

            if limit > 0 and len(entries) >= limit:
                entries = entries[:limit]
                break
    else:
        # ── Fallback: blind range-page pagination ──
        num1 = 1
        page_num = 1

        while True:
            num2 = num1 + _RANGE_PAGE_SIZE - 1
            url = _range_url(doc_type, num1, num2)

            LOGGER.info(
                "Fetching bill index: %s %04d-%04d (page %d)",
                doc_type,
                num1,
                num2,
                page_num,
            )

            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
            time.sleep(request_delay)

            page_entries = _parse_range_page(resp.text, doc_type)
            if not page_entries:
                LOGGER.info("No more %s entries found; stopping.", doc_type)
                break

            # Recycled-data guard (fallback path only)
            first_bill_num_match = re.search(r"(\d+)$", page_entries[0].bill_number)
            if first_bill_num_match:
                first_bill_num = int(first_bill_num_match.group(1))
                if first_bill_num < num1:
                    LOGGER.info(
                        "Range page %s %04d-%04d returned %s (outside requested "
                        "range); end of %s index reached.",
                        doc_type,
                        num1,
                        num2,
                        page_entries[0].bill_number,
                        doc_type,
                    )
                    break

            entries.extend(page_entries)
            LOGGER.info(
                "  ✓ %d entries: %s to %s (total %s: %d)",
                len(page_entries),
                page_entries[0].bill_number,
                page_entries[-1].bill_number,
                doc_type,
                len(entries),
            )

            if save_checkpoints:
                existing_checkpoint = _load_index_checkpoint()
                existing_checkpoint[doc_type] = entries
                _save_index_checkpoint(existing_checkpoint)

            if limit > 0 and len(entries) >= limit:
                entries = entries[:limit]
                break
            if len(page_entries) < _RANGE_PAGE_SIZE:
                break
            num1 = num2 + 1
            page_num += 1

    return entries


def _checkpoint_looks_complete(
    existing: list[BillIndexEntry],
    limit: int,
) -> bool:
    """Return True if an index checkpoint has enough entries for *limit*.

    When ``limit == 0`` (all bills) we can't know the true total, so we
    use a heuristic: if the checkpoint has more entries than the default
    page size (100) it's likely a genuine partial-or-complete scrape.
    If it has *exactly* ``_RANGE_PAGE_SIZE`` entries, it was almost
    certainly produced by the (now-fixed) single-page truncation bug
    and should be discarded.
    """
    if not existing:
        return False
    if limit > 0:
        return len(existing) >= limit
    # limit == 0 → want ALL bills
    # Accept checkpoint only if it has more than one page's worth
    return len(existing) > _RANGE_PAGE_SIZE


def scrape_all_bill_indexes(
    sb_limit: int = 100,
    hb_limit: int = 100,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
    save_checkpoints: bool = True,
) -> list[BillIndexEntry]:
    """Discover ALL doc types from /Legislation and scrape their indexes.

    Fetches the ILGA Legislation page once to discover every document type
    (SB, HB, SR, HR, SJR, HJR, SJRCA, HJRCA, EO, JSR, AM) and their
    exact range pages.  Then scrapes each range page.

    ``sb_limit`` / ``hb_limit`` control SB and HB counts (0 = all).
    All other types are always scraped in full (they're small).

    Parameters
    ----------
    save_checkpoints:
        If True, saves progress to checkpoint file as scraping proceeds.
        If interrupted and rerun, will resume from checkpoint.
    """
    sess = session or _build_session()

    # ── Step 1: Discover all doc types and their exact ranges ──
    doc_types = _discover_doc_types(sess, timeout, request_delay)

    if not doc_types:
        LOGGER.warning(
            "Could not discover doc types from /Legislation page. "
            "Falling back to SB + HB with blind pagination."
        )
        doc_types = [
            DocTypeInfo("SB", "Senate Bills", []),
            DocTypeInfo("HB", "House Bills", []),
        ]

    # ── Step 2: Per-type limits (SB/HB respect CLI flags; others = all) ──
    limits: dict[str, int] = {}
    for dt in doc_types:
        if dt.doc_type == "SB":
            limits[dt.doc_type] = sb_limit
        elif dt.doc_type == "HB":
            limits[dt.doc_type] = hb_limit
        else:
            limits[dt.doc_type] = 0  # 0 = all (small types)

    # ── Step 3: Scrape each type (with checkpoint resume) ──
    checkpoint = _load_index_checkpoint()
    all_entries: list[BillIndexEntry] = []

    for dt in doc_types:
        limit = limits.get(dt.doc_type, 0)
        existing = checkpoint.get(dt.doc_type, [])

        if _checkpoint_looks_complete(existing, limit):
            LOGGER.info(
                "✓ %s (%s): loaded %d from checkpoint",
                dt.doc_type,
                dt.label,
                len(existing),
            )
            entries = existing[:limit] if limit > 0 else existing
        else:
            entries = scrape_bill_index(
                doc_type=dt.doc_type,
                limit=limit,
                session=sess,
                timeout=timeout,
                request_delay=request_delay,
                save_checkpoints=save_checkpoints,
                range_urls=dt.range_urls if dt.range_urls else None,
            )

        all_entries.extend(entries)
        LOGGER.info(
            "  %s (%s): %d entries indexed",
            dt.doc_type,
            dt.label,
            len(entries),
        )

    # ── Step 4: Clean up and log summary ──
    if save_checkpoints:
        _clear_index_checkpoint()

    by_type: dict[str, int] = {}
    for e in all_entries:
        by_type[e.doc_type] = by_type.get(e.doc_type, 0) + 1
    summary_parts = [f"{count} {dt}" for dt, count in sorted(by_type.items())]
    LOGGER.info(
        "Total bill index: %s = %d entries",
        " + ".join(summary_parts),
        len(all_entries),
    )

    return all_entries


def tail_only_bill_indexes(
    highest_per_type: dict[str, int],
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> list[BillIndexEntry]:
    """Fast index scan: only fetch range pages beyond the highest known bill.

    Instead of walking all 125 index pages (~30 min), this fetches just the
    last page per doc type where new bills would appear.  For doc types not
    yet in ``highest_per_type``, all pages are fetched (they're new).

    Returns only the **new** entries (bill numbers higher than what's cached).
    Typically finishes in ~1-2 minutes vs 30+ for a full scan.
    """
    sess = session or _build_session()

    doc_types = _discover_doc_types(sess, timeout, request_delay)
    if not doc_types:
        LOGGER.warning("Tail scan: could not discover doc types, falling back to full.")
        return scrape_all_bill_indexes(
            sb_limit=0,
            hb_limit=0,
            session=sess,
            timeout=timeout,
            request_delay=request_delay,
        )

    new_entries: list[BillIndexEntry] = []

    for dt in doc_types:
        highest = highest_per_type.get(dt.doc_type, 0)

        if not dt.range_urls:
            continue

        if highest == 0:
            # Never scanned this type before — fetch all pages
            LOGGER.info(
                "Tail scan %s: no prior data, fetching all %d pages",
                dt.doc_type,
                len(dt.range_urls),
            )
            pages_to_fetch = dt.range_urls
        else:
            # Only fetch pages whose range includes or exceeds our highest
            # E.g. if highest SB is 4052 and a page covers 4001-4100, fetch it
            pages_to_fetch = [(n1, n2, url) for n1, n2, url in dt.range_urls if n2 >= highest]
            if not pages_to_fetch:
                # All range pages are below our highest — nothing new
                LOGGER.info(
                    "Tail scan %s: highest=%d, all %d pages below — skip",
                    dt.doc_type,
                    highest,
                    len(dt.range_urls),
                )
                continue
            LOGGER.info(
                "Tail scan %s: highest=%d, checking %d of %d pages",
                dt.doc_type,
                highest,
                len(pages_to_fetch),
                len(dt.range_urls),
            )

        for n1, n2, url in pages_to_fetch:
            try:
                resp = sess.get(url, timeout=timeout)
                resp.raise_for_status()
                time.sleep(request_delay)
            except requests.RequestException as exc:
                LOGGER.warning("Tail scan: failed to fetch %s: %s", url, exc)
                continue

            page_entries = _parse_range_page(resp.text, dt.doc_type)
            if not page_entries:
                continue

            # Filter to only truly new entries (beyond our highest)
            new_on_page = [e for e in page_entries if _bill_number_to_int(e.bill_number) > highest]
            if new_on_page:
                new_entries.extend(new_on_page)
                LOGGER.info(
                    "  ✓ %s: %d new entries (%s to %s)",
                    dt.doc_type,
                    len(new_on_page),
                    new_on_page[0].bill_number,
                    new_on_page[-1].bill_number,
                )

    by_type: dict[str, int] = {}
    for e in new_entries:
        by_type[e.doc_type] = by_type.get(e.doc_type, 0) + 1

    if new_entries:
        summary = ", ".join(f"{c} {dt}" for dt, c in sorted(by_type.items()))
        LOGGER.info("Tail scan found %d new entries: %s", len(new_entries), summary)
    else:
        LOGGER.info("Tail scan: no new bills found.")

    return new_entries


def _bill_number_to_int(bill_number: str) -> int:
    """Extract the numeric suffix from a bill number (e.g. 'SB0042' -> 42)."""
    m = re.search(r"(\d+)$", bill_number)
    return int(m.group(1)) if m else 0


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
    date_str = normalize_date(parts[0].strip()) if parts else ""
    chamber = normalize_chamber(parts[1].rstrip(":").strip()) if len(parts) > 1 else ""

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
        date = normalize_date(cells[0].get_text(strip=True))
        chamber = normalize_chamber(cells[1].get_text(strip=True))
        action = cells[2].get_text(strip=True)
        if date:
            entries.append(ActionEntry(date=date, chamber=chamber, action=action))
    return entries


def _scrape_bill_status_with_html(
    bill_status_url: str,
    index_entry: BillIndexEntry | None = None,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> tuple[Bill | None, str]:
    """Fetch a BillStatus page, parse metadata, return ``(Bill, raw_html)``.

    Returns ``(None, "")`` on fetch failure.  The raw HTML is kept so
    callers (e.g. ``scrape_bill_complete``) can pass it to
    ``scrape_votes_from_html`` without a second HTTP request.
    """
    sess = session or _build_session()

    try:
        resp = sess.get(bill_status_url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch BillStatus %s: %s", bill_status_url, exc)
        return None, ""

    raw_html = resp.text
    time.sleep(request_delay)
    soup = BeautifulSoup(raw_html, "html.parser")

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
        return None, ""

    title_h5 = soup.find("h5", class_="fw-bold")
    description = ""
    if index_entry:
        description = index_entry.description
    elif title_h5:
        description = title_h5.get_text(strip=True)

    last_action_date, _la_chamber, last_action = _parse_last_action(soup)
    primary_sponsor, senate_ids, house_ids = _parse_sponsors(soup)

    dt_upper = doc_type.upper()
    if dt_upper.startswith("S"):
        chamber = "S"
    elif dt_upper.startswith("H"):
        chamber = "H"
    elif dt_upper in ("EO", "JSR", "AM"):
        chamber = "J"
    else:
        chamber = "H"

    synopsis = _parse_synopsis(soup)
    action_history = _parse_action_history(soup)

    bill = Bill(
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
    return bill, raw_html


def scrape_bill_status(
    bill_status_url: str,
    index_entry: BillIndexEntry | None = None,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> Bill | None:
    """Scrape a single BillStatus page and return a full Bill object."""
    bill, _html = _scrape_bill_status_with_html(
        bill_status_url, index_entry, session, timeout, request_delay
    )
    return bill


_STALLED_ACTIONS = frozenset(
    {
        "filed with secretary",
        "filed with the clerk",
        "first reading",
        "referred to assignments",
        "referred to rules committee",
        "added as co-sponsor",
        "added as chief co-sponsor",
        "chief co-sponsor changed",
        "alternate chief co-sponsor changed",
        "alternate chief sponsor changed",
        "rule 19(a) / re-referred to rules committee",
        "session sine die",
    }
)


def _is_stalled_bill(bill: Bill) -> bool:
    """Return True if a bill has only intro/assignment actions.

    Bills stuck at these stages have never had a committee hearing or
    floor vote, so votes/slips/fulltext work can be skipped.
    """
    if not bill.action_history:
        return True
    for entry in bill.action_history:
        action_lower = entry.action.lower().strip()
        if not any(action_lower.startswith(s) for s in _STALLED_ACTIONS):
            return False
    return True


def scrape_bill_complete(
    bill_status_url: str,
    index_entry: BillIndexEntry | None = None,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
    include_votes: bool = True,
    include_slips: bool = True,
    include_fulltext: bool = False,
) -> Bill | None:
    """Scrape all data for a single bill in one pass.

    Fetches the BillStatus page **once** and reuses the HTML for
    metadata parsing *and* vote tab URL extraction.  This eliminates
    the duplicate BillStatus fetch that the old pipeline performed.

    Stalled bills (intro/assignments only) skip votes, slips, and
    fulltext automatically.
    """
    from .full_text import scrape_bill_full_text
    from .votes import scrape_votes_from_html
    from .witness_slips import scrape_witness_slips

    sess = session or _build_session()

    # 1. Single BillStatus fetch → metadata + raw HTML
    bill, bill_status_html = _scrape_bill_status_with_html(
        bill_status_url,
        index_entry=index_entry,
        session=sess,
        timeout=timeout,
        request_delay=request_delay,
    )
    if bill is None:
        return None

    # 2. Stalled-bill shortcut — skip sub-scrapes
    if _is_stalled_bill(bill):
        return bill

    # 3. Votes (reuse already-fetched HTML — no second BillStatus request)
    if include_votes and bill_status_html:
        try:
            bill.vote_events = scrape_votes_from_html(
                bill_status_html,
                bill_status_url,
                session=sess,
                timeout=timeout,
                request_delay=request_delay,
            )
        except Exception:
            LOGGER.exception("Votes failed for %s", bill.bill_number)

    # 4. Witness slips
    if include_slips:
        try:
            bill.witness_slips = scrape_witness_slips(
                bill_status_url,
                session=sess,
                timeout=timeout,
                request_delay=request_delay,
            )
        except Exception:
            LOGGER.exception("Slips failed for %s", bill.bill_number)

    # 5. Full text (written to cache/fulltext/ immediately)
    if include_fulltext:
        try:
            text = scrape_bill_full_text(
                bill_status_url,
                bill_number=bill.bill_number,
                session=sess,
                timeout=timeout,
                request_delay=request_delay,
            )
            if text:
                bill.full_text = text
                save_fulltext(bill)
        except Exception:
            LOGGER.exception("Fulltext failed for %s", bill.bill_number)

    return bill


def scrape_all_bills(
    index: list[BillIndexEntry],
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
    max_workers: int = 3,
    use_cache: bool = True,
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
    if use_cache:
        cached = load_bill_cache()
        if cached is not None:
            return cached

    _existing_cache = load_bill_cache() or {}

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
                    # Preserve vote/slip data from any existing cache —
                    # these are scraped separately and must not be lost.
                    old = _existing_cache.get(bill.leg_id)
                    if old:
                        bill.vote_events = old.vote_events
                        bill.witness_slips = old.witness_slips
                        bill.full_text = old.full_text
                    bills[bill.leg_id] = bill
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    log_progress(LOGGER, completed, total, bill.bill_number, elapsed, eta)
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
    """Serialize a Bill to a JSON-safe dict.

    ``full_text`` is externalized to ``cache/fulltext/<bill_number>.txt``
    and replaced with ``has_full_text: true/false`` in the JSON.
    """
    d = asdict(bill)
    has_ft = bool(d.pop("full_text", ""))
    d["has_full_text"] = has_ft
    return d


def _bill_from_dict(d: dict) -> Bill:
    """Deserialize a Bill from a cache dict."""
    # Lazy import to avoid circular dependency at module load
    try:
        from ..ml.action_classifier import classify_action

        _classify = classify_action
    except Exception:
        _classify = None

    action_history = []
    for a in d.get("action_history", []):
        if isinstance(a, dict) and "date" in a and "chamber" in a and "action" in a:
            action_text = a["action"]
            entry = ActionEntry(
                date=normalize_date(a["date"]),
                chamber=normalize_chamber(a["chamber"]),
                action=action_text,
            )
            # Classify if classifier is available
            if _classify is not None:
                try:
                    ca = _classify(action_text)
                    entry.action_category = ca.category_id
                    entry.action_category_label = ca.category_label
                    entry.outcome_signal = ca.outcome_signal
                    entry.meaning = ca.meaning
                except Exception:
                    pass
            action_history.append(entry)

    vote_events = []
    for v in d.get("vote_events", []):
        if isinstance(v, dict):
            vote_events.append(
                VoteEvent(
                    bill_number=v.get("bill_number", ""),
                    date=normalize_date(v.get("date", "")),
                    description=v.get("description", ""),
                    chamber=normalize_chamber(v.get("chamber", "")),
                    yea_votes=v.get("yea_votes", []),
                    nay_votes=v.get("nay_votes", []),
                    present_votes=v.get("present_votes", []),
                    nv_votes=v.get("nv_votes", []),
                    pdf_url=v.get("pdf_url", ""),
                    vote_type=v.get("vote_type", "floor"),
                )
            )

    witness_slips = []
    for ws in d.get("witness_slips", []):
        if isinstance(ws, dict):
            witness_slips.append(
                WitnessSlip(
                    name=ws.get("name", ""),
                    organization=ws.get("organization", ""),
                    representing=ws.get("representing", ""),
                    position=ws.get("position", ""),
                    hearing_committee=ws.get("hearing_committee", ""),
                    hearing_date=normalize_date(ws.get("hearing_date", "")),
                    testimony_type=ws.get("testimony_type", "Record of Appearance Only"),
                    bill_number=ws.get("bill_number", ""),
                )
            )

    bill = Bill(
        bill_number=d["bill_number"],
        leg_id=d["leg_id"],
        description=d["description"],
        chamber=normalize_chamber(d["chamber"]),
        last_action=d.get("last_action", ""),
        last_action_date=normalize_date(d.get("last_action_date", "")),
        primary_sponsor=d.get("primary_sponsor", ""),
        synopsis=d.get("synopsis", ""),
        status_url=d.get("status_url", ""),
        sponsor_ids=d.get("sponsor_ids", []),
        house_sponsor_ids=d.get("house_sponsor_ids", []),
        action_history=action_history,
        vote_events=vote_events,
        witness_slips=witness_slips,
        full_text=d.get("full_text", ""),
    )

    # Externalized full text: load from cache/fulltext/<bill>.txt when the
    # JSON has "has_full_text" but no inline text (new format).
    if not bill.full_text and d.get("has_full_text"):
        ft_path = FULLTEXT_DIR / f"{bill.bill_number}.txt"
        if ft_path.exists():
            bill.full_text = ft_path.read_text(encoding="utf-8")

    return bill


def save_fulltext(bill: Bill) -> None:
    """Write a single bill's full_text to ``cache/fulltext/<bill_number>.txt``."""
    if not bill.full_text:
        return
    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    ft_path = FULLTEXT_DIR / f"{bill.bill_number}.txt"
    ft_path.write_text(bill.full_text, encoding="utf-8")


def save_bill_cache(bills: dict[str, Bill]) -> None:
    """Save bills dict to cache/bills.json (atomic write).

    Full text is written to individual files under ``cache/fulltext/``
    and excluded from ``bills.json`` to keep checkpoint writes fast.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Write externalized full-text files
    for b in bills.values():
        save_fulltext(b)

    data = {lid: _bill_to_dict(b) for lid, b in bills.items()}
    tmp_path = BILLS_CACHE_FILE.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(BILLS_CACHE_FILE)
    LOGGER.info("Saved %d bills to %s", len(data), BILLS_CACHE_FILE)


def load_bill_cache() -> dict[str, Bill] | None:
    """Load bills from the current data dir (cache or mocks)."""
    path = get_data_dir() / "bills.json"
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    validate_bill_cache(raw)
    bills = {lid: _bill_from_dict(d) for lid, d in raw.items()}
    LOGGER.info("Loaded %d bills from %s.", len(bills), path)
    return bills


def save_scrape_metadata(
    bill_count: int,
    *,
    scan_type: str = "full",
    highest_per_type: dict[str, int] | None = None,
) -> None:
    """Save scrape metadata (timestamps, counts, index scan info).

    Parameters
    ----------
    scan_type:
        ``"full"`` or ``"tail"`` — which kind of index scan was performed.
    highest_per_type:
        ``{doc_type: highest_bill_number}`` e.g. ``{"SB": 4052, "HB": 5623}``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            existing = json.load(f)

    now = datetime.now(timezone.utc).isoformat()
    existing["last_bill_scrape_at"] = now
    existing["bill_index_count"] = bill_count

    if scan_type == "full":
        existing["last_full_scan"] = now
        existing["last_tail_scan"] = now  # full implies tail is also fresh
    elif scan_type == "tail":
        existing["last_tail_scan"] = now

    if highest_per_type is not None:
        # Merge — keep the highest we've ever seen per type
        prev = existing.get("highest_bill_per_type", {})
        for dt, num in highest_per_type.items():
            prev[dt] = max(prev.get(dt, 0), num)
        existing["highest_bill_per_type"] = prev

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    LOGGER.info("Updated scrape metadata: %s", METADATA_FILE)


def load_scrape_metadata() -> dict:
    """Load scrape metadata, or empty dict if missing."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _hours_since(iso_ts: str | None) -> float:
    """Return hours elapsed since an ISO-8601 timestamp, or inf if None."""
    if not iso_ts:
        return float("inf")
    try:
        ts = datetime.fromisoformat(iso_ts)
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    except (ValueError, TypeError):
        return float("inf")


def _extract_highest_bill_numbers(
    bills: dict[str, Bill],
) -> dict[str, int]:
    """Scan cached bills and find the highest numeric suffix per doc type.

    E.g. ``{"SB": 4052, "HB": 5623, "SR": 614}``.
    """
    highest: dict[str, int] = {}
    for bill in bills.values():
        bn = bill.bill_number  # e.g. "SB0042", "HB5623"
        m = re.match(r"([A-Z]+)(\d+)", bn)
        if m:
            dt, num_str = m.group(1), m.group(2)
            num = int(num_str)
            if num > highest.get(dt, 0):
                highest[dt] = num
    return highest


# ── Incremental scraping ─────────────────────────────────────────────────────


# ── Tiered scan thresholds ────────────────────────────────────────────────────
# These control which kind of index scan happens automatically.
TAIL_SCAN_THRESHOLD_HOURS = 24.0  # < 24h since last tail → skip index
FULL_SCAN_THRESHOLD_HOURS = 168.0  # < 7 days since last full → tail only


def incremental_bill_scrape(
    sb_limit: int = 100,
    hb_limit: int = 100,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
    max_workers: int = 10,
    rescrape_recent_days: int = 14,
    force_full: bool = False,
    include_votes: bool = True,
    include_slips: bool = True,
    include_fulltext: bool = False,
    checkpoint_interval: int = 50,
    use_hearing_signals: bool = True,
    use_report_signals: bool = True,
) -> dict[str, Bill]:
    """Unified incremental scrape — metadata + votes + slips in one pass.

    Tier decision (based on metadata timestamps):
      - ``force_full=True``  → Full index walk (all 125 pages)
      - No metadata / >7 days since full scan → Full index walk
      - >24h since last tail scan → Tail-only scan (last page per doc type)
      - <24h since last tail scan → Skip index, just re-scrape recent bills

    Each bill is scraped via :func:`scrape_bill_complete` which fetches
    the BillStatus page once and reuses the HTML for votes, slips, and
    optional full text.  Checkpoints every *checkpoint_interval* bills.
    """
    sess = session or _build_session(request_delay=request_delay, pool_size=max_workers)

    # Load existing cache
    existing = load_bill_cache() or {}
    existing_ids = set(existing.keys())

    # ── Tiered scan decision ──────────────────────────────────────────────
    meta = load_scrape_metadata()
    hours_since_full = _hours_since(meta.get("last_full_scan"))
    hours_since_tail = _hours_since(meta.get("last_tail_scan"))
    highest_per_type = meta.get("highest_bill_per_type") or {}

    if existing and not highest_per_type:
        highest_per_type = _extract_highest_bill_numbers(existing)

    scan_type: str
    index: list[BillIndexEntry] = []

    t_index_start = time.perf_counter()

    if force_full or not existing or hours_since_full > FULL_SCAN_THRESHOLD_HOURS:
        reason = (
            "forced"
            if force_full
            else "no cache"
            if not existing
            else f"full scan {hours_since_full:.0f}h ago (>{FULL_SCAN_THRESHOLD_HOURS:.0f}h)"
        )
        LOGGER.info("Index strategy: FULL SCAN (%s)", reason)
        index = scrape_all_bill_indexes(
            sb_limit=sb_limit,
            hb_limit=hb_limit,
            session=sess,
            timeout=timeout,
            request_delay=request_delay,
        )
        scan_type = "full"
    elif hours_since_tail > TAIL_SCAN_THRESHOLD_HOURS:
        LOGGER.info(
            "Index strategy: TAIL SCAN (last tail %.0fh ago, full %.0fh ago)",
            hours_since_tail,
            hours_since_full,
        )
        index = tail_only_bill_indexes(
            highest_per_type=highest_per_type,
            session=sess,
            timeout=timeout,
            request_delay=request_delay,
        )
        scan_type = "tail"
    else:
        LOGGER.info(
            "Index strategy: SKIP (last tail %.1fh ago — within %dh threshold)",
            hours_since_tail,
            int(TAIL_SCAN_THRESHOLD_HOURS),
        )
        scan_type = "skip"

    log_phase(
        LOGGER,
        "Phase 1: Index scan",
        time.perf_counter() - t_index_start,
        f"{len(index)} entries" if index else "skipped",
    )

    # ── Identify new bills from index scan ────────────────────────────────
    fresh_ids = {e.leg_id for e in index}
    new_ids = fresh_ids - existing_ids
    if index:
        LOGGER.info(
            "Index result: %d entries, %d new bills to scrape.",
            len(index),
            len(new_ids),
        )

    # ── Re-scrape recently active bills (regardless of scan type) ─────────
    rescrape_ids: set[str] = set()
    if rescrape_recent_days > 0:
        cutoff = datetime.now()
        for lid, bill in existing.items():
            if bill.last_action_date:
                try:
                    parsed = datetime.strptime(bill.last_action_date, "%Y-%m-%d")
                    age_days = (cutoff - parsed).days
                    if age_days <= rescrape_recent_days:
                        rescrape_ids.add(lid)
                except ValueError:
                    try:
                        parsed = datetime.strptime(bill.last_action_date, "%m/%d/%Y")
                        age_days = (cutoff - parsed).days
                        if age_days <= rescrape_recent_days:
                            rescrape_ids.add(lid)
                    except ValueError:
                        pass
        if rescrape_ids:
            LOGGER.info(
                "Re-scrape: %d bills with activity in last %d days.",
                len(rescrape_ids),
                rescrape_recent_days,
            )

    # ── Phase 0: Signal collection (hearings, reports) ─────────────────────
    signal_bill_numbers: set[str] = set()
    if use_hearing_signals and existing:
        t_sig = time.perf_counter()
        try:
            hearings = scrape_hearing_schedules(
                chambers=["Senate", "House"],
                session=sess,
                timeout=timeout,
                request_delay=request_delay,
            )
            signal_bill_numbers |= hearings_to_bill_numbers(hearings)
            save_hearings_cache(hearings)
            if signal_bill_numbers:
                LOGGER.info(
                    "Hearing signals: %d bill numbers from %d hearings (in %s)",
                    len(signal_bill_numbers),
                    len(hearings),
                    fmt_duration(time.perf_counter() - t_sig),
                )
        except Exception as e:
            LOGGER.warning("Hearing signal scrape failed (continuing): %s", e)
    if use_report_signals and existing:
        try:
            from .reports import reports_to_bill_numbers, scrape_common_reports

            report_bills = reports_to_bill_numbers(
                scrape_common_reports(session=sess, timeout=timeout, request_delay=request_delay)
            )
            signal_bill_numbers |= report_bills
            if report_bills:
                LOGGER.info("Report signals: %d bill numbers", len(report_bills))
        except Exception as e:
            LOGGER.warning("Report signal scrape failed (continuing): %s", e)

    # ── Build scrape list ─────────────────────────────────────────────────
    to_scrape: list[BillIndexEntry] = []
    already_queued: set[str] = set()

    to_scrape.extend(e for e in index if e.leg_id in new_ids)
    already_queued |= new_ids

    for lid in rescrape_ids:
        if lid in already_queued:
            continue
        bill = existing.get(lid)
        if bill and bill.status_url:
            dt_match = re.match(r"([A-Z]+)", bill.bill_number)
            to_scrape.append(
                BillIndexEntry(
                    bill_number=bill.bill_number,
                    leg_id=lid,
                    description=bill.description or "",
                    doc_type=dt_match.group(1) if dt_match else "",
                    status_url=bill.status_url,
                )
            )
            already_queued.add(lid)

    # Bills from hearing/report signals (in cache but not yet queued)
    bill_number_to_bill = {b.bill_number: b for b in existing.values()}
    for bn in signal_bill_numbers:
        bill = bill_number_to_bill.get(bn)
        if not bill or bill.leg_id in already_queued or not bill.status_url:
            continue
        dt_match = re.match(r"([A-Z]+)", bill.bill_number)
        to_scrape.append(
            BillIndexEntry(
                bill_number=bill.bill_number,
                leg_id=bill.leg_id,
                description=bill.description or "",
                doc_type=dt_match.group(1) if dt_match else "",
                status_url=bill.status_url,
            )
        )
        already_queued.add(bill.leg_id)

    if not to_scrape:
        LOGGER.info("Incremental: nothing to scrape, cache is up to date.")
        save_scrape_metadata(
            len(existing),
            scan_type=scan_type,
            highest_per_type=highest_per_type or _extract_highest_bill_numbers(existing),
        )
        return existing

    LOGGER.info(
        "Incremental: scraping %d bills (%d new + %d re-check) [votes=%s, slips=%s, fulltext=%s]",
        len(to_scrape),
        len(new_ids),
        len(rescrape_ids - new_ids),
        include_votes,
        include_slips,
        include_fulltext,
    )

    # ── Unified scrape loop ──────────────────────────────────────────────
    t_start = time.perf_counter()
    total_to_scrape = len(to_scrape)
    completed = 0
    last_checkpoint = 0
    total_votes = 0
    total_slips = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_entry = {
            pool.submit(
                scrape_bill_complete,
                entry.status_url,
                entry,
                sess,
                timeout,
                request_delay,
                include_votes,
                include_slips,
                include_fulltext,
            ): entry
            for entry in to_scrape
        }
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            completed += 1
            try:
                bill = future.result()
                if bill:
                    # For re-scraped bills, preserve data from sub-scrapes
                    # that were not included in this run.
                    old = existing.get(bill.leg_id)
                    if old:
                        if not include_votes and old.vote_events:
                            bill.vote_events = old.vote_events
                        if not include_slips and old.witness_slips:
                            bill.witness_slips = old.witness_slips
                        if not include_fulltext and old.full_text:
                            bill.full_text = old.full_text

                    total_votes += len(bill.vote_events)
                    total_slips += len(bill.witness_slips)
                    existing[bill.leg_id] = bill

                    elapsed = time.perf_counter() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_to_scrape - completed) / rate if rate > 0 else 0
                    log_progress(
                        LOGGER,
                        completed,
                        total_to_scrape,
                        entry.bill_number,
                        elapsed,
                        eta,
                    )
                else:
                    LOGGER.warning(
                        "  [%d/%d] Failed: %s", completed, total_to_scrape, entry.bill_number
                    )
            except Exception:
                LOGGER.exception(
                    "  [%d/%d] Error scraping %s", completed, total_to_scrape, entry.bill_number
                )

            # ── Checkpoint ────────────────────────────────────────────
            if (
                checkpoint_interval > 0
                and completed - last_checkpoint >= checkpoint_interval
                and existing
            ):
                LOGGER.info(
                    "  Checkpoint: saving %d bills at %d/%d...",
                    len(existing),
                    completed,
                    total_to_scrape,
                )
                save_bill_cache(existing)
                last_checkpoint = completed

    elapsed = time.perf_counter() - t_start
    detail = f"{len(to_scrape)} bills"
    if include_votes or include_slips:
        detail += f" ({total_votes} votes, {total_slips} slips)"
    log_phase(LOGGER, "Phase 2: Bill scrape", elapsed, detail)

    # Final save + metadata
    save_bill_cache(existing)
    save_scrape_metadata(
        len(existing),
        scan_type=scan_type,
        highest_per_type=_extract_highest_bill_numbers(existing),
    )

    return existing
