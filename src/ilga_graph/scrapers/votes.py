"""Level 4 – Behavior Layer: Roll-call vote scraper.

Downloads vote-record PDFs from the ILGA website, extracts the text with
*pdfplumber*, and parses the vote codes + member names via regex.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup

from ..models import VoteEvent

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://www.ilga.gov/"
VOTE_CACHE_DIR = Path("cache")
VOTE_CACHE_FILE = VOTE_CACHE_DIR / "vote_events.json"
VOTE_MOCK_DEV_FILE = Path("mocks") / "dev" / "vote_events.json"

# ── Regex patterns ────────────────────────────────────────────────────────────

# Vote code + member name.  NV must precede N so we don't greedily match 'N'
# when the actual code is 'NV'.  Lazy capture stops at the next vote code or EOL.
# Codes: Y=Yea, N=Nay, P=Present, NV=Not Voting, E=Excused, A=Absent
#
# The tricky part: 'A' can also be a middle initial (e.g. "Kimberly A" in committee
# PDFs with full names).  We disambiguate: treat 'A' as a vote code ONLY when it is
# NOT immediately followed by another vote code (Y/N/P/E/NV).  In "Kimberly A Y Murphy"
# the A is followed by Y (a vote code) → middle initial.  In "Davidsmeyer A Jones"
# the A is followed by "Jones" (a name) → Absent vote code.
_SMART_A = r"A(?!\s+(?:NV|Y|N|P|E)\s)"
_RE_VOTE_ENTRY = re.compile(
    rf"\b(NV|Y|N|P|E|{_SMART_A})\s+"
    r"([\w][\w\s,.\'\-\"]+?)"
    rf"(?=\s+(?:NV|Y|N|P|E|{_SMART_A})\s|\s*$)",
)

# Metadata lines at the bottom of the PDF
_RE_DATE = re.compile(
    r"((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+\d{4})"
)
_RE_BILL_NUMBER = re.compile(r"(?:Senate|House)\s+Bill\s+No\.\s*(\d+)", re.IGNORECASE)
_RE_VOTE_DESCRIPTION = re.compile(
    r"(THIRD READING|SECOND READING|FIRST READING|"
    r"MOTION TO CONCUR|CONCURRENCE|"
    r"MOTION TO SUSPEND RULE|"
    r"PASSAGE OF .+?|"
    r"(?:DO PASS|RECOMMEND DO ADOPT).*)",
    re.IGNORECASE,
)
_RE_TALLY = re.compile(r"(\d+)\s+YEAS?\s+(\d+)\s+NAYS?\s+(\d+)\s+PRESENT", re.IGNORECASE)


# ── PDF parsing ──────────────────────────────────────────────────────────────


def _parse_vote_text(raw_text: str) -> dict[str, Any]:
    """Parse the extracted text of a single vote PDF.

    Returns a dict with keys:
        yeas, nays, present, nv   – lists of member name strings
        date, description, bill_number – metadata (may be empty strings)
        tally – dict with expected yea/nay/present counts (or None)
    """
    yeas: list[str] = []
    nays: list[str] = []
    present: list[str] = []
    nv: list[str] = []

    bucket_map = {"Y": yeas, "N": nays, "P": present, "NV": nv, "E": nv, "A": nv}

    for line in raw_text.splitlines():
        # Skip metadata / header lines that start with digits (tally) or known keywords
        stripped = line.strip()
        if not stripped:
            continue
        # Only parse lines that look like vote entries (contain a vote code)
        for match in _RE_VOTE_ENTRY.finditer(stripped):
            code = match.group(1)
            name = match.group(2).strip().rstrip(",").strip()
            if name and name != "Mr. President":
                bucket_map[code].append(name)
            elif name == "Mr. President":
                # Include as-is
                bucket_map[code].append(name)

    # ── Extract metadata ──
    date_match = _RE_DATE.search(raw_text)
    date_str = date_match.group(1) if date_match else ""

    desc_match = _RE_VOTE_DESCRIPTION.search(raw_text)
    description = desc_match.group(1).strip().title() if desc_match else ""

    bill_match = _RE_BILL_NUMBER.search(raw_text)
    bill_num_digits = bill_match.group(1) if bill_match else ""

    tally_match = _RE_TALLY.search(raw_text)
    tally = None
    if tally_match:
        tally = {
            "yeas": int(tally_match.group(1)),
            "nays": int(tally_match.group(2)),
            "present": int(tally_match.group(3)),
        }

    return {
        "yeas": sorted(yeas),
        "nays": sorted(nays),
        "present": sorted(present),
        "nv": sorted(nv),
        "date": date_str,
        "description": description,
        "bill_number_digits": bill_num_digits,
        "tally": tally,
    }


def scrape_vote_pdf(
    pdf_url: str,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Download a vote PDF and parse it into structured vote data."""
    sess = session or requests.Session()
    LOGGER.info("  Downloading vote PDF: %s", pdf_url)
    t0 = time.perf_counter()

    resp = sess.get(pdf_url, timeout=timeout)
    resp.raise_for_status()

    download_ms = (time.perf_counter() - t0) * 1000

    # Extract text with pdfplumber
    t1 = time.perf_counter()
    full_text = ""
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

    parse_ms = (time.perf_counter() - t1) * 1000

    result = _parse_vote_text(full_text)
    result["pdf_url"] = pdf_url

    LOGGER.info(
        "    PDF parsed: %d Y, %d N, %d P, %d NV  (download %.0fms, parse %.0fms)",
        len(result["yeas"]),
        len(result["nays"]),
        len(result["present"]),
        len(result["nv"]),
        download_ms,
        parse_ms,
    )

    # Validate against tally if available
    if result["tally"]:
        expected_y = result["tally"]["yeas"]
        expected_n = result["tally"]["nays"]
        if len(result["yeas"]) != expected_y:
            LOGGER.warning(
                "    Tally mismatch: expected %d yeas, got %d",
                expected_y,
                len(result["yeas"]),
            )
        if len(result["nays"]) != expected_n:
            LOGGER.warning(
                "    Tally mismatch: expected %d nays, got %d",
                expected_n,
                len(result["nays"]),
            )

    return result


# ── Vote history page parsing ────────────────────────────────────────────────


def _parse_vote_history_page(html: str, page_url: str) -> list[dict[str, str]]:
    """Parse the vote history HTML page and return a list of PDF info dicts.

    Each dict has keys: pdf_url, label, chamber, vote_type.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, str]] = []

    tables = soup.find_all("table", class_="table")
    for table in tables:
        # Determine vote type from the <th> header
        header_th = table.find("th")
        if not header_th:
            continue
        header_text = header_th.get_text(strip=True)
        if "Voting Record" in header_text:
            vote_type = "floor"
        elif "Committee" in header_text:
            vote_type = "committee"
        else:
            vote_type = "unknown"

        tbody = table.find("tbody", recursive=False)
        tbody_rows = tbody.find_all("tr") if tbody else []
        for row in tbody_rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            link = cells[0].find("a", href=True)
            if not link:
                continue
            href = link["href"]
            if not href.lower().endswith(".pdf"):
                continue
            pdf_url = urljoin(page_url, href)
            label = link.get_text(strip=True)
            chamber = cells[1].get_text(strip=True)
            results.append(
                {
                    "pdf_url": pdf_url,
                    "label": label,
                    "chamber": chamber,
                    "vote_type": vote_type,
                }
            )

    return results


def _extract_votes_tab_url(bill_status_html: str, page_url: str) -> str | None:
    """Extract the 'Votes' tab href from a bill status page."""
    soup = BeautifulSoup(bill_status_html, "html.parser")
    # Look for the Votes link in the tab navigation
    for link in soup.find_all("a", href=True):
        href = link["href"]
        text = link.get_text(strip=True)
        if text == "Votes" and "VoteHistory" in href:
            return urljoin(page_url, href)
    return None


def _extract_description_from_label(label: str) -> str:
    """Extract a description like 'Third Reading' from the PDF link label.

    Label format: 'SB0852 - Third Reading - May 31, 2025'
                  'SB0852 - Judiciary - Criminal - May 29, 2025'
                  'SFA0001 - Executive - May 21, 2025'
    """
    parts = label.split(" - ")
    if len(parts) >= 3:
        # Middle parts are the description (excluding first=bill and last=date)
        return " - ".join(parts[1:-1]).strip()
    if len(parts) == 2:
        return parts[1].strip()
    return label


def _extract_date_from_label(label: str) -> str:
    """Extract the date from the PDF link label."""
    match = _RE_DATE.search(label)
    return match.group(1) if match else ""


# ── Public API ───────────────────────────────────────────────────────────────


def scrape_bill_votes(
    bill_status_url: str,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> list[VoteEvent]:
    """Scrape all vote PDFs for a single bill.

    Args:
        bill_status_url: URL to the bill status page (BillStatus?DocNum=...).
        session: Optional requests session for connection reuse.
        timeout: HTTP timeout in seconds.
        request_delay: Delay between requests to avoid rate-limiting.

    Returns:
        List of VoteEvent objects (floor + committee), ordered chronologically.
    """
    sess = session or requests.Session()
    t_bill_start = time.perf_counter()

    # Step 1: Get bill status page, extract Votes tab URL
    LOGGER.info("Fetching bill status: %s", bill_status_url)
    resp = sess.get(bill_status_url, timeout=timeout)
    resp.raise_for_status()
    time.sleep(request_delay)

    votes_url = _extract_votes_tab_url(resp.text, bill_status_url)
    if not votes_url:
        LOGGER.warning("  No 'Votes' tab found on bill status page.")
        return []

    # Step 2: Get vote history page
    LOGGER.info("  Fetching vote history: %s", votes_url)
    resp = sess.get(votes_url, timeout=timeout)
    resp.raise_for_status()
    time.sleep(request_delay)

    # Step 3: Parse all PDF links from both tables
    pdf_infos = _parse_vote_history_page(resp.text, votes_url)
    if not pdf_infos:
        LOGGER.info("  No vote PDFs found for this bill.")
        return []

    LOGGER.info("  Found %d vote PDFs to process.", len(pdf_infos))

    # Step 4: Download and parse each PDF
    events: list[VoteEvent] = []
    for info in pdf_infos:
        time.sleep(request_delay)
        try:
            parsed = scrape_vote_pdf(info["pdf_url"], session=sess, timeout=timeout)

            # Build bill_number from the label (first part before ' - ')
            label_parts = info["label"].split(" - ")
            bill_number = label_parts[0].strip() if label_parts else ""

            # Use label-derived metadata, fall back to PDF-parsed metadata
            description = _extract_description_from_label(info["label"])
            date = _extract_date_from_label(info["label"]) or parsed.get("date", "")

            event = VoteEvent(
                bill_number=bill_number,
                date=date,
                description=description or parsed.get("description", ""),
                chamber=info["chamber"],
                yea_votes=parsed["yeas"],
                nay_votes=parsed["nays"],
                present_votes=parsed["present"],
                nv_votes=parsed["nv"],
                pdf_url=info["pdf_url"],
                vote_type=info["vote_type"],
            )
            events.append(event)
        except Exception:
            LOGGER.exception("  Failed to parse vote PDF: %s", info["pdf_url"])

    elapsed_ms = (time.perf_counter() - t_bill_start) * 1000
    LOGGER.info(
        "  Bill complete: %d vote events scraped in %.0fms",
        len(events),
        elapsed_ms,
    )
    return events


def scrape_bills_from_range(
    doc_type: str = "SB",
    num_start: int = 1,
    num_end: int = 100,
    max_bills: int = 3,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> list[VoteEvent]:
    """Scrape votes from bills in a range, returning up to *max_bills* that have votes.

    Iterates the bill range listing, follows each bill to its vote history,
    and collects VoteEvent objects.  Stops once *max_bills* bills with votes
    have been found.
    """
    sess = session or requests.Session()
    t_total_start = time.perf_counter()

    # Construct the range listing URL
    range_url = (
        f"{BASE_URL}Legislation/RegularSession/{doc_type}"
        f"?num1={num_start:04d}&num2={num_end:04d}"
        f"&DocTypeID={doc_type}&GaId=18&SessionId=114"
    )
    LOGGER.info("Fetching bill range listing: %s", range_url)
    resp = sess.get(range_url, timeout=timeout)
    resp.raise_for_status()
    time.sleep(request_delay)

    # Parse the listing table to get bill status URLs
    soup = BeautifulSoup(resp.text, "html.parser")
    bill_links: list[str] = []
    table = soup.find("table", class_="table")
    if table:
        for row in table.find_all("tr"):
            link = row.find("a", href=True)
            if link and "BillStatus" in link["href"]:
                bill_links.append(urljoin(range_url, link["href"]))

    # Deduplicate (same bill appears twice in the row – number + description)
    seen: set[str] = set()
    unique_links: list[str] = []
    for url in bill_links:
        if url not in seen:
            seen.add(url)
            unique_links.append(url)

    LOGGER.info(
        "Found %d bills in range %s %04d-%04d.", len(unique_links), doc_type, num_start, num_end
    )

    all_events: list[VoteEvent] = []
    bills_with_votes = 0

    for bill_url in unique_links:
        if bills_with_votes >= max_bills:
            break

        time.sleep(request_delay)
        try:
            events = scrape_bill_votes(
                bill_url,
                session=sess,
                timeout=timeout,
                request_delay=request_delay,
            )
            if events:
                all_events.extend(events)
                bills_with_votes += 1
                LOGGER.info(
                    "  [%d/%d bills with votes found]",
                    bills_with_votes,
                    max_bills,
                )
        except Exception:
            LOGGER.exception("Failed to scrape votes for: %s", bill_url)

    elapsed_s = time.perf_counter() - t_total_start
    LOGGER.info(
        "Vote scraping complete: %d events from %d bills in %.1fs",
        len(all_events),
        bills_with_votes,
        elapsed_s,
    )
    return all_events


def _load_vote_cache(*, seed_fallback: bool = False) -> list[VoteEvent] | None:
    """Load vote events from disk: cache/ first, then mocks/dev/ if requested."""
    if VOTE_CACHE_FILE.exists():
        with open(VOTE_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        LOGGER.info("Loaded %d vote events from cache (%s).", len(data), VOTE_CACHE_FILE)
        return [_vote_event_from_dict(v) for v in data]
    if seed_fallback and VOTE_MOCK_DEV_FILE.exists():
        with open(VOTE_MOCK_DEV_FILE, encoding="utf-8") as f:
            data = json.load(f)
        LOGGER.info("Loaded %d vote events from mocks/dev (%s).", len(data), VOTE_MOCK_DEV_FILE)
        return [_vote_event_from_dict(v) for v in data]
    return None


def _vote_event_from_dict(v: dict) -> VoteEvent:
    """Build a VoteEvent from a JSON dict."""
    return VoteEvent(
        bill_number=v["bill_number"],
        date=v["date"],
        description=v["description"],
        chamber=v["chamber"],
        yea_votes=v["yea_votes"],
        nay_votes=v["nay_votes"],
        present_votes=v["present_votes"],
        nv_votes=v["nv_votes"],
        pdf_url=v["pdf_url"],
        vote_type=v["vote_type"],
    )


def _save_vote_cache(events: list[VoteEvent]) -> None:
    """Save vote events to disk cache."""
    VOTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(VOTE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in events], f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved %d vote events to cache (%s).", len(events), VOTE_CACHE_FILE)


def scrape_specific_bills(
    bill_status_urls: list[str],
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
    use_cache: bool = True,
    seed_fallback: bool = False,
) -> list[VoteEvent]:
    """Scrape votes for a specific list of bill status URLs.

    Parameters
    ----------
    bill_status_urls:
        List of bill status URLs to scrape.
    session:
        Optional requests session to reuse.
    timeout:
        HTTP timeout in seconds.
    request_delay:
        Delay between requests in seconds.
    use_cache:
        If True, load from cache or seed (when seed_fallback) if available.
    seed_fallback:
        If True and cache/ is missing, load from mocks/dev/vote_events.json.
    """
    # Check cache first, then seed when requested
    if use_cache:
        cached = _load_vote_cache(seed_fallback=seed_fallback)
        if cached is not None:
            return cached

    # Suppress verbose per-PDF logs during scraping
    old_level = LOGGER.level
    LOGGER.setLevel(logging.WARNING)

    sess = session or requests.Session()

    all_events: list[VoteEvent] = []
    for i, url in enumerate(bill_status_urls, 1):
        LOGGER.info("━━━ Bill %d/%d ━━━", i, len(bill_status_urls))
        t_bill = time.perf_counter()
        try:
            events = scrape_bill_votes(
                url,
                session=sess,
                timeout=timeout,
                request_delay=request_delay,
            )
            all_events.extend(events)
            LOGGER.info(
                "  Bill %d/%d: %d events (%.1fs)",
                i,
                len(bill_status_urls),
                len(events),
                time.perf_counter() - t_bill,
            )
        except Exception:
            LOGGER.exception("Failed to scrape bill %d: %s", i, url)

    # Restore log level
    LOGGER.setLevel(old_level)

    # Save to cache for next time
    if all_events:
        _save_vote_cache(all_events)

    return all_events
