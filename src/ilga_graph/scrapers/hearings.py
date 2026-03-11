"""Scrape Senate/House hearing schedules (Month view) for change-detection signals.

Extracts committee hearings with date, time, location, committee, bill numbers
from subject matter, posting date, and status. Used to build a re-scrape queue
of bills that appear in upcoming hearings.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import BASE_URL, CACHE_DIR
from ..models import Hearing

LOGGER = logging.getLogger(__name__)

HEARINGS_CACHE_FILE = CACHE_DIR / "hearings.json"

# Bill number patterns: SB1234, HB 5652, HB5652, etc.
_RE_BILL = re.compile(r"\b(SB|HB)\s*(\d+)\b", re.IGNORECASE)


def _normalize_bill_number(prefix: str, num: str) -> str:
    """Return e.g. SB0076, HB5652 (zero-pad to 4 digits for SB/HB)."""
    n = int(num)
    if prefix.upper() in ("SB", "HB"):
        return f"{prefix.upper()}{n:04d}"
    return f"{prefix.upper()}{n}"


def extract_bill_numbers_from_text(text: str) -> list[str]:
    """Extract and normalize bill numbers from subject matter or any text."""
    if not text or not text.strip():
        return []
    seen: set[str] = set()
    result: list[str] = []
    for m in _RE_BILL.finditer(text):
        pref, num = m.group(1), m.group(2)
        bn = _normalize_bill_number(pref, num)
        if bn not in seen:
            seen.add(bn)
            result.append(bn)
    return result


def _build_session(request_delay: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _parse_schedule_table(soup: BeautifulSoup, chamber: str) -> list[Hearing]:
    """Parse hearing table(s) from a schedule page. Returns list of Hearing."""
    hearings: list[Hearing] = []
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header_cells = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]
        if not header_cells:
            continue
        # Look for columns that suggest a hearing schedule (time, committee, location, subject)
        has_time = any("time" in h or "date" in h for h in header_cells)
        has_committee = any("committee" in h for h in header_cells)
        has_subject = any(
            "subject" in h or "matter" in h or "legislation" in h for h in header_cells
        )
        has_location = any("location" in h or "room" in h or "place" in h for h in header_cells)
        if not (has_time and (has_committee or has_subject or has_location)):
            continue

        for tr in rows[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            texts = [c.get_text(" ", strip=True) for c in cells]
            full_row = " ".join(texts)

            # Status from markers in row text
            status = "normal"
            if "**** canceled ****" in full_row.lower() or "canceled" in full_row.lower():
                status = "canceled"
            elif "**** changed ****" in full_row.lower() or "changed" in full_row.lower():
                status = "changed"

            # First cell often date/time (e.g. "3/11/2026 8:30 AM" or "3/12/2026 9:00 AM")
            date_str = ""
            time_str = ""
            if len(texts) >= 1:
                first = texts[0]
                parts = first.split(None, 2)
                if len(parts) >= 2:
                    date_str = parts[0]
                    time_str = parts[1] if len(parts) > 1 else ""
                else:
                    date_str = first

            # Normalize date to YYYY-MM-DD for consistency
            try:
                if "/" in date_str:
                    dt = datetime.strptime(date_str.split()[0], "%m/%d/%Y")
                    date_str = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

            # Committee: second column often, or search for a cell with committee-like name
            committee_name = ""
            committee_id = ""
            location = ""
            posting_date = ""
            subject_text = ""

            for i, t in enumerate(texts):
                if i == 0:
                    continue
                # Subject matter often contains bill numbers
                if re.search(_RE_BILL, t):
                    subject_text = t
                if "capitol" in t.lower() or "springfield" in t.lower() or "-" in t:
                    location = t
                if "posting" in header_cells[i].lower() if i < len(header_cells) else False:
                    posting_date = t
                if not committee_name and len(t) > 2 and len(t) < 80:
                    if not re.match(r"^\d", t) and ":" not in t[:10]:
                        committee_name = t

            if not committee_name and len(texts) >= 2:
                committee_name = texts[1]

            bills = extract_bill_numbers_from_text(subject_text or full_row)

            hearings.append(
                Hearing(
                    date=date_str,
                    time=time_str,
                    location=location,
                    committee_name=committee_name,
                    committee_id=committee_id,
                    bills=bills,
                    posting_date=posting_date,
                    status=status,
                    chamber=chamber,
                )
            )

    return hearings


# ILGA schedule URLs: hearings.asp with Scheduled=M (month) or W (week) returns HTML tables.
_SCHEDULE_URLS = {
    "Senate": "senate/schedules/hearings.asp?Scheduled=M",
    "House": "house/schedules/hearings.asp?Scheduled=W",
}


def scrape_hearing_schedules(
    chambers: list[str] | None = None,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> list[Hearing]:
    """Scrape Senate and/or House schedule pages (Month/Week view) and return hearings.

    ILGA uses hearings.asp?Scheduled=M (Senate month) and hearings.asp?Scheduled=W
    (House week). We parse tables that look like a hearing list and extract bill
    numbers from Subject Matter (or full row) for re-scrape signals.
    """
    chambers = chambers or ["Senate", "House"]
    sess = session or _build_session(request_delay=request_delay)
    all_hearings: list[Hearing] = []
    base = BASE_URL.rstrip("/") + "/"

    for chamber in chambers:
        path = _SCHEDULE_URLS.get(chamber)
        if not path:
            LOGGER.warning("Unknown chamber %r, skipping schedule.", chamber)
            continue
        url = base + path
        try:
            if request_delay > 0:
                time.sleep(request_delay)
            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            LOGGER.warning("Failed to fetch %s schedule: %s", chamber, e)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        hearings = _parse_schedule_table(soup, chamber)
        all_hearings.extend(hearings)
        LOGGER.info("Scraped %s schedule: %d hearings", chamber, len(hearings))

    return all_hearings


def hearings_to_bill_numbers(hearings: list[Hearing]) -> set[str]:
    """Return the set of all bill numbers mentioned in hearings (for re-scrape queue)."""
    out: set[str] = set()
    for h in hearings:
        out.update(h.bills)
    return out


def save_hearings_cache(hearings: list[Hearing]) -> None:
    """Write hearings and timestamp to cache/hearings.json."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "hearings": [asdict(h) for h in hearings],
    }
    path = HEARINGS_CACHE_FILE
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)
    LOGGER.info("Saved %d hearings to %s", len(hearings), path)


def load_hearings_cache() -> list[Hearing] | None:
    """Load hearings from cache/hearings.json, or None if missing."""
    if not HEARINGS_CACHE_FILE.exists():
        return None
    try:
        with open(HEARINGS_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        LOGGER.warning("Failed to load hearings cache: %s", e)
        return None
    raw = data.get("hearings") or []
    hearings = []
    for r in raw:
        if isinstance(r, dict):
            hearings.append(
                Hearing(
                    date=r.get("date", ""),
                    time=r.get("time", ""),
                    location=r.get("location", ""),
                    committee_name=r.get("committee_name", ""),
                    committee_id=r.get("committee_id", ""),
                    bills=list(r.get("bills") or []),
                    posting_date=r.get("posting_date", ""),
                    status=r.get("status", "normal"),
                    chamber=r.get("chamber", ""),
                )
            )
    return hearings
