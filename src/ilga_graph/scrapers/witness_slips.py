"""Level 5 – Influence Layer: Witness slip scraper.

Fetches witness-slip data from the ILGA export endpoint (pipe-delimited text)
and parses it into :class:`WitnessSlip` objects.

The witness slips table on ilga.gov is loaded dynamically via JavaScript, so
we use the ``ExportWitnessSlips`` endpoint on ``my.ilga.gov`` which returns
clean pipe-delimited text.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

from ..models import WitnessSlip

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://www.ilga.gov/"
EXPORT_BASE = "https://my.ilga.gov/Legislation/BillStatus/ExportWitnessSlips"

# ── Cache / seed paths ───────────────────────────────────────────────────────

WS_CACHE_DIR = Path("cache")
WS_CACHE_FILE = WS_CACHE_DIR / "witness_slips.json"
WS_MOCK_DEV_FILE = Path("mocks") / "dev" / "witness_slips.json"

# ── Internal helpers ──────────────────────────────────────────────────────────


def _witness_slips_tab_url(bill_status_url: str) -> str:
    """Convert a BillStatus URL to the WitnessSlips tab URL.

    Replaces the path component so that ``/Legislation/BillStatus?...``
    becomes ``/Legislation/BillStatus/WitnessSlips?...``.
    """
    parsed = urlparse(bill_status_url)
    # Normalise the path: strip trailing slash, append /WitnessSlips
    base_path = parsed.path.rstrip("/")
    # If path already ends with a sub-tab (e.g. /VoteHistory), strip it
    if base_path.endswith("/WitnessSlips"):
        return bill_status_url
    parts = base_path.rsplit("/", 1)
    if len(parts) == 2 and parts[1] not in ("BillStatus",):
        # e.g. /Legislation/BillStatus/VoteHistory → /Legislation/BillStatus
        base_path = parts[0]
    new_path = base_path + "/WitnessSlips"
    return parsed._replace(path=new_path).geturl()


def _extract_leg_doc_ids(html: str, page_url: str) -> list[tuple[str, str]]:
    """Extract ``(LegDocId, label)`` tuples from the witness slips overview page.

    The page contains sub-tab links like::

        <a href="...?LegDocId=196535&DocNum=1075&...">HB1075</a>
        <a href="...?LegDocId=204772&...">Senate Amendment 001</a>

    We parse each ``LegDocId`` query parameter and its link text.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "LegDocId" not in href and "legdocid" not in href.lower():
            continue
        # Only follow links that point to the WitnessSlips endpoint
        if "WitnessSlips" not in href:
            continue
        qs = parse_qs(urlparse(href).query)
        leg_doc_id = qs.get("LegDocId", qs.get("legdocid", [None]))[0]
        if leg_doc_id and leg_doc_id not in seen:
            label = link.get_text(strip=True)
            results.append((leg_doc_id, label))
            seen.add(leg_doc_id)

    return results


def _parse_export_text(text: str) -> list[WitnessSlip]:
    """Parse pipe-delimited export text into :class:`WitnessSlip` objects.

    Expected format (first line is header)::

        Legislation|Name|Firm|Representation|Position|Committee|ScheduledDateTime
        HB1075|Paul Makarewicz|AES Clean Energy|AES Clean Energy|Proponent|Executive|
            2025-05-31 17:00
    """
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return []

    # Skip the header line
    slips: list[WitnessSlip] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 7:
            LOGGER.warning("Skipping malformed export line: %r", line)
            continue

        slips.append(
            WitnessSlip(
                bill_number=parts[0].strip(),
                name=parts[1].strip(),
                organization=parts[2].strip(),
                representing=parts[3].strip(),
                position=parts[4].strip(),
                hearing_committee=parts[5].strip(),
                hearing_date=parts[6].strip(),
            )
        )

    return slips


# ── Public API ────────────────────────────────────────────────────────────────

# Regex to extract DocTypeID + DocNum from a BillStatus URL → e.g. "HB0034"
_RE_BILL_FROM_URL = re.compile(r"DocNum=(\d+).*?DocTypeID=(\w+)", re.IGNORECASE)


def _bill_number_from_url(bill_status_url: str) -> str | None:
    """Derive a canonical bill number (e.g. 'HB0034') from a BillStatus URL."""
    m = _RE_BILL_FROM_URL.search(bill_status_url)
    if not m:
        return None
    doc_num = m.group(1).zfill(4)
    doc_type = m.group(2).upper()
    return f"{doc_type}{doc_num}"


def scrape_witness_slips(
    bill_status_url: str,
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> list[WitnessSlip]:
    """Scrape all witness slips for a bill via the ILGA export endpoint.

    Args:
        bill_status_url: URL to the bill status page (``BillStatus?DocNum=...``).
        session: Optional requests session for connection reuse.
        timeout: HTTP timeout in seconds.
        request_delay: Delay between requests to avoid rate-limiting.

    Returns:
        List of :class:`WitnessSlip` objects for every version of the bill
        (original + amendments).  Amendment slips have their ``bill_number``
        normalised to the parent bill number so lookups by bill work correctly.
    """
    sess = session or requests.Session()
    t_start = time.perf_counter()

    # Derive the canonical bill number from the URL for normalisation
    parent_bill = _bill_number_from_url(bill_status_url)

    # Step 1: Fetch the WitnessSlips tab page
    ws_url = _witness_slips_tab_url(bill_status_url)
    LOGGER.info("Fetching witness slips page: %s", ws_url)
    resp = sess.get(ws_url, timeout=timeout)
    resp.raise_for_status()
    time.sleep(request_delay)

    # Step 2: Extract LegDocId(s) from the sub-tab links
    leg_doc_ids = _extract_leg_doc_ids(resp.text, ws_url)
    if not leg_doc_ids:
        LOGGER.warning("  No LegDocId links found on witness slips page.")
        return []

    LOGGER.info("  Found %d bill version(s) with witness slips.", len(leg_doc_ids))

    # Step 3: Fetch export data for each LegDocId
    all_slips: list[WitnessSlip] = []
    for leg_doc_id, label in leg_doc_ids:
        export_url = f"{EXPORT_BASE}?legdocid={leg_doc_id}&legislationname={label}"
        LOGGER.info("  Fetching export for %s (LegDocId=%s)", label, leg_doc_id)

        try:
            resp = sess.get(export_url, timeout=timeout)
            resp.raise_for_status()
            time.sleep(request_delay)

            slips = _parse_export_text(resp.text)
            LOGGER.info("    Parsed %d witness slips.", len(slips))
            all_slips.extend(slips)
        except Exception:
            LOGGER.exception("  Failed to fetch/parse export for LegDocId=%s", leg_doc_id)

    # Step 4: Normalise bill_number for amendment slips so they link to the
    # parent bill (e.g. "House Amendment 001" → "HB0034").
    if parent_bill:
        for slip in all_slips:
            if slip.bill_number != parent_bill:
                slip.bill_number = parent_bill

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    LOGGER.info(
        "Witness slips complete: %d total slips for %s in %.0fms",
        len(all_slips),
        parent_bill or "unknown bill",
        elapsed_ms,
    )
    return all_slips


# ── Cache helpers ─────────────────────────────────────────────────────────────


def _load_ws_cache(*, seed_fallback: bool = False) -> list[WitnessSlip] | None:
    """Load witness slips from disk: cache/ first, then mocks/dev/ if requested."""
    if WS_CACHE_FILE.exists():
        with open(WS_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        LOGGER.info("Loaded %d witness slips from cache (%s).", len(data), WS_CACHE_FILE)
        return [WitnessSlip(**ws) for ws in data]
    if seed_fallback and WS_MOCK_DEV_FILE.exists():
        with open(WS_MOCK_DEV_FILE, encoding="utf-8") as f:
            data = json.load(f)
        LOGGER.info("Loaded %d witness slips from mocks/dev (%s).", len(data), WS_MOCK_DEV_FILE)
        return [WitnessSlip(**ws) for ws in data]
    return None


def _save_ws_cache(slips: list[WitnessSlip]) -> None:
    """Save witness slips to disk cache."""
    WS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(WS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(ws) for ws in slips], f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved %d witness slips to cache (%s).", len(slips), WS_CACHE_FILE)


# ── Multi-bill scraper (mirrors scrape_specific_bills for votes) ──────────────


def scrape_all_witness_slips(
    bill_status_urls: list[str],
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
    use_cache: bool = True,
    seed_fallback: bool = False,
) -> list[WitnessSlip]:
    """Scrape witness slips for a list of bill status URLs.

    Uses the same cache-first / seed-fallback pattern as vote scraping.

    Parameters
    ----------
    bill_status_urls:
        List of bill status URLs to scrape slips for.
    session:
        Optional requests session to reuse.
    timeout:
        HTTP timeout in seconds.
    request_delay:
        Delay between requests in seconds.
    use_cache:
        If True, load from cache or seed (when seed_fallback) if available.
    seed_fallback:
        If True and cache/ is missing, load from mocks/dev/witness_slips.json.
    """
    # Check cache first, then seed when requested
    if use_cache:
        cached = _load_ws_cache(seed_fallback=seed_fallback)
        if cached is not None:
            return cached

    sess = session or requests.Session()
    t_total_start = time.perf_counter()

    all_slips: list[WitnessSlip] = []
    for i, url in enumerate(bill_status_urls, 1):
        LOGGER.info("━━━ Witness slips: bill %d/%d ━━━", i, len(bill_status_urls))
        try:
            slips = scrape_witness_slips(
                url,
                session=sess,
                timeout=timeout,
                request_delay=request_delay,
            )
            all_slips.extend(slips)
            LOGGER.info(
                "  Bill %d/%d: %d witness slips",
                i,
                len(bill_status_urls),
                len(slips),
            )
        except Exception:
            LOGGER.exception("Failed to scrape witness slips for bill %d: %s", i, url)

    elapsed_s = time.perf_counter() - t_total_start
    LOGGER.info(
        "Witness slip scraping complete: %d slips from %d bills in %.1fs",
        len(all_slips),
        len(bill_status_urls),
        elapsed_s,
    )

    # Save to cache for next time
    if all_slips:
        _save_ws_cache(all_slips)

    return all_slips
