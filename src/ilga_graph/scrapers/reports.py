"""Scrape ILGA Common Reports for milestone bill lists (passed both houses, pending governor).

Used as change-detection signals: bills on these reports are high-priority for re-scraping.
"""

from __future__ import annotations

import logging
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import BASE_URL, GA_ID, SESSION_ID
from .hearings import extract_bill_numbers_from_text

LOGGER = logging.getLogger(__name__)

# Common report paths. ILGA uses GaId/SessionId (same as BillStatus and legislation index).
_REPORT_PATHS = [
    f"legislation/PassedBills.asp?GaId={GA_ID}&SessionId={SESSION_ID}",
    f"legislation/PendingBills.asp?GaId={GA_ID}&SessionId={SESSION_ID}",
]


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


def scrape_common_reports(
    session: requests.Session | None = None,
    timeout: int = 20,
    request_delay: float = 0.5,
) -> list[str]:
    """Fetch Common Report pages and extract all bill numbers.

    Returns a flat list of bill numbers (e.g. SB1234, HB5678) from tables on
    Passed Both Houses, Pending Governor, and similar report pages.
    """
    sess = session or _build_session(request_delay=request_delay)
    base = BASE_URL.rstrip("/") + "/"
    all_bills: list[str] = []
    seen: set[str] = set()

    for path in _REPORT_PATHS:
        url = urljoin(base, path)
        try:
            if request_delay > 0:
                time.sleep(request_delay)
            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            LOGGER.warning("Failed to fetch report %s: %s", url, e)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for table in soup.find_all("table"):
            text = table.get_text(" ", strip=True)
            for bn in extract_bill_numbers_from_text(text):
                if bn not in seen:
                    seen.add(bn)
                    all_bills.append(bn)
        # Also scan links for bill number hrefs (e.g. BillStatus?DocNum=1234&DocTypeID=SB)
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            text = a.get_text(" ", strip=True)
            for bn in extract_bill_numbers_from_text(text or href):
                if bn not in seen:
                    seen.add(bn)
                    all_bills.append(bn)

    return all_bills


def reports_to_bill_numbers(bill_list: list[str]) -> set[str]:
    """Return set of bill numbers from scrape_common_reports result."""
    return set(bill_list)
