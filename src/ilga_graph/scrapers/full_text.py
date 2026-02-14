"""Full bill text scraper: downloads PDFs from the ILGA FullText tab.

For each bill, the scraper:
1. **Fast path** — predicts the PDF URL from the bill number and downloads
   directly in a single HTTP request.
2. **Fallback** — if the predicted URL returns a non-200, falls back to the
   original two-request flow: fetch the FullText tab HTML, parse the PDF
   link, then download the PDF.
3. Applies a 10 MB size guard on the download.
4. Extracts text using pdfplumber (already a project dependency).
5. Cleans the text (strips headers, line numbers, normalises whitespace).

**Why PDF?**  ILGA does not render full text inline for large bills — the
page says "too large for display" and only provides the PDF link.  PDF is
the only method that works for *every* bill.

Text is stored on ``Bill.full_text`` with NO truncation.  Truncation to
the first ~5000 words happens at feature-building time in ``features.py``.
"""

from __future__ import annotations

import io
import logging
import re
import time
from urllib.parse import urljoin, urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup

from ..config import BASE_URL, GA_NUMBER

LOGGER = logging.getLogger(__name__)

# Maximum PDF file size we'll download (bytes).  Bills beyond this
# threshold are likely 1000+ page omnibus appropriation PDFs that
# would be slow to download, slow to parse, and add mostly noise.
MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB


# ── URL helpers ──────────────────────────────────────────────────────────────


def _full_text_tab_url(bill_status_url: str) -> str:
    """Convert a BillStatus URL to the FullText tab URL.

    Replaces the path component so that ``/Legislation/BillStatus?...``
    becomes ``/Legislation/BillStatus/FullText?...``.

    Same pattern as ``_witness_slips_tab_url`` in ``witness_slips.py``.
    """
    parsed = urlparse(bill_status_url)
    base_path = parsed.path.rstrip("/")
    # If path already ends with a sub-tab (e.g. /VoteHistory), strip it
    if base_path.endswith("/FullText"):
        return bill_status_url
    parts = base_path.rsplit("/", 1)
    if len(parts) == 2 and parts[1] not in ("BillStatus",):
        # e.g. /Legislation/BillStatus/VoteHistory → /Legislation/BillStatus
        base_path = parts[0]
    new_path = base_path + "/FullText"
    return parsed._replace(path=new_path).geturl()


_RE_BILL_PARTS = re.compile(r"^([A-Z]+)(\d+)$")


def _predict_pdf_url(bill_number: str, ga_number: int = GA_NUMBER) -> str:
    """Construct the most-likely PDF URL directly from bill metadata.

    ILGA PDF URLs follow a consistent pattern::

        /legislation/{GA}/{DocType}/PDF/{GA}00{DocType}{Num:04d}lv.pdf

    Example: bill ``SB0005`` with GA 104 →
    ``https://www.ilga.gov/legislation/104/SB/PDF/10400SB0005lv.pdf``

    The ``lv`` suffix means "latest version" and is present for virtually
    every bill.  If the predicted URL returns a 404, the caller falls back
    to fetching the FullText HTML page to find the real link.

    Returns an empty string if *bill_number* can't be parsed.
    """
    m = _RE_BILL_PARTS.match(bill_number)
    if not m:
        return ""
    doc_type = m.group(1)
    doc_num = int(m.group(2))
    filename = f"{ga_number}00{doc_type}{doc_num:04d}lv.pdf"
    base = BASE_URL.rstrip("/")
    return f"{base}/legislation/{ga_number}/{doc_type}/PDF/{filename}"


# ── HTML parsing ─────────────────────────────────────────────────────────────


def _parse_pdf_link(html: str, page_url: str) -> str | None:
    """Parse the FullText tab HTML to find the PDF download link.

    Looks for the "Open PDF" button::

        <a href="../../documents/legislation/104/SB/PDF/10400SB0005lv.pdf"
           target="_blank" class="btn btn-primary">
            <i class="fas fa-file-pdf"></i> Open PDF
        </a>

    Returns the resolved absolute URL, or ``None`` if not found.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Strategy 1: look for <a> with class "btn btn-primary" whose href ends in .pdf
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf"):
            classes = link.get("class", [])
            if "btn-primary" in classes or "btn" in classes:
                return urljoin(page_url, href)

    # Strategy 2: any <a> whose href ends in .pdf (less specific fallback)
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf"):
            return urljoin(page_url, href)

    return None


# ── PDF text extraction ──────────────────────────────────────────────────────


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a bill PDF using pdfplumber.

    Concatenates all pages.  Same approach as ``scrape_vote_pdf`` in
    ``scrapers/votes.py``.
    """
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


# ── Text cleaning ────────────────────────────────────────────────────────────

# Common page-header patterns in IL bill PDFs
_RE_PAGE_HEADER = re.compile(
    r"^(?:"
    r"\d{3}(?:ST|ND|RD|TH)\s+GENERAL\s+ASSEMBLY"  # "104TH GENERAL ASSEMBLY"
    r"|State\s+of\s+Illinois"
    r"|\d{4}\s+and\s+\d{4}"  # year ranges
    r"|(?:SB|HB|SR|HR|SJR|HJR|SJRCA|HJRCA)\d{4}"  # bill number headers
    r"|Public\s+Act\s+\d+-\d+"
    r"|-\s*\d+\s*-"  # centered page numbers like "- 3 -"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Leading line numbers at the start of each line (common in bill text PDFs)
_RE_LINE_NUMBER = re.compile(r"^\s{0,4}\d{1,4}\s{2,}", re.MULTILINE)

# Multiple blank lines
_RE_MULTI_BLANK = re.compile(r"\n{3,}")


def _clean_bill_text(raw_text: str) -> str:
    """Clean extracted PDF text for storage.

    - Strips page headers and footers
    - Strips leading line numbers
    - Collapses excessive whitespace
    - Normalises unicode (curly quotes, em-dashes)
    - Does NOT truncate — truncation happens at feature-building time
    """
    if not raw_text:
        return ""

    text = raw_text

    # Normalise unicode punctuation
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes
    text = text.replace("\u2013", "-").replace("\u2014", "--")  # en/em dashes
    text = text.replace("\u00a0", " ")  # non-breaking space

    # Strip page headers
    text = _RE_PAGE_HEADER.sub("", text)

    # Strip leading line numbers
    text = _RE_LINE_NUMBER.sub("", text)

    # Collapse multiple blank lines to one
    text = _RE_MULTI_BLANK.sub("\n\n", text)

    # Collapse runs of spaces (but not newlines) to single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Final strip
    text = text.strip()

    return text


# ── PDF download + size guard ────────────────────────────────────────────────


def _download_pdf(
    pdf_url: str,
    sess: requests.Session,
    timeout: int,
) -> str | bytes | None:
    """Download a PDF, enforcing the size guard.

    Returns
    -------
    bytes
        The raw PDF bytes on success.
    str
        ``"[SKIPPED: PDF too large]"`` if the PDF exceeds :data:`MAX_PDF_BYTES`.
    None
        If the download failed (non-200, network error, etc.).
    """
    try:
        pdf_resp = sess.get(pdf_url, timeout=timeout, stream=True)
        if pdf_resp.status_code != 200:
            pdf_resp.close()
            return None
        # Check Content-Length header if available
        content_length = pdf_resp.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_PDF_BYTES:
            LOGGER.warning(
                "PDF too large (%s bytes), skipping: %s",
                content_length,
                pdf_url,
            )
            pdf_resp.close()
            return "[SKIPPED: PDF too large]"
        # Read the body
        pdf_bytes = pdf_resp.content
        if len(pdf_bytes) > MAX_PDF_BYTES:
            LOGGER.warning(
                "PDF body too large (%d bytes), skipping: %s",
                len(pdf_bytes),
                pdf_url,
            )
            return "[SKIPPED: PDF too large]"
        return pdf_bytes
    except requests.RequestException as exc:
        LOGGER.warning("Failed to download PDF %s: %s", pdf_url, exc)
        return None


def _extract_and_clean(pdf_bytes: bytes, pdf_url: str) -> str | None:
    """Extract text from PDF bytes and clean it.  Returns cleaned text or None."""
    try:
        raw_text = _extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        LOGGER.warning("Failed to parse PDF %s: %s", pdf_url, exc)
        return None
    cleaned = _clean_bill_text(raw_text)
    return cleaned if cleaned else None


# ── Main per-bill entry point ────────────────────────────────────────────────


def scrape_bill_full_text(
    bill_status_url: str,
    bill_number: str = "",
    session: requests.Session | None = None,
    timeout: int = 30,
    request_delay: float = 0.5,
) -> str | None:
    """Scrape the full text of a single bill via PDF download.

    **Fast path** — when *bill_number* is provided, the function predicts
    the PDF URL directly and attempts a single-request download.  If the
    predicted URL fails (e.g. 404), it falls back to the two-request HTML
    flow automatically.

    Parameters
    ----------
    bill_status_url:
        The BillStatus URL (e.g. ``/Legislation/BillStatus?DocNum=...``).
    bill_number:
        The bill identifier (e.g. ``"SB0005"``).  When provided, enables
        the direct-PDF fast path that skips the FullText tab HTML fetch.
    session:
        Optional requests session with retry/pooling.
    timeout:
        HTTP request timeout in seconds.
    request_delay:
        Delay between HTTP requests (politeness).

    Returns
    -------
    str or None
        Cleaned bill text, or ``None`` if no PDF was found / download failed.
        Returns ``"[SKIPPED: PDF too large]"`` if PDF exceeds the size guard.
    """
    sess = session or requests.Session()

    # ── Fast path: predict PDF URL and download directly (1 request) ──────
    if bill_number:
        predicted_url = _predict_pdf_url(bill_number)
        if predicted_url:
            result = _download_pdf(predicted_url, sess, timeout)
            if isinstance(result, str):
                # Size-guard skip marker
                return result
            if isinstance(result, bytes):
                LOGGER.debug("Direct PDF hit for %s", bill_number)
                return _extract_and_clean(result, predicted_url)
            # result is None → predicted URL failed, fall through to slow path
            LOGGER.debug("Direct PDF miss for %s, falling back to HTML", bill_number)

    # ── Slow path: fetch FullText tab HTML → parse PDF link → download ────
    ft_url = _full_text_tab_url(bill_status_url)

    try:
        resp = sess.get(ft_url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch FullText tab %s: %s", ft_url, exc)
        return None

    time.sleep(request_delay)

    pdf_url = _parse_pdf_link(resp.text, ft_url)
    if not pdf_url:
        LOGGER.debug("No PDF link found on FullText tab: %s", ft_url)
        return None

    result = _download_pdf(pdf_url, sess, timeout)
    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return _extract_and_clean(result, pdf_url)
    return None
