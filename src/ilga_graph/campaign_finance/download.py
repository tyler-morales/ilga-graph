"""Download official SBE Campaign Disclosure files via HTTP Range chunks.

Full ``Receipts.txt`` is ~1GB and starts in the 1990s. We binary-search
by ``RcvDate`` and only pull the recent tail.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SBE_DOWNLOAD_BASE = "https://downloads.elections.il.gov"
USER_AGENT = "ilga-graph-sbe-ingest/0.1 (+https://github.com/tyler-morales/ilga-graph)"
CHUNK_SIZE = 200_000
MAX_RETRIES = 6

FILE_SIZES_FALLBACK = {
    "Candidates.txt": 3_441_568,
    "CanElections.txt": 2_933_610,
    "CmteCandidateLinks.txt": 668_288,
    "Committees.txt": 8_961_902,
    "Receipts.txt": 1_055_044_029,
}


def _request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None) -> bytes:
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    req = urllib.request.Request(url, method=method, headers=hdrs)
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def content_length(url: str, fallback: int | None = None) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return int(resp.headers.get("Content-Length", "0"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        if fallback:
            LOGGER.warning("HEAD failed for %s (%s); using fallback size %s", url, exc, fallback)
            return fallback
        raise


def download_range(url: str, start: int, end: int) -> bytes:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return _request(url, headers={"Range": f"bytes={start}-{end}"})
        except Exception as exc:  # noqa: BLE001 — retries cover flaky CDN
            last_err = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"range download failed {url} {start}-{end}: {last_err}")


def download_chunked(url: str, dest: Path, *, size: int | None = None, start_at: int = 0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = size or content_length(url)
    LOGGER.info("Downloading %s (%s bytes from offset %s)", url, total, start_at)
    mode = "ab" if start_at else "wb"
    with dest.open(mode) as handle:
        pos = start_at
        while pos < total:
            end = min(total - 1, pos + CHUNK_SIZE - 1)
            data = download_range(url, pos, end)
            handle.write(data)
            pos = end + 1
            if (pos - start_at) % (CHUNK_SIZE * 10) < CHUNK_SIZE:
                LOGGER.info("  %s / %s", pos, total)
    return dest


def _date_near(url: str, offset: int, size: int) -> str | None:
    data = download_range(url, offset, min(size - 1, offset + 16_383))
    text = data.decode("latin-1", errors="replace")
    for line in text.splitlines()[1:16]:
        parts = line.split("\t")
        if len(parts) > 7 and parts[6][:4].isdigit():
            return parts[6][:10]
    return None


def find_receipts_offset(url: str, since: date, size: int | None = None) -> int:
    """Binary-search Receipts.txt for the first row on/after ``since``."""
    total = size or content_length(url, FILE_SIZES_FALLBACK["Receipts.txt"])
    target = since.isoformat()
    lo, hi = 0, total - 1
    best = total
    for _ in range(24):
        mid = (lo + hi) // 2
        found = _date_near(url, mid, total)
        if found is None:
            hi = mid - 1
            continue
        if found >= target:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
        time.sleep(0.05)
    remaining_mb = (total - best) / 1e6
    LOGGER.info("Receipts offset for %s ≈ %s (%.1f MB remaining)", target, best, remaining_mb)
    return max(0, best)


def download_receipts_since(dest: Path, since: date, *, base: str = SBE_DOWNLOAD_BASE) -> Path:
    url = f"{base}/Receipts.txt"
    size = content_length(url, FILE_SIZES_FALLBACK["Receipts.txt"])
    offset = find_receipts_offset(url, since, size)
    # Header row lives at byte 0; write it first, then the recent tail.
    header = download_range(url, 0, 800).split(b"\n", 1)[0] + b"\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(header)
    return download_chunked(url, dest, size=size, start_at=offset)


def download_sbe_files(
    dest_dir: Path,
    *,
    since: date,
    base: str = SBE_DOWNLOAD_BASE,
    files: tuple[str, ...] = (
        "Candidates.txt",
        "Committees.txt",
        "CmteCandidateLinks.txt",
    ),
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        url = f"{base}/{name}"
        size = content_length(url, FILE_SIZES_FALLBACK.get(name))
        download_chunked(url, dest_dir / name, size=size)
    download_receipts_since(dest_dir / "Receipts.txt", since, base=base)
    return dest_dir
