#!/usr/bin/env python3
"""Refresh only member photo_url from ILGA member detail pages.

Use when you have cache/members.json (and cache/bills.json) but photo_url
is missing or stale. Fetches each member's detail page, extracts the
img.member-photo src, updates members and saves cache. Does not re-scrape
bills or votes.

Usage::

    make refresh-photos
    # or
    PYTHONPATH=src python scripts/refresh_member_photos.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ilga_graph.config import CACHE_DIR
from ilga_graph.scraper import (
    ILGAScraper,
    load_normalized_cache,
    save_normalized_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
LOGGER = logging.getLogger("refresh_photos")


def main() -> int:
    cached = load_normalized_cache(seed_fallback=False)
    if cached is None:
        LOGGER.error(
            "No cache found. Need cache/members.json and cache/bills.json. "
            "Run make scrape or make scrape-members first."
        )
        return 1

    members, bills_lookup = cached
    if not members:
        LOGGER.error("No members in cache.")
        return 1

    scraper = ILGAScraper(request_delay=0.3)
    updated = 0
    for m in members:
        if not m.member_url:
            continue
        try:
            detail = scraper.scrape_details(m.member_url, m.chamber)
            current = getattr(m, "photo_url", "") or ""
            if detail and detail.photo_url and detail.photo_url != current:
                m.photo_url = detail.photo_url
                updated += 1
        except Exception:
            LOGGER.exception("Failed to refresh photo for %s (%s)", m.name, m.member_url)

    save_normalized_cache(members, bills_lookup)
    LOGGER.info("Refreshed %d member photos. Cache saved to %s", updated, CACHE_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
