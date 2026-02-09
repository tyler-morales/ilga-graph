#!/usr/bin/env python3
"""Standalone scraping CLI -- scrape ilga.gov to cache/ without starting a server.

Usage::

    python scripts/scrape.py                    # full scrape (all members)
    python scripts/scrape.py --limit 20         # scrape 20 members per chamber
    python scripts/scrape.py --export           # scrape + export vault
    python scripts/scrape.py --export-only      # skip scraping, just export from cache
    python scripts/scrape.py --force-refresh    # ignore existing cache, re-scrape
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Ensure the project is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ilga_graph.main import compute_analytics, export_vault, get_bill_status_urls, load_or_scrape_data  # noqa: E402
from ilga_graph.scrapers.votes import scrape_specific_bills  # noqa: E402
from ilga_graph.scrapers.witness_slips import scrape_all_witness_slips  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="ILGA Graph scraper CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max members to scrape per chamber (0 = all)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Also export the Obsidian vault after scraping",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip scraping; export vault from existing cache",
    )
    parser.add_argument(
        "--bill-limit",
        type=int,
        default=None,
        help="Max bills to export (default: all)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Delete cache/ before scraping to force a full refresh",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use dev-mode request delay (0.25s instead of 1.0s)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("scrape")

    if args.force_refresh:
        cache_dir = ROOT / "cache"
        if cache_dir.exists():
            logger.info("Removing cache directory: %s", cache_dir)
            shutil.rmtree(cache_dir)

    # Never use seed fallback when explicitly scraping
    seed_mode = args.export_only

    logger.info(
        "Loading data (limit=%d, seed_mode=%s, dev_delay=%s)...",
        args.limit,
        seed_mode,
        args.fast,
    )
    data = load_or_scrape_data(
        limit=args.limit,
        dev_mode=args.fast,
        seed_mode=seed_mode,
    )
    logger.info(
        "Loaded %d members, %d committees.",
        len(data.members),
        len(data.committees),
    )

    # ── Scrape votes + witness slips for the configured bill URLs ──
    if not args.export_only:
        bill_urls = get_bill_status_urls()
        delay = 0.25 if args.fast else 0.5
        logger.info(
            "Scraping votes + witness slips for %d bill(s)...", len(bill_urls)
        )

        vote_events = scrape_specific_bills(
            bill_urls,
            request_delay=delay,
            use_cache=not args.force_refresh,
            seed_fallback=False,
        )
        logger.info("Scraped %d vote events.", len(vote_events))

        witness_slips = scrape_all_witness_slips(
            bill_urls,
            request_delay=delay,
            use_cache=not args.force_refresh,
            seed_fallback=False,
        )
        logger.info("Scraped %d witness slips.", len(witness_slips))

    if args.export or args.export_only:
        logger.info("Computing analytics...")
        scorecards, moneyball = compute_analytics(data.members)
        logger.info("Exporting vault...")
        export_vault(
            data,
            scorecards,
            moneyball,
            bill_export_limit=args.bill_limit,
        )
        logger.info("Vault exported to ILGA_Graph_Vault/")

    logger.info("Done.")


if __name__ == "__main__":
    main()
