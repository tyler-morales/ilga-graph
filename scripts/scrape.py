#!/usr/bin/env python3
"""Standalone scraping CLI -- scrape ilga.gov to cache/ without starting a server.

Usage::

    python scripts/scrape.py                    # full scrape (members + 100 SB + 100 HB)
    python scripts/scrape.py --incremental      # incremental: only new/changed bills
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

from ilga_graph.config import CACHE_DIR, MOCK_DEV_DIR, get_bill_status_urls  # noqa: E402
from ilga_graph.etl import (  # noqa: E402
    compute_analytics,
    export_vault,
    load_from_cache,
    load_or_scrape_data,
)
from ilga_graph.analytics_cache import load_analytics_cache, save_analytics_cache  # noqa: E402
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
        "--incremental",
        action="store_true",
        help="Incremental scrape: only fetch new/changed bills",
    )
    parser.add_argument(
        "--sb-limit",
        type=int,
        default=100,
        help="Max Senate bills to scrape from index (default: 100)",
    )
    parser.add_argument(
        "--hb-limit",
        type=int,
        default=100,
        help="Max House bills to scrape from index (default: 100)",
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
        "Loading data (limit=%d, seed_mode=%s, dev_delay=%s, incremental=%s)...",
        args.limit,
        seed_mode,
        args.fast,
        args.incremental,
    )
    if args.export_only:
        data = load_from_cache(seed_fallback=seed_mode)
        if data is None:
            logger.error("No cache found. Run without --export-only to scrape first.")
            sys.exit(1)
    else:
        data = load_or_scrape_data(
            limit=args.limit,
            dev_mode=args.fast,
            seed_mode=seed_mode,
            incremental=args.incremental,
            sb_limit=args.sb_limit,
            hb_limit=args.hb_limit,
        )
    logger.info(
        "Loaded %d members, %d committees, %d bills.",
        len(data.members),
        len(data.committees),
        len(data.bills_lookup),
    )

    # ── Scrape votes + witness slips for the configured bill URLs ──
    if not args.export_only:
        bill_urls = get_bill_status_urls()
        delay = 0.25 if args.fast else 0.5
        logger.info("Scraping votes + witness slips for %d bill(s)...", len(bill_urls))

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
        cached = load_analytics_cache(CACHE_DIR, MOCK_DEV_DIR, seed_mode)
        if cached is not None:
            scorecards, moneyball = cached
            logger.info("Using cached analytics.")
        else:
            scorecards, moneyball = compute_analytics(data.members, data.committee_rosters)
            save_analytics_cache(scorecards, moneyball, CACHE_DIR)
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
