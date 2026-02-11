#!/usr/bin/env python3
"""Standalone scraping CLI -- populate cache/ for the API (no server).

Usage::

    make scrape          # 300 SB + 300 HB (prod-style)
    make scrape-200      # 200 SB + 200 HB (test pagination: 2 range pages per type)
    make scrape-full     # full index: all ~9600+ bills (slow)
    make scrape-dev      # light: 20/chamber, 100+100, fast
    make scrape-incremental   # only new/changed bills

    python scripts/scrape.py --sb-limit 0 --hb-limit 0 --export   # full index (~9600+ bills)
    python scripts/scrape.py --sb-limit 200 --hb-limit 200 --export # 200 per type
    python scripts/scrape.py --export-only   # re-export vault from cache only
    python scripts/scrape.py --force-refresh # clear cache and re-scrape
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

from ilga_graph.config import CACHE_DIR, MOCK_DEV_DIR  # noqa: E402
from ilga_graph.etl import (  # noqa: E402
    compute_analytics,
    export_vault,
    load_from_cache,
    load_or_scrape_data,
)
from ilga_graph.analytics_cache import load_analytics_cache, save_analytics_cache  # noqa: E402
from ilga_graph.scraper import save_normalized_cache  # noqa: E402


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
        help="Max Senate bills from index (0 = all pages, ~4800; default: 100)",
    )
    parser.add_argument(
        "--hb-limit",
        type=int,
        default=100,
        help="Max House bills from index (0 = all pages, ~4800; default: 100)",
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
        # ══════════════════════════════════════════════════════════════════
        # PHASE 1: EXTRACT — scrape all raw data (no intermediate saves)
        # ══════════════════════════════════════════════════════════════════
        data = load_or_scrape_data(
            limit=args.limit,
            dev_mode=args.fast,
            seed_mode=seed_mode,
            incremental=args.incremental,
            sb_limit=args.sb_limit,
            hb_limit=args.hb_limit,
            save_cache=False,  # defer save until all data is assembled
        )
        logger.info(
            "Extracted %d members, %d committees, %d bills.",
            len(data.members),
            len(data.committees),
            len(data.bills_lookup),
        )

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2: PERSIST — save members + bills cache
        # ══════════════════════════════════════════════════════════════════
        # Note: votes/slips are scraped incrementally via a separate command:
        #   make scrape-votes          (next 10 bills, resumable)
        #   make scrape-votes LIMIT=0  (all remaining bills)
        # Existing per-bill vote_events/witness_slips in bills.json are
        # preserved -- this scrape does NOT clear them.
        save_normalized_cache(data.members, data.bills_lookup)
        logger.info(
            "Saved cache: %d members, %d bills.",
            len(data.members), len(data.bills_lookup),
        )
        logger.info(
            "Votes/slips: run 'make scrape-votes' to incrementally add vote + slip data."
        )

    # ── PHASE 4: ANALYTICS + EXPORT (optional) ───────────────────────────
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
