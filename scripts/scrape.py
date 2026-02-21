#!/usr/bin/env python3
"""Unified ILGA data pipeline — one command for everything.

Smart incremental scraping with tiered index scanning:

  Phase 1: Index scan (tiered: full / tail / skip)
  Phase 2: Per-bill scrape — metadata + votes + slips in one pass
  Phase 3: Analytics + Obsidian export (optional, --export)

Each bill's BillStatus page is fetched **once**; the HTML is reused for
metadata parsing and vote-tab URL extraction.  Stalled bills (intro/
assignments only) skip votes/slips/fulltext automatically.

Usage::

    make scrape                  # smart incremental (daily, ~2 min)
    make scrape FULL=1           # force full index walk (~30 min)
    make scrape FRESH=1          # nuke cache and re-scrape from scratch
    make scrape FULLTEXT=1       # include full text in same pass
    make scrape WORKERS=10       # more parallel workers

    python scripts/scrape.py                   # smart tiered scan
    python scripts/scrape.py --full            # force full index walk
    python scripts/scrape.py --fresh           # clear cache, re-scrape
    python scripts/scrape.py --skip-votes      # metadata only (no votes/slips)
    python scripts/scrape.py --fulltext        # include full text
    python scripts/scrape.py --export          # include vault export
    python scripts/scrape.py --export-only     # export from cache only
    python scripts/scrape.py --workers 10      # parallel workers
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

# Ensure the project is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ilga_graph.analytics_cache import load_analytics_cache, save_analytics_cache  # noqa: E402
from ilga_graph.config import CACHE_DIR  # noqa: E402
from ilga_graph.etl import (  # noqa: E402
    compute_analytics,
    export_vault,
    load_from_cache,
    load_or_scrape_data,
)
from ilga_graph.run_log import RunLogger  # noqa: E402
from ilga_graph.scraper import save_normalized_cache  # noqa: E402
from ilga_graph.scrapers._log import fmt_duration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ILGA Graph unified data pipeline.",
    )

    # ── Core options ──────────────────────────────────────────────────────
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete cache/ and re-scrape everything from scratch.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full index walk (all 125 pages). Default: smart tiered scan.",
    )
    parser.add_argument(
        "--skip-votes",
        action="store_true",
        help="Skip votes + witness slips (metadata-only scrape).",
    )
    parser.add_argument(
        "--fulltext",
        action="store_true",
        help="Include full text scraping in the same pass.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Include analytics + Obsidian vault export.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip scraping entirely; just export vault from cache.",
    )
    parser.add_argument(
        "--members-only",
        action="store_true",
        help="Only fetch members (and committees). Load bills from cache.",
    )

    # ── Tuning ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel workers for bill scraping (default: 10).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use shorter request delays (0.1s).",
    )

    # ── Legacy/advanced (hidden from help) ────────────────────────────────
    parser.add_argument("--limit", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--sb-limit", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--hb-limit", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--bill-limit", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vote-limit", type=int, default=0, help=argparse.SUPPRESS)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("scrape")

    # ── Fresh start ──────────────────────────────────────────────────────
    if args.fresh:
        cache_dir = ROOT / "cache"
        if cache_dir.exists():
            logger.info("Removing cache directory: %s", cache_dir)
            shutil.rmtree(cache_dir)

    include_votes = not args.skip_votes
    include_slips = not args.skip_votes
    include_fulltext = args.fulltext

    meta = {
        "full": args.full,
        "skip_votes": args.skip_votes,
        "fulltext": include_fulltext,
        "export": args.export or args.export_only,
    }

    t_total = time.perf_counter()

    with RunLogger("scrape", meta=meta) as log:
        if args.export_only:
            t0 = time.perf_counter()
            data = load_from_cache()
            if data is None:
                logger.error("No cache found. Run without --export-only to scrape first.")
                sys.exit(1)
            log.phase(
                "Load cache",
                duration_s=time.perf_counter() - t0,
                detail=f"{len(data.members)} members",
            )
        else:
            # ══════════════════════════════════════════════════════════════
            # PHASE 1+2: Members + Bills (unified)
            # ══════════════════════════════════════════════════════════════
            logger.info("=" * 72)
            logger.info(
                "  Unified scrape: members + bills + votes + slips%s",
                " + fulltext" if include_fulltext else "",
            )
            logger.info(
                "  Workers: %d  |  Fast: %s  |  Full index: %s",
                args.workers,
                args.fast,
                args.full,
            )
            logger.info("=" * 72)
            t0 = time.perf_counter()
            data = load_or_scrape_data(
                limit=args.limit,
                dev_mode=args.fast,
                incremental=not args.members_only,
                sb_limit=args.sb_limit,
                hb_limit=args.hb_limit,
                save_cache=False,
                force_full_index=args.full,
                members_only=args.members_only,
                include_votes=include_votes,
                include_slips=include_slips,
                include_fulltext=include_fulltext,
                max_workers=args.workers,
            )
            logger.info(
                "  %d members, %d committees, %d bills.",
                len(data.members),
                len(data.committees),
                len(data.bills_lookup),
            )
            save_normalized_cache(data.members, data.bills_lookup)
            logger.info("  Saved to cache/.")
            log.phase(
                "Members + Bills (unified)",
                duration_s=time.perf_counter() - t0,
                detail=f"{len(data.members)} members, {len(data.bills_lookup)} bills",
            )

        # ══════════════════════════════════════════════════════════════════
        # Analytics + Obsidian export (optional)
        # ══════════════════════════════════════════════════════════════════
        if args.export or args.export_only:
            logger.info("")
            logger.info("=" * 72)
            logger.info("  Analytics + Vault export")
            logger.info("=" * 72)
            t0 = time.perf_counter()
            cached = load_analytics_cache(CACHE_DIR)
            if cached is not None:
                scorecards, moneyball = cached
                logger.info("  Using cached analytics.")
            else:
                scorecards, moneyball = compute_analytics(data.members, data.committee_rosters)
                save_analytics_cache(scorecards, moneyball, CACHE_DIR)
            logger.info("  Exporting vault...")
            export_vault(
                data,
                scorecards,
                moneyball,
                bill_export_limit=args.bill_limit,
            )
            logger.info("  Vault exported to ILGA_Graph_Vault/")
            log.phase(
                "Analytics + Export",
                duration_s=time.perf_counter() - t0,
                detail="vault exported",
            )

    total_elapsed = time.perf_counter() - t_total
    logger.info("")
    logger.info("═" * 50)
    logger.info("Total scrape: %s", fmt_duration(total_elapsed))
    logger.info("═" * 50)


if __name__ == "__main__":
    main()
