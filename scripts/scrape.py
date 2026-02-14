#!/usr/bin/env python3
"""Unified ILGA data pipeline — one command for everything.

Smart incremental scraping with tiered index scanning:

  Phase 1+2: Members + Bills
    - <24h since last scan → SKIP index (just re-scrape recently active bills)
    - <7 days since full scan → TAIL-ONLY scan (~2 min, checks last page per
      doc type for new bills)
    - >7 days / first run → FULL scan (all 125 pages, ~30 min)
    Vote/slip data is always preserved when re-scraping bill details.

  Phase 3: Votes + witness slips (incremental, resumable)
  Phase 4: Analytics + Obsidian export (optional, --export)

Usage::

    make scrape                  # smart incremental (daily, ~2 min)
    make scrape FULL=1           # force full index walk (~30 min)
    make scrape FRESH=1          # nuke cache and re-scrape from scratch
    make scrape LIMIT=100        # limit votes/slips to 100 bills
    make scrape WORKERS=10       # more parallel workers for votes

    python scripts/scrape.py                   # smart tiered scan
    python scripts/scrape.py --full            # force full index walk
    python scripts/scrape.py --fresh           # clear cache, re-scrape
    python scripts/scrape.py --skip-votes      # phases 1-2 only (no votes)
    python scripts/scrape.py --export          # include vault export
    python scripts/scrape.py --export-only     # export from cache only
    python scripts/scrape.py --workers 10      # more vote/slip workers
    python scripts/scrape.py --vote-limit 50   # limit vote/slip bills
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ilga_graph.analytics_cache import load_analytics_cache, save_analytics_cache  # noqa: E402
from ilga_graph.config import CACHE_DIR, MOCK_DEV_DIR  # noqa: E402
from ilga_graph.etl import (  # noqa: E402
    compute_analytics,
    export_vault,
    load_from_cache,
    load_or_scrape_data,
)
from ilga_graph.run_log import RunLogger  # noqa: E402
from ilga_graph.scraper import save_normalized_cache  # noqa: E402


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
        help="Skip Phase 3 (votes + witness slips).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Include Phase 4: analytics + Obsidian vault export.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip scraping entirely; just export vault from cache.",
    )

    # ── Tuning ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Parallel workers for vote/slip scraping (default: 5).",
    )
    parser.add_argument(
        "--vote-limit",
        type=int,
        default=0,
        help="Max bills to scrape votes/slips for (0 = all remaining).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use shorter request delays.",
    )

    # ── Legacy/advanced (hidden from help) ────────────────────────────────
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sb-limit",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hb-limit",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bill-limit",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

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

    seed_mode = args.export_only
    meta = {
        "full": args.full,
        "skip_votes": args.skip_votes,
        "export": args.export or args.export_only,
    }

    with RunLogger("scrape", meta=meta) as log:
        if args.export_only:
            t0 = time.perf_counter()
            data = load_from_cache(seed_fallback=seed_mode)
            if data is None:
                logger.error("No cache found. Run without --export-only to scrape first.")
                sys.exit(1)
            log.phase(
                "Load cache",
                duration_s=time.perf_counter() - t0,
                detail=f"{len(data.members)} members",
            )
        else:
            # ══════════════════════════════════════════════════════════════════
            # PHASE 1+2: Members + Bill index + Bill details
            # ══════════════════════════════════════════════════════════════════
            logger.info("=" * 72)
            logger.info("  PHASE 1+2: Members + Bills (incremental)")
            logger.info("=" * 72)
            t0 = time.perf_counter()
            data = load_or_scrape_data(
                limit=args.limit,
                dev_mode=args.fast,
                seed_mode=seed_mode,
                incremental=True,  # always incremental
                sb_limit=args.sb_limit,
                hb_limit=args.hb_limit,
                save_cache=False,
                force_full_index=args.full,
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
                "Members + Bills",
                duration_s=time.perf_counter() - t0,
                detail=f"{len(data.members)} members, {len(data.bills_lookup)} bills",
            )

            # ══════════════════════════════════════════════════════════════════
            # PHASE 3: Votes + witness slips (incremental)
            # ══════════════════════════════════════════════════════════════════
            if not args.skip_votes:
                logger.info("")
                logger.info("=" * 72)
                logger.info("  PHASE 3: Votes + witness slips (incremental)")
                logger.info("=" * 72)

                vote_cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "scrape_votes.py"),
                    "--limit",
                    str(args.vote_limit),
                    "--workers",
                    str(args.workers),
                    "--fast",
                ]
                logger.info("  Running: %s", " ".join(vote_cmd))
                t0 = time.perf_counter()
                result = subprocess.run(vote_cmd, cwd=str(ROOT))
                t3 = time.perf_counter() - t0
                log.phase("Votes + Slips", duration_s=t3, detail=f"exit {result.returncode}")
                if result.returncode != 0:
                    logger.warning(
                        "  Vote/slip scraping exited with code %d (partial data saved).",
                        result.returncode,
                    )
                else:
                    logger.info("  Vote/slip scraping complete.")

                data = load_from_cache(seed_fallback=False)
                if data is None:
                    logger.error("Failed to reload cache after vote scraping.")
                    sys.exit(1)

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 4: Analytics + Obsidian export (optional)
        # ══════════════════════════════════════════════════════════════════════
        if args.export or args.export_only:
            logger.info("")
            logger.info("=" * 72)
            logger.info("  PHASE 4: Analytics + Vault export")
            logger.info("=" * 72)
            t0 = time.perf_counter()
            cached = load_analytics_cache(CACHE_DIR, MOCK_DEV_DIR, seed_mode)
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

    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
