#!/usr/bin/env python3
"""Ingest Illinois SBE Committees + Receipts and join them onto Member IDs.

Official files: https://downloads.elections.il.gov/
(linked from https://elections.il.gov/CampaignDisclosure/DownloadCDDataFiles.aspx)

Usage::

    PYTHONPATH=src python scripts/ingest_sbe_money.py
    PYTHONPATH=src python scripts/ingest_sbe_money.py --from-dir tests/fixtures/sbe
    PYTHONPATH=src python scripts/ingest_sbe_money.py --since 2025-01-01

Writes SQLite tables (sbe_*) plus processed/campaign_finance/index.json.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ilga_graph.campaign_finance.ingest import (
    DEFAULT_WINDOW_START,
    ingest_from_dir,
    ingest_from_sbe,
)
from ilga_graph.data_source import get_data_dir
from ilga_graph.db import DB_PATH
from ilga_graph.scraper import load_normalized_cache

LOGGER = logging.getLogger("ingest_sbe_money")
GOLD_PATH = ROOT / "docs" / "canonical" / "sbe_committee_member_gold.json"


def _load_members():
    cached = load_normalized_cache()
    if cached is None:
        raise SystemExit(
            "No members.json in the current data dir. Run make scrape or use mocks/dev."
        )
    members, _bills = cached
    if not members:
        raise SystemExit("members.json is empty")
    return members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        default=DEFAULT_WINDOW_START,
        help="Keep receipts on/after this date (YYYY-MM-DD). Default: 2025-01-01",
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        help="Parse local SBE .txt files instead of downloading",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "cache" / "sbe",
        help="Where to store downloaded SBE files",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="SQLite path for sbe_* tables (default: ILGA_DB_PATH)",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=ROOT / "processed" / "campaign_finance" / "index.json",
        help="JSON index loaded at app startup",
    )
    parser.add_argument(
        "--also-write-data-dir",
        action="store_true",
        help="Also copy the index to get_data_dir()/campaign_finance.json",
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    members = _load_members()
    gold = GOLD_PATH if GOLD_PATH.exists() else None

    if args.from_dir:
        result = ingest_from_dir(
            args.from_dir,
            members,
            db_path=args.db_path,
            index_path=args.index_path,
            window_start=args.since,
            gold_path=gold,
        )
    else:
        result = ingest_from_sbe(
            members,
            work_dir=args.work_dir,
            db_path=args.db_path,
            index_path=args.index_path,
            window_start=args.since,
            gold_path=gold,
            skip_download=args.skip_download,
        )

    if args.also_write_data_dir:
        dest = get_data_dir() / "campaign_finance.json"
        dest.write_bytes(Path(result.index_path).read_bytes())
        LOGGER.info("Wrote %s", dest)

    LOGGER.info(
        "Ingest complete: receipts=%s members_matched=%s match_rate=%.1f%% index=%s review=%s",
        result.receipts_stored,
        result.members_matched,
        result.match_rate * 100,
        result.index_path,
        result.unmatched_path,
    )


if __name__ == "__main__":
    main()
