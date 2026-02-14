#!/usr/bin/env python3
"""Incremental full-text bill scraper.

Downloads bill PDFs from the ILGA FullText tab and extracts text using
pdfplumber.  Only scrapes substantive bills (SB, HB) — resolutions are
skipped because the synopsis is sufficient for those.

**Speed optimisations** (v2):

- Predicts the PDF URL directly from the bill number, cutting HTTP requests
  from 2 per bill to 1 for most bills (falls back to HTML if 404).
- Configurable delay via ``--delay`` (including ``--delay 0`` for no delay).
- Default save interval raised to 100 to reduce disk I/O stalls.

Progress is saved after every batch of bills, so you can Ctrl+C at any
time without losing work.  Run again to resume from where you left off.

Usage::

    make scrape-fulltext                 # next 100 bills (default)
    make scrape-fulltext LIMIT=500       # next 500 bills
    make scrape-fulltext LIMIT=0         # all remaining SB/HB bills
    make scrape-fulltext FAST=1          # shorter request delay
    make scrape-fulltext DELAY=0         # no delay at all (fastest)

    python scripts/scrape_fulltext.py                # next 100
    python scripts/scrape_fulltext.py --limit 0      # all remaining
    python scripts/scrape_fulltext.py --workers 5    # 5 parallel workers
    python scripts/scrape_fulltext.py --fast          # shorter delay (0.15s)
    python scripts/scrape_fulltext.py --delay 0      # exact delay (overrides --fast)
    python scripts/scrape_fulltext.py --reset         # wipe progress, start over
    python scripts/scrape_fulltext.py --verify        # rebuild progress from data
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ilga_graph.config import CACHE_DIR  # noqa: E402
from ilga_graph.models import Bill  # noqa: E402
from ilga_graph.scrapers.bills import (  # noqa: E402
    load_bill_cache,
    save_bill_cache,
)
from ilga_graph.scrapers.full_text import scrape_bill_full_text  # noqa: E402

PROGRESS_FILE = CACHE_DIR / "fulltext_progress.json"

# Batch save interval: write bills.json every N completed bills.
# Higher values reduce disk I/O stalls (serializing 150 MB bills.json blocks workers).
SAVE_INTERVAL = 100

# Only scrape these bill types (substantive bills).
# Resolutions (SR, HR, SJR, etc.) are short enough that synopsis suffices.
SUBSTANTIVE_TYPES = frozenset({"SB", "HB"})

# Extract bill type prefix from bill number
_RE_BILL_TYPE = re.compile(r"^([A-Z]+)")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scrape_fulltext")
logger.setLevel(logging.INFO)


# ── Progress helpers ─────────────────────────────────────────────────────────


def load_progress() -> dict:
    """Load progress data from disk."""
    if not PROGRESS_FILE.exists():
        return {
            "scraped_bill_numbers": [],
            "checked_no_pdf": [],
            "skipped_too_large": [],
        }
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("scraped_bill_numbers", [])
    data.setdefault("checked_no_pdf", [])
    data.setdefault("skipped_too_large", [])
    return data


def save_progress(progress_data: dict) -> None:
    """Persist progress data to disk (atomic write)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    progress_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    progress_data["scraped_bill_numbers"] = sorted(set(progress_data["scraped_bill_numbers"]))
    progress_data["checked_no_pdf"] = sorted(set(progress_data.get("checked_no_pdf", [])))
    progress_data["skipped_too_large"] = sorted(set(progress_data.get("skipped_too_large", [])))
    tmp = PROGRESS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)
    tmp.replace(PROGRESS_FILE)


def rebuild_progress_from_data(bills_lookup: dict[str, Bill]) -> dict:
    """Rebuild progress file from actual Bill.full_text data.

    A bill is 'scraped' if it has a non-empty full_text that isn't a
    skip marker.  Bills with "[SKIPPED: PDF too large]" are tracked in
    ``skipped_too_large``.
    """
    scraped = []
    skipped = []
    for bill in bills_lookup.values():
        if bill.full_text:
            if bill.full_text.startswith("[SKIPPED:"):
                skipped.append(bill.bill_number)
            else:
                scraped.append(bill.bill_number)

    progress = {
        "scraped_bill_numbers": sorted(scraped),
        "checked_no_pdf": [],
        "skipped_too_large": sorted(skipped),
    }
    save_progress(progress)
    logger.info(
        "Rebuilt progress from data: %d scraped, %d skipped-too-large.",
        len(scraped),
        len(skipped),
    )
    return progress


def _get_bill_type(bill_number: str) -> str:
    """Extract bill type prefix (e.g. 'SB' from 'SB0042')."""
    m = _RE_BILL_TYPE.match(bill_number)
    return m.group(1) if m else ""


# ── Per-bill worker ──────────────────────────────────────────────────────────


def scrape_one_bill(
    bill_number: str,
    status_url: str,
    delay: float,
) -> tuple[str, str | None, float]:
    """Worker function: scrape full text for one bill.

    Returns (bill_number, full_text_or_none, elapsed_seconds).
    """
    t0 = time.perf_counter()
    try:
        text = scrape_bill_full_text(
            status_url,
            bill_number=bill_number,
            timeout=30,
            request_delay=delay,
        )
    except Exception:
        logger.exception("Worker crashed for %s", bill_number)
        text = None
    elapsed = time.perf_counter() - t0
    return bill_number, text, elapsed


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Scrape full bill text PDFs from ILGA (incremental, resumable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of bills to scrape this run (0 = all remaining; default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3, lower than votes since PDFs are larger)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use shorter request delay (0.15s instead of 0.5s)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Exact request delay in seconds (overrides --fast; e.g. --delay 0 for none)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe progress file and start from scratch",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rebuild progress from actual bills.json data (fixes stale progress)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=SAVE_INTERVAL,
        help=f"Save bills.json every N bills (default: {SAVE_INTERVAL})",
    )
    args = parser.parse_args()

    # ── Load bill cache ──────────────────────────────────────────────────
    bills_lookup = load_bill_cache()
    if not bills_lookup:
        logger.error("No bill cache found. Run 'make scrape' first to populate cache/bills.json.")
        sys.exit(1)

    # ── Reset progress if requested ──────────────────────────────────────
    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        logger.info("Progress reset. Starting from scratch.")

    # ── Verify / rebuild progress from actual data ────────────────────────
    if args.verify or args.reset:
        progress = rebuild_progress_from_data(bills_lookup)
    else:
        progress = load_progress()

    # ── Validate progress against actual data ────────────────────────────
    # Remove bills from progress that claim "scraped" but have no full_text
    validated_scraped = []
    dropped = 0
    bn_to_bill = {b.bill_number: b for b in bills_lookup.values()}
    for bn in progress["scraped_bill_numbers"]:
        bill = bn_to_bill.get(bn)
        if bill and bill.full_text and not bill.full_text.startswith("[SKIPPED:"):
            validated_scraped.append(bn)
        else:
            dropped += 1
    if dropped:
        logger.info(
            "Dropped %d stale progress entries (marked done but no full_text in bills.json).",
            dropped,
        )
        progress["scraped_bill_numbers"] = validated_scraped
        save_progress(progress)

    already_scraped = set(progress["scraped_bill_numbers"])
    already_no_pdf = set(progress.get("checked_no_pdf", []))
    already_too_large = set(progress.get("skipped_too_large", []))
    skip_set = already_scraped | already_no_pdf | already_too_large

    # ── Also skip bills that already have full_text in memory ────────────
    for bill in bills_lookup.values():
        if bill.full_text:
            skip_set.add(bill.bill_number)

    # ── Build TODO list: SB/HB only, sorted by bill number ──────────────
    bill_by_number: dict[str, Bill] = {}
    for bill in bills_lookup.values():
        if not bill.status_url or not bill.status_url.strip():
            continue
        bt = _get_bill_type(bill.bill_number)
        if bt not in SUBSTANTIVE_TYPES:
            continue
        bill_by_number[bill.bill_number] = bill

    todo = sorted(
        [(bn, bill_by_number[bn].status_url) for bn in bill_by_number if bn not in skip_set],
        key=lambda t: t[0],  # alphabetical: HB before SB, then numeric
    )

    total_eligible = len(bill_by_number)
    total_done = len(already_scraped)
    total_skipped = len(skip_set)

    if not todo:
        logger.info(
            "All %d substantive bills already processed "
            "(%d with text, %d no PDF, %d too large). Nothing to do.",
            total_eligible,
            total_done,
            len(already_no_pdf),
            len(already_too_large),
        )
        return

    # Apply limit
    if args.limit > 0:
        batch = todo[: args.limit]
    else:
        batch = todo

    batch_size = len(batch)
    # Delay precedence: --delay (exact) > --fast (0.15s) > default (0.5s)
    if args.delay is not None:
        delay = args.delay
    elif args.fast:
        delay = 0.15
    else:
        delay = 0.5

    print(flush=True)
    print("=" * 80, flush=True)
    print(
        f"  Scraping full text PDFs for {batch_size} bill(s)  "
        f"({total_done} done / {total_skipped} processed "
        f"/ {total_eligible} substantive)",
        flush=True,
    )
    print(
        f"  Workers: {args.workers}  |  Delay: {delay}s  |  "
        f"Limit: {args.limit or 'all'}  |  "
        f"Save every: {args.save_interval} bills",
        flush=True,
    )
    print("=" * 80, flush=True)
    print(flush=True)

    # ── Shared state ─────────────────────────────────────────────────────
    save_lock = threading.Lock()
    scraped_bills = set(already_scraped)
    no_pdf_bills = set(already_no_pdf)
    too_large_bills = set(already_too_large)
    completed_count = 0
    unsaved_count = 0
    total_words_scraped = 0
    bills_with_text = 0
    interrupted = False
    t_start = time.perf_counter()

    def _handle_sigint(signum, frame):
        nonlocal interrupted
        if interrupted:
            print("\nForce exit.", flush=True)
            sys.exit(1)
        interrupted = True
        print(
            "\nInterrupted! Waiting for in-flight workers to finish... (Ctrl+C again to force)",
            flush=True,
        )

    signal.signal(signal.SIGINT, _handle_sigint)

    def _do_save():
        """Save bills.json and progress (called under save_lock)."""
        nonlocal unsaved_count
        save_bill_cache(bills_lookup)
        progress["scraped_bill_numbers"] = sorted(scraped_bills)
        progress["checked_no_pdf"] = sorted(no_pdf_bills)
        progress["skipped_too_large"] = sorted(too_large_bills)
        save_progress(progress)
        unsaved_count = 0

    # ── Thread pool ──────────────────────────────────────────────────────
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for bill_number, status_url in batch:
            if interrupted:
                break
            fut = pool.submit(scrape_one_bill, bill_number, status_url, delay)
            futures[fut] = bill_number

        for fut in as_completed(futures):
            if interrupted and not fut.done():
                continue
            try:
                bill_number, text, elapsed = fut.result()
            except Exception:
                bill_number = futures[fut]
                logger.exception("Worker crashed for %s", bill_number)
                text, elapsed = None, 0.0

            # ── Merge + batch save under lock ────────────────────────
            with save_lock:
                completed_count += 1
                unsaved_count += 1

                word_count = 0
                status_label = "no PDF"

                # Update the in-memory bill object
                bill = bill_by_number.get(bill_number)
                if bill and text:
                    # Find the bill in bills_lookup by leg_id for persistence
                    for lid, b in bills_lookup.items():
                        if b.bill_number == bill_number:
                            b.full_text = text
                            break

                    if text.startswith("[SKIPPED:"):
                        too_large_bills.add(bill_number)
                        status_label = "SKIPPED (too large)"
                    else:
                        scraped_bills.add(bill_number)
                        word_count = len(text.split())
                        total_words_scraped += word_count
                        bills_with_text += 1
                        status_label = f"{word_count:,} words"
                elif bill:
                    no_pdf_bills.add(bill_number)
                    status_label = "no PDF"

                # Batch save (every N bills, on interrupt, or last bill)
                is_last = completed_count == batch_size
                if unsaved_count >= args.save_interval or interrupted or is_last:
                    _do_save()

                # Print real-time line
                elapsed_total = time.perf_counter() - t_start
                rate = completed_count / elapsed_total if elapsed_total else 0
                eta_s = (batch_size - completed_count) / rate if rate > 0 else 0
                eta_min = eta_s / 60

                print(
                    f"[{completed_count:>{len(str(batch_size))}}/{batch_size}] "
                    f"{bill_number:<10}  "
                    f"{status_label:<20}  "
                    f"({elapsed:.1f}s)  "
                    f"ETA: {eta_min:.1f}m",
                    flush=True,
                )

    # ── Final save if anything unsaved ────────────────────────────────────
    if unsaved_count > 0:
        with save_lock:
            _do_save()

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start

    print(flush=True)
    print("-" * 80, flush=True)
    if interrupted:
        print(
            f"  Stopped: {completed_count}/{batch_size} bills  |  "
            f"{bills_with_text} with text  |  "
            f"{total_words_scraped:,} total words  |  "
            f"{elapsed_total:.1f}s",
            flush=True,
        )
    else:
        print(
            f"  Done: {completed_count}/{batch_size} bills  |  "
            f"{bills_with_text} with text  |  "
            f"{len(no_pdf_bills) - len(already_no_pdf)} no PDF  |  "
            f"{len(too_large_bills) - len(already_too_large)} too large  |  "
            f"{total_words_scraped:,} total words  |  "
            f"{elapsed_total:.1f}s",
            flush=True,
        )
    print(
        f"  Overall: {len(scraped_bills)} scraped / "
        f"{len(no_pdf_bills)} no-PDF / "
        f"{len(too_large_bills)} too-large / "
        f"{total_eligible} substantive total",
        flush=True,
    )
    print("-" * 80, flush=True)
    print("  Progress saved. Run again to continue.", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
