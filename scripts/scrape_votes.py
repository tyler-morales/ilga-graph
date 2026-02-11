#!/usr/bin/env python3
"""Incremental votes + witness slips scraper.

Scrapes vote events and witness slips for bills in cache/bills.json that
haven't been scraped yet.  Progress is saved after **each bill**, so you
can Ctrl+C at any time without losing work.  Run again to resume.

Supports two-phase strategy:
1. **Sample phase**: Scrape every Nth bill first (fast representative dataset)
2. **Gap-fill phase**: Fill in remaining bills after sample completes

Usage::

    make scrape-votes              # next 10 bills (default)
    make scrape-votes LIMIT=50     # next 50 bills
    make scrape-votes LIMIT=0      # all remaining bills

    python scripts/scrape_votes.py              # next 10
    python scripts/scrape_votes.py --limit 50   # next 50
    python scripts/scrape_votes.py --limit 0    # all remaining
    python scripts/scrape_votes.py --workers 8  # 8 parallel workers
    python scripts/scrape_votes.py --fast       # shorter request delay
    python scripts/scrape_votes.py --reset      # wipe progress, start over

    # Sample strategy (10% representative sample first)
    python scripts/scrape_votes.py --sample 10  # every 10th bill
    python scripts/scrape_votes.py --sample 10 --limit 0  # all sampled bills
"""

from __future__ import annotations

import argparse
import json
import logging
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
from ilga_graph.scrapers.bills import load_bill_cache, save_bill_cache  # noqa: E402
from ilga_graph.scrapers.votes import scrape_bill_votes  # noqa: E402
from ilga_graph.scrapers.witness_slips import scrape_witness_slips  # noqa: E402

PROGRESS_FILE = CACHE_DIR / "votes_slips_progress.json"

logging.basicConfig(
    level=logging.WARNING,  # quiet by default; real-time lines replace logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scrape_votes")
logger.setLevel(logging.INFO)


# ── Progress helpers ──────────────────────────────────────────────────────────


def load_progress() -> dict:
    """Load progress data including scraped bills and sample phase info."""
    if not PROGRESS_FILE.exists():
        return {
            "scraped_bill_numbers": [],
            "sample_phase": False,
            "sample_rate": None,
            "sample_complete": False,
        }
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    # Ensure all keys exist
    data.setdefault("sample_phase", False)
    data.setdefault("sample_rate", None)
    data.setdefault("sample_complete", False)
    return data


def save_progress(progress_data: dict) -> None:
    """Persist progress data to disk (atomic write)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    progress_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    progress_data["scraped_bill_numbers"] = sorted(progress_data["scraped_bill_numbers"])
    tmp = PROGRESS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)
    tmp.replace(PROGRESS_FILE)


# ── Per-bill worker ──────────────────────────────────────────────────────────


def scrape_one_bill(
    bill_number: str,
    status_url: str,
    request_delay: float,
) -> tuple[str, list, list, float]:
    """Scrape votes + witness slips for a single bill.

    Returns (bill_number, vote_events, witness_slips, elapsed_seconds).
    """
    import requests as _req

    sess = _req.Session()
    t0 = time.perf_counter()

    votes = []
    slips = []
    try:
        votes = scrape_bill_votes(status_url, session=sess, request_delay=request_delay)
    except Exception:
        logger.exception("  Failed votes for %s", bill_number)
    try:
        slips = scrape_witness_slips(status_url, session=sess, request_delay=request_delay)
    except Exception:
        logger.exception("  Failed slips for %s", bill_number)

    elapsed = time.perf_counter() - t0
    return bill_number, votes, slips, elapsed


# ── Sample strategy helpers ──────────────────────────────────────────────────


def get_sample_bills(
    all_bills: list[tuple[str, str]],
    sample_rate: int,
) -> list[tuple[str, str]]:
    """Extract every Nth bill for sample strategy.

    Args:
        all_bills: List of (bill_number, status_url) tuples sorted by bill_number
        sample_rate: Take every Nth bill (e.g., 10 = every 10th bill = 10% sample)

    Returns:
        Sampled bills in original order
    """
    return [bill for i, bill in enumerate(all_bills) if i % sample_rate == 0]


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally scrape votes + witness slips for cached bills.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of bills to scrape this run (0 = all remaining; default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use shorter request delay (0.15s instead of 0.4s)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe progress file and start from scratch",
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help=(
            "Sample strategy: scrape every Nth bill first (e.g., --sample 10 = 10%% sample). "
            "Run again without --sample to fill gaps."
        ),
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

    # ── Load progress ────────────────────────────────────────────────────
    progress = load_progress()
    already_done = set(progress["scraped_bill_numbers"])

    # Index by bill_number for quick lookup + merge later
    bill_by_number: dict[str, Bill] = {}
    for bill in bills_lookup.values():
        if bill.status_url and bill.status_url.strip():
            bill_by_number[bill.bill_number] = bill

    all_todo = sorted(
        [(bn, bill_by_number[bn].status_url) for bn in bill_by_number if bn not in already_done],
        key=lambda t: t[0],
    )

    # ── Apply sample strategy if requested ───────────────────────────────
    if args.sample:
        if progress["sample_complete"]:
            # Sample already done, do gap fill
            todo = all_todo
            phase_label = "gap-fill"
        else:
            # First run with --sample: take every Nth bill
            todo = get_sample_bills(all_todo, args.sample)
            progress["sample_phase"] = True
            progress["sample_rate"] = args.sample
            phase_label = f"sample (every {args.sample}th bill)"
    else:
        # No sampling: scrape sequentially
        todo = all_todo
        phase_label = "sequential"

    total_eligible = len(bill_by_number)
    total_done = len(already_done)

    if not todo:
        logger.info(
            "All %d bills with status URLs already have votes/slips data. Nothing to do.",
            total_eligible,
        )
        return

    # Apply limit
    if args.limit > 0:
        batch = todo[: args.limit]
    else:
        batch = todo

    batch_size = len(batch)
    delay = 0.15 if args.fast else 0.4

    print(flush=True)
    print("=" * 72, flush=True)
    print(
        f"  Scraping votes + slips for {batch_size} bill(s)  "
        f"({total_done} done / {total_eligible} total with status_url)",
        flush=True,
    )
    print(
        f"  Workers: {args.workers}  |  Delay: {delay}s  |  "
        f"Limit: {args.limit or 'all'}  |  Phase: {phase_label}",
        flush=True,
    )
    print("=" * 72, flush=True)
    print(flush=True)

    # ── Shared state ─────────────────────────────────────────────────────
    save_lock = threading.Lock()
    scraped_bills = set(already_done)  # mutable copy for this batch
    completed_count = 0
    total_votes_scraped = 0
    total_slips_scraped = 0
    interrupted = False
    t_start = time.perf_counter()

    def _handle_sigint(signum, frame):
        nonlocal interrupted
        if interrupted:
            # Second Ctrl+C: force exit
            print("\nForce exit.", flush=True)
            sys.exit(1)
        interrupted = True
        print(
            "\nInterrupted! Waiting for in-flight workers to finish... (Ctrl+C again to force)",
            flush=True,
        )

    signal.signal(signal.SIGINT, _handle_sigint)

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
                bill_number, votes, slips, elapsed = fut.result()
            except Exception:
                bill_number = futures[fut]
                logger.exception("Worker crashed for %s", bill_number)
                votes, slips, elapsed = [], [], 0.0

            # ── Merge + save under lock ──────────────────────────────
            with save_lock:
                completed_count += 1
                total_votes_scraped += len(votes)
                total_slips_scraped += len(slips)

                # Update the in-memory bill object
                bill = bill_by_number.get(bill_number)
                if bill:
                    bill.vote_events = votes
                    bill.witness_slips = slips

                # Save bills.json (atomic)
                save_bill_cache(bills_lookup)

                # Update progress
                scraped_bills.add(bill_number)
                progress["scraped_bill_numbers"] = list(scraped_bills)

                # Mark sample complete if we just finished a sample batch
                if args.sample and progress["sample_phase"] and not progress["sample_complete"]:
                    # Check if we've done all sample bills
                    sample_set = set(
                        bn
                        for bn, _ in get_sample_bills(
                            sorted([(b, bill_by_number[b].status_url) for b in bill_by_number]),
                            args.sample,
                        )
                    )
                    if sample_set.issubset(scraped_bills):
                        progress["sample_complete"] = True

                save_progress(progress)

                # Print real-time line
                print(
                    f"[{completed_count:>{len(str(batch_size))}}/{batch_size}] "
                    f"{bill_number:<10}  --  {len(votes)} votes, {len(slips)} slips  "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    new_total_done = len(scraped_bills)

    print(flush=True)
    print("-" * 72, flush=True)
    if interrupted:
        print(
            f"  Stopped: {completed_count}/{batch_size} bills  |  "
            f"{total_votes_scraped} vote events  |  {total_slips_scraped} slips  |  "
            f"{elapsed_total:.1f}s",
            flush=True,
        )
    else:
        print(
            f"  Done: {completed_count}/{batch_size} bills  |  "
            f"{total_votes_scraped} vote events  |  {total_slips_scraped} slips  |  "
            f"{elapsed_total:.1f}s",
            flush=True,
        )
    print(
        f"  Saved to cache/bills.json.  Progress: {new_total_done}/{total_eligible} bills "
        f"have votes/slips data.",
        flush=True,
    )

    # Sample-specific messaging
    if args.sample and progress["sample_complete"] and not interrupted:
        print(
            f"  ✓ Sample phase complete ({args.sample}% of bills).",
            flush=True,
        )
        remaining_after_sample = total_eligible - new_total_done
        if remaining_after_sample > 0:
            print(
                "  Run again WITHOUT --sample to fill "
                f"{remaining_after_sample} remaining bills (gap-fill phase).",
                flush=True,
            )
    else:
        remaining = total_eligible - new_total_done
        if remaining > 0:
            print(
                f"  {remaining} bills remaining. Run again to continue.",
                flush=True,
            )
        else:
            print("  All bills complete!", flush=True)

    print("-" * 72, flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
