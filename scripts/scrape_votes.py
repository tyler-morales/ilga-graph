#!/usr/bin/env python3
"""Incremental votes + witness slips scraper.

Scrapes vote events and witness slips for bills in cache/bills.json that
haven't been scraped yet.  Progress is saved after **each batch**, so you
can Ctrl+C at any time without losing work.  Run again to resume.

**Robustness:** The "already done" check is dual-source — a bill is skipped
only when it appears in the progress file AND actually has vote/slip data
(or was explicitly checked and found empty).  This prevents the old bug where
the progress file said "done" but bills.json had empty arrays.

**Speed improvements over original:**
  - Batch saves: bills.json written every N bills (default 25), not per-bill
  - Smart ordering: SB/HB first (most important), then resolutions, AM last
  - Heuristic skip: bills stuck at intro/assignments get marked as checked
    without making HTTP requests (saves ~3800 HTTP round-trips)
  - Merge standalone: optionally merges vote_events.json / witness_slips.json

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
    python scripts/scrape_votes.py --verify     # rebuild progress from actual data

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

# Batch save interval: write bills.json every N completed bills (not per-bill)
SAVE_INTERVAL = 25

# Stalled-bill action patterns — bills whose ONLY actions are these are very
# unlikely to have votes or witness slips.  We skip HTTP requests for them.
_STALLED_ACTIONS = frozenset(
    {
        "filed with secretary",
        "filed with the clerk",
        "first reading",
        "referred to assignments",
        "referred to rules committee",
        "added as co-sponsor",
        "added as chief co-sponsor",
        "chief co-sponsor changed",
        "alternate chief co-sponsor changed",
        "alternate chief sponsor changed",
        "rule 19(a) / re-referred to rules committee",
        "session sine die",
    }
)

# Bill type sort priority — lower number = scraped first
_TYPE_PRIORITY = {
    "SB": 0,
    "HB": 1,  # substantive bills first
    "SR": 2,
    "HR": 3,  # resolutions
    "SJR": 4,
    "HJR": 5,
    "SJRCA": 6,
    "HJRCA": 7,
    "JSR": 8,
    "EO": 9,
    "AM": 10,  # amendments last (least important)
}

logging.basicConfig(
    level=logging.WARNING,
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
            "checked_no_data": [],
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
    data.setdefault("checked_no_data", [])
    return data


def save_progress(progress_data: dict) -> None:
    """Persist progress data to disk (atomic write)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    progress_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    progress_data["scraped_bill_numbers"] = sorted(set(progress_data["scraped_bill_numbers"]))
    progress_data["checked_no_data"] = sorted(set(progress_data.get("checked_no_data", [])))
    tmp = PROGRESS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)
    tmp.replace(PROGRESS_FILE)


def rebuild_progress_from_data(bills_lookup: dict[str, Bill]) -> dict:
    """Rebuild progress file from actual data in bills.json.

    A bill is considered 'scraped' if it has non-empty vote_events or
    witness_slips.  This fixes stale progress files that claim 'done'
    when data is actually missing.
    """
    scraped = []
    for bill in bills_lookup.values():
        if bill.vote_events or bill.witness_slips:
            scraped.append(bill.bill_number)

    progress = {
        "scraped_bill_numbers": sorted(scraped),
        "checked_no_data": [],
        "sample_phase": False,
        "sample_rate": None,
        "sample_complete": False,
    }
    save_progress(progress)
    logger.info(
        "Rebuilt progress from data: %d bills with actual vote/slip data.",
        len(scraped),
    )
    return progress


# ── Heuristic helpers ─────────────────────────────────────────────────────────


def _bill_type_sort_key(bill_number: str) -> tuple[int, str]:
    """Sort key that puts SB/HB first, AM last."""
    prefix = ""
    for c in bill_number:
        if c.isalpha():
            prefix += c
        else:
            break
    priority = _TYPE_PRIORITY.get(prefix, 5)
    return (priority, bill_number)


def _is_stalled_bill(bill: Bill) -> bool:
    """Check if a bill is stalled (only intro/assignment actions).

    These bills have never had a committee hearing or floor vote, so they
    won't have vote events or witness slips.  We can skip them without
    making any HTTP requests.
    """
    if not bill.action_history:
        return True

    for entry in bill.action_history:
        action_lower = entry.action.lower().strip()
        # Strip leading sponsor names like "Filed with Secretary bySen. X"
        # Normalise by checking if the action starts with a known stalled prefix
        is_stalled = False
        for stalled in _STALLED_ACTIONS:
            if action_lower.startswith(stalled):
                is_stalled = True
                break
        if not is_stalled:
            return False

    return True


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
    """Extract every Nth bill for sample strategy."""
    return [bill for i, bill in enumerate(all_bills) if i % sample_rate == 0]


# ── Merge standalone cache files ─────────────────────────────────────────────


def merge_standalone_caches(bills_lookup: dict[str, Bill]) -> int:
    """Merge vote_events.json and witness_slips.json into bills that lack data.

    Returns the number of bills updated.
    """
    merged = 0

    # Vote events
    ve_file = CACHE_DIR / "vote_events.json"
    if ve_file.exists():
        with open(ve_file, encoding="utf-8") as f:
            ve_data = json.load(f)
        ve_by_bill: dict[str, list] = {}
        for v in ve_data:
            ve_by_bill.setdefault(v.get("bill_number", ""), []).append(v)

        bn_to_bill = {b.bill_number: b for b in bills_lookup.values()}
        for bn, events in ve_by_bill.items():
            bill = bn_to_bill.get(bn)
            if bill and not bill.vote_events:
                from ilga_graph.models import VoteEvent

                bill.vote_events = [
                    VoteEvent(
                        bill_number=v.get("bill_number", ""),
                        date=v.get("date", ""),
                        description=v.get("description", ""),
                        chamber=v.get("chamber", ""),
                        yea_votes=v.get("yea_votes", []),
                        nay_votes=v.get("nay_votes", []),
                        present_votes=v.get("present_votes", []),
                        nv_votes=v.get("nv_votes", []),
                        pdf_url=v.get("pdf_url", ""),
                        vote_type=v.get("vote_type", "floor"),
                    )
                    for v in events
                ]
                merged += 1

    # Witness slips
    ws_file = CACHE_DIR / "witness_slips.json"
    if ws_file.exists():
        with open(ws_file, encoding="utf-8") as f:
            ws_data = json.load(f)
        ws_by_bill: dict[str, list] = {}
        for s in ws_data:
            ws_by_bill.setdefault(s.get("bill_number", ""), []).append(s)

        bn_to_bill = {b.bill_number: b for b in bills_lookup.values()}
        for bn, slips in ws_by_bill.items():
            bill = bn_to_bill.get(bn)
            if bill and not bill.witness_slips:
                from ilga_graph.models import WitnessSlip

                bill.witness_slips = [
                    WitnessSlip(
                        name=s.get("name", ""),
                        organization=s.get("organization", ""),
                        representing=s.get("representing", ""),
                        position=s.get("position", ""),
                        hearing_committee=s.get("hearing_committee", ""),
                        hearing_date=s.get("hearing_date", ""),
                        testimony_type=s.get("testimony_type", "Record of Appearance Only"),
                        bill_number=s.get("bill_number", ""),
                    )
                    for s in slips
                ]
                merged += 1

    if merged:
        logger.info("Merged standalone cache data into %d bills.", merged)

    return merged


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: C901
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
        "--verify",
        action="store_true",
        help=("Rebuild progress from actual bills.json data (fixes stale progress)."),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=SAVE_INTERVAL,
        help=f"Save bills.json every N bills (default: {SAVE_INTERVAL})",
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help=(
            "Sample strategy: scrape every Nth bill first "
            "(e.g., --sample 10 = 10%% sample). "
            "Run again without --sample to fill gaps."
        ),
    )
    args = parser.parse_args()

    # ── Load bill cache ──────────────────────────────────────────────────
    bills_lookup = load_bill_cache()
    if not bills_lookup:
        logger.error("No bill cache found. Run 'make scrape' first to populate cache/bills.json.")
        sys.exit(1)

    # ── Merge standalone caches (vote_events.json, witness_slips.json) ───
    merged_count = merge_standalone_caches(bills_lookup)
    if merged_count:
        save_bill_cache(bills_lookup)
        logger.info("Saved %d merged standalone records into bills.json.", merged_count)

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

    # ── Validate progress against actual data ─────────────────────────────
    # Remove bills from progress that claim "scraped" but have no data
    # (prevents the stale-progress bug)
    bn_to_bill = {b.bill_number: b for b in bills_lookup.values()}
    validated_scraped = []
    dropped = 0
    for bn in progress["scraped_bill_numbers"]:
        bill = bn_to_bill.get(bn)
        if bill and (bill.vote_events or bill.witness_slips):
            validated_scraped.append(bn)
        else:
            dropped += 1
    if dropped:
        logger.info(
            "Dropped %d stale progress entries (marked done but no data in bills.json).",
            dropped,
        )
        progress["scraped_bill_numbers"] = validated_scraped
        save_progress(progress)

    already_done = set(progress["scraped_bill_numbers"])
    already_checked = set(progress.get("checked_no_data", []))
    skip_set = already_done | already_checked

    # ── Index by bill_number ─────────────────────────────────────────────
    bill_by_number: dict[str, Bill] = {}
    for bill in bills_lookup.values():
        if bill.status_url and bill.status_url.strip():
            bill_by_number[bill.bill_number] = bill

    # ── Heuristic skip: mark stalled bills as checked ─────────────────────
    stalled_count = 0
    new_checked: list[str] = []
    for bn, bill in bill_by_number.items():
        if bn in skip_set:
            continue
        if _is_stalled_bill(bill):
            new_checked.append(bn)
            stalled_count += 1

    if new_checked:
        progress.setdefault("checked_no_data", [])
        progress["checked_no_data"].extend(new_checked)
        save_progress(progress)
        skip_set.update(new_checked)
        logger.info(
            "Skipped %d stalled bills (intro/assignments only — no votes/slips possible).",
            stalled_count,
        )

    # ── Build TODO list with smart ordering ──────────────────────────────
    all_todo = sorted(
        [(bn, bill_by_number[bn].status_url) for bn in bill_by_number if bn not in skip_set],
        key=lambda t: _bill_type_sort_key(t[0]),
    )

    # ── Apply sample strategy if requested ───────────────────────────────
    if args.sample:
        if progress["sample_complete"]:
            todo = all_todo
            phase_label = "gap-fill"
        else:
            todo = get_sample_bills(all_todo, args.sample)
            progress["sample_phase"] = True
            progress["sample_rate"] = args.sample
            phase_label = f"sample (every {args.sample}th bill)"
    else:
        todo = all_todo
        phase_label = "sequential"

    total_eligible = len(bill_by_number)
    total_done = len(already_done)
    total_checked = len(skip_set)

    if not todo:
        logger.info(
            "All %d bills with status URLs already processed "
            "(%d with data, %d checked empty, %d stalled-skip). Nothing to do.",
            total_eligible,
            total_done,
            len(already_checked) + len(new_checked),
            stalled_count,
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
    print("=" * 80, flush=True)
    print(
        f"  Scraping votes + slips for {batch_size} bill(s)  "
        f"({total_done} with data / {total_checked} processed "
        f"/ {total_eligible} total)",
        flush=True,
    )
    print(
        f"  Workers: {args.workers}  |  Delay: {delay}s  |  "
        f"Limit: {args.limit or 'all'}  |  Phase: {phase_label}  |  "
        f"Save every: {args.save_interval} bills",
        flush=True,
    )
    if stalled_count:
        print(
            f"  Heuristic skip: {stalled_count} stalled bills (intro/assignments only)",
            flush=True,
        )
    print("=" * 80, flush=True)
    print(flush=True)

    # ── Shared state ─────────────────────────────────────────────────────
    save_lock = threading.Lock()
    scraped_bills = set(already_done)
    checked_no_data = set(progress.get("checked_no_data", []))
    completed_count = 0
    unsaved_count = 0
    total_votes_scraped = 0
    total_slips_scraped = 0
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
        progress["scraped_bill_numbers"] = list(scraped_bills)
        progress["checked_no_data"] = list(checked_no_data)
        if args.sample and progress["sample_phase"] and not progress["sample_complete"]:
            sample_set = set(
                bn
                for bn, _ in get_sample_bills(
                    sorted([(b, bill_by_number[b].status_url) for b in bill_by_number]),
                    args.sample,
                )
            )
            if sample_set.issubset(scraped_bills | checked_no_data):
                progress["sample_complete"] = True
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
                bill_number, votes, slips, elapsed = fut.result()
            except Exception:
                bill_number = futures[fut]
                logger.exception("Worker crashed for %s", bill_number)
                votes, slips, elapsed = [], [], 0.0

            # ── Merge + batch save under lock ─────────────────────────
            with save_lock:
                completed_count += 1
                unsaved_count += 1
                total_votes_scraped += len(votes)
                total_slips_scraped += len(slips)

                # Update the in-memory bill object
                bill = bill_by_number.get(bill_number)
                if bill:
                    bill.vote_events = votes
                    bill.witness_slips = slips

                # Track progress
                if votes or slips:
                    scraped_bills.add(bill_number)
                else:
                    checked_no_data.add(bill_number)

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
                    f"{len(votes)} votes, {len(slips)} slips  "
                    f"({elapsed:.1f}s)  "
                    f"ETA: {eta_min:.0f}m",
                    flush=True,
                )

    # ── Final save if anything unsaved ────────────────────────────────────
    if unsaved_count > 0:
        with save_lock:
            _do_save()

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    new_total_done = len(scraped_bills)
    new_total_checked = len(scraped_bills) + len(checked_no_data)

    print(flush=True)
    print("-" * 80, flush=True)
    if interrupted:
        print(
            f"  Stopped: {completed_count}/{batch_size} bills  |  "
            f"{total_votes_scraped} vote events  |  "
            f"{total_slips_scraped} slips  |  "
            f"{elapsed_total:.1f}s",
            flush=True,
        )
    else:
        print(
            f"  Done: {completed_count}/{batch_size} bills  |  "
            f"{total_votes_scraped} vote events  |  "
            f"{total_slips_scraped} slips  |  "
            f"{elapsed_total:.1f}s",
            flush=True,
        )
    print(
        f"  Saved to cache/bills.json.  "
        f"Progress: {new_total_done} with data, "
        f"{new_total_checked} total checked "
        f"/ {total_eligible} bills.",
        flush=True,
    )

    # Sample-specific messaging
    if args.sample and progress["sample_complete"] and not interrupted:
        print(
            f"  ✓ Sample phase complete ({args.sample}% of bills).",
            flush=True,
        )
        remaining = total_eligible - new_total_checked
        if remaining > 0:
            print(
                "  Run again WITHOUT --sample to fill "
                f"{remaining} remaining bills (gap-fill phase).",
                flush=True,
            )
    else:
        remaining = total_eligible - new_total_checked
        if remaining > 0:
            print(
                f"  {remaining} bills remaining. Run again to continue.",
                flush=True,
            )
        else:
            print("  All bills complete!", flush=True)

    print("-" * 80, flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
