#!/usr/bin/env python3
"""Sample prod cache into mocks/dev/ for a small, representative dev seed.

Reads from cache/ (prod) and writes a subset to mocks/dev/ so that:
- New contributors can run make dev without scraping (use mocks).
- Mocks stay in sync with schema and a recent subset of real data.

Run after a full scrape when you want to refresh the committed mocks:
  make snapshot-mocks

Does not run automatically on scrape; run explicitly and commit the result.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_SOURCE = ROOT / "cache"
MOCKS_TARGET = ROOT / "mocks" / "dev"

# Subset sizes (tune as needed)
MAX_SENATE_MEMBERS = 20
MAX_HOUSE_MEMBERS = 20
MAX_BILLS = 150
MIN_BILLS_WITH_VOTES = 50
MAX_COMMITTEES = 40

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    if not (CACHE_SOURCE / "members.json").exists():
        logger.error("No cache found at %s. Run 'make scrape' first.", CACHE_SOURCE)
        return 1
    if not (CACHE_SOURCE / "bills.json").exists():
        logger.error("No bills.json in cache. Run 'make scrape' first.")
        return 1

    # Load from prod cache
    with open(CACHE_SOURCE / "members.json", encoding="utf-8") as f:
        all_members = json.load(f)
    with open(CACHE_SOURCE / "bills.json", encoding="utf-8") as f:
        all_bills_raw = json.load(f)

    # Subset members: 20 Senate + 20 House (stable order by id)
    senate = sorted([m for m in all_members if m.get("chamber") == "Senate"], key=lambda m: m["id"])
    house = sorted([m for m in all_members if m.get("chamber") == "House"], key=lambda m: m["id"])
    subset_members = senate[:MAX_SENATE_MEMBERS] + house[:MAX_HOUSE_MEMBERS]
    member_ids = {m["id"] for m in subset_members}

    # Bills: sponsored/co-sponsored by subset members first, then bills with votes/slips
    leg_ids_from_members = set()
    for m in subset_members:
        leg_ids_from_members.update(m.get("sponsored_bill_ids", []))
        leg_ids_from_members.update(m.get("co_sponsor_bill_ids", []))

    bills_with_data = []
    other_bills = []
    for leg_id, b in all_bills_raw.items():
        if leg_id in leg_ids_from_members:
            continue
        has_votes = bool(b.get("vote_events"))
        has_slips = bool(b.get("witness_slips"))
        if has_votes or has_slips:
            bills_with_data.append((leg_id, b))
        else:
            other_bills.append((leg_id, b))

    subset_bills_raw = {leg_id: all_bills_raw[leg_id] for leg_id in leg_ids_from_members}
    # Add bills that have vote/slip data up to MIN_BILLS_WITH_VOTES
    for leg_id, b in bills_with_data[: max(0, MIN_BILLS_WITH_VOTES - len(subset_bills_raw))]:
        subset_bills_raw[leg_id] = b
    # Cap total
    if len(subset_bills_raw) > MAX_BILLS:
        keys = list(subset_bills_raw.keys())[:MAX_BILLS]
        subset_bills_raw = {k: subset_bills_raw[k] for k in keys}

    subset_leg_ids = set(subset_bills_raw.keys())
    subset_bill_numbers = {
        b.get("bill_number") for b in subset_bills_raw.values() if b.get("bill_number")
    }

    # Trim member bill id lists to only include bills we kept
    for m in subset_members:
        m["sponsored_bill_ids"] = [
            x for x in m.get("sponsored_bill_ids", []) if x in subset_leg_ids
        ]
        m["co_sponsor_bill_ids"] = [
            x for x in m.get("co_sponsor_bill_ids", []) if x in subset_leg_ids
        ]

    # Committees: load and filter to those with roster overlap or bill overlap
    committees_path = CACHE_SOURCE / "committees.json"
    if committees_path.exists():
        with open(committees_path, encoding="utf-8") as f:
            all_committees = json.load(f)
        subset_committees = []
        for c in all_committees:
            roster = c.get("roster", [])
            bill_nums = c.get("bill_numbers", []) or []
            roster_ids = {r.get("member_id") for r in roster if r.get("member_id")}
            if not (roster_ids & member_ids) and not (set(bill_nums) & subset_bill_numbers):
                continue
            # Keep committee; filter roster and bill_numbers to our subset
            new_roster = [r for r in roster if r.get("member_id") in member_ids]
            new_bill_nums = [bn for bn in bill_nums if bn in subset_bill_numbers]
            entry = {**c, "roster": new_roster, "bill_numbers": new_bill_nums}
            subset_committees.append(entry)
            if len(subset_committees) >= MAX_COMMITTEES:
                break
    else:
        subset_committees = []

    # Write to mocks/dev/
    MOCKS_TARGET.mkdir(parents=True, exist_ok=True)

    with open(MOCKS_TARGET / "members.json", "w", encoding="utf-8") as f:
        json.dump(subset_members, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s: %d members", MOCKS_TARGET / "members.json", len(subset_members))

    with open(MOCKS_TARGET / "bills.json", "w", encoding="utf-8") as f:
        json.dump(subset_bills_raw, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s: %d bills", MOCKS_TARGET / "bills.json", len(subset_bills_raw))

    if subset_committees:
        with open(MOCKS_TARGET / "committees.json", "w", encoding="utf-8") as f:
            json.dump(subset_committees, f, indent=2, ensure_ascii=False)
        logger.info(
            "Wrote %s: %d committees",
            MOCKS_TARGET / "committees.json",
            len(subset_committees),
        )

    logger.info("Done. Commit mocks/dev/ to refresh the dev seed for everyone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
