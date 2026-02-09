#!/usr/bin/env python3
"""Generate trimmed dev mock data from the full cache/.

Produces files in ``mocks/dev/``:

- ``members.json`` -- a diverse subset of members (metadata + bill_ids)
- ``bills.json`` -- the N most recent bills (trimmed, not all referenced bills)
- ``committees.json`` -- all committees (flat list, small)

Usage::

    python scripts/generate_seed.py                     # defaults: 10/chamber, 100 bills
    python scripts/generate_seed.py --limit 5 --bills 50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
MOCK_DEV_DIR = ROOT / "mocks" / "dev"


def _load_json(path: Path) -> list | dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: list | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {path} ({_size_label(path)})")


def _size_label(path: Path) -> str:
    size = path.stat().st_size
    if size > 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size > 1_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size} B"


def _parse_date(date_str: str) -> datetime:
    """Parse M/D/YYYY into datetime. Unparseable dates sort last."""
    try:
        return datetime.strptime(date_str, "%m/%d/%Y")
    except (ValueError, TypeError):
        return datetime.min


def _pick_diverse_members(members: list[dict], limit: int) -> list[dict]:
    """Select a diverse set of members by party and bill count."""
    if len(members) <= limit:
        return members

    def _bill_count(m: dict) -> int:
        return len(m.get("sponsored_bill_ids", m.get("primary_bill_ids", [])))

    dems = [m for m in members if m.get("party", "").lower() == "democrat"]
    reps = [m for m in members if m.get("party", "").lower() == "republican"]
    others = [m for m in members if m not in dems and m not in reps]

    for group in (dems, reps, others):
        group.sort(key=_bill_count, reverse=True)

    picked: list[dict] = []
    seen_ids: set[str] = set()
    sources = [dems, reps, others]
    indices = [0, 0, 0]

    while len(picked) < limit:
        added = False
        for i, source in enumerate(sources):
            if len(picked) >= limit:
                break
            while indices[i] < len(source):
                candidate = source[indices[i]]
                indices[i] += 1
                if candidate.get("id") not in seen_ids:
                    picked.append(candidate)
                    seen_ids.add(candidate.get("id", ""))
                    added = True
                    break
        if not added:
            break

    return picked


def _pick_recent_bills(bills_data: dict, limit: int) -> dict:
    """Select the N most recent bills by last_action_date."""
    sorted_items = sorted(
        bills_data.items(),
        key=lambda item: _parse_date(item[1].get("last_action_date", "")),
        reverse=True,
    )
    return dict(sorted_items[:limit])


def generate_seed(member_limit: int = 10, bill_limit: int = 100) -> None:
    """Generate dev mock data from the full cache/.

    Produces exactly 3 files: members.json, bills.json, committees.json.
    """
    print(f"Generating dev mock data ({member_limit} members/chamber, {bill_limit} bills)...")
    print(f"  Source: {CACHE_DIR}")
    print(f"  Destination: {MOCK_DEV_DIR}")
    print()

    # ── Committees (flat list, ship all of them) ──
    committees_data = _load_json(CACHE_DIR / "committees.json")
    if committees_data is None:
        committees_data = _load_json(MOCK_DEV_DIR / "committees.json")
    if committees_data is None:
        print("ERROR: committees.json not found. Run a full scrape first.")
        sys.exit(1)
    # Handle legacy unified format (dict with "committees" key)
    if isinstance(committees_data, dict) and "committees" in committees_data:
        committees_data = committees_data["committees"]
    _save_json(MOCK_DEV_DIR / "committees.json", committees_data)

    # ── Committee rosters and bills (so GraphQL committee.roster is populated) ──
    rosters_data = _load_json(CACHE_DIR / "committee_rosters.json")
    if rosters_data is not None:
        _save_json(MOCK_DEV_DIR / "committee_rosters.json", rosters_data)
        print(f"  committee_rosters.json  {len(rosters_data)} committees with rosters")
    bills_by_committee = _load_json(CACHE_DIR / "committee_bills.json")
    if bills_by_committee is not None:
        _save_json(MOCK_DEV_DIR / "committee_bills.json", bills_by_committee)
        print(f"  committee_bills.json    {len(bills_by_committee)} committees with bills")

    # ── Vote events (copy from cache if available) ──
    vote_events_data = _load_json(CACHE_DIR / "vote_events.json")
    if vote_events_data is not None:
        _save_json(MOCK_DEV_DIR / "vote_events.json", vote_events_data)
        print(f"  vote_events.json        {len(vote_events_data)} vote events")

    # ── Witness slips (copy from cache if available) ──
    ws_data = _load_json(CACHE_DIR / "witness_slips.json")
    if ws_data is not None:
        _save_json(MOCK_DEV_DIR / "witness_slips.json", ws_data)
        print(f"  witness_slips.json      {len(ws_data)} witness slips")

    # ── Members + Bills (normalized format) ──
    members_data = _load_json(CACHE_DIR / "members.json")
    bills_data = _load_json(CACHE_DIR / "bills.json")
    if members_data is None or bills_data is None:
        print("ERROR: cache/members.json + cache/bills.json not found. Run a full scrape first.")
        sys.exit(1)

    assert isinstance(members_data, list)
    assert isinstance(bills_data, dict)

    # Pick diverse members
    senate = [m for m in members_data if m.get("chamber") == "Senate"]
    house = [m for m in members_data if m.get("chamber") == "House"]
    picked_senate = _pick_diverse_members(senate, member_limit)
    picked_house = _pick_diverse_members(house, member_limit)
    picked_all = picked_senate + picked_house

    # Pick most recent bills (not all referenced bills)
    seed_bills = _pick_recent_bills(bills_data, bill_limit)
    kept_ids = set(seed_bills.keys())

    # Prune member bill_ids to only reference bills we kept
    for m in picked_all:
        m["sponsored_bill_ids"] = [bid for bid in m.get("sponsored_bill_ids", []) if bid in kept_ids]
        m["co_sponsor_bill_ids"] = [bid for bid in m.get("co_sponsor_bill_ids", []) if bid in kept_ids]

    _save_json(MOCK_DEV_DIR / "members.json", picked_all)
    _save_json(MOCK_DEV_DIR / "bills.json", seed_bills)

    print(f"  Selected {len(picked_senate)}/{len(senate)} Senate members")
    print(f"  Selected {len(picked_house)}/{len(house)} House members")
    print(f"  Included {len(seed_bills)}/{len(bills_data)} bills (most recent)")

    # ── Clean up old files that should no longer exist ──
    stale_files = [
        "senate_members.json",
        "house_members.json",
    ]
    for name in stale_files:
        path = MOCK_DEV_DIR / name
        if path.exists():
            path.unlink()
            print(f"  Removed stale {path.name}")

    print()
    print("Done! Dev mock data written to mocks/dev/")
    print(f"  members.json     {len(picked_all)} members")
    print(f"  bills.json       {len(seed_bills)} bills")
    print(f"  committees.json  {len(committees_data)} committees")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dev mock data from cache/.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of members per chamber to include (default: 10)",
    )
    parser.add_argument(
        "--bills",
        type=int,
        default=100,
        help="Number of most recent bills to include (default: 100)",
    )
    args = parser.parse_args()
    generate_seed(member_limit=args.limit, bill_limit=args.bills)


if __name__ == "__main__":
    main()
