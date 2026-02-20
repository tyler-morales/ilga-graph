#!/usr/bin/env python3
"""Sample prod cache into mocks/dev/ for a small, representative dev seed.

Reads from cache/ (prod) and writes a subset to mocks/dev/ so that:
- New contributors can run make dev without scraping (use mocks).
- Mocks stay in sync with schema and a recent subset of real data.
- All cache JSON types the app can use from mocks are represented.

Cache files and whether they go to mocks:
- members.json, bills.json, committees.json  -> yes (core seed data)
- vote_events.json, witness_slips.json       -> yes, when present in cache
- scorecards.json, moneyball.json            -> yes, subset by member id (dev uses these)
- house_committees.json, senate_committees.json -> no; app uses unified committees.json
- scrape_metadata.json                       -> no; scrape state only
- zip_to_district.json                       -> yes; subset for mock districts (ZIP search works)

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

    # Vote events: subset by bill numbers we kept (standalone cache used when seed_fallback)
    vote_events_path = CACHE_SOURCE / "vote_events.json"
    if vote_events_path.exists():
        with open(vote_events_path, encoding="utf-8") as f:
            all_vote_events = json.load(f)
        subset_vote_events = [
            v for v in all_vote_events if v.get("bill_number") in subset_bill_numbers
        ]
        with open(MOCKS_TARGET / "vote_events.json", "w", encoding="utf-8") as f:
            json.dump(subset_vote_events, f, indent=2, ensure_ascii=False)
        logger.info(
            "Wrote %s: %d vote events",
            MOCKS_TARGET / "vote_events.json",
            len(subset_vote_events),
        )
    else:
        logger.info("No %s in cache; skipping vote_events.json", vote_events_path)

    # Witness slips: subset by bill numbers we kept
    witness_slips_path = CACHE_SOURCE / "witness_slips.json"
    if witness_slips_path.exists():
        with open(witness_slips_path, encoding="utf-8") as f:
            all_witness_slips = json.load(f)
        subset_witness_slips = [
            s for s in all_witness_slips if s.get("bill_number") in subset_bill_numbers
        ]
        with open(MOCKS_TARGET / "witness_slips.json", "w", encoding="utf-8") as f:
            json.dump(subset_witness_slips, f, indent=2, ensure_ascii=False)
        logger.info(
            "Wrote %s: %d witness slips",
            MOCKS_TARGET / "witness_slips.json",
            len(subset_witness_slips),
        )
    else:
        logger.info("No %s in cache; skipping witness_slips.json", witness_slips_path)

    # Scorecards: subset by member ids we kept (dev loads from mocks when seed_mode)
    sc_path = CACHE_SOURCE / "scorecards.json"
    mb_path = CACHE_SOURCE / "moneyball.json"
    if sc_path.exists() and mb_path.exists():
        with open(sc_path, encoding="utf-8") as f:
            all_scorecards = json.load(f)
        with open(mb_path, encoding="utf-8") as f:
            mb_raw = json.load(f)
        subset_sc = {mid: all_scorecards[mid] for mid in member_ids if mid in all_scorecards}
        subset_profiles = {
            mid: mb_raw["profiles"][mid] for mid in member_ids if mid in mb_raw.get("profiles", {})
        }

        # Filter ranking lists to subset only, preserve order
        def filter_ranking(rank_list: list) -> list:
            return [x for x in rank_list if x in member_ids]

        mb_subset = {
            "profiles": subset_profiles,
            "rankings_overall": filter_ranking(mb_raw.get("rankings_overall", [])),
            "rankings_house": filter_ranking(mb_raw.get("rankings_house", [])),
            "rankings_senate": filter_ranking(mb_raw.get("rankings_senate", [])),
            "rankings_house_non_leadership": filter_ranking(
                mb_raw.get("rankings_house_non_leadership", [])
            ),
            "rankings_senate_non_leadership": filter_ranking(
                mb_raw.get("rankings_senate_non_leadership", [])
            ),
            "mvp_house_non_leadership": mb_raw.get("mvp_house_non_leadership")
            if mb_raw.get("mvp_house_non_leadership") in member_ids
            else None,
            "mvp_senate_non_leadership": mb_raw.get("mvp_senate_non_leadership")
            if mb_raw.get("mvp_senate_non_leadership") in member_ids
            else None,
            "weights_used": mb_raw.get("weights_used", {}),
        }
        with open(MOCKS_TARGET / "scorecards.json", "w", encoding="utf-8") as f:
            json.dump(subset_sc, f, indent=0, ensure_ascii=False)
        with open(MOCKS_TARGET / "moneyball.json", "w", encoding="utf-8") as f:
            json.dump(mb_subset, f, indent=0, ensure_ascii=False)
        logger.info(
            "Wrote %s: %d scorecards; %s: %d profiles",
            MOCKS_TARGET / "scorecards.json",
            len(subset_sc),
            MOCKS_TARGET / "moneyball.json",
            len(subset_profiles),
        )
    else:
        logger.info("No scorecards.json or moneyball.json in cache; skipping (dev will recompute).")

    # ZIP crosswalk: keep only ZIPs that map to (il_senate, il_house) pairs in our mock members,
    # so every ZIP search can resolve to Your Senator + Your Rep from the 40 members.
    senate_districts = {m["district"] for m in subset_members if m.get("chamber") == "Senate"}
    house_districts = {m["district"] for m in subset_members if m.get("chamber") == "House"}
    zip_path = CACHE_SOURCE / "zip_to_district.json"
    if zip_path.exists():
        with open(zip_path, encoding="utf-8") as f:
            all_zip = json.load(f)
        subset_zip = {
            zcta: info
            for zcta, info in all_zip.items()
            if info.get("il_senate") in senate_districts and info.get("il_house") in house_districts
        }
        with open(MOCKS_TARGET / "zip_to_district.json", "w", encoding="utf-8") as f:
            json.dump(subset_zip, f, indent=2, ensure_ascii=False)
        logger.info(
            "Wrote %s: %d ZIPs (districts covered: %d Senate, %d House)",
            MOCKS_TARGET / "zip_to_district.json",
            len(subset_zip),
            len(senate_districts),
            len(house_districts),
        )
    else:
        logger.info("No zip_to_district.json in cache; skipping (dev uses hardcoded seed ZIPs).")

    logger.info("Done. Commit mocks/dev/ to refresh the dev seed for everyone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
