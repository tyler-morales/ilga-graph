#!/usr/bin/env python3
"""Review unmatched SBE Active Candidate committees.

Does not invent member links or disclosure amounts. Prints why each
committee is unmatched and a gold stub you can confirm by hand.

Usage::

    PYTHONPATH=src python scripts/review_sbe_unmatched.py --from-dir tests/fixtures/sbe
    PYTHONPATH=src python scripts/review_sbe_unmatched.py --unmatched-json processed/campaign_finance/unmatched.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ilga_graph.campaign_finance.match import (  # noqa: E402
    format_unmatched_review,
    match_committees_to_members,
    unmatched_review_rows,
)
from ilga_graph.campaign_finance.parse import parse_sbe_dir  # noqa: E402
from ilga_graph.scraper import load_normalized_cache  # noqa: E402

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
        "--from-dir",
        type=Path,
        help="Parse local SBE .txt files and rematch against sitting members",
    )
    parser.add_argument(
        "--unmatched-json",
        type=Path,
        help="Read an existing unmatched.json instead of rematching",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print review rows as JSON",
    )
    args = parser.parse_args()
    if bool(args.from_dir) == bool(args.unmatched_json):
        raise SystemExit("Pass exactly one of --from-dir or --unmatched-json")

    if args.unmatched_json:
        rows = json.loads(args.unmatched_json.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise SystemExit("unmatched.json must be a JSON list")
    else:
        members = _load_members()
        gold = GOLD_PATH if GOLD_PATH.exists() else None
        parsed = parse_sbe_dir(args.from_dir)
        report = match_committees_to_members(parsed, members, gold_path=gold)
        rows = unmatched_review_rows(report)
        print(
            f"match_rate={report.match_rate:.4f} "
            f"accepted={len(report.accepted)} review={len(report.unmatched)} "
            f"legislative_committees={report.legislative_committees}",
            file=sys.stderr,
        )

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(format_unmatched_review(rows), end="")


if __name__ == "__main__":
    main()
