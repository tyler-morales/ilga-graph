"""ETL pipeline: flatten cache/*.json into processed/*.parquet star schema.

Reads the nested JSON caches produced by the scraper pipeline and normalizes
them into a columnar Parquet star schema optimized for ML workloads:

    processed/
        dim_members.parquet          - Member dimension table
        dim_bills.parquet            - Bill dimension table
        fact_bill_actions.parquet    - Bill action history (fact)
        fact_vote_events.parquet     - Aggregated vote events (fact)
        fact_vote_casts_raw.parquet  - Individual vote casts, pre-resolution (fact)
        fact_witness_slips.parquet   - Witness slip filings (fact)

Uses Polars for speed and native nested-struct handling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import polars as pl

from .action_classifier import action_category_for_etl as _classify_action

LOGGER = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("cache")
PROCESSED_DIR = Path("processed")

BILLS_PATH = CACHE_DIR / "bills.json"
MEMBERS_PATH = CACHE_DIR / "members.json"
VOTE_EVENTS_PATH = CACHE_DIR / "vote_events.json"


def _hash_id(*parts: str) -> str:
    """Generate a stable 16-char hex ID from concatenated parts."""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:16]


# ── Bill number parsing ──────────────────────────────────────────────────────

_BILL_NUM_RE = re.compile(r"^([A-Z]+)\s*0*(\d+)(?:\s*\((.+)\))?$")


def _parse_bill_number(raw: str) -> tuple[str, int | None, str]:
    """Parse 'SB0009' -> ('SB', 9, '') or 'HB123 (SCA1)' -> ('HB', 123, 'SCA1')."""
    m = _BILL_NUM_RE.match(raw.strip())
    if m:
        return m.group(1), int(m.group(2)), m.group(3) or ""
    return raw, None, ""


# ── Date parsing helpers ─────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%m/%d/%Y",  # 1/13/2025
    "%B %d, %Y",  # May 31, 2025
    "%Y-%m-%d %H:%M",  # 2025-03-11 09:00
    "%Y-%m-%d",  # 2025-03-11
]


_date_parse_failures: set[str] = set()  # track unique failures to avoid log spam


def _parse_date_str(s: str) -> str | None:
    """Try multiple date formats, return ISO date string or None."""
    if not s or not s.strip():
        return None
    from datetime import datetime

    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Log once per unique unparseable value
    if s not in _date_parse_failures:
        _date_parse_failures.add(s)
        LOGGER.warning("Could not parse date string: %r", s[:60])
    return None


# ── Action category mapping ──────────────────────────────────────────────────
# Uses the comprehensive action_classifier module which handles ILGA's
# inconsistent formatting and properly distinguishes bill vs. amendment actions.

# ── 1. dim_members ───────────────────────────────────────────────────────────


def process_members() -> pl.DataFrame:
    """Load members.json -> dim_members.parquet."""
    LOGGER.info("Processing members...")
    with open(MEMBERS_PATH) as f:
        members_raw = json.load(f)

    rows = []
    for m in members_raw:
        career_start = None
        career_end = None
        if m.get("career_ranges"):
            career_start = min(
                (cr["start_year"] for cr in m["career_ranges"] if cr.get("start_year")),
                default=None,
            )
            career_end = max(
                (cr["end_year"] for cr in m["career_ranges"] if cr.get("end_year")),
                default=None,
            )

        rows.append(
            {
                "member_id": m["id"],
                "name": m["name"],
                "name_clean": m["name"].upper().strip(),
                "party": m.get("party", ""),
                "chamber": m.get("chamber", ""),
                "district": int(m["district"]) if m.get("district", "").isdigit() else None,
                "career_start": career_start,
                "career_end": career_end,
                "role": m.get("role") or "",
                "email": m.get("email") or "",
                "sponsored_bill_count": len(m.get("sponsored_bill_ids", [])),
                "cosponsor_bill_count": len(m.get("co_sponsor_bill_ids", [])),
            }
        )

    df = pl.DataFrame(rows)
    out_path = PROCESSED_DIR / "dim_members.parquet"
    df.write_parquet(out_path)
    LOGGER.info("  dim_members: %d rows -> %s", len(df), out_path)
    return df


# ── 2. dim_bills + fact tables ───────────────────────────────────────────────


def _load_bills_raw() -> dict:
    """Load the bills JSON dict (leg_id -> bill object)."""
    with open(BILLS_PATH) as f:
        return json.load(f)


def process_bills() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load bills.json -> dim_bills, fact_bill_actions, fact_vote_events,
    fact_vote_casts_raw, fact_witness_slips parquet files.

    Returns (dim_bills, fact_actions, fact_vote_events, fact_vote_casts_raw).
    """
    LOGGER.info("Processing bills...")
    bills_raw = _load_bills_raw()

    bill_rows = []
    action_rows = []
    vote_event_rows = []
    vote_cast_rows = []
    slip_rows = []

    for leg_id, b in bills_raw.items():
        bill_number_raw = b.get("bill_number", "")
        bill_type, bill_number_int, amendment_suffix = _parse_bill_number(bill_number_raw)

        # Find introduction date from first action
        introduction_date = None
        if b.get("action_history"):
            introduction_date = _parse_date_str(b["action_history"][0].get("date", ""))

        # Primary sponsor ID = first in sponsor_ids
        sponsor_ids = b.get("sponsor_ids", [])
        primary_sponsor_id = sponsor_ids[0] if sponsor_ids else None

        # Chamber: map S->Senate, H->House
        chamber_raw = b.get("chamber", "")
        chamber_origin = {"S": "Senate", "H": "House"}.get(chamber_raw, chamber_raw)

        bill_rows.append(
            {
                "bill_id": leg_id,
                "bill_number_raw": bill_number_raw,
                "bill_type": bill_type,
                "bill_number_int": bill_number_int,
                "amendment_suffix": amendment_suffix,
                "description": b.get("description", ""),
                "synopsis_text": b.get("synopsis", ""),
                "chamber_origin": chamber_origin,
                "primary_sponsor": b.get("primary_sponsor", ""),
                "primary_sponsor_id": primary_sponsor_id,
                "introduction_date": introduction_date,
                "last_action": b.get("last_action", ""),
                "last_action_date": _parse_date_str(b.get("last_action_date", "")),
                "sponsor_count": len(sponsor_ids) + len(b.get("house_sponsor_ids", [])),
                "status_url": b.get("status_url", ""),
                "full_text": b.get("full_text", ""),
            }
        )

        # ── Actions ──────────────────────────────────────────────────────
        for act in b.get("action_history", []):
            action_date = _parse_date_str(act.get("date", ""))
            action_text = act.get("action", "")
            action_rows.append(
                {
                    "action_id": _hash_id(leg_id, act.get("date", ""), action_text),
                    "bill_id": leg_id,
                    "date": action_date,
                    "chamber": act.get("chamber", ""),
                    "action_text": action_text,
                    "action_category": _classify_action(action_text),
                }
            )

        # ── Vote events ──────────────────────────────────────────────────
        for ve in b.get("vote_events", []):
            ve_date = _parse_date_str(ve.get("date", ""))
            ve_desc = ve.get("description", "")
            ve_id = _hash_id(leg_id, ve.get("date", ""), ve_desc)
            yea = ve.get("yea_votes", [])
            nay = ve.get("nay_votes", [])
            present = ve.get("present_votes", [])
            nv = ve.get("nv_votes", [])

            total_yea = len(yea)
            total_nay = len(nay)

            # Determine outcome
            if total_yea > total_nay:
                outcome = "passed"
            elif total_yea < total_nay:
                outcome = "lost"
            else:
                outcome = "tied"

            vote_event_rows.append(
                {
                    "vote_event_id": ve_id,
                    "bill_id": leg_id,
                    "bill_number": ve.get("bill_number", bill_number_raw),
                    "date": ve_date,
                    "chamber": ve.get("chamber", ""),
                    "vote_type": ve.get("vote_type", "floor"),
                    "description": ve_desc,
                    "outcome": outcome,
                    "total_yea": total_yea,
                    "total_nay": total_nay,
                    "total_present": len(present),
                    "total_nv": len(nv),
                    "pdf_url": ve.get("pdf_url", ""),
                }
            )

            # ── Individual vote casts ────────────────────────────────────
            for name in yea:
                vote_cast_rows.append(
                    {
                        "vote_event_id": ve_id,
                        "bill_id": leg_id,
                        "raw_voter_name": name,
                        "vote_cast": "Y",
                        "chamber": ve.get("chamber", ""),
                        "date": ve_date,
                    }
                )
            for name in nay:
                vote_cast_rows.append(
                    {
                        "vote_event_id": ve_id,
                        "bill_id": leg_id,
                        "raw_voter_name": name,
                        "vote_cast": "N",
                        "chamber": ve.get("chamber", ""),
                        "date": ve_date,
                    }
                )
            for name in present:
                vote_cast_rows.append(
                    {
                        "vote_event_id": ve_id,
                        "bill_id": leg_id,
                        "raw_voter_name": name,
                        "vote_cast": "P",
                        "chamber": ve.get("chamber", ""),
                        "date": ve_date,
                    }
                )
            for name in nv:
                vote_cast_rows.append(
                    {
                        "vote_event_id": ve_id,
                        "bill_id": leg_id,
                        "raw_voter_name": name,
                        "vote_cast": "NV",
                        "chamber": ve.get("chamber", ""),
                        "date": ve_date,
                    }
                )

        # ── Witness slips ────────────────────────────────────────────────
        for slip in b.get("witness_slips", []):
            slip_name = slip.get("name", "")
            slip_org = slip.get("organization", "")
            slip_date = _parse_date_str(slip.get("hearing_date", ""))
            slip_rows.append(
                {
                    "slip_id": _hash_id(leg_id, slip_name, slip_org, slip.get("hearing_date", "")),
                    "bill_id": leg_id,
                    "bill_number": slip.get("bill_number", bill_number_raw),
                    "name_raw": slip_name,
                    "name_clean": slip_name.upper().strip(),
                    "organization_raw": slip_org,
                    "organization_clean": slip_org.upper().strip(),
                    "representing": slip.get("representing", ""),
                    "position": slip.get("position", ""),
                    "testimony_type": slip.get("testimony_type", ""),
                    "hearing_committee": slip.get("hearing_committee", ""),
                    "hearing_date": slip_date,
                }
            )

    # ── Write parquet files ──────────────────────────────────────────────

    # dim_bills
    df_bills = pl.DataFrame(bill_rows) if bill_rows else pl.DataFrame()
    df_bills.write_parquet(PROCESSED_DIR / "dim_bills.parquet")
    LOGGER.info("  dim_bills: %d rows", len(df_bills))

    # fact_bill_actions
    df_actions = pl.DataFrame(action_rows) if action_rows else pl.DataFrame()
    df_actions.write_parquet(PROCESSED_DIR / "fact_bill_actions.parquet")
    LOGGER.info("  fact_bill_actions: %d rows", len(df_actions))

    # fact_vote_events
    df_vote_events = pl.DataFrame(vote_event_rows) if vote_event_rows else pl.DataFrame()
    df_vote_events.write_parquet(PROCESSED_DIR / "fact_vote_events.parquet")
    LOGGER.info("  fact_vote_events: %d rows", len(df_vote_events))

    # fact_vote_casts_raw
    df_vote_casts = pl.DataFrame(vote_cast_rows) if vote_cast_rows else pl.DataFrame()
    df_vote_casts.write_parquet(PROCESSED_DIR / "fact_vote_casts_raw.parquet")
    LOGGER.info("  fact_vote_casts_raw: %d rows", len(df_vote_casts))

    # fact_witness_slips
    df_slips = pl.DataFrame(slip_rows) if slip_rows else pl.DataFrame()
    df_slips.write_parquet(PROCESSED_DIR / "fact_witness_slips.parquet")
    LOGGER.info("  fact_witness_slips: %d rows", len(df_slips))

    return df_bills, df_actions, df_vote_events, df_vote_casts


# ── Also process standalone vote_events.json (supplements per-bill data) ─────


def process_standalone_vote_events() -> pl.DataFrame | None:
    """Process cache/vote_events.json (standalone, not per-bill).

    These are vote events scraped independently. They may overlap with per-bill
    data but provide additional coverage. Merged into the main fact tables.
    """
    if not VOTE_EVENTS_PATH.exists():
        return None

    LOGGER.info("Processing standalone vote_events.json...")
    with open(VOTE_EVENTS_PATH) as f:
        events_raw = json.load(f)

    if not events_raw:
        return None

    rows = []
    cast_rows = []

    for ve in events_raw:
        ve_date = _parse_date_str(ve.get("date", ""))
        ve_desc = ve.get("description", "")
        bill_number = ve.get("bill_number", "")
        ve_id = _hash_id("standalone", bill_number, ve.get("date", ""), ve_desc)

        yea = ve.get("yea_votes", [])
        nay = ve.get("nay_votes", [])
        present = ve.get("present_votes", [])
        nv = ve.get("nv_votes", [])

        rows.append(
            {
                "vote_event_id": ve_id,
                "bill_id": "",  # No leg_id in standalone
                "bill_number": bill_number,
                "date": ve_date,
                "chamber": ve.get("chamber", ""),
                "vote_type": ve.get("vote_type", "floor"),
                "description": ve_desc,
                "outcome": "passed" if len(yea) > len(nay) else "lost",
                "total_yea": len(yea),
                "total_nay": len(nay),
                "total_present": len(present),
                "total_nv": len(nv),
                "pdf_url": ve.get("pdf_url", ""),
            }
        )

        for name in yea:
            cast_rows.append(
                {
                    "vote_event_id": ve_id,
                    "bill_id": "",
                    "raw_voter_name": name,
                    "vote_cast": "Y",
                    "chamber": ve.get("chamber", ""),
                    "date": ve_date,
                }
            )
        for name in nay:
            cast_rows.append(
                {
                    "vote_event_id": ve_id,
                    "bill_id": "",
                    "raw_voter_name": name,
                    "vote_cast": "N",
                    "chamber": ve.get("chamber", ""),
                    "date": ve_date,
                }
            )
        for name in present:
            cast_rows.append(
                {
                    "vote_event_id": ve_id,
                    "bill_id": "",
                    "raw_voter_name": name,
                    "vote_cast": "P",
                    "chamber": ve.get("chamber", ""),
                    "date": ve_date,
                }
            )
        for name in nv:
            cast_rows.append(
                {
                    "vote_event_id": ve_id,
                    "bill_id": "",
                    "raw_voter_name": name,
                    "vote_cast": "NV",
                    "chamber": ve.get("chamber", ""),
                    "date": ve_date,
                }
            )

    df_events = pl.DataFrame(rows) if rows else pl.DataFrame()
    df_casts = pl.DataFrame(cast_rows) if cast_rows else pl.DataFrame()

    LOGGER.info("  standalone vote_events: %d events, %d casts", len(df_events), len(df_casts))
    return df_events


# ── Main pipeline runner ─────────────────────────────────────────────────────


def run_pipeline() -> dict[str, pl.DataFrame]:
    """Run the full ETL pipeline. Returns dict of table_name -> DataFrame."""
    PROCESSED_DIR.mkdir(exist_ok=True)

    results: dict[str, pl.DataFrame] = {}

    # Dimension tables
    results["dim_members"] = process_members()

    # Fact tables from bills
    df_bills, df_actions, df_vote_events, df_vote_casts = process_bills()
    results["dim_bills"] = df_bills
    results["fact_bill_actions"] = df_actions
    results["fact_vote_events"] = df_vote_events
    results["fact_vote_casts_raw"] = df_vote_casts

    # Merge standalone vote events if they exist
    df_standalone = process_standalone_vote_events()
    if df_standalone is not None and len(df_standalone) > 0:
        # Deduplicate by vote_event_id (per-bill data takes precedence)
        if len(df_vote_events) > 0:
            existing_ids = set(df_vote_events["vote_event_id"].to_list())
        else:
            existing_ids = set()
        new_events = df_standalone.filter(~pl.col("vote_event_id").is_in(list(existing_ids)))
        if len(new_events) > 0:
            merged = pl.concat([df_vote_events, new_events])
            merged.write_parquet(PROCESSED_DIR / "fact_vote_events.parquet")
            LOGGER.info(
                "  Merged %d standalone vote events (total: %d)",
                len(new_events),
                len(merged),
            )
            results["fact_vote_events"] = merged

    # Read back slips (already written by process_bills)
    slips_path = PROCESSED_DIR / "fact_witness_slips.parquet"
    if slips_path.exists():
        results["fact_witness_slips"] = pl.read_parquet(slips_path)

    # Summary
    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline complete. Tables:")
    for name, df in results.items():
        LOGGER.info("  %-30s %d rows", name, len(df))
    LOGGER.info("Output: %s/", PROCESSED_DIR)
    LOGGER.info("=" * 60)

    return results
