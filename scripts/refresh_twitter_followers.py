#!/usr/bin/env python3
"""Refresh cached X/Twitter follower counts for legislators with twitter_handle set.

Reads members from cache (members.json), merges docs/canonical/legislator_twitter_handles.json,
calls Twitter API v2 only for handles missing or stale (older than --max-age-days).
Optionally cap API calls per run with --max to minimize credits.

Usage::

    TWITTER_BEARER_TOKEN=xxx PYTHONPATH=src python scripts/refresh_twitter_followers.py
    python scripts/refresh_twitter_followers.py --max-age-days 30 --max 5  # 5 calls, skip fresh
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from ilga_graph.config import TWITTER_BEARER_TOKEN  # noqa: E402
from ilga_graph.scraper import load_normalized_cache  # noqa: E402
from ilga_graph.twitter_followers import (  # noqa: E402
    get_follower_cache_path,
    load_follower_cache_raw,
    save_follower_counts,
)


def merge_legislator_twitter_handles(members: list) -> None:
    """Overlay docs/canonical/legislator_twitter_handles.json onto members (mutates in place)."""
    path = ROOT / "docs" / "canonical" / "legislator_twitter_handles.json"
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        LOGGER.warning("Could not load legislator Twitter handles from %s: %s", path, e)
        return
    if not isinstance(data, dict):
        return
    lookup = {m.id: m for m in members}
    merged = 0
    for member_id, username in data.items():
        if not isinstance(username, str) or not username.strip():
            continue
        handle = username.strip().lstrip("@")
        member = lookup.get(member_id)
        if member is not None:
            member.twitter_handle = handle
            merged += 1
    if merged:
        LOGGER.info("Merged %d Twitter handles from %s", merged, path.name)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
LOGGER = logging.getLogger("refresh_twitter_followers")

TWITTER_API_BASE = "https://api.twitter.com/2"
REQUEST_DELAY = 0.5  # seconds between requests to stay under rate limit

_credits_depleted_hint_logged = False


def fetch_followers_count(username: str) -> int | None:
    """Return followers_count for username from Twitter API v2, or None on error/404."""
    token = TWITTER_BEARER_TOKEN
    if not token:
        LOGGER.error("TWITTER_BEARER_TOKEN not set. Set it in .env or environment.")
        return None
    handle = username.strip().lstrip("@")
    url = f"{TWITTER_API_BASE}/users/by/username/{handle}"
    params = {"user.fields": "public_metrics"}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
    except requests.RequestException as e:
        LOGGER.warning("Request failed for @%s: %s", handle, e)
        return None
    if r.status_code == 429:
        reset = r.headers.get("x-rate-limit-reset")
        if reset:
            try:
                wait = int(reset) - int(time.time()) + 15
                if wait > 0:
                    LOGGER.warning("Rate limited; waiting %ds until reset", wait)
                    time.sleep(wait)
            except (ValueError, TypeError):
                time.sleep(60)
        else:
            time.sleep(60)
        return fetch_followers_count(username)  # retry once
    if r.status_code != 200:
        LOGGER.warning("Twitter API @%s: %s %s", handle, r.status_code, r.text[:200])
        if r.status_code == 402:
            global _credits_depleted_hint_logged
            if not _credits_depleted_hint_logged:
                _credits_depleted_hint_logged = True
                LOGGER.warning(
                    "402 CreditsDepleted: Bearer Token may be for a different app than the one "
                    "with credits. In developer.x.com open the project with the $5 balance → Apps "
                    "→ use that app's Bearer Token in TWITTER_BEARER_TOKEN."
                )
        return None
    data = r.json()
    user = data.get("data")
    if not user:
        return None
    metrics = user.get("public_metrics") or {}
    return metrics.get("followers_count")


def _parse_updated_at(entry: dict) -> datetime | None:
    """Return updated_at as timezone-aware datetime, or None if missing/invalid."""
    raw = entry.get("updated_at")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh cached X/Twitter follower counts (incremental; minimal credits)."
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Refresh only handles updated more than N days ago (default: 30). Use 0 for all.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Cap API calls per run (e.g. 5). Oldest-updated handles first. No limit if unset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache age; refresh all (still respects --max).",
    )
    args = parser.parse_args()

    if not TWITTER_BEARER_TOKEN:
        LOGGER.error("Set TWITTER_BEARER_TOKEN to refresh follower counts.")
        return 1
    cached = load_normalized_cache()
    if cached is None:
        LOGGER.error("No cache. Run from project root with cache/members.json present.")
        return 1
    members, _ = cached
    merge_legislator_twitter_handles(members)
    all_handles = [
        (m, (m.twitter_handle or "").strip().lstrip("@"))
        for m in members
        if getattr(m, "twitter_handle", None) and (m.twitter_handle or "").strip()
    ]
    all_handles = [(m, h) for m, h in all_handles if h]
    if not all_handles:
        LOGGER.warning(
            "No members with twitter_handle. Add to members.json or "
            "docs/canonical/legislator_twitter_handles.json."
        )
        return 0

    existing = load_follower_cache_raw()
    now = datetime.now(timezone.utc)
    cutoff = (
        now - timedelta(days=args.max_age_days) if args.max_age_days and not args.force else None
    )

    to_refresh: list[tuple[datetime | None, str]] = []
    for _m, handle in all_handles:
        if cutoff is None:
            to_refresh.append((None, handle))
            continue
        entry = existing.get(handle)
        if not entry:
            to_refresh.append((None, handle))
            continue
        updated = _parse_updated_at(entry)
        if updated is None or updated < cutoff:
            to_refresh.append((updated, handle))

    if not to_refresh:
        LOGGER.info(
            "All %d handles are fresh (within %d days). No API calls.",
            len(all_handles),
            args.max_age_days or 0,
        )
        return 0

    to_refresh.sort(
        key=lambda x: (0 if x[0] is None else 1, x[0] or datetime.min.replace(tzinfo=timezone.utc))
    )
    if args.max is not None and args.max > 0:
        to_refresh = to_refresh[: args.max]
    to_refresh = [h for _updated, h in to_refresh]
    LOGGER.info(
        "Refreshing %d handle(s) (max_age_days=%s, max=%s)",
        len(to_refresh),
        args.max_age_days,
        args.max,
    )

    new_entries: dict[str, dict] = {}
    for handle in to_refresh:
        count = fetch_followers_count(handle)
        if count is not None:
            new_entries[handle] = {
                "followers_count": count,
                "updated_at": now.isoformat(),
            }
        time.sleep(REQUEST_DELAY)

    merged = dict(existing)
    merged.update(new_entries)
    save_follower_counts(merged)
    LOGGER.info(
        "Wrote %d new counts; cache has %d total at %s",
        len(new_entries),
        len(merged),
        get_follower_cache_path(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
