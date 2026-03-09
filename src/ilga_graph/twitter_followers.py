"""Load and save cached X/Twitter follower counts for legislators.

Used by the intelligence raw table (Legislator Twitter tab). Counts are persisted
in cache/twitter_follower_counts.json and refreshed by scripts/refresh_twitter_followers.py
(or an admin endpoint); no live API calls in the request path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import config as cfg

LOGGER = logging.getLogger(__name__)

CACHE_FILENAME = "twitter_follower_counts.json"


def get_follower_cache_path() -> Path:
    """Path to the JSON cache file (uses configured CACHE_DIR)."""
    return cfg.CACHE_DIR / CACHE_FILENAME


def load_follower_cache_raw() -> dict[str, dict[str, Any]]:
    """Load full cache { username: { followers_count, updated_at } } for incremental refresh."""
    path = get_follower_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        LOGGER.warning("Could not load Twitter follower cache %s: %s", path, e)
        return {}
    result: dict[str, dict[str, Any]] = {}
    for username, entry in data.items():
        key = str(username).strip().lstrip("@")
        if isinstance(entry, dict) and "followers_count" in entry:
            result[key] = dict(entry)
        elif isinstance(entry, (int, float)):
            result[key] = {"followers_count": int(entry), "updated_at": None}
    return result


def load_follower_counts() -> dict[str, int]:
    """Load username -> followers_count from cache file. Returns empty dict if missing/invalid."""
    raw = load_follower_cache_raw()
    return {k: int(v["followers_count"]) for k, v in raw.items() if "followers_count" in v}


def save_follower_counts(entries: dict[str, dict[str, Any]]) -> None:
    """Save cache with structure { username: { followers_count: N, updated_at: ISO8601 } }."""
    path = get_follower_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
