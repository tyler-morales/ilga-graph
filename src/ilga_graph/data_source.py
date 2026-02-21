"""Single resolution point for where the app reads legislative JSON data.

Prod: always cache/ (CACHE_DIR). Dev: cache/dev if it has members.json (or
ILGA_USE_CACHE_ONLY=1), else mocks/dev. All loaders use get_data_dir() so
the cache-then-mocks decision lives in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import config as cfg

_DEV_SENTINEL = "members.json"


def get_data_dir() -> Path:
    """Return the directory to read legislative JSON from (members, bills, etc.).

    - Prod: cache/ (CACHE_DIR).
    - Dev: cache/dev if it contains members.json or ILGA_USE_CACHE_ONLY=1;
      otherwise mocks/dev.
    """
    if cfg.PROFILE != "dev":
        return cfg.CACHE_DIR
    if os.getenv("ILGA_USE_CACHE_ONLY") == "1":
        return cfg.CACHE_DIR
    if (cfg.CACHE_DIR / _DEV_SENTINEL).exists():
        return cfg.CACHE_DIR
    return cfg.MOCK_DEV_DIR


def is_using_mocks() -> bool:
    """True if the app is reading from mocks/dev (dev with no cache/dev data)."""
    if cfg.PROFILE != "dev":
        return False
    return get_data_dir() == cfg.MOCK_DEV_DIR
