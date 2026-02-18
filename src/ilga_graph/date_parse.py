"""Date parsing helpers for bills/actions. Used by GraphQL and intelligence routes."""

import functools
import logging
from datetime import datetime

LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=16384)
def parse_bill_date(date_str: str) -> datetime:
    """Parse 'M/D/YYYY' for sorting. Unparseable -> datetime.max."""
    try:
        return datetime.strptime(date_str, "%m/%d/%Y")
    except (ValueError, TypeError):
        return datetime.max


def parse_action_date(date_str: str) -> datetime:
    """Parse action date (YYYY-MM-DD or M/D/YYYY). Unparseable -> datetime.min."""
    if not date_str:
        return datetime.min
    try:
        if "-" in date_str and len(date_str) == 10:
            return datetime.strptime(date_str, "%Y-%m-%d")
        return datetime.strptime(date_str, "%m/%d/%Y")
    except (ValueError, TypeError):
        return datetime.min


def safe_parse_date(date_str: str, param_name: str) -> datetime | None:
    """Parse ISO date string; return None and log on failure."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        LOGGER.warning("Invalid date for %s: %r", param_name, date_str)
        return None
