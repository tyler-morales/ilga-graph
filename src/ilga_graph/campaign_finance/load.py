"""Load the campaign-finance index into AppState."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .ingest import CampaignFinanceIndex, load_index_json

LOGGER = logging.getLogger(__name__)


def attach_index(state: Any, index: CampaignFinanceIndex | None) -> None:
    state.campaign_finance = index


def _candidate_paths(data_dir: Path) -> list[Path]:
    return [
        data_dir / "campaign_finance.json",
        Path("processed/campaign_finance/index.json"),
    ]


def load_campaign_finance_index(data_dir: Path) -> CampaignFinanceIndex | None:
    for path in _candidate_paths(data_dir):
        if not path.exists():
            continue
        try:
            index = load_index_json(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Could not load campaign finance index %s: %s", path, exc)
            continue
        LOGGER.info(
            "Campaign finance index: %s committees, %s matched members, source=%s",
            len(index.committees_by_id),
            len(index.matches_by_member),
            index.source,
        )
        return index
    return None
