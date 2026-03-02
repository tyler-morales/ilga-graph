"""Campaign config loader: hero copy, default topic, brief PDF, one-pager points.

Loaded from config/campaign.json (or path in ILGA_CAMPAIGN_CONFIG). Enables
swapping campaign by replacing the JSON without code changes. See
docs/architecture/campaign-decoupling.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Repo root: from ilga_graph package file, go up to project root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "campaign.json"


def _default_one_pager_points() -> list[str]:
    """Fallback when campaign config missing (Kei campaign)."""
    return [
        "Kei vehicles are federally legal to import (25-year rule).",
        "21+ states already allow registration—Illinois is the outlier.",
        "The current Illinois statute has an ambiguity, not a prohibition.",
        "The fix is a narrow clarifying amendment—no new regulatory framework.",
        "This affects real Illinois residents who own legal vehicles they can't register.",
    ]


def _default_error_facts() -> list[dict]:
    """Fallback when campaign config missing (empty = use app fallback in main)."""
    return []


@dataclass
class CampaignConfig:
    """Campaign-specific copy and settings for advocacy (and optionally home)."""

    campaign_name: str = ""
    primary_color: str = ""
    issue_summary: str = ""
    hero_headline: str = ""
    hero_headline_line1: str = ""
    hero_headline_line1_prefix: str = ""
    hero_headline_line1_highlight: str = ""
    hero_headline_line1_suffix: str = ""
    hero_headline_line2: str = ""
    hero_headline_line2_prefix: str = ""
    hero_headline_highlight: str = ""
    hero_headline_line2_suffix: str = ""
    hero_subhead: str = ""
    advocacy_hero_headline: str = ""
    advocacy_hero_headline_line1: str = ""
    advocacy_hero_headline_line1_prefix: str = ""
    advocacy_hero_headline_line1_highlight: str = ""
    advocacy_hero_headline_line1_suffix: str = ""
    advocacy_hero_headline_line2: str = ""
    advocacy_hero_headline_line2_prefix: str = ""
    advocacy_hero_headline_highlight: str = ""
    advocacy_hero_headline_line2_suffix: str = ""
    advocacy_hero_subhead_line1: str = ""
    advocacy_hero_subhead_line2: str = ""
    default_topic: str = "Transportation"
    brief_pdf_filename: str = ""
    brief_pdf_url_path: str = "/advocacy/brief.pdf"
    one_pager_points: list[str] = field(default_factory=_default_one_pager_points)
    poll_slug: str = ""
    # Poll goal (e.g. 1000). When 0, use content_constants.KEI_POLL_GOAL_RESPONSES.
    poll_goal_responses: int = 0
    # White-label: poll prompt + welcome email; when empty, code uses "kei" / issue_summary.
    poll_prompt_query: str = "kei"
    welcome_email_intro: str = ""
    welcome_email_poll_link_text: str = ""
    strategic_mission: str = ""
    mission_attribution: str = ""
    error_page_facts: list[dict] = field(default_factory=_default_error_facts)
    error_page_fact_label: str = "Kei vehicles"
    bill_status_urls: list[str] = field(default_factory=list)


_cached: CampaignConfig | None = None


def _load_raw() -> dict | None:
    """Load campaign JSON from configured path. Returns None on missing/invalid."""
    import os

    path_str = os.environ.get("ILGA_CAMPAIGN_CONFIG", "").strip()
    path = Path(path_str) if path_str else _DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        LOGGER.debug("Campaign config not found: %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        LOGGER.warning("Failed to load campaign config from %s: %s", path, e)
        return None


def get_campaign_config() -> CampaignConfig:
    """Return campaign config from JSON; cached after first load. Fallback when missing."""
    global _cached
    if _cached is not None:
        return _cached
    raw = _load_raw()
    if not raw:
        _cached = CampaignConfig()
        return _cached
    try:
        _cached = CampaignConfig(
            campaign_name=raw.get("campaign_name", "").strip(),
            primary_color=raw.get("primary_color", "").strip(),
            issue_summary=raw.get("issue_summary", "").strip(),
            hero_headline=raw.get("hero_headline", ""),
            hero_headline_line1=raw.get("hero_headline_line1", ""),
            hero_headline_line1_prefix=raw.get("hero_headline_line1_prefix", ""),
            hero_headline_line1_highlight=raw.get("hero_headline_line1_highlight", ""),
            hero_headline_line1_suffix=raw.get("hero_headline_line1_suffix", ""),
            hero_headline_line2=raw.get("hero_headline_line2", ""),
            hero_headline_line2_prefix=raw.get("hero_headline_line2_prefix", ""),
            hero_headline_highlight=raw.get("hero_headline_highlight", ""),
            hero_headline_line2_suffix=raw.get("hero_headline_line2_suffix", ""),
            hero_subhead=raw.get("hero_subhead", ""),
            advocacy_hero_headline=raw.get("advocacy_hero_headline", ""),
            advocacy_hero_headline_line1=raw.get("advocacy_hero_headline_line1", ""),
            advocacy_hero_headline_line1_prefix=raw.get("advocacy_hero_headline_line1_prefix", ""),
            advocacy_hero_headline_line1_highlight=raw.get(
                "advocacy_hero_headline_line1_highlight", ""
            ),
            advocacy_hero_headline_line1_suffix=raw.get("advocacy_hero_headline_line1_suffix", ""),
            advocacy_hero_headline_line2=raw.get("advocacy_hero_headline_line2", ""),
            advocacy_hero_headline_line2_prefix=raw.get("advocacy_hero_headline_line2_prefix", ""),
            advocacy_hero_headline_highlight=raw.get("advocacy_hero_headline_highlight", ""),
            advocacy_hero_headline_line2_suffix=raw.get("advocacy_hero_headline_line2_suffix", ""),
            advocacy_hero_subhead_line1=raw.get("advocacy_hero_subhead_line1", ""),
            advocacy_hero_subhead_line2=raw.get("advocacy_hero_subhead_line2", ""),
            default_topic=raw.get("default_topic", "Transportation"),
            brief_pdf_filename=raw.get("brief_pdf_filename", ""),
            brief_pdf_url_path=raw.get("brief_pdf_url_path", "/advocacy/brief.pdf"),
            one_pager_points=list(raw.get("one_pager_points", [])),
            poll_slug=raw.get("poll_slug", "").strip(),
            poll_goal_responses=int(raw.get("poll_goal_responses", 0) or 0),
            poll_prompt_query=(raw.get("poll_prompt_query") or "kei").strip(),
            welcome_email_intro=(raw.get("welcome_email_intro") or "").strip(),
            welcome_email_poll_link_text=(raw.get("welcome_email_poll_link_text") or "").strip(),
            strategic_mission=(raw.get("strategic_mission") or "").strip(),
            mission_attribution=(raw.get("mission_attribution") or "").strip(),
            error_page_facts=list(raw.get("error_page_facts", []))
            if isinstance(raw.get("error_page_facts"), list)
            else [],
            error_page_fact_label=(raw.get("error_page_fact_label") or "Kei vehicles").strip(),
            bill_status_urls=[u.strip() for u in raw.get("bill_status_urls", []) if u]
            if isinstance(raw.get("bill_status_urls"), list)
            else [],
        )
    except (TypeError, ValueError) as e:
        LOGGER.warning("Invalid campaign config: %s", e)
        _cached = CampaignConfig()
    return _cached


def get_kei_poll_goal() -> int:
    """Return poll goal (e.g. 1000). From campaign config or content_constants default."""
    from .routers.content_constants import KEI_POLL_GOAL_RESPONSES

    c = get_campaign_config()
    return c.poll_goal_responses or KEI_POLL_GOAL_RESPONSES
