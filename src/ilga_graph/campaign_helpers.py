"""Campaign (action alert) helpers: active campaign, visibility by ZIP, counts, deactivation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import Request
from sqlalchemy import func, select, update

from .db_models import Campaign, OutreachEvent
from .zip_crosswalk import ZipDistrictInfo

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def get_current_action_campaign_for_template(request: Request) -> object | None:
    """Return active campaign from request.state (set by middleware) for use in Jinja globals."""
    return getattr(request.state, "current_action_campaign", None)


def _as_utc(dt: datetime | None) -> datetime | None:
    """Return dt as timezone-aware UTC; if naive, assume UTC. For DB datetime comparison."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def get_active_campaign(db: AsyncSession, *, for_admin: bool = False) -> Campaign | None:
    """Return the single active campaign, or None.

    Filters by is_active=True. When for_admin=False (default), enforces
    start_at <= now <= end_at when those are set; when for_admin=True, returns
    the campaign marked active regardless of start_at/end_at (for admin dashboard).
    """
    now = datetime.now(timezone.utc)
    q = (
        select(Campaign)
        .where(Campaign.is_active.is_(True))
        .order_by(Campaign.created_at.desc())
        .limit(1)
    )
    result = await db.execute(q)
    campaign = result.scalar_one_or_none()
    if campaign is None:
        return None
    if for_admin:
        return campaign
    start_at = _as_utc(campaign.start_at)
    end_at = _as_utc(campaign.end_at)
    if start_at is not None and now < start_at:
        return None
    if end_at is not None and now > end_at:
        return None
    return campaign


def _parse_district_ids(raw: str | None) -> set[str]:
    """Parse target_district_ids JSON into a set of 'chamber:num' strings."""
    if not raw or not raw.strip():
        return set()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return {str(x).strip() for x in data if x}
        return set()
    except (json.JSONDecodeError, TypeError):
        return set()


def is_campaign_visible_to_zip(
    campaign: Campaign,
    zip_code: str | None,
    zip_to_district: dict[str, ZipDistrictInfo],
) -> bool:
    """Return True if this campaign should be shown to a user in the given ZIP.

    For target_type='all', always True. For target_type='by_district', True only
    when the ZIP maps to a targeted district. target_district_ids format: JSON
    list of 'chamber:num' e.g. ["il_house:9", "il_senate:5"].
    """
    if campaign.target_type == "all":
        return True
    if not zip_code or not zip_code.strip():
        return False
    zip_code = zip_code.strip()
    if zip_code not in zip_to_district:
        return False
    target_ids = _parse_district_ids(campaign.target_district_ids)
    if not target_ids:
        return True
    info = zip_to_district[zip_code]
    user_districts = {
        f"il_house:{info.il_house}",
        f"il_senate:{info.il_senate}",
        f"us_house:{info.us_house}",
    }
    return bool(user_districts & target_ids)


async def campaign_outreach_count(db: AsyncSession, campaign_id: int) -> int:
    """Return total outreach events for this campaign."""
    q = (
        select(func.count())
        .select_from(OutreachEvent)
        .where(OutreachEvent.campaign_id == campaign_id)
    )
    r = await db.execute(q)
    return r.scalar() or 0


async def deactivate_other_campaigns(db: AsyncSession, active_campaign_id: int) -> None:
    """Set is_active=False for all campaigns except the given id."""
    await db.execute(
        update(Campaign).where(Campaign.id != active_campaign_id).values(is_active=False)
    )
    await db.flush()
