"""Intelligence dashboard routes: summary, raw tabs, deep-dives (member/bill), witness slips."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from .. import config as cfg
from ..app_state import state
from ..campaign_config import get_campaign_config
from ..constants import KEI_STATUS_OPTIONS
from ..date_parse import parse_action_date
from ..intelligence_helpers import (
    bill_description_for_slip_bill_number,
    canonical_organization_name,
)
from ..ml.rule_engine import get_bill_to_law_process
from ..models import Bill
from ..routers.content import STRATEGIC_FIVE_POINTS
from ..session_schedule import (
    get_milestone_by_id,
    get_next_deadline_safe,
    get_session_date_range,
    get_session_dates_set,
    session_label,
)
from ..twitter_followers import load_follower_counts


def _is_session_date(date_str: str, session_dates: set[str]) -> bool:
    """True if date_str (YYYY-MM-DD or M/D/YYYY) falls on a House/Senate session day."""
    if not date_str or not session_dates:
        return False
    dt = parse_action_date(date_str)
    if dt == datetime.min:
        return False
    return dt.strftime("%Y-%m-%d") in session_dates


def _ilga_bill_url(bill_number: str, leg_id: str | None = None) -> str:
    """Build ILGA BillStatus URL from bill number (e.g. SB0005) and optional leg_id."""
    m = re.match(r"^([A-Z]+)(\d+)$", (bill_number or "").strip())
    if not m:
        return ""
    doc_type, doc_num = m.group(1), m.group(2)
    url = (
        f"{cfg.BASE_URL}Legislation/BillStatus?DocNum={doc_num}&GAID={cfg.GA_ID}"
        f"&DocTypeID={doc_type}&SessionID={cfg.SESSION_ID}"
    )
    if leg_id:
        url += f"&LegId={leg_id}"
    return url


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = _REPO_ROOT / "processed"
router = APIRouter()
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["dev_available"] = cfg.DEV_MODE
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["site_name"] = cfg.SITE_NAME
_campaign = get_campaign_config()
templates.env.globals["campaign_name"] = _campaign.campaign_name or cfg.SITE_NAME
templates.env.globals["primary_color"] = _campaign.primary_color or "#e55a1a"
templates.env.globals["issue_summary"] = _campaign.issue_summary
templates.env.globals["meta_description"] = cfg.META_DESCRIPTION
templates.env.globals["og_image_url"] = cfg.OG_IMAGE_URL
templates.env.globals["umami_enabled"] = cfg.PROFILE == "prod" and bool(cfg.UMAMI_WEBSITE_ID)
templates.env.globals["umami_website_id"] = cfg.UMAMI_WEBSITE_ID
templates.env.globals["umami_script_url"] = cfg.UMAMI_SCRIPT_URL
templates.env.globals["show_beta_banner"] = cfg.BETA_BANNER
templates.env.globals["beta_banner_feedback_url"] = cfg.BETA_BANNER_REPORT_URL
templates.env.globals["footer_last_updated"] = cfg.FOOTER_LAST_UPDATED
templates.env.globals["footer_last_updated_iso"] = cfg.FOOTER_LAST_UPDATED_ISO
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS
templates.env.globals["features"] = cfg.get_client_features()

from ..campaign_helpers import (  # noqa: E402
    get_current_action_campaign_for_template,
    get_poll_campaign_for_template,
)

templates.env.globals["get_current_action_campaign"] = get_current_action_campaign_for_template
templates.env.globals["get_poll_campaign_for_template"] = get_poll_campaign_for_template
templates.env.globals["get_milestone_by_id"] = get_milestone_by_id
templates.env.globals["get_next_deadline"] = get_next_deadline_safe
templates.env.globals["kei_status_options"] = KEI_STATUS_OPTIONS
templates.env.globals["is_session_date"] = _is_session_date

# Procedural/routing committees: bills are assigned here after passing substantive
# committees (e.g. "Referred to Rules * Reports"). "Advanced" in our pipeline
# means last_action = Do Pass/Reported Out, so these show 0% and are misleading.
_PROCEDURAL_COMMITTEE_NAMES = frozenset(
    {
        "rules * reports",
        "assignments * reports",
        "committee of the whole",
        "assignments",
        "rules committee",
    }
)


def _get_productive_days() -> tuple[list[dict], str | None]:
    """Aggregate bill actions by date for the current session; return (rows, error).

    Each row has: date, total_actions, unique_bills, by_chamber (House/Senate counts).
    If parquet or schedule is missing, returns ([], error_message).
    """
    try:
        import polars as pl
    except ImportError:
        return [], "Polars not installed (run 'make ml-setup')."

    actions_path = PROCESSED_DIR / "fact_bill_actions.parquet"
    if not actions_path.exists():
        return [], "Run the ML pipeline (make ml-run) to generate action data."

    try:
        df = pl.read_parquet(actions_path)
    except Exception as e:
        return [], f"Could not read action data: {e!s}"

    if df.is_empty() or "date" not in df.columns:
        return [], None

    # Filter to session date range
    try:
        date_range = get_session_date_range()
    except (FileNotFoundError, ValueError):
        date_range = None
    if date_range:
        min_date, max_date = date_range
        df = df.filter(
            pl.col("date").is_not_null()
            & (pl.col("date") >= min_date)
            & (pl.col("date") <= max_date)
        )
    else:
        # No Session events in schedule: use actions that have a valid date
        df = df.filter(pl.col("date").is_not_null())

    if df.is_empty():
        return [], None

    # Aggregate by date: total actions, unique bills, and chamber counts
    if "chamber" in df.columns:
        agg = (
            df.group_by("date")
            .agg(
                pl.len().alias("total_actions"),
                pl.col("bill_id").n_unique().alias("unique_bills"),
                pl.col("chamber").eq("House").sum().alias("house_actions"),
                pl.col("chamber").eq("Senate").sum().alias("senate_actions"),
            )
            .sort("total_actions", descending=True)
        )
    else:
        agg = (
            df.group_by("date")
            .agg(
                pl.len().alias("total_actions"),
                pl.col("bill_id").n_unique().alias("unique_bills"),
            )
            .sort("total_actions", descending=True)
        )
        agg = agg.with_columns(
            pl.lit(0).alias("house_actions"),
            pl.lit(0).alias("senate_actions"),
        )

    rows = [
        {
            "date": r["date"],
            "total_actions": r["total_actions"],
            "unique_bills": r["unique_bills"],
            "by_chamber": {
                "House": r.get("house_actions", 0),
                "Senate": r.get("senate_actions", 0),
            },
        }
        for r in agg.to_dicts()
    ]
    return rows, None


def _get_actions_by_day_all_days() -> tuple[
    list[list[dict | None]],
    list[dict],
    str | None,
    str | None,
    str | None,
    list[str | None],
]:
    """Aggregate bill actions by date for all days in parquet (session + non-session).

    Returns (heatmap_grid, days_table, date_range_start, date_range_end, error, month_labels).
    heatmap_grid: 7 rows (Sunday–Saturday), each row list of cell dicts or None per week.
    days_table: list of day dicts (date, total_actions, unique_bills, by_chamber, is_session_day).
    month_labels: one label per week column for the month timeline (None where no month starts).
    """
    empty_months: list[str | None] = []
    try:
        import polars as pl
    except ImportError:
        return [], [], "", "", "Polars not installed (run 'make ml-setup').", empty_months

    actions_path = PROCESSED_DIR / "fact_bill_actions.parquet"
    if not actions_path.exists():
        return (
            [],
            [],
            "",
            "",
            "Run the ML pipeline (make ml-run) to generate action data.",
            empty_months,
        )

    try:
        df = pl.read_parquet(actions_path)
    except Exception as e:
        return [], [], "", "", f"Could not read action data: {e!s}", empty_months

    if df.is_empty() or "date" not in df.columns:
        return [], [], "", "", None, empty_months

    df = df.filter(pl.col("date").is_not_null())
    if df.is_empty():
        return [], [], "", "", None, empty_months

    try:
        session_dates = get_session_dates_set()
    except (FileNotFoundError, ValueError):
        session_dates = set()

    if "chamber" in df.columns:
        agg = df.group_by("date").agg(
            pl.len().alias("total_actions"),
            pl.col("bill_id").n_unique().alias("unique_bills"),
            pl.col("chamber").eq("House").sum().alias("house_actions"),
            pl.col("chamber").eq("Senate").sum().alias("senate_actions"),
        )
    else:
        agg = df.group_by("date").agg(
            pl.len().alias("total_actions"),
            pl.col("bill_id").n_unique().alias("unique_bills"),
        )
        agg = agg.with_columns(
            pl.lit(0).alias("house_actions"),
            pl.lit(0).alias("senate_actions"),
        )

    counts_by_date: dict[str, dict] = {}
    for r in agg.to_dicts():
        d = r["date"]
        if d:
            counts_by_date[d] = {
                "total_actions": r["total_actions"],
                "unique_bills": r["unique_bills"],
                "house_actions": r.get("house_actions", 0),
                "senate_actions": r.get("senate_actions", 0),
            }

    dates_sorted = sorted(counts_by_date.keys())
    if not dates_sorted:
        return [], [], "", "", None, []

    min_date_str = dates_sorted[0]
    max_date_str = dates_sorted[-1]
    start_d = datetime.strptime(min_date_str, "%Y-%m-%d").date()
    end_d = datetime.strptime(max_date_str, "%Y-%m-%d").date()
    span_days = (end_d - start_d).days + 1
    if span_days > 366:
        start_d = end_d - timedelta(days=365)
        span_days = 366

    n_weeks = (span_days + 6) // 7
    if n_weeks < 1:
        n_weeks = 1

    grid: list[list[dict | None]] = [[None] * n_weeks for _ in range(7)]

    days_table: list[dict] = []
    current = start_d
    end_inclusive = start_d + timedelta(days=span_days - 1)

    while current <= end_inclusive:
        d_str = current.isoformat()
        info = counts_by_date.get(d_str)
        if info:
            total = info["total_actions"]
            unique = info["unique_bills"]
            house = info["house_actions"]
            senate = info["senate_actions"]
        else:
            total = unique = house = senate = 0

        is_session = d_str in session_dates
        cell = {
            "date": d_str,
            "total_actions": total,
            "unique_bills": unique,
            "is_session_day": is_session,
            "by_chamber": {"House": house, "Senate": senate},
        }
        days_table.append(cell)

        week_idx = (current - start_d).days // 7
        dow = (current.weekday() + 1) % 7
        if week_idx < n_weeks:
            grid[dow][week_idx] = cell

        current += timedelta(days=1)

    days_table.sort(key=lambda x: x["date"])

    # Month timeline: label at the first week of each month for heatmap orientation
    month_labels: list[str | None] = [None] * n_weeks
    prev_year: int | None = None
    month_start = start_d.replace(day=1)
    while month_start <= end_inclusive:
        first_in_range = month_start if month_start >= start_d else start_d
        week_idx = (first_in_range - start_d).days // 7
        if week_idx < n_weeks:
            year = month_start.year
            label = (
                month_start.strftime("%b %Y") if year != prev_year else month_start.strftime("%b")
            )
            prev_year = year
            month_labels[week_idx] = label
        if month_start.month == 12:
            month_start = month_start.replace(year=month_start.year + 1, month=1)
        else:
            month_start = month_start.replace(month=month_start.month + 1)

    return (
        grid,
        days_table,
        start_d.isoformat(),
        end_inclusive.isoformat(),
        None,
        month_labels,
    )


def _get_day_details(date_str: str) -> dict:
    """Return summary + actions for one day. Always 200; empty data if parquet missing."""
    try:
        session_dates = get_session_dates_set()
    except (FileNotFoundError, ValueError):
        session_dates = set()
    is_session = date_str in session_dates
    empty_payload = {
        "date": date_str,
        "total_actions": 0,
        "unique_bills": 0,
        "by_chamber": {"House": 0, "Senate": 0},
        "is_session_day": is_session,
        "actions": [],
    }
    try:
        import polars as pl
    except ImportError:
        return empty_payload
    actions_path = PROCESSED_DIR / "fact_bill_actions.parquet"
    if not actions_path.exists():
        return empty_payload
    try:
        df = pl.read_parquet(actions_path)
    except Exception:
        return empty_payload
    if df.is_empty() or "date" not in df.columns:
        return empty_payload
    day_df = df.filter(pl.col("date").cast(pl.Utf8) == date_str)
    if day_df.is_empty():
        return {
            "date": date_str,
            "total_actions": 0,
            "unique_bills": 0,
            "by_chamber": {"House": 0, "Senate": 0},
            "is_session_day": is_session,
            "actions": [],
        }
    total = len(day_df)
    unique = day_df["bill_id"].n_unique()
    house = day_df.filter(pl.col("chamber") == "House").height if "chamber" in day_df.columns else 0
    senate = (
        day_df.filter(pl.col("chamber") == "Senate").height if "chamber" in day_df.columns else 0
    )
    actions = []
    for r in day_df.select(["bill_id", "chamber", "action_text"]).to_dicts():
        actions.append(
            {
                "bill_id": r.get("bill_id", ""),
                "chamber": r.get("chamber", ""),
                "action_text": (r.get("action_text") or "")[:200],
            }
        )
    return {
        "date": date_str,
        "total_actions": total,
        "unique_bills": unique,
        "by_chamber": {"House": house, "Senate": senate},
        "is_session_day": is_session,
        "actions": actions,
    }


@router.get("/")
async def intelligence_summary(request: Request):
    """Executive summary: narrative-driven intelligence overview."""
    ml = state.ml
    available = ml and ml.available

    if not available:
        return templates.TemplateResponse(
            request,
            "intelligence_summary.html",
            {"request": request, "title": "Intelligence", "available": False},
        )

    trust_level = ml.quality.get("trust_assessment", {}).get("overall", "")
    roc_auc = ml.quality.get("test_set_metrics", {}).get("roc_auc")
    accuracy_pct = None
    if ml.accuracy_history:
        latest = ml.accuracy_history[-1]
        accuracy_pct = latest.accuracy * 100

    total_bills_scored = len(ml.bill_scores)
    n_coalitions = len(set(m.coalition_id for m in ml.coalitions))
    flagged_anomalies = sum(1 for a in ml.anomalies if a.is_anomaly)

    open_bills = [s for s in ml.bill_scores if s.lifecycle_status == "OPEN"]
    bills_to_watch = []

    advance_preds = sorted(
        [s for s in open_bills if s.predicted_outcome == "ADVANCE" and s.confidence >= 0.75],
        key=lambda s: -s.prob_advance,
    )
    for s in advance_preds[:3]:
        why = f"High confidence advance prediction ({s.confidence:.0%})"
        if s.days_since_action > 60:
            why += f" despite {s.days_since_action} days idle"
        bills_to_watch.append(
            {
                "bill_id": s.bill_id,
                "bill_number": s.bill_number,
                "description": s.description,
                "sponsor": s.sponsor,
                "prob_advance": s.prob_advance,
                "prob_law": getattr(s, "prob_law", 0.0),
                "predicted_outcome": s.predicted_outcome,
                "predicted_destination": getattr(s, "predicted_destination", "Stuck"),
                "confidence": s.confidence,
                "stage_label": s.stage_label,
                "forecast_score": getattr(s, "forecast_score", 0.0),
                "forecast_confidence": getattr(s, "forecast_confidence", ""),
                "why": why,
            }
        )

    surprises = sorted(
        [
            s
            for s in open_bills
            if s.predicted_outcome == "ADVANCE"
            and s.prob_advance >= 0.65
            and s.current_stage in ("IN_COMMITTEE", "FILED")
        ],
        key=lambda s: -s.prob_advance,
    )
    for s in surprises[:2]:
        if not any(b["bill_id"] == s.bill_id for b in bills_to_watch):
            bills_to_watch.append(
                {
                    "bill_id": s.bill_id,
                    "bill_number": s.bill_number,
                    "description": s.description,
                    "sponsor": s.sponsor,
                    "prob_advance": s.prob_advance,
                    "prob_law": getattr(s, "prob_law", 0.0),
                    "predicted_outcome": s.predicted_outcome,
                    "predicted_destination": getattr(s, "predicted_destination", "Stuck"),
                    "confidence": s.confidence,
                    "stage_label": s.stage_label,
                    "forecast_score": getattr(s, "forecast_score", 0.0),
                    "forecast_confidence": getattr(s, "forecast_confidence", ""),
                    "why": (
                        f"Surprise: still in {s.stage_label} but "
                        f"{s.prob_advance:.0%} chance of advancing"
                    ),
                }
            )

    stuck_surprises = sorted(
        [
            s
            for s in open_bills
            if s.predicted_outcome == "STUCK"
            and s.confidence >= 0.80
            and s.current_stage in ("PASSED_COMMITTEE", "FLOOR_VOTE")
        ],
        key=lambda s: s.prob_advance,
    )
    for s in stuck_surprises[:2]:
        if not any(b["bill_id"] == s.bill_id for b in bills_to_watch):
            bills_to_watch.append(
                {
                    "bill_id": s.bill_id,
                    "bill_number": s.bill_number,
                    "description": s.description,
                    "sponsor": s.sponsor,
                    "prob_advance": s.prob_advance,
                    "prob_law": getattr(s, "prob_law", 0.0),
                    "predicted_outcome": s.predicted_outcome,
                    "predicted_destination": getattr(s, "predicted_destination", "Stuck"),
                    "confidence": s.confidence,
                    "stage_label": s.stage_label,
                    "forecast_score": getattr(s, "forecast_score", 0.0),
                    "forecast_confidence": getattr(s, "forecast_confidence", ""),
                    "why": (
                        f"Warning: reached {s.stage_label} but model "
                        f"predicts stall ({s.confidence:.0%} confidence)"
                    ),
                }
            )

    power_movers = []
    influence_profiles = sorted(
        state.influence.values(),
        key=lambda p: p.influence_score,
        reverse=True,
    )
    for p in influence_profiles[:8]:
        member = state.member_lookup_by_id.get(p.member_id)
        if member:
            power_movers.append(
                {
                    "member_id": p.member_id,
                    "name": p.member_name,
                    "party": p.party,
                    "chamber": p.chamber,
                    "district": member.district,
                    "score": p.influence_score,
                    "label": p.influence_label,
                    "rank": p.rank_overall,
                    "signals": p.influence_signals,
                }
            )

    coalitions_summary = []
    prof_map = {cp.coalition_id: cp for cp in ml.coalition_profiles}
    ci_map = {ci.coalition_id: ci for ci in state.coalition_influence}
    coalition_groups: dict[int, list] = {}
    for m in ml.coalitions:
        coalition_groups.setdefault(m.coalition_id, []).append(m)

    for cid, members in sorted(coalition_groups.items()):
        prof = prof_map.get(cid)
        ci = ci_map.get(cid)
        dem = sum(1 for m in members if m.party == "Democrat")
        rep = sum(1 for m in members if m.party == "Republican")
        coalitions_summary.append(
            {
                "name": prof.name if prof else f"Coalition {cid + 1}",
                "size": len(members),
                "dem": dem,
                "rep": rep,
                "cohesion": prof.cohesion if prof else 0,
                "focus_areas": prof.focus_areas if prof else [],
                "top_influencer": ci.top_influencer_name if ci else None,
            }
        )

    top_anomalies = []
    for a in sorted(ml.anomalies, key=lambda a: -a.anomaly_score):
        if a.is_anomaly:
            top_anomalies.append(
                {
                    "bill_number": a.bill_number,
                    "description": a.description,
                    "reason": a.anomaly_reason,
                    "total_slips": a.total_slips,
                }
            )
            if len(top_anomalies) >= 5:
                break

    return templates.TemplateResponse(
        request,
        "intelligence_summary.html",
        {
            "request": request,
            "title": "Intelligence",
            "available": True,
            "trust_level": trust_level,
            "roc_auc": roc_auc,
            "accuracy_pct": accuracy_pct,
            "total_bills_scored": total_bills_scored,
            "n_coalitions": n_coalitions,
            "flagged_anomalies": flagged_anomalies,
            "bills_to_watch": bills_to_watch,
            "power_movers": power_movers,
            "coalitions_summary": coalitions_summary,
            "top_anomalies": top_anomalies,
            "last_run": ml.last_run_date,
        },
    )


@router.get("/raw")
async def intelligence_raw(request: Request):
    """Raw data tables — the original tabbed ML dashboard for power users."""
    ml = state.ml
    available = ml and ml.available

    summary = {}
    if available:
        scores = ml.bill_scores
        advance_count = sum(1 for s in scores if s.predicted_outcome == "ADVANCE")
        stuck_count = sum(1 for s in scores if s.predicted_outcome == "STUCK")
        forecast_count = sum(1 for s in scores if not s.label_reliable)
        flagged = sum(1 for a in ml.anomalies if a.is_anomaly)
        n_coalitions = len(set(m.coalition_id for m in ml.coalitions))
        dest_law = sum(
            1
            for s in scores
            if getattr(s, "predicted_destination", "").startswith("→ Law")
            or getattr(s, "predicted_destination", "") == "Became Law"
        )
        dest_floor = sum(
            1
            for s in scores
            if getattr(s, "predicted_destination", "") in ("→ Floor", "→ Passed", "→ Governor")
        )
        dest_stuck = sum(1 for s in scores if getattr(s, "predicted_destination", "") == "Stuck")
        summary = {
            "total_predictions": len(scores),
            "advance_count": advance_count,
            "stuck_count": stuck_count,
            "dest_law": dest_law,
            "dest_floor": dest_floor,
            "dest_stuck": dest_stuck,
            "forecast_count": forecast_count,
            "flagged_anomalies": flagged,
            "total_anomalies": len(ml.anomalies),
            "n_coalitions": n_coalitions,
            "n_coalition_members": len(ml.coalitions),
            "last_run": ml.last_run_date,
            "model": ml.quality.get("model_selected", ""),
            "trust": ml.quality.get("trust_assessment", {}).get("overall", ""),
            "roc_auc": ml.quality.get("test_set_metrics", {}).get("roc_auc"),
            "accuracy_runs": len(ml.accuracy_history),
            "n_committees": len(state.committees),
            "active_committees": sum(
                1 for cs in state.committee_stats.values() if cs.total_bills >= 10
            ),
        }

    # Vote summary (from state; not ML-dependent)
    deciding_vote_events = sum(
        1 for e in state.vote_events if abs(len(e.yea_votes) - len(e.nay_votes)) == 1
    )
    summary["bills_with_votes"] = len(state.vote_lookup)
    summary["total_vote_events"] = len(state.vote_events)
    summary["deciding_vote_events"] = deciding_vote_events

    return templates.TemplateResponse(
        request,
        "intelligence.html",
        {
            "request": request,
            "title": "ML Intelligence",
            "available": available,
            "summary": summary,
            "ml": ml,
        },
    )


def _event_sort_key(ev: dict) -> tuple[str, int, str]:
    """Sort vote events: chamber, then committee before floor, then date."""
    chamber = (ev.get("chamber") or "").lower()
    vote_type = ev.get("vote_type") or "floor"
    committee_first = 0 if vote_type == "committee" else 1
    date_str = ev.get("date") or ""
    return (chamber, committee_first, date_str)


def _vote_events_for_bill(bill_number: str) -> list[dict]:
    """Build sorted list of vote event dicts (with deciding_voters) for a single bill."""
    events = state.vote_lookup.get(bill_number, [])
    if not events:
        return []
    event_dicts = []
    for e in events:
        yea = len(e.yea_votes)
        nay = len(e.nay_votes)
        margin = abs(yea - nay)
        outcome = "passed" if yea > nay else ("lost" if yea < nay else "tied")
        deciding_voters = []
        if margin == 1:
            deciding_voters = list(e.yea_votes) if yea > nay else list(e.nay_votes)
        event_dicts.append(
            {
                "date": e.date,
                "chamber": e.chamber,
                "vote_type": e.vote_type,
                "description": e.description,
                "yea": yea,
                "nay": nay,
                "present": len(e.present_votes),
                "nv": len(e.nv_votes),
                "outcome": outcome,
                "margin": margin,
                "deciding_voters": deciding_voters,
            }
        )
    event_dicts.sort(key=_event_sort_key)
    return event_dicts


def _witness_slips_for_bill(bill_number: str) -> list[dict]:
    """Build list of witness slip dicts for template (name, organization, position, etc.)."""
    slips = getattr(state, "witness_slips_lookup", {}).get(bill_number, [])
    return [
        {
            "name": getattr(ws, "name", "") or "",
            "organization": getattr(ws, "organization", "") or "",
            "representing": getattr(ws, "representing", "") or "",
            "position": getattr(ws, "position", "") or "",
            "hearing_committee": getattr(ws, "hearing_committee", "") or "",
            "hearing_date": getattr(ws, "hearing_date", "") or "",
            "testimony_type": getattr(ws, "testimony_type", "") or "",
        }
        for ws in slips
    ]


def _bills_with_votes_data():
    """Build list of bills with vote events and per-event deciding voters for the Votes tab."""
    bills_list = []
    for bill_number, events in state.vote_lookup.items():
        bill = state.bill_lookup.get(bill_number)
        bill_id = bill.leg_id if bill else bill_number

        event_dicts = []
        for e in events:
            yea = len(e.yea_votes)
            nay = len(e.nay_votes)
            margin = abs(yea - nay)
            outcome = "passed" if yea > nay else ("lost" if yea < nay else "tied")
            deciding_voters = []
            if margin == 1:
                deciding_voters = list(e.yea_votes) if yea > nay else list(e.nay_votes)
            event_dicts.append(
                {
                    "date": e.date,
                    "chamber": e.chamber,
                    "vote_type": e.vote_type,
                    "description": e.description,
                    "yea": yea,
                    "nay": nay,
                    "present": len(e.present_votes),
                    "nv": len(e.nv_votes),
                    "outcome": outcome,
                    "margin": margin,
                    "deciding_voters": deciding_voters,
                }
            )

        event_dicts.sort(key=_event_sort_key)

        # Most recent vote date for bill sort (across events)
        latest_date = max((ed["date"] for ed in event_dicts), default="")

        bills_list.append(
            {
                "bill_number": bill_number,
                "bill_id": bill_id,
                "events": event_dicts,
                "latest_date": latest_date,
            }
        )

    bills_list.sort(key=lambda b: (b["latest_date"], b["bill_number"]), reverse=True)
    return bills_list


@router.get("/votes")
async def intelligence_votes(request: Request):
    """Tab: bills with vote data, totals per event, deciding voters (margin-of-one)."""
    if not state.vote_events:
        return templates.TemplateResponse(
            request,
            "_intelligence_votes.html",
            {
                "request": request,
                "bills_with_votes": [],
                "total_vote_events": 0,
                "deciding_vote_events": 0,
            },
        )

    deciding_count = sum(
        1 for e in state.vote_events if abs(len(e.yea_votes) - len(e.nay_votes)) == 1
    )
    bills_with_votes = _bills_with_votes_data()
    return templates.TemplateResponse(
        request,
        "_intelligence_votes.html",
        {
            "request": request,
            "bills_with_votes": bills_with_votes,
            "total_vote_events": len(state.vote_events),
            "deciding_vote_events": deciding_count,
        },
    )


@router.get("/productive-days")
async def intelligence_productive_days(request: Request):
    """Most productive days: bill actions per calendar day for the current session."""
    productive_days, error = _get_productive_days()
    session_label_str = session_label() if not error else ""

    if request.query_params.get("format") == "json" or "application/json" in (
        request.headers.get("accept") or ""
    ):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={
                "session_label": session_label_str,
                "productive_days": productive_days,
                "error": error,
            }
        )

    return templates.TemplateResponse(
        request,
        "intelligence_productive_days.html",
        {
            "request": request,
            "title": "Most Productive Days",
            "productive_days": productive_days,
            "session_label": session_label_str,
            "error": error,
        },
    )


@router.get("/activity-calendar")
async def intelligence_activity_calendar(request: Request):
    """Activity by Day: heatmap and table of bill actions per day (session + non-session)."""
    heatmap_grid, days_table, date_range_start, date_range_end, error, month_labels = (
        _get_actions_by_day_all_days()
    )
    session_label_str = session_label() if not error else ""

    return templates.TemplateResponse(
        request,
        "_intelligence_activity_calendar.html",
        {
            "request": request,
            "heatmap_grid": heatmap_grid,
            "days_table": days_table,
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "session_label": session_label_str,
            "error": error,
            "month_labels": month_labels,
        },
    )


@router.get("/activity-calendar/day")
async def intelligence_activity_calendar_day(date: str):
    """JSON: summary + actions for one calendar day (heatmap click). Always 200 for valid date."""
    if not date or len(date) != 10:
        from fastapi.responses import JSONResponse

        return JSONResponse({"error": "Invalid date"}, status_code=400)
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        from fastapi.responses import JSONResponse

        return JSONResponse({"error": "Invalid date format"}, status_code=400)
    return _get_day_details(date)


@router.get("/predictions")
async def intelligence_predictions(request: Request):
    """Tab: bill predictions."""
    ml = state.ml
    if not ml or not ml.available:
        return templates.TemplateResponse(
            request,
            "_intelligence_predictions.html",
            {"request": request, "predictions": [], "ml": None},
        )

    predictions = sorted(ml.bill_scores, key=lambda s: -s.prob_advance)
    return templates.TemplateResponse(
        request,
        "_intelligence_predictions.html",
        {"request": request, "predictions": predictions, "ml": ml},
    )


@router.get("/coalitions")
async def intelligence_coalitions(request: Request):
    """Tab: voting coalitions."""
    ml = state.ml
    if not ml or not ml.available:
        return templates.TemplateResponse(
            request,
            "_intelligence_coalitions.html",
            {"request": request, "groups": [], "ml": None},
        )

    groups: dict[int, list] = {}
    for m in ml.coalitions:
        groups.setdefault(m.coalition_id, []).append(m)

    prof_map = {p.coalition_id: p for p in ml.coalition_profiles}

    coalition_list = []
    for cid, members in sorted(groups.items()):
        dem = sum(1 for m in members if m.party == "Democrat")
        rep = sum(1 for m in members if m.party == "Republican")
        members_sorted = sorted(members, key=lambda m: (m.party, m.name))
        prof = prof_map.get(cid)
        coalition_list.append(
            {
                "id": cid,
                "name": prof.name if prof else f"Coalition {cid + 1}",
                "size": len(members),
                "dem": dem,
                "rep": rep,
                "cross_party": dem > 0 and rep > 0,
                "focus_areas": prof.focus_areas if prof else [],
                "yes_rate": prof.yes_rate if prof else 0.0,
                "cohesion": prof.cohesion if prof else 0.0,
                "signature_bills": (prof.signature_bills[:5] if prof else []),
                "members": members_sorted,
            }
        )

    return templates.TemplateResponse(
        request,
        "_intelligence_coalitions.html",
        {"request": request, "groups": coalition_list, "ml": ml},
    )


@router.get("/anomalies")
async def intelligence_anomalies(request: Request):
    """Tab: anomaly detection."""
    ml = state.ml
    if not ml or not ml.available:
        return templates.TemplateResponse(
            request,
            "_intelligence_anomalies.html",
            {"request": request, "anomalies": [], "ml": None},
        )

    anomalies = sorted(ml.anomalies, key=lambda a: -a.anomaly_score)
    return templates.TemplateResponse(
        request,
        "_intelligence_anomalies.html",
        {"request": request, "anomalies": anomalies, "ml": ml},
    )


@router.get("/influence")
async def intelligence_influence(request: Request):
    """Tab: influence leaderboard."""
    profiles = list(state.influence.values())
    if not profiles:
        return templates.TemplateResponse(
            request,
            "_intelligence_influence.html",
            {"request": request, "profiles": [], "coalition_influence": []},
        )

    profiles.sort(key=lambda p: p.influence_score, reverse=True)

    profile_dicts = [
        {
            "rank_overall": p.rank_overall,
            "name": p.member_name,
            "chamber": p.chamber,
            "party": p.party,
            "score": p.influence_score,
            "label": p.influence_label,
            "moneyball_pct": round(p.moneyball_normalized * 100, 1),
            "betweenness_pct": round(p.betweenness_normalized * 100, 1),
            "pivotality_pct": round(p.pivotality_normalized * 100, 1),
            "pull_pct": round(p.pull_normalized * 100, 1),
            "signals": p.influence_signals,
        }
        for p in profiles
    ]

    ci_dicts = [
        {
            "coalition_id": ci.coalition_id,
            "coalition_name": ci.coalition_name,
            "total_members": ci.total_members,
            "avg_influence": ci.avg_influence,
            "high_influence_count": ci.high_influence_count,
            "top_influencer_name": ci.top_influencer_name,
            "top_influencer_score": ci.top_influencer_score,
            "top_influencer_label": ci.top_influencer_label,
            "bridge_member_name": ci.bridge_member_name,
            "bridge_member_betweenness": ci.bridge_member_betweenness,
        }
        for ci in state.coalition_influence
    ]

    return templates.TemplateResponse(
        request,
        "_intelligence_influence.html",
        {
            "request": request,
            "profiles": profile_dicts,
            "coalition_influence": ci_dicts,
        },
    )


@router.get("/recruitment")
async def intelligence_recruitment(request: Request):
    """Topic-specific recruitment recommendations."""
    ml = state.ml
    topics: list[dict] = []
    meta: dict = {}
    value_scores: list[dict] = []

    if ml and ml.topic_recruitment:
        meta = ml.member_value_meta or {}
        for topic_name in sorted(ml.topic_recruitment.keys()):
            rankings = ml.topic_recruitment[topic_name]
            topics.append(
                {
                    "name": topic_name,
                    "slug": (topic_name.lower().replace(" ", "-").replace("&", "and")),
                    "count": len(rankings),
                }
            )

    if ml and ml.member_value_scores:
        sorted_scores = sorted(
            ml.member_value_scores.values(),
            key=lambda s: -s.value_residual,
        )
        for s in sorted_scores:
            value_scores.append(
                {
                    "member_id": s.member_id,
                    "member_name": s.member_name,
                    "party": s.party,
                    "chamber": s.chamber,
                    "predicted_effectiveness": s.predicted_effectiveness,
                    "actual_effectiveness": s.actual_effectiveness,
                    "value_residual": s.value_residual,
                    "value_percentile": s.value_percentile,
                    "value_label": s.value_label,
                    "moneyball_score": s.moneyball_score,
                    "top_recruitment_topics": s.top_recruitment_topics,
                }
            )

    return templates.TemplateResponse(
        request,
        "intelligence_recruitment.html",
        {
            "request": request,
            "topics": topics,
            "meta": meta,
            "value_scores": value_scores,
        },
    )


@router.get("/recruitment/{topic}")
async def intelligence_recruitment_topic(
    request: Request,
    topic: str,
):
    """HTMX partial: per-topic recruitment rankings."""
    ml = state.ml
    if not ml or not ml.topic_recruitment:
        return templates.TemplateResponse(
            request,
            "_recruitment_topic_partial.html",
            {"request": request, "topic": topic, "rankings": []},
        )

    rankings_raw = ml.topic_recruitment.get(topic, [])
    rankings = []
    for r in rankings_raw[:30]:
        rankings.append(
            {
                "member_id": r.get("member_id", ""),
                "member_name": r.get("member_name", ""),
                "party": r.get("party", ""),
                "chamber": r.get("chamber", ""),
                "recruitment_score": r.get("recruitment_score", 0),
                "affinity_score": r.get("affinity_score", 0),
                "effectiveness_score": r.get("effectiveness_score", 0),
                "persuadability_score": r.get("persuadability_score", 0),
                "network_reach": r.get("network_reach", 0),
                "coalition_tier": r.get("coalition_tier", ""),
                "value_label": r.get("value_label", ""),
                "yes_rate": r.get("yes_rate", 0),
            }
        )

    return templates.TemplateResponse(
        request,
        "_recruitment_topic_partial.html",
        {"request": request, "topic": topic, "rankings": rankings},
    )


@router.get("/committees")
async def intelligence_committees(request: Request):
    """Tab: committee power dashboard."""
    if not state.committees:
        return templates.TemplateResponse(
            request,
            "_intelligence_committees.html",
            {
                "request": request,
                "committees": [],
                "top_by_volume": [],
                "top_by_passage": [],
                "top_law_factories": [],
            },
        )

    committee_dicts = []
    for c in state.committees:
        cstats = state.committee_stats.get(c.code)
        roster = state.committee_rosters.get(c.code, [])

        chamber = "Senate" if c.code.startswith("S") else "House"
        is_procedural = c.name.strip().lower() in _PROCEDURAL_COMMITTEE_NAMES

        chair_name = None
        chair_id = None
        for cmr in roster:
            if cmr.role.lower() == "chair":
                chair_name = cmr.member_name
                chair_id = cmr.member_id
                break

        committee_dicts.append(
            {
                "code": c.code,
                "name": c.name,
                "chamber": chamber,
                "total_bills": cstats.total_bills if cstats else 0,
                "advanced_count": cstats.advanced_count if cstats else 0,
                "passed_count": cstats.passed_count if cstats else 0,
                "advancement_rate": cstats.advancement_rate if cstats else 0.0,
                "pass_rate": cstats.pass_rate if cstats else 0.0,
                "chair": chair_name,
                "chair_id": chair_id,
                "member_count": len(roster),
                "is_procedural": is_procedural,
            }
        )

    committee_dicts.sort(key=lambda x: -x["total_bills"])

    substantive = [c for c in committee_dicts if not c["is_procedural"]]
    active = [c for c in substantive if c["total_bills"] >= 10]
    top_by_volume = sorted(committee_dicts, key=lambda x: -x["total_bills"])[:10]
    top_by_passage = sorted(active, key=lambda x: -x["advancement_rate"])[:10]
    top_law_factories = sorted(
        [c for c in substantive if c["passed_count"] > 0],
        key=lambda x: -x["passed_count"],
    )[:10]

    return templates.TemplateResponse(
        request,
        "_intelligence_committees.html",
        {
            "request": request,
            "committees": committee_dicts,
            "top_by_volume": top_by_volume,
            "top_by_passage": top_by_passage,
            "top_law_factories": top_law_factories,
        },
    )


@router.get("/accuracy")
async def intelligence_accuracy(request: Request):
    """Tab: accuracy history / feedback loop."""
    ml = state.ml
    if not ml or not ml.available:
        return templates.TemplateResponse(
            request,
            "_intelligence_accuracy.html",
            {"request": request, "history": [], "quality": {}, "ml": None},
        )

    return templates.TemplateResponse(
        request,
        "_intelligence_accuracy.html",
        {
            "request": request,
            "history": ml.accuracy_history,
            "quality": ml.quality,
            "ml": ml,
        },
    )


@router.get("/legislator-twitter")
async def intelligence_legislator_twitter(request: Request):
    """Tab: legislators with Twitter/X handle and follower count, rankable by followers."""
    counts = load_follower_counts()
    rows = []
    for m in state.members:
        handle = (getattr(m, "twitter_handle", None) or "").strip().lstrip("@")
        if not handle:
            continue
        followers = counts.get(handle)
        rows.append(
            {
                "member_id": m.id,
                "name": m.name,
                "chamber": m.chamber,
                "party": m.party,
                "district": m.district,
                "twitter_handle": handle,
                "followers_count": followers,
                "member_url": getattr(m, "member_url", "") or "",
            }
        )
    rows.sort(key=lambda r: (-(r["followers_count"] or 0), (r["name"] or "").lower()))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
        r["followers_display"] = (
            f"{r['followers_count']:,}" if r.get("followers_count") is not None else None
        )
    return templates.TemplateResponse(
        request,
        "_intelligence_legislator_twitter.html",
        {
            "request": request,
            "rows": rows,
            "twitter_configured": bool(cfg.TWITTER_BEARER_TOKEN),
        },
    )


@router.get("/witness-slips")
async def intelligence_witness_slips(request: Request):
    """Tab: witness slips and organization/lobbying influence on bills."""
    lookup = getattr(state, "witness_slips_lookup", {})
    if not lookup:
        return templates.TemplateResponse(
            request,
            "_intelligence_witness_slips.html",
            {
                "request": request,
                "bill_slips": [],
                "top_organizations": [],
                "anomaly_by_bill": {},
            },
        )

    anomaly_by_bill = {}
    ml = getattr(state, "ml", None)
    if ml and getattr(ml, "anomalies", None):
        for a in ml.anomalies:
            if getattr(a, "is_anomaly", False) and getattr(a, "bill_number", None):
                anomaly_by_bill[a.bill_number] = a
        for a in ml.anomalies:
            if getattr(a, "is_anomaly", False) and getattr(a, "bill_id", None):
                bn = getattr(a, "bill_number", None) or a.bill_id
                if bn and bn not in anomaly_by_bill:
                    anomaly_by_bill[bn] = a

    bill_slips = []
    for bill_number, slips in lookup.items():
        pro = sum(1 for s in slips if s.position == "Proponent")
        opp = sum(1 for s in slips if s.position == "Opponent")
        no_pos = sum(1 for s in slips if s.position and "no position" in s.position.lower())
        total = len(slips)
        controversy = (opp / (pro + opp)) if (pro + opp) > 0 else 0.0
        desc = bill_description_for_slip_bill_number(bill_number)
        org_counts = {}
        for s in slips:
            org = canonical_organization_name(s.organization or "")
            if org not in org_counts:
                org_counts[org] = {"total": 0, "pro": 0, "opp": 0}
            org_counts[org]["total"] += 1
            if s.position == "Proponent":
                org_counts[org]["pro"] += 1
            elif s.position == "Opponent":
                org_counts[org]["opp"] += 1
        top_orgs = sorted(
            [
                {"name": org, "total": d["total"], "pro": d["pro"], "opp": d["opp"]}
                for org, d in org_counts.items()
            ],
            key=lambda x: -x["total"],
        )[:10]
        anomaly = anomaly_by_bill.get(bill_number)
        bill_slips.append(
            {
                "bill_number": bill_number,
                "description": desc,
                "total_count": total,
                "proponent_count": pro,
                "opponent_count": opp,
                "no_position_count": no_pos,
                "controversy": controversy,
                "top_organizations": top_orgs,
                "is_flagged": bool(anomaly),
                "anomaly_reason": getattr(anomaly, "anomaly_reason", "") if anomaly else "",
            }
        )
    bill_slips.sort(key=lambda x: -x["total_count"])

    org_global = {}
    for slips in lookup.values():
        for s in slips:
            org = canonical_organization_name(s.organization or "")
            org_global[org] = org_global.get(org, 0) + 1
    top_organizations = sorted(org_global.items(), key=lambda x: -x[1])[:50]

    return templates.TemplateResponse(
        request,
        "_intelligence_witness_slips.html",
        {
            "request": request,
            "bill_slips": bill_slips,
            "top_organizations": top_organizations,
            "anomaly_by_bill": anomaly_by_bill,
        },
    )


@router.get("/member/{member_id}")
async def intelligence_member_detail(request: Request, member_id: str):
    """Deep-dive on a single legislator's influence profile."""
    member = state.member_lookup_by_id.get(member_id)
    if not member:
        member = state.member_lookup.get(member_id)
    if not member:
        return templates.TemplateResponse(
            request,
            "intelligence_member.html",
            {
                "request": request,
                "member": None,
                "influence": None,
                "moneyball": None,
                "narrative": None,
                "top_bills": [],
                "coalition": None,
                "value": None,
            },
        )

    ip = state.influence.get(member_id)
    influence_dict = None
    if ip:
        piv = state.pivotality.get(member.name)
        sp = state.sponsor_pull.get(member_id)
        influence_dict = {
            "score": ip.influence_score,
            "label": ip.influence_label,
            "rank_overall": ip.rank_overall,
            "rank_chamber": ip.rank_chamber,
            "moneyball_pct": round(ip.moneyball_normalized * 100, 1),
            "betweenness_pct": round(ip.betweenness_normalized * 100, 1),
            "pivotality_pct": round(ip.pivotality_normalized * 100, 1),
            "pull_pct": round(ip.pull_normalized * 100, 1),
            "signals": ip.influence_signals,
            "pivotal_winning": piv.pivotal_winning if piv else 0,
            "swing_votes": piv.swing_votes if piv else 0,
            "close_votes_total": piv.close_votes_total if piv else 0,
            "sponsor_lift": sp.sponsor_lift if sp else 0,
            "cosponsor_lift": sp.cosponsor_lift if sp else 0,
        }

    mb = state.moneyball.profiles.get(member_id) if state.moneyball else None
    moneyball_dict = None
    if mb:
        moneyball_dict = {
            "laws_passed": mb.laws_passed,
            "effectiveness_rate": mb.effectiveness_rate,
            "magnet_score": mb.magnet_score,
            "bridge_score": mb.bridge_score,
            "unique_collaborators": mb.unique_collaborators,
            "moneyball_score": mb.moneyball_score,
            "badges": mb.badges,
        }

    narrative_parts = []
    if ip:
        narrative_parts.append(
            f"{member.name} ranks #{ip.rank_overall} overall in the Illinois General Assembly"
        )
        if ip.influence_label == "High":
            narrative_parts.append("with high legislative influence")
        elif ip.influence_label == "Moderate":
            narrative_parts.append("with moderate influence")

    if mb:
        if mb.laws_passed > 0:
            narrative_parts.append(
                f"They have passed {mb.laws_passed} law{'s' if mb.laws_passed != 1 else ''} "
                f"with a {mb.effectiveness_rate:.0%} effectiveness rate"
            )
        if mb.unique_collaborators > 20:
            narrative_parts.append(
                f"and collaborate with {mb.unique_collaborators} different legislators"
            )
        if mb.bridge_score > 0.3:
            narrative_parts.append(
                f"({mb.bridge_score:.0%} of their laws have cross-party co-sponsors)"
            )

    if ip and ip.influence_signals:
        narrative_parts.append(". " + ip.influence_signals[0])

    narrative = ", ".join(narrative_parts) + "." if narrative_parts else None

    top_bills = []
    ml = state.ml
    if ml and ml.available:
        member_bills = [
            s for s in ml.bill_scores if s.sponsor and member.name and member.name in s.sponsor
        ]
        member_bills.sort(key=lambda s: -s.prob_advance)
        for s in member_bills[:10]:
            top_bills.append(
                {
                    "bill_id": s.bill_id,
                    "bill_number": s.bill_number,
                    "description": s.description,
                    "prob_advance": s.prob_advance,
                    "prob_law": getattr(s, "prob_law", 0.0),
                    "predicted_destination": getattr(s, "predicted_destination", "Stuck"),
                    "stage_label": s.stage_label,
                    "lifecycle_status": s.lifecycle_status,
                    "forecast_score": getattr(s, "forecast_score", 0.0),
                    "forecast_confidence": getattr(s, "forecast_confidence", ""),
                }
            )

    coalition = None
    if ml and ml.coalitions:
        for cm in ml.coalitions:
            if cm.member_id == member_id:
                prof_map = {p.coalition_id: p for p in ml.coalition_profiles}
                prof = prof_map.get(cm.coalition_id)
                coalition_members = [m for m in ml.coalitions if m.coalition_id == cm.coalition_id]
                coalition = {
                    "name": (prof.name if prof else f"Coalition {cm.coalition_id + 1}"),
                    "size": len(coalition_members),
                    "cohesion": prof.cohesion if prof else 0,
                    "focus_areas": prof.focus_areas if prof else [],
                }
                break

    value_dict = None
    if ml and ml.member_value_scores:
        vs = ml.member_value_scores.get(member_id)
        if vs:
            value_dict = {
                "predicted_effectiveness": vs.predicted_effectiveness,
                "actual_effectiveness": vs.actual_effectiveness,
                "value_residual": vs.value_residual,
                "value_percentile": vs.value_percentile,
                "value_label": vs.value_label,
                "top_recruitment_topics": vs.top_recruitment_topics,
            }

    return templates.TemplateResponse(
        request,
        "intelligence_member.html",
        {
            "request": request,
            "member": member,
            "influence": influence_dict,
            "moneyball": moneyball_dict,
            "narrative": narrative,
            "top_bills": top_bills,
            "coalition": coalition,
            "value": value_dict,
        },
    )


@router.get("/bill/{bill_id}")
async def intelligence_bill_detail(request: Request, bill_id: str):
    """Deep-dive on a single bill's prediction and context."""
    ml = state.ml
    bill = None
    if ml and ml.available:
        for s in ml.bill_scores:
            if s.bill_id == bill_id or s.bill_number == bill_id:
                bill = s
                break

    if not bill:
        try:
            bill_to_law_process = get_bill_to_law_process()
        except Exception:
            bill_to_law_process = []
        try:
            session_dates = get_session_dates_set()
        except (FileNotFoundError, ValueError):
            session_dates = set()
        return templates.TemplateResponse(
            request,
            "intelligence_bill.html",
            {
                "request": request,
                "bill": None,
                "sponsor_influence": None,
                "anomaly": None,
                "action_history": [],
                "bill_to_law_process": bill_to_law_process,
                "bill_votes": [],
                "witness_slips": [],
                "session_dates": session_dates,
            },
        )

    sponsor_influence = None
    sponsor_member = None
    for m in state.members:
        if m.name and bill.sponsor and m.name in bill.sponsor:
            sponsor_member = m
            break

    if sponsor_member:
        ip = state.influence.get(sponsor_member.id)
        sp = state.sponsor_pull.get(sponsor_member.id)
        if ip:
            sponsor_influence = {
                "member_id": sponsor_member.id,
                "label": ip.influence_label,
                "rank": ip.rank_overall,
                "signals": ip.influence_signals,
                "sponsor_lift": sp.sponsor_lift if sp else 0,
            }
        bill_dict_extra = {"sponsor_id": sponsor_member.id}
    else:
        bill_dict_extra = {"sponsor_id": None}

    anomaly = None
    if ml and ml.anomalies:
        for a in ml.anomalies:
            if a.bill_id == bill_id or a.bill_number == bill_id:
                anomaly = {
                    "total_slips": a.total_slips,
                    "n_proponent": a.n_proponent,
                    "n_opponent": a.n_opponent,
                    "unique_orgs": a.unique_orgs,
                    "anomaly_score": a.anomaly_score,
                    "is_anomaly": a.is_anomaly,
                    "anomaly_reason": a.anomaly_reason,
                }
                break

    class _BillCtx:
        """Template-friendly bill context."""

        def __init__(self, score, extras):
            self.bill_id = score.bill_id
            self.bill_number = score.bill_number
            self.description = score.description
            self.synopsis = ""  # Set from bill_obj when resolved
            self.sponsor = score.sponsor
            self.prob_advance = score.prob_advance
            self.prob_law = getattr(score, "prob_law", 0.0)
            self.predicted_outcome = score.predicted_outcome
            self.predicted_destination = getattr(score, "predicted_destination", "Stuck")
            self.confidence = score.confidence
            self.label_reliable = score.label_reliable
            self.chamber_origin = score.chamber_origin
            self.introduction_date = score.introduction_date
            self.current_stage = score.current_stage
            self.stage_progress = score.stage_progress
            self.stage_label = score.stage_label
            self.days_since_action = score.days_since_action
            self.last_action_text = score.last_action_text
            self.last_action_date = score.last_action_date
            self.stuck_status = score.stuck_status
            self.stuck_reason = score.stuck_reason
            self.lifecycle_status = score.lifecycle_status
            self.rule_context = getattr(score, "rule_context", "")
            self.forecast_score = getattr(score, "forecast_score", 0.0)
            self.forecast_confidence = getattr(score, "forecast_confidence", "")
            self.sponsor_id = extras.get("sponsor_id")

    bill_ctx = _BillCtx(bill, bill_dict_extra)

    action_history = []
    bill_obj = None
    if hasattr(state, "bills_lookup"):
        bill_obj = state.bills_lookup.get(bill_id)
    if bill_obj is None and hasattr(state, "bill_lookup"):
        bill_obj = state.bill_lookup.get(bill.bill_number)
    if bill_obj is None and hasattr(state, "bills_lookup") and state.bills_lookup:
        candidates = [b for b in state.bills_lookup.values() if b.bill_number == bill.bill_number]
        if candidates:

            def _latest_action_date(b: Bill) -> datetime:
                if not b.action_history:
                    return datetime.min
                return max(parse_action_date(ae.date) for ae in b.action_history)

            bill_obj = max(candidates, key=_latest_action_date)
    if bill_obj:
        bill_ctx.synopsis = getattr(bill_obj, "synopsis", "") or ""
    if bill_obj and bill_obj.action_history:
        for ae in bill_obj.action_history:
            action_history.append(
                {
                    "date": ae.date,
                    "action": ae.action,
                    "chamber": ae.chamber,
                    "action_category": ae.action_category or "other",
                    "action_category_label": ae.action_category_label or "Other",
                    "outcome_signal": ae.outcome_signal or "neutral",
                    "meaning": ae.meaning or "",
                    "rule_reference": getattr(ae, "rule_reference", "") or "",
                }
            )
        action_history.sort(key=lambda a: parse_action_date(a["date"]))
        last_act = action_history[-1]
        bill_ctx.last_action_date = last_act["date"]
        bill_ctx.last_action_text = last_act["action"]
        last_dt = parse_action_date(last_act["date"])
        if last_dt != datetime.min:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            bill_ctx.days_since_action = max(0, (today - last_dt).days)

    try:
        bill_to_law_process = get_bill_to_law_process()
    except Exception:
        bill_to_law_process = []

    bill_votes = _vote_events_for_bill(bill.bill_number)
    witness_slips = _witness_slips_for_bill(bill.bill_number)
    try:
        session_dates = get_session_dates_set()
    except (FileNotFoundError, ValueError):
        session_dates = set()

    ilga_bill_url = ""
    if bill_obj and getattr(bill_obj, "status_url", ""):
        ilga_bill_url = bill_obj.status_url
    else:
        leg_id = getattr(bill_obj, "leg_id", None) if bill_obj else None
        ilga_bill_url = _ilga_bill_url(bill.bill_number, leg_id)

    return templates.TemplateResponse(
        request,
        "intelligence_bill.html",
        {
            "request": request,
            "bill": bill_ctx,
            "ilga_bill_url": ilga_bill_url,
            "sponsor_influence": sponsor_influence,
            "anomaly": anomaly,
            "action_history": action_history,
            "bill_to_law_process": bill_to_law_process,
            "bill_votes": bill_votes,
            "witness_slips": witness_slips,
            "session_dates": session_dates,
        },
    )
