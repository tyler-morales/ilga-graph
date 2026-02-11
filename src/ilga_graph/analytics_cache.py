"""Persist and load scorecards and Moneyball report to skip recomputation on startup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .analytics import MemberScorecard
from .moneyball import MoneyballProfile, MoneyballReport, MoneyballWeights

LOGGER = logging.getLogger(__name__)

_SCORECARDS_FILE = "scorecards.json"
_MONEYBALL_FILE = "moneyball.json"
_MEMBERS_FILE = "members.json"
_BILLS_FILE = "bills.json"


def _source_data_mtime(cache_dir: Path, mock_dir: Path, seed_mode: bool) -> float:
    """Return the latest mtime of source data files (members + bills).

    Analytics depend on BOTH members.json and bills.json.  If either is
    updated (e.g. a full bill scrape), the analytics cache should be
    recomputed.  Returns the *maximum* mtime of the two, or 0.0 if
    neither file exists.
    """
    data_dir = mock_dir if seed_mode else cache_dir
    mtimes: list[float] = []
    for fname in (_MEMBERS_FILE, _BILLS_FILE):
        try:
            mtimes.append((data_dir / fname).stat().st_mtime)
        except OSError:
            pass
    # Also check bills in cache_dir when seed_mode (members from mock, bills from cache)
    if seed_mode:
        try:
            mtimes.append((cache_dir / _BILLS_FILE).stat().st_mtime)
        except OSError:
            pass
    return max(mtimes) if mtimes else 0.0


def _scorecards_path(cache_dir: Path) -> Path:
    return cache_dir / _SCORECARDS_FILE


def _moneyball_path(cache_dir: Path) -> Path:
    return cache_dir / _MONEYBALL_FILE


def load_analytics_cache(
    cache_dir: Path,
    mock_dir: Path,
    seed_mode: bool,
) -> tuple[dict[str, MemberScorecard], MoneyballReport] | None:
    """Load scorecards and moneyball from disk if present and not stale.

    Staleness: analytics cache is valid only if both scorecards.json and
    moneyball.json exist and are newer than the member data file. Returns
    None if any file is missing or stale.
    """
    source_mtime = _source_data_mtime(cache_dir, mock_dir, seed_mode)
    if source_mtime <= 0:
        return None

    sc_path = _scorecards_path(cache_dir)
    mb_path = _moneyball_path(cache_dir)
    if not sc_path.exists() or not mb_path.exists():
        return None
    try:
        if sc_path.stat().st_mtime < source_mtime or mb_path.stat().st_mtime < source_mtime:
            return None
    except OSError:
        return None

    try:
        with open(sc_path, encoding="utf-8") as f:
            sc_raw = json.load(f)
        with open(mb_path, encoding="utf-8") as f:
            mb_raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        LOGGER.warning("Failed to read analytics cache: %s", e)
        return None

    try:
        scorecards = {
            mid: MemberScorecard(**d)
            for mid, d in sc_raw.items()
        }
        profiles = {
            mid: MoneyballProfile(**d)
            for mid, d in mb_raw["profiles"].items()
        }
        weights = MoneyballWeights(**mb_raw["weights_used"])
        report = MoneyballReport(
            profiles=profiles,
            rankings_overall=mb_raw["rankings_overall"],
            rankings_house=mb_raw["rankings_house"],
            rankings_senate=mb_raw["rankings_senate"],
            rankings_house_non_leadership=mb_raw["rankings_house_non_leadership"],
            rankings_senate_non_leadership=mb_raw["rankings_senate_non_leadership"],
            mvp_house_non_leadership=mb_raw.get("mvp_house_non_leadership"),
            mvp_senate_non_leadership=mb_raw.get("mvp_senate_non_leadership"),
            weights_used=weights,
        )
        LOGGER.info(
            "Loaded analytics from cache (%d scorecards, %d profiles).",
            len(scorecards),
            len(profiles),
        )
        return scorecards, report
    except (TypeError, KeyError) as e:
        LOGGER.warning("Analytics cache format error: %s", e)
        return None


def save_analytics_cache(
    scorecards: dict[str, MemberScorecard],
    moneyball: MoneyballReport,
    cache_dir: Path,
) -> None:
    """Write scorecards and moneyball to cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    sc_path = _scorecards_path(cache_dir)
    sc_raw = {mid: _scorecard_to_dict(sc) for mid, sc in scorecards.items()}
    with open(sc_path, "w", encoding="utf-8") as f:
        json.dump(sc_raw, f, indent=0)

    mb_path = _moneyball_path(cache_dir)
    mb_raw = {
        "profiles": {mid: _profile_to_dict(p) for mid, p in moneyball.profiles.items()},
        "rankings_overall": moneyball.rankings_overall,
        "rankings_house": moneyball.rankings_house,
        "rankings_senate": moneyball.rankings_senate,
        "rankings_house_non_leadership": moneyball.rankings_house_non_leadership,
        "rankings_senate_non_leadership": moneyball.rankings_senate_non_leadership,
        "mvp_house_non_leadership": moneyball.mvp_house_non_leadership,
        "mvp_senate_non_leadership": moneyball.mvp_senate_non_leadership,
        "weights_used": _weights_to_dict(moneyball.weights_used),
    }
    with open(mb_path, "w", encoding="utf-8") as f:
        json.dump(mb_raw, f, indent=0)

    LOGGER.info("Saved analytics cache to %s and %s", sc_path, mb_path)


def _scorecard_to_dict(sc: MemberScorecard) -> dict:
    return {
        "primary_bill_count": sc.primary_bill_count,
        "passed_count": sc.passed_count,
        "vetoed_count": sc.vetoed_count,
        "stuck_count": sc.stuck_count,
        "in_progress_count": sc.in_progress_count,
        "success_rate": sc.success_rate,
        "heat_score": sc.heat_score,
        "effectiveness_score": sc.effectiveness_score,
        "law_heat_score": sc.law_heat_score,
        "law_passed_count": sc.law_passed_count,
        "law_success_rate": sc.law_success_rate,
        "magnet_score": sc.magnet_score,
        "bridge_score": sc.bridge_score,
        "resolutions_count": sc.resolutions_count,
        "resolutions_passed_count": sc.resolutions_passed_count,
        "resolution_pass_rate": sc.resolution_pass_rate,
    }


def _profile_to_dict(p: MoneyballProfile) -> dict:
    return {
        "member_id": p.member_id,
        "member_name": p.member_name,
        "chamber": p.chamber,
        "party": p.party,
        "district": p.district,
        "role": p.role,
        "is_leadership": p.is_leadership,
        "laws_filed": p.laws_filed,
        "laws_passed": p.laws_passed,
        "effectiveness_rate": p.effectiveness_rate,
        "magnet_score": p.magnet_score,
        "bridge_score": p.bridge_score,
        "resolutions_filed": p.resolutions_filed,
        "resolutions_passed": p.resolutions_passed,
        "pipeline_depth_avg": p.pipeline_depth_avg,
        "pipeline_depth_normalized": p.pipeline_depth_normalized,
        "network_centrality": p.network_centrality,
        "unique_collaborators": p.unique_collaborators,
        "collaborator_republicans": p.collaborator_republicans,
        "collaborator_democrats": p.collaborator_democrats,
        "collaborator_other": p.collaborator_other,
        "magnet_vs_chamber": p.magnet_vs_chamber,
        "cosponsor_passage_rate": p.cosponsor_passage_rate,
        "cosponsor_passage_multiplier": p.cosponsor_passage_multiplier,
        "chamber_median_cosponsor_rate": p.chamber_median_cosponsor_rate,
        "passage_rate_vs_caucus": p.passage_rate_vs_caucus,
        "caucus_avg_passage_rate": p.caucus_avg_passage_rate,
        "total_primary_bills": p.total_primary_bills,
        "total_passed": p.total_passed,
        "institutional_weight": p.institutional_weight,
        "moneyball_score": p.moneyball_score,
        "rank_overall": p.rank_overall,
        "rank_chamber": p.rank_chamber,
        "rank_non_leadership": p.rank_non_leadership,
        "badges": list(p.badges),
    }


def _weights_to_dict(w: MoneyballWeights) -> dict:
    return {
        "effectiveness": w.effectiveness,
        "pipeline": w.pipeline,
        "magnet": w.magnet,
        "bridge": w.bridge,
        "centrality": w.centrality,
        "institutional": w.institutional,
    }
