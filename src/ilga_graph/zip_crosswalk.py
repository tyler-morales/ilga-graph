"""ZIP (ZCTA) -> Illinois Legislative District crosswalk.

Ported from the Google Sheets ``buildZipToDistrict_IL()`` Apps Script.

Uses official Census 2020 relationship files (pipe-delimited) to build a
mapping from 5-digit ZCTA codes to Illinois House, Senate, and US House
district numbers.  When a ZCTA overlaps multiple districts, the district
with the largest ``AREALAND_PART`` wins (dominant by land area).

Public entry point: :func:`load_zip_crosswalk`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from . import config as cfg

LOGGER = logging.getLogger(__name__)

IL_STATEFP = "17"  # Illinois FIPS code

# Official Census 2020 relationship files (pipe-delimited, first row is header).
URL_CD_ZCTA = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/cd-sld/tab20_cd11920_zcta520_natl.txt"
)
URL_SLDL_ZCTA = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "cd-sld/tab20_sldl202220_zcta520_natl.txt"
)
URL_SLDU_ZCTA = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "cd-sld/tab20_sldu202220_zcta520_natl.txt"
)

# Cache file path.
_CACHE_FILE = "zip_to_district.json"


@dataclass
class ZipDistrictInfo:
    """District numbers for a single ZCTA."""

    il_house: str  # IL State House district (numeric string, e.g. "9")
    il_senate: str  # IL State Senate district (numeric string, e.g. "5")
    us_house: str  # US Congressional district (numeric string, e.g. "4")


# ── Seed-mode fallback (ZIPs that match the 20-member dev mock) ─────────────────
# All pairings below use Senate+House districts that exist in mocks/dev/members.json
# so dev/seed mode shows Your Senator, Your Rep, and Power Broker (no red banners).

_SEED_CROSSWALK: dict[str, ZipDistrictInfo] = {
    # Primary dev ZIPs — use these to test; all 3 cards (senator, rep, broker) render.
    "60601": ZipDistrictInfo(il_house="26", il_senate="18", us_house="4"),  # Cunningham + Buckner
    "60605": ZipDistrictInfo(il_house="106", il_senate="53", us_house="16"),  # Balkema + Bunting
    "60120": ZipDistrictInfo(il_house="26", il_senate="22", us_house="8"),  # Castro + Buckner
    "60540": ZipDistrictInfo(il_house="90", il_senate="41", us_house="11"),  # Curran + Cabello
    "60201": ZipDistrictInfo(il_house="26", il_senate="9", us_house="9"),  # Fine + Buckner
    "60643": ZipDistrictInfo(il_house="26", il_senate="18", us_house="1"),  # Cunningham + Buckner
    "60608": ZipDistrictInfo(il_house="26", il_senate="18", us_house="4"),  # Cunningham + Buckner
    "60614": ZipDistrictInfo(il_house="90", il_senate="41", us_house="11"),  # Curran + Cabello
    "60637": ZipDistrictInfo(il_house="26", il_senate="57", us_house="1"),  # Belt + Buckner
    "61032": ZipDistrictInfo(il_house="90", il_senate="47", us_house="16"),  # Anderson + Cabello
    "61701": ZipDistrictInfo(il_house="106", il_senate="53", us_house="16"),  # Balkema + Bunting
    "60532": ZipDistrictInfo(il_house="85", il_senate="22", us_house="11"),  # Castro + Avelar
}


# ── Census file downloader / parser ──────────────────────────────────────────


def _build_dominant_map_by_zcta(
    url: str,
    statefp: str,
    district_col: str,
    zcta_col: str,
    land_col: str,
) -> dict[str, str]:
    """Download a Census relationship file and return ZCTA -> best district GEOID.

    For each ZCTA that overlaps multiple districts within the given state,
    keep only the district with the largest ``AREALAND_PART``.

    Parameters
    ----------
    url:
        URL to the pipe-delimited Census relationship file.
    statefp:
        2-digit state FIPS code to filter on (e.g. ``"17"`` for Illinois).
    district_col:
        Column name for the district GEOID.
    zcta_col:
        Column name for the ZCTA5 code.
    land_col:
        Column name for the land area overlap metric.

    Returns
    -------
    dict mapping ZCTA5 -> district GEOID (the one with max land area overlap).
    """
    LOGGER.info("Downloading Census crosswalk: %s", url.split("/")[-1])
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    lines = [ln for ln in resp.text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty Census file from {url}")

    header = lines[0].split("|")
    col_idx = {name: i for i, name in enumerate(header)}

    i_district = col_idx.get(district_col)
    i_zcta = col_idx.get(zcta_col)
    i_land = col_idx.get(land_col)

    if i_district is None or i_zcta is None or i_land is None:
        raise ValueError(
            f"Missing expected columns in {url}. "
            f"Wanted {district_col}, {zcta_col}, {land_col}; "
            f"found: {', '.join(header)}"
        )

    # For each ZCTA, keep the district with the largest land area overlap.
    best: dict[str, tuple[str, float]] = {}  # zcta -> (district_geoid, land_area)

    for line in lines[1:]:
        parts = line.split("|")
        dist = parts[i_district] if i_district < len(parts) else ""
        zcta = parts[i_zcta] if i_zcta < len(parts) else ""
        land_str = parts[i_land] if i_land < len(parts) else "0"

        if not dist or not zcta:
            continue
        if not dist.startswith(statefp):
            continue  # Keep only Illinois districts

        land = float(land_str or 0)
        prev = best.get(zcta)
        if prev is None or land > prev[1]:
            best[zcta] = (dist, land)

    return {zcta: info[0] for zcta, info in best.items()}


def _extract_district_number(geoid: str, prefix_len: int = 2) -> str:
    """Extract the numeric district number from a Census GEOID.

    SLDL/SLDU GEOIDs: 2-digit state + 3-digit district (e.g. ``"17001"``).
    CD GEOIDs: 2-digit state + 2-digit district (e.g. ``"1707"``).

    Returns the district number as a string with leading zeros stripped.
    """
    raw = geoid[prefix_len:]
    return str(int(raw)) if raw else ""


# ── Builder ──────────────────────────────────────────────────────────────────


def build_zip_to_district() -> dict[str, ZipDistrictInfo]:
    """Download Census files and build the full ZCTA -> district crosswalk.

    This makes three HTTP requests to census.gov and takes ~10-30 seconds
    depending on connection speed (files are ~10-30 MB each).
    """
    cd_map = _build_dominant_map_by_zcta(
        URL_CD_ZCTA,
        IL_STATEFP,
        "GEOID_CD119_20",
        "GEOID_ZCTA5_20",
        "AREALAND_PART",
    )
    sldl_map = _build_dominant_map_by_zcta(
        URL_SLDL_ZCTA,
        IL_STATEFP,
        "GEOID_SLDL2022_20",
        "GEOID_ZCTA5_20",
        "AREALAND_PART",
    )
    sldu_map = _build_dominant_map_by_zcta(
        URL_SLDU_ZCTA,
        IL_STATEFP,
        "GEOID_SLDU2022_20",
        "GEOID_ZCTA5_20",
        "AREALAND_PART",
    )

    # Union of all ZCTAs seen for Illinois.
    all_zctas = sorted(set(cd_map) | set(sldl_map) | set(sldu_map))

    result: dict[str, ZipDistrictInfo] = {}
    for zcta in all_zctas:
        il_house = _extract_district_number(sldl_map[zcta]) if zcta in sldl_map else ""
        il_senate = _extract_district_number(sldu_map[zcta]) if zcta in sldu_map else ""
        us_house = _extract_district_number(cd_map[zcta]) if zcta in cd_map else ""
        result[zcta] = ZipDistrictInfo(
            il_house=il_house,
            il_senate=il_senate,
            us_house=us_house,
        )

    LOGGER.info("Built ZIP-to-district crosswalk: %d ZCTAs.", len(result))
    return result


# ── Cache persistence ────────────────────────────────────────────────────────


def _save_cache(crosswalk: dict[str, ZipDistrictInfo], cache_dir: Path) -> None:
    """Persist the crosswalk to ``cache/zip_to_district.json``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / _CACHE_FILE
    data = {zcta: asdict(info) for zcta, info in crosswalk.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    LOGGER.info("Saved ZIP crosswalk cache to %s (%d entries).", path, len(data))


def _load_cache(cache_dir: Path) -> dict[str, ZipDistrictInfo] | None:
    """Load the crosswalk from cache, or return ``None`` if unavailable."""
    path = cache_dir / _CACHE_FILE
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        result = {zcta: ZipDistrictInfo(**info) for zcta, info in data.items()}
        LOGGER.info("Loaded ZIP crosswalk from cache: %d entries.", len(result))
        return result
    except Exception:
        LOGGER.warning("Failed to load ZIP crosswalk cache; will rebuild.", exc_info=True)
        return None


# ── Public entry point ───────────────────────────────────────────────────────


def load_zip_crosswalk(
    *,
    seed_mode: bool | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, ZipDistrictInfo]:
    """Load (or build) the ZCTA -> district crosswalk.

    Resolution order:

    1. If ``seed_mode`` is true (or ``cfg.SEED_MODE``), return the hardcoded
       seed dictionary immediately (no network, no cache).
    2. If a cache file exists and ``force_refresh`` is false, load from cache.
    3. Otherwise download Census files, build the crosswalk, cache it, and return.

    Parameters
    ----------
    seed_mode:
        Override for ``cfg.SEED_MODE``.  When ``None``, reads from config.
    cache_dir:
        Override for ``cfg.CACHE_DIR``.
    force_refresh:
        When true, ignore cache and re-download from Census.

    Returns
    -------
    dict mapping 5-digit ZCTA strings to :class:`ZipDistrictInfo`.
    """
    if seed_mode is None:
        seed_mode = cfg.SEED_MODE
    if cache_dir is None:
        cache_dir = cfg.CACHE_DIR

    # Seed mode: try mocks/dev/ first (snapshot covers all 40 mock members' districts).
    if seed_mode:
        mock_path = cfg.MOCK_DEV_DIR / _CACHE_FILE
        if mock_path.exists():
            try:
                with open(mock_path, encoding="utf-8") as f:
                    data = json.load(f)
                result = {zcta: ZipDistrictInfo(**info) for zcta, info in data.items()}
                if result:
                    LOGGER.info(
                        "ZIP crosswalk: using mocks/dev (%d ZIPs covering mock districts).",
                        len(result),
                    )
                    return result
            except Exception:
                pass
        LOGGER.info("ZIP crosswalk: using seed-mode fallback (%d ZIPs).", len(_SEED_CROSSWALK))
        return dict(_SEED_CROSSWALK)

    # Try cache first.
    if not force_refresh:
        cached = _load_cache(cache_dir)
        if cached is not None:
            return cached

    # Build from Census files.
    try:
        crosswalk = build_zip_to_district()
        _save_cache(crosswalk, cache_dir)
        return crosswalk
    except Exception:
        LOGGER.error(
            "Failed to build ZIP crosswalk from Census; falling back to seed data.", exc_info=True
        )
        return dict(_SEED_CROSSWALK)
