"""Entity resolution: map raw vote names to canonical member IDs.

Strategy (ordered by confidence):
1. **Exact match** -- existing variant-map logic from vote_name_normalizer.py
2. **Fuzzy match** -- rapidfuzz on candidate pool filtered by chamber
3. **Human-in-the-loop** -- present ambiguous cases to user for confirmation

Produces:
- ``processed/entity_gold.json`` -- user-confirmed mappings (persistent)
- ``processed/fact_vote_casts.parquet`` -- resolved vote casts with member_id FK
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process

LOGGER = logging.getLogger(__name__)

PROCESSED_DIR = Path("processed")
GOLD_PATH = PROCESSED_DIR / "entity_gold.json"

# Suffixes to strip for matching
_SUFFIX_RE = re.compile(r",?\s+(?:Jr\.?|Sr\.?|III|II|IV)\s*$", re.IGNORECASE)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class MemberCandidate:
    """A potential match for a raw vote name."""

    member_id: str
    name: str
    party: str
    chamber: str
    district: int | None
    score: float = 0.0


@dataclass
class ResolutionResult:
    """Outcome of resolving one unique (raw_name, chamber) pair."""

    raw_name: str
    chamber: str
    member_id: str | None = None
    member_name: str | None = None
    method: str = ""  # "exact", "fuzzy", "gold", "unresolved"
    confidence: float = 0.0
    occurrence_count: int = 0  # How many vote casts use this name


@dataclass
class ResolutionReport:
    """Summary of a full entity resolution run."""

    total_unique_names: int = 0
    resolved_exact: int = 0
    resolved_fuzzy: int = 0
    resolved_gold: int = 0
    unresolved: int = 0
    results: list[ResolutionResult] = field(default_factory=list)

    @property
    def total_resolved(self) -> int:
        return self.resolved_exact + self.resolved_fuzzy + self.resolved_gold

    @property
    def resolution_rate(self) -> float:
        if self.total_unique_names == 0:
            return 0.0
        return self.total_resolved / self.total_unique_names


# ── Gold standard persistence ────────────────────────────────────────────────


def load_gold_mappings() -> dict[tuple[str, str], str]:
    """Load user-confirmed (raw_name, chamber) -> member_id mappings."""
    if not GOLD_PATH.exists():
        return {}
    with open(GOLD_PATH) as f:
        data = json.load(f)
    # Stored as list of {"raw_name": ..., "chamber": ..., "member_id": ...}
    return {(d["raw_name"], d["chamber"]): d["member_id"] for d in data}


def save_gold_mappings(mappings: dict[tuple[str, str], str]) -> None:
    """Persist user-confirmed mappings."""
    data = [
        {"raw_name": k[0], "chamber": k[1], "member_id": v} for k, v in sorted(mappings.items())
    ]
    PROCESSED_DIR.mkdir(exist_ok=True)
    with open(GOLD_PATH, "w") as f:
        json.dump(data, f, indent=2)
    LOGGER.info("Saved %d gold mappings to %s", len(data), GOLD_PATH)


# ── Variant map (reuses vote_name_normalizer logic) ──────────────────────────


def _norm_key(name: str) -> str:
    """Produce a lowercase key with periods stripped and whitespace collapsed."""
    key = name.lower().replace(".", "")
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _parse_member_name(full_name: str) -> tuple[str, str, str]:
    """Parse 'First M. Last' -> (first_middle, last, suffix)."""
    suffix = ""
    m = _SUFFIX_RE.search(full_name)
    if m:
        suffix = m.group().strip().lstrip(",").strip()
        full_name = full_name[: m.start()].strip()
    parts = full_name.split()
    if len(parts) <= 1:
        return ("", full_name, suffix)
    last = parts[-1]
    first_middle = " ".join(parts[:-1])
    return (first_middle, last, suffix)


# Regex to extract quoted nicknames: 'Elizabeth "Lisa" Hernandez'
_NICKNAME_RE = re.compile(r'"([^"]+)"')

# Known non-member strings in vote PDFs
_NON_MEMBER_NAMES = {"Mr. President", "Mr. Speaker", "Presiding"}

# PDF artifact: leading vote code "Y ", "N " etc.
_VOTE_CODE_PREFIX_RE = re.compile(r"^(Y|N|NV|P|E|A)\s+(.+)$")


def build_variant_map(
    df_members: pl.DataFrame,
) -> dict[tuple[str, str], MemberCandidate]:
    """Build {(chamber, norm_key): MemberCandidate} from dim_members.

    Generates multiple key variants per member so that any vote-PDF format
    can resolve back to the canonical member.
    """
    variant_map: dict[tuple[str, str], MemberCandidate] = {}

    # Count last names per chamber for ambiguity detection
    last_counts: dict[tuple[str, str], int] = {}
    parsed: list[tuple[dict, str, str, str]] = []

    members = df_members.to_dicts()
    for m in members:
        first_middle, last, suffix = _parse_member_name(m["name"])
        parsed.append((m, first_middle, last, suffix))
        key = (m["chamber"], _norm_key(last))
        last_counts[key] = last_counts.get(key, 0) + 1

    for m, first_middle, last, suffix in parsed:
        candidate = MemberCandidate(
            member_id=m["member_id"],
            name=m["name"],
            party=m["party"],
            chamber=m["chamber"],
            district=m["district"],
        )
        chamber = m["chamber"]
        full_name = m["name"]

        # 1. Full canonical name
        variant_map[(chamber, _norm_key(full_name))] = candidate

        # 2. "Last, First M" variants
        fm_stripped = first_middle.replace(".", "").strip()
        if fm_stripped:
            for sep in [", ", ","]:
                v = f"{last}{sep}{fm_stripped}"
                variant_map[(chamber, _norm_key(v))] = candidate

                if suffix:
                    v_s = f"{last} {suffix}{sep}{fm_stripped}"
                    variant_map[(chamber, _norm_key(v_s))] = candidate

            # First-name-only
            first_only = fm_stripped.split()[0] if fm_stripped else ""
            if first_only and first_only != fm_stripped:
                for sep in [", ", ","]:
                    key = (chamber, _norm_key(f"{last}{sep}{first_only}"))
                    if key not in variant_map:
                        variant_map[key] = candidate

            # First-initial-only
            first_initial = fm_stripped[0] if fm_stripped else ""
            if first_initial:
                for sep in [", ", ","]:
                    key = (chamber, _norm_key(f"{last}{sep}{first_initial}"))
                    if key not in variant_map:
                        variant_map[key] = candidate

        # 3. Bare last name (only when unambiguous)
        if last_counts.get((chamber, _norm_key(last)), 0) == 1:
            key = (chamber, _norm_key(last))
            if key not in variant_map:
                variant_map[key] = candidate

        # 4. Compound last names: "Loughran Cappel", "Glowiak Hilton", etc.
        #    For members with 3+ name parts, try compound last names.
        #    e.g. "Meg Loughran Cappel" -> bare key "Loughran Cappel"
        #         "Laura Faver Dias" -> bare key "Faver Dias"
        #         "Terra Costa Howard" -> bare key "Costa Howard"
        name_no_suffix = _SUFFIX_RE.sub("", full_name).strip()
        # Remove quoted nicknames for parsing
        name_clean = _NICKNAME_RE.sub("", name_no_suffix).strip()
        name_clean = re.sub(r"\s+", " ", name_clean)
        parts = name_clean.split()
        if len(parts) >= 3:
            # Try compound last name = last 2 words
            compound_last = " ".join(parts[-2:])
            key = (chamber, _norm_key(compound_last))
            if key not in variant_map:
                variant_map[key] = candidate

            # Also try "CompoundLast, First" format
            fm_parts = parts[:-2]
            if fm_parts:
                fm = " ".join(fm_parts).replace(".", "").strip()
                for sep in [", ", ","]:
                    key = (chamber, _norm_key(f"{compound_last}{sep}{fm}"))
                    if key not in variant_map:
                        variant_map[key] = candidate

        # 5. Nickname variants: 'William "Will" Davis' -> "Davis,Will"
        nick_match = _NICKNAME_RE.search(full_name)
        if nick_match:
            nickname = nick_match.group(1)
            for sep in [", ", ","]:
                key = (chamber, _norm_key(f"{last}{sep}{nickname}"))
                if key not in variant_map:
                    variant_map[key] = candidate

            # Also try compound last + nickname
            if len(parts) >= 3:
                compound_last = " ".join(parts[-2:])
                for sep in [", ", ","]:
                    key = (chamber, _norm_key(f"{compound_last}{sep}{nickname}"))
                    if key not in variant_map:
                        variant_map[key] = candidate

    return variant_map


# ── Fuzzy matching ───────────────────────────────────────────────────────────


def fuzzy_match_candidates(
    raw_name: str,
    chamber: str,
    df_members: pl.DataFrame,
    top_n: int = 5,
    threshold: float = 50.0,
) -> list[MemberCandidate]:
    """Find top-N member candidates for a raw vote name using fuzzy matching.

    Filters to members in the same chamber, then ranks by rapidfuzz score.
    """
    # Filter members by chamber
    chamber_members = df_members.filter(pl.col("chamber") == chamber).to_dicts()
    if not chamber_members:
        return []

    # Build candidate name list
    candidate_names = [m["name"] for m in chamber_members]

    # Fuzzy match using token_sort_ratio (handles name reordering)
    results = process.extract(
        raw_name,
        candidate_names,
        scorer=fuzz.token_sort_ratio,
        limit=top_n,
        score_cutoff=threshold,
    )

    candidates = []
    for matched_name, score, idx in results:
        m = chamber_members[idx]
        candidates.append(
            MemberCandidate(
                member_id=m["member_id"],
                name=m["name"],
                party=m["party"],
                chamber=m["chamber"],
                district=m["district"],
                score=score,
            )
        )

    return sorted(candidates, key=lambda c: -c.score)


# ── Full resolution engine ───────────────────────────────────────────────────


def resolve_all_names(
    df_vote_casts: pl.DataFrame,
    df_members: pl.DataFrame,
    *,
    fuzzy_auto_threshold: float = 95.0,
) -> ResolutionReport:
    """Resolve all unique (raw_name, chamber) pairs in vote casts.

    Resolution strategy:
    1. Gold mappings (user-confirmed from previous sessions)
    2. Exact variant-map match
    3. High-confidence fuzzy match (score >= fuzzy_auto_threshold)
    4. Mark as unresolved (for human review)
    """
    gold = load_gold_mappings()
    variant_map = build_variant_map(df_members)
    member_by_id = {m["member_id"]: m for m in df_members.to_dicts()}

    # Get unique (raw_name, chamber) pairs with counts
    unique_pairs = (
        df_vote_casts.group_by(["raw_voter_name", "chamber"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    report = ResolutionReport(total_unique_names=len(unique_pairs))

    for row in unique_pairs.to_dicts():
        raw_name = row["raw_voter_name"]
        chamber = row["chamber"]
        count = row["count"]

        result = ResolutionResult(
            raw_name=raw_name,
            chamber=chamber,
            occurrence_count=count,
        )

        # 0. Skip known non-member names
        if raw_name.strip() in _NON_MEMBER_NAMES:
            result.method = "non_member"
            result.confidence = 1.0
            report.resolved_exact += 1
            report.results.append(result)
            continue

        # 1. Check gold mappings
        gold_key = (raw_name, chamber)
        if gold_key in gold:
            mid = gold[gold_key]
            if mid == "__NOT_A_MEMBER__":
                result.method = "non_member"
                result.confidence = 1.0
                report.resolved_gold += 1
                report.results.append(result)
                continue
            if mid in member_by_id:
                result.member_id = mid
                result.member_name = member_by_id[mid]["name"]
                result.method = "gold"
                result.confidence = 1.0
                report.resolved_gold += 1
                report.results.append(result)
                continue

        # 2. Exact variant-map match
        norm = _norm_key(raw_name)
        candidate = variant_map.get((chamber, norm))
        if candidate is not None:
            result.member_id = candidate.member_id
            result.member_name = candidate.name
            result.method = "exact"
            result.confidence = 1.0
            report.resolved_exact += 1
            report.results.append(result)
            continue

        # 2b. Try stripping PDF artifact prefix ("Y Murphy, Laura" -> "Murphy, Laura")
        pdf_match = _VOTE_CODE_PREFIX_RE.match(raw_name.strip())
        if pdf_match:
            stripped_name = pdf_match.group(2).strip()
            norm_stripped = _norm_key(stripped_name)
            candidate = variant_map.get((chamber, norm_stripped))
            if candidate is not None:
                result.member_id = candidate.member_id
                result.member_name = candidate.name
                result.method = "exact"
                result.confidence = 0.95
                report.resolved_exact += 1
                report.results.append(result)
                continue

        # 3. Fuzzy match
        candidates = fuzzy_match_candidates(raw_name, chamber, df_members)
        if candidates and candidates[0].score >= fuzzy_auto_threshold:
            best = candidates[0]
            result.member_id = best.member_id
            result.member_name = best.name
            result.method = "fuzzy"
            result.confidence = best.score / 100.0
            report.resolved_fuzzy += 1
            report.results.append(result)
            continue

        # 4. Unresolved
        result.method = "unresolved"
        result.confidence = 0.0
        report.unresolved += 1
        report.results.append(result)

    return report


def apply_resolution(
    df_vote_casts: pl.DataFrame,
    report: ResolutionReport,
) -> pl.DataFrame:
    """Apply resolution results to vote casts, adding member_id column.

    Returns a new DataFrame with ``member_id`` and ``member_name`` columns.
    """
    # Build lookup: (raw_name, chamber) -> (member_id, member_name, method)
    lookup: dict[tuple[str, str], tuple[str, str, str]] = {}
    for r in report.results:
        if r.member_id:
            lookup[(r.raw_name, r.chamber)] = (r.member_id, r.member_name or "", r.method)

    # Map each row
    member_ids = []
    member_names = []
    methods = []

    for row in df_vote_casts.to_dicts():
        key = (row["raw_voter_name"], row["chamber"])
        if key in lookup:
            mid, mname, method = lookup[key]
            member_ids.append(mid)
            member_names.append(mname)
            methods.append(method)
        else:
            member_ids.append(None)
            member_names.append(None)
            methods.append("unresolved")

    return df_vote_casts.with_columns(
        pl.Series("member_id", member_ids, dtype=pl.Utf8),
        pl.Series("member_name", member_names, dtype=pl.Utf8),
        pl.Series("resolution_method", methods, dtype=pl.Utf8),
    )


def save_resolved_casts(df: pl.DataFrame) -> None:
    """Write the resolved vote casts to parquet."""
    out_path = PROCESSED_DIR / "fact_vote_casts.parquet"
    df.write_parquet(out_path)
    resolved = df.filter(pl.col("member_id").is_not_null())
    total = len(df)
    LOGGER.info(
        "Saved %d resolved vote casts (%d/%d = %.1f%% resolved) to %s",
        len(resolved),
        len(resolved),
        total,
        100 * len(resolved) / total if total > 0 else 0,
        out_path,
    )
