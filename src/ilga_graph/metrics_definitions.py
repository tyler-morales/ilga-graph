"""Single source of truth for what every metric means.

Use this for UI tooltips, API descriptions, and the vault so users see
one clear story: empirical stats first, derived metrics explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Empirical (raw) metrics: directly from bill/member data ──────────────────


@dataclass
class MetricDefinition:
    """One metric with a human-readable definition."""

    id: str
    name: str
    short_definition: str
    formula: str = ""


# Counts and rates we compute from bills only. No weighting or blending.
EMPIRICAL_METRICS: list[MetricDefinition] = [
    MetricDefinition(
        id="laws_filed",
        name="Laws filed (HB/SB)",
        short_definition="Number of substantive bills (HB/SB) the member is primary sponsor of. Excludes resolutions (HR/SR) and shell/technical placeholders.",
        formula="Count of primary-sponsored HB/SB, excluding shell bills.",
    ),
    MetricDefinition(
        id="laws_passed",
        name="Laws passed",
        short_definition="Number of those bills that became law (signed by Governor or adopted both houses).",
        formula="Count of primary-sponsored HB/SB with last_action indicating passage.",
    ),
    MetricDefinition(
        id="law_success_rate",
        name="Passage rate (laws)",
        short_definition="Share of the member's substantive bills that passed. Simple percentage.",
        formula="laws_passed ÷ laws_filed (0–100%).",
    ),
    MetricDefinition(
        id="passed_count",
        name="All bills passed",
        short_definition="Total primary-sponsored bills that passed (including resolutions).",
    ),
    MetricDefinition(
        id="vetoed_count",
        name="Bills vetoed",
        short_definition="Primary-sponsored bills that were vetoed.",
    ),
    MetricDefinition(
        id="stuck_count",
        name="Bills stuck",
        short_definition="Primary-sponsored bills that never moved out of committee/referral.",
    ),
    MetricDefinition(
        id="in_progress_count",
        name="Bills in progress",
        short_definition="Primary-sponsored bills currently moving through the process (not passed, vetoed, or stuck).",
    ),
    MetricDefinition(
        id="magnet_score",
        name="Avg co-sponsors per law",
        short_definition="Average number of co-sponsors on the member's substantive bills. Higher = more colleagues signing on.",
        formula="Total co-sponsors on member's HB/SB ÷ number of those bills.",
    ),
    MetricDefinition(
        id="bridge_score",
        name="Cross-party co-sponsorship %",
        short_definition="Share of the member's substantive bills that have at least one co-sponsor from the other party.",
        formula="(Bills with ≥1 opposite-party co-sponsor) ÷ laws_filed (0–100%).",
    ),
    MetricDefinition(
        id="pipeline_depth_avg",
        name="Pipeline depth (avg 0–6)",
        short_definition="How far the member's bills typically go. 0 = filed only, 6 = signed by Governor.",
        formula="Stages: 0 filed → 1 committee → 2 committee passed → 3 second reading → 4 chamber passed → 5 both chambers → 6 signed. Average over member's HB/SB.",
    ),
    MetricDefinition(
        id="network_centrality",
        name="Co-sponsorship network centrality",
        short_definition="How connected the member is in the co-sponsorship graph: share of other legislators they have co-sponsored with.",
        formula="(Unique co-sponsorship partners) ÷ (total legislators − 1). 0–1.",
    ),
    MetricDefinition(
        id="resolutions_passed",
        name="Resolutions passed",
        short_definition="Primary-sponsored resolutions (HR/SR/HJR/SJR) that were adopted.",
    ),
]


# ── Derived metrics: combinations or weighted composites ────────────────────


@dataclass
class MoneyballComponent:
    """One ingredient of the Moneyball composite, with weight and definition."""

    id: str
    weight_pct: float
    name: str
    short_definition: str


def get_moneyball_components() -> list[MoneyballComponent]:
    """Return the current Moneyball formula components (matches moneyball.MoneyballWeights)."""
    return [
        MoneyballComponent(
            id="effectiveness",
            weight_pct=24.0,
            name="Passage rate (laws)",
            short_definition="Laws passed ÷ laws filed (HB/SB only).",
        ),
        MoneyballComponent(
            id="pipeline",
            weight_pct=16.0,
            name="Pipeline depth",
            short_definition="How far the member's bills go on average (0–6 scale, normalized to 0–1).",
        ),
        MoneyballComponent(
            id="magnet",
            weight_pct=16.0,
            name="Avg co-sponsors per law",
            short_definition="Normalized against the member with the most co-sponsors.",
        ),
        MoneyballComponent(
            id="bridge",
            weight_pct=12.0,
            name="Cross-party co-sponsorship %",
            short_definition="Share of bills with at least one opposite-party co-sponsor.",
        ),
        MoneyballComponent(
            id="centrality",
            weight_pct=12.0,
            name="Network centrality",
            short_definition="How many unique colleagues they co-sponsor with (degree in co-sponsorship graph).",
        ),
        MoneyballComponent(
            id="institutional",
            weight_pct=20.0,
            name="Institutional power",
            short_definition="Bonus for leadership roles: President/Leader/Speaker (1.0), Chair/Spokesperson (0.5), Whip/Caucus Chair (0.25).",
        ),
    ]


MONEYBALL_ONE_LINER: str = (
    "Moneyball is a 0–100 composite that ranks legislators by combining "
    "passage rate, how far their bills go, co-sponsorship pull, cross-party work, "
    "network connectedness, and institutional role. We use it to surface high-impact "
    "targets beyond name recognition."
)


def get_effectiveness_score_definition() -> MetricDefinition:
    """Clarify the legacy 'effectiveness_score' (volume × rate) so it's not confused with passage rate."""
    return MetricDefinition(
        id="effectiveness_score",
        name="Volume‑weighted passage (legacy)",
        short_definition="Primary bill count × overall passage rate. Conflates volume and success; we prefer showing laws passed and passage rate separately.",
        formula="primary_bill_count × success_rate (all primary bills).",
    )


# ── Glossary for API / UI ────────────────────────────────────────────────────


@dataclass
class MetricsGlossary:
    """Full glossary: empirical metrics, derived explanations, Moneyball formula."""

    empirical: list[dict] = field(default_factory=list)
    effectiveness_score: dict = field(default_factory=dict)
    moneyball_one_liner: str = ""
    moneyball_components: list[dict] = field(default_factory=list)


def get_metrics_glossary() -> MetricsGlossary:
    """Return a glossary suitable for JSON/GraphQL and server-rendered UI."""
    empirical = [
        {
            "id": m.id,
            "name": m.name,
            "short_definition": m.short_definition,
            "formula": m.formula or None,
        }
        for m in EMPIRICAL_METRICS
    ]
    effectiveness = get_effectiveness_score_definition()
    return MetricsGlossary(
        empirical=empirical,
        effectiveness_score={
            "id": effectiveness.id,
            "name": effectiveness.name,
            "short_definition": effectiveness.short_definition,
            "formula": effectiveness.formula or None,
        },
        moneyball_one_liner=MONEYBALL_ONE_LINER,
        moneyball_components=[
            {
                "id": c.id,
                "weight_pct": c.weight_pct,
                "name": c.name,
                "short_definition": c.short_definition,
            }
            for c in get_moneyball_components()
        ],
    )
