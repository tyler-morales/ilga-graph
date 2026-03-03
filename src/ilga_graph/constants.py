"""Shared constants (e.g. category → committee mapping for advocacy and explore)."""

# Committee codes that most or all bills go through (gatekeeper + appropriations).
# Used to limit "Why we recommend them" chair chips to topic + general committees.
GENERAL_COMMITTEE_CODES: list[str] = [
    "SAPP",  # Appropriations
    "SCOA",  # Assignments * Reports (Senate gatekeeper)
]

# Policy categories → Senate committee codes. Used by advocacy search and Power Map.
CATEGORY_COMMITTEES: dict[str, list[str]] = {
    "": [],
    "Transportation": ["STRN"],
    "Agriculture": ["SAGR"],
    "Commerce & Small Business": ["SCOM", "SBTE"],
    "Criminal Justice": ["SCRL", "SHRJ"],
    "Education": ["SESE", "SCHE"],
    "Energy & Environment": ["SENE", "SNVR"],
    "Healthcare & Human Services": ["SBMH", "SCHW", "SHUM"],
    "Housing": ["SHOU"],
    "Insurance & Finance": ["SINS", "SFIC"],
    "Labor": ["SLAB"],
    "Revenue & Pensions": ["SREV", "SPEN"],
    "State Government": ["SGOA", "SHEE", "SEXC"],
}

# State of kei: user self-report (has kei / does not have kei with sub-options).
KEI_STATUS_SLUGS: frozenset[str] = frozenset(
    {
        "registered",
        "revoked",
        "denied",
        "would_want",
        "would_not_want",
    }
)
# Labels match poll form (Question 2) and results chart.
KEI_STATUS_OPTIONS: list[tuple[str, str]] = [
    ("registered", "Registered"),
    ("revoked", "Had one; registration was revoked"),
    ("denied", "Was denied registration"),
    ("would_want", "Yes, I'd want one"),
    ("would_not_want", "No, I wouldn't want one"),
]

# First poll question: two bubbles (animate up). Second question shows status sub-options.
KEI_FIRST_OPTIONS: list[tuple[str, str]] = [
    ("have_or_had", "I have or had a kei"),
    ("dont_have", "I don't have a kei"),
]
KEI_STATUS_BY_FIRST: dict[str, list[tuple[str, str]]] = {
    "have_or_had": [
        ("registered", "Registered"),
        ("revoked", "Had one; registration was revoked"),
        ("denied", "Was denied registration"),
    ],
    "dont_have": [
        ("would_want", "Yes, I'd want one"),
        ("would_not_want", "No, I wouldn't want one"),
    ],
}

# Main poll Q3: "What would this fix mean for you?" — benefit-oriented for legislator narrative.
KEI_POLL_IMPACT_SLUGS: frozenset[str] = frozenset(
    {"direct_owner", "want_to_buy", "know_someone", "support_cause"}
)
KEI_POLL_IMPACT_OPTIONS: list[tuple[str, str]] = [
    ("direct_owner", "I could keep (or get) my kei legally registered"),
    ("want_to_buy", "I could buy a kei I've been wanting"),
    ("know_someone", "It would help someone I know"),
    ("support_cause", "I support clearer rules for kei vehicles"),
]
KEI_IMPACT_SLUG_COOKIE = "kei_impact_slug"

# Impact options for script personalization (Step 3 after status).
# Key = kei_status; value = (slug, label).
KEI_IMPACT_OPTIONS: dict[str, list[tuple[str, str]]] = {
    "registered": [
        ("work_commute", "Daily commute"),
        ("recreation", "Recreation"),
        ("worried_revoked", "Worried it could be revoked"),
        ("other", "Other"),
    ],
    "revoked": [
        ("sitting_unused", "Sitting in my garage"),
        ("lost_commute", "Lost my way to work"),
        ("cost_money", "Cost me money"),
        ("other", "Other"),
    ],
    "denied": [
        ("sitting_unused", "Sitting in my garage"),
        ("lost_commute", "Lost my way to work"),
        ("cost_money", "Cost me money"),
        ("other", "Other"),
    ],
    "would_want": [
        ("for_work", "Want it for work"),
        ("recreation", "Recreation"),
        ("small_business", "Small business"),
        ("other", "Other"),
    ],
    "would_not_want": [
        ("support_cause", "I support the cause"),
        ("know_someone", "I know someone affected"),
        ("civic_duty", "Civic duty"),
        ("other", "Other"),
    ],
}

# All impact (slug, label) for results — poll Q3 first, then status-specific, legacy.
KEI_IMPACT_ALL_OPTIONS: list[tuple[str, str]] = [
    ("direct_owner", "I could keep (or get) my kei legally registered"),
    ("want_to_buy", "I could buy a kei I've been wanting"),
    ("know_someone", "It would help someone I know"),
    ("support_cause", "I support clearer rules for kei vehicles"),
    ("civic_duty", "Civic duty"),
    ("work_commute", "Daily commute"),
    ("recreation", "Recreation"),
    ("worried_revoked", "Worried it could be revoked"),
    ("sitting_unused", "Sitting in my garage"),
    ("lost_commute", "Lost my way to work"),
    ("cost_money", "Cost me money"),
    ("for_work", "Want it for work"),
    ("small_business", "Small business"),
    ("other", "Other"),
]

# Advocacy intro: outreach preference. Cookie + User.call_pref; merged on signup.
# Values: no (email only), yes (call and email), call_only, elevator (30-sec script scaffold).
ADV_CALL_PREF_COOKIE = "adv_call_pref"
ADV_CALL_PREF_MAX_AGE = 365 * 24 * 60 * 60  # 1 year
ADV_CALL_PREF_VALUES = ("no", "yes", "call_only", "elevator")

CATEGORY_CHOICES: list[tuple[str, str]] = [
    ("", "All categories"),
    ("Transportation", "Transportation"),
    ("Agriculture", "Agriculture"),
    ("Commerce & Small Business", "Commerce & Small Business"),
    ("Criminal Justice", "Criminal Justice"),
    ("Education", "Education"),
    ("Energy & Environment", "Energy & Environment"),
    ("Healthcare & Human Services", "Healthcare & Human Services"),
    ("Housing", "Housing"),
    ("Insurance & Finance", "Insurance & Finance"),
    ("Labor", "Labor"),
    ("Revenue & Pensions", "Revenue & Pensions"),
    ("State Government", "State Government"),
]
