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
KEI_STATUS_OPTIONS: list[tuple[str, str]] = [
    ("registered", "I have a kei (registered)"),
    ("revoked", "I had a kei; registration was revoked"),
    ("denied", "I was denied registration"),
    ("would_want", "I don't have a kei but would want one"),
    ("would_not_want", "I don't have a kei and wouldn't want one"),
]

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
