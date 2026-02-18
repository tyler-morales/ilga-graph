"""Shared constants (e.g. category → committee mapping for advocacy and explore)."""

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
