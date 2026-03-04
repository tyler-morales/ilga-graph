---
name: canonical-content-sources
description: When writing or editing substantive site content (copy, key points, FAQs, brief text), uses only approved canonical sources (docs/canonical/briefs/, docs/canonical/books/, content_constants.py, templates) and cites which source. Use when adding or changing copy on the legislator brief, The Issue, home, advocacy scripts, or any key points or messaging.
---

# Canonical Content Sources

When you add or edit **substantive content** (copy, key points, FAQs, brief sections, email/call script wording), use **only** the approved canonical sources. Never invent facts, statistics, or messaging.

## Quick reference

| What you're writing | Use only |
|---------------------|----------|
| Mission, vision, 5-point message | content.py: STRATEGIC_MISSION, STRATEGIC_VISION, STRATEGIC_FIVE_POINTS |
| Legislator brief page (any section or key points) | legislator_brief.html existing sections + STRATEGIC_FIVE_POINTS (for a key points block) |
| The Issue page / constituent narrative | the_issue.html prose + content.py STRATEGIC_* |
| FAQs | content.py: FAQ_LAW, FAQ_ADVOCACY (The Issue), FAQ_LEGISLATORS (legislator brief) |
| State table, bills, doc links | content.py: BRIEF_STATE_STATUS, BRIEF_*, BRIEF_SOURCES, BRIEF_DOCUMENTS |

## One-pager and constituent brief continuity

- **Legislator one-pager:** Canonical text is `docs/canonical/briefs/legislator-brief.txt` (loaded at runtime by content.py). The PDF at `/static/advocacy/IL_Kei_Vehicle_Registration_Fix_Brief.pdf` should match. When writing anything “from the one-pager,” use that template text only.
- **Constituent brief:** Canonical text is `docs/canonical/briefs/constituent-brief.txt` (loaded at runtime for The Issue page). Key points from STRATEGIC_FIVE_POINTS. Rendered narrative is in `the_issue.html`. The PDF at `/static/images/Illinois_Kei_Vehicle_Registration_Constituent_Brief.pdf` is the authoritative doc; if you don’t have its text in the repo, use the_issue.html + STRATEGIC_* and do not invent. If the user adds `docs/canonical/constituent-brief-extract.md` (or similar), use that for exact wording.

## Workflow

1. **Identify the audience** — Legislator (brief page, one-pager refs) vs constituent (Issue page, email/call script).
2. **Open the right source** — Brief text in `docs/canonical/briefs/`; strategy books in `docs/canonical/books/`; site copy (STRATEGIC_*, FAQ_*, etc.) in content_constants.py and templates. Full list: `docs/canonical/README.md`.
3. **Draft only from those sources** — Paraphrase or quote; do not add new claims or points that aren’t there.
4. **Cite in your response** — e.g. “Key point 1 is from STRATEGIC_FIVE_POINTS; key point 2 is a condensed version of ‘Issue in one sentence’ in legislator_brief.html.”

If the user asks for content that would require wording not in any canonical source, say so and ask them to provide the text or add it to content.py / docs/canonical/.

Full list: `docs/canonical/README.md`. Rule: `.cursor/rules/no-hallucination-content.mdc`.
