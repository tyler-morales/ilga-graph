---
name: canonical-content-sources
description: When writing or editing substantive site content (copy, key points, FAQs, brief text), uses only approved canonical sources (content.py, legislator_brief.html, the_issue.html, one-pager/constituent PDFs) and cites which source. Use when adding or changing copy on the legislator brief, The Issue, home, advocacy scripts, or any key points or messaging.
---

# Canonical Content Sources

When you add or edit **substantive content** (copy, key points, FAQs, brief sections, email/call script wording), use **only** the approved canonical sources. Never invent facts, statistics, or messaging.

## Quick reference

| What you're writing | Use only |
|---------------------|----------|
| Mission, vision, 5-point message | content.py: STRATEGIC_MISSION, STRATEGIC_VISION, STRATEGIC_FIVE_POINTS |
| Legislator brief page (any section or key points) | legislator_brief.html existing sections + STRATEGIC_FIVE_POINTS (for a key points block) |
| The Issue page / constituent narrative | the_issue.html prose + content.py STRATEGIC_* |
| FAQs | content.py: FAQ_ADVOCATES, FAQ_LEGISLATORS |
| State table, bills, doc links | content.py: BRIEF_STATE_STATUS, BRIEF_*, BRIEF_SOURCES, BRIEF_DOCUMENTS |

## One-pager and constituent brief continuity

- **Legislator one-pager:** The canonical text is the content of `legislator_brief.html` (Issue in one sentence, What SOS is relying on, Why narrow statute, Proposed concept, What we are asking). The PDF at `/static/advocacy/IL_Kei_Vehicle_Registration_Fix_Brief.pdf` should match. When writing anything “from the one-pager,” use that template text only.
- **Constituent brief:** The canonical constituent-facing text is in `the_issue.html` and STRATEGIC_FIVE_POINTS. The PDF at `/static/images/Illinois_Kei_Vehicle_Registration_Constituent_Brief.pdf` is the authoritative doc; if you don’t have its text in the repo, use the_issue.html + STRATEGIC_* and do not invent. If the user adds `docs/canonical/constituent-brief-extract.md` (or similar), use that for exact wording.

## Workflow

1. **Identify the audience** — Legislator (brief page, one-pager refs) vs constituent (Issue page, email/call script).
2. **Open the right source** — content.py, legislator_brief.html, or the_issue.html (and docs/canonical/ if present).
3. **Draft only from those sources** — Paraphrase or quote; do not add new claims or points that aren’t there.
4. **Cite in your response** — e.g. “Key point 1 is from STRATEGIC_FIVE_POINTS; key point 2 is a condensed version of ‘Issue in one sentence’ in legislator_brief.html.”

If the user asks for content that would require wording not in any canonical source, say so and ask them to provide the text or add it to content.py / docs/canonical/.

Full list: `docs/canonical/README.md`. Rule: `.cursor/rules/no-hallucination-content.mdc`.
