# Canonical Content Sources

This doc defines the **approved sources** for all substantive content on the site. The rule **no-hallucination-content** requires that copy, key points, FAQs, and messaging come **only** from these sources. No invented facts, stats, or wording.

## Why this matters

- **Continuity** — Legislator one-pager (PDF + leg brief page), constituent brief (PDF + The Issue page), and in-app copy (email, call script) must stay aligned.
- **Accuracy** — Legal and policy claims must match the official briefs and content.py.
- **Single source of truth** — When adding or editing content, the agent (and you) pull from here instead of inventing.

## Canonical sources

### 1. Strategic message (mission, vision, success measure, 5 points)

- **Where:** `src/ilga_graph/routers/content.py` — `STRATEGIC_MISSION`, `STRATEGIC_VISION`, `STRATEGIC_SUCCESS_MEASURE`, `STRATEGIC_FIVE_POINTS`
- **Used on:** Home (mission block + success measure), The Issue (Key points + vision callout), and anywhere we need the shared message
- **Rule:** Any “key points” or mission/vision text must match these constants. Do not add new bullets that aren’t in STRATEGIC_FIVE_POINTS or approved by you.

### 2. Legislator one-pager / official brief

- **PDF:** `/static/advocacy/IL_Kei_Vehicle_Registration_Fix_Brief.pdf` (same as “IL Kei Vehicle Registration Fix Brief” in sidebar)
- **Canonical text in repo:** `src/ilga_graph/templates/legislator_brief.html` — sections:
  - Issue in one sentence (+ core ambiguity)
  - What the Secretary of State is relying on
  - Why a narrowly tailored statute
  - Proposed legislative concept
  - What we are asking your office to do (3 items)
  - Attachments & reference, Point of contact
- **Rule:** Legislator-facing copy (brief page, email/call script references to “the one-pager”) must match or summarize this text. Do not invent new brief content. If you add a “Key points” section on the leg brief page, use STRATEGIC_FIVE_POINTS or condensed versions of these section headings/text only.

### 3. Constituent one-pager / constituent brief

- **PDF:** `/static/images/Illinois_Kei_Vehicle_Registration_Constituent_Brief.pdf` (linked from The Issue sidebar as “Illinois Kei Vehicle Registration Constituent Brief”)
- **Canonical text in repo:** `src/ilga_graph/templates/the_issue.html` — narrative sections (The Issue in Plain English, Why This Is Happening, What This Means, The Narrow Fix, Why This Matters, etc.) plus `content.py` STRATEGIC_* for Key points and vision
- **Rule:** Constituent-facing copy must align with the Issue page and STRATEGIC_*. If you need wording that should come from the constituent PDF and it isn’t in the template or content.py, add a text extract to this folder (e.g. `constituent-brief-extract.md`) or ask the user to provide it. Do not invent.

### 4. Fact sheet for volunteers

- **PDF:** `/static/advocacy/Kei_Registration_Fact_Sheet.pdf` (linked from The Issue sidebar as "Fact sheet for volunteers"). Place the file at this path; you can generate it by printing the `/fact-sheet` page to PDF.
- **Canonical text in repo:** `content.py` — `FACT_SHEET_ISSUE`, `FACT_SHEET_POSITION`, plus `STRATEGIC_FIVE_POINTS` and selected `FAQ_ADVOCACY` items. The web page at `/fact-sheet` renders this content.
- **Rule:** Fact sheet copy must match these constants and `docs/advocacy/focused-next-steps-1-2-4-5-6.md` §5. Do not invent.

### 5. FAQs

- **Where:** `content.py` — `FAQ_LAW`, `FAQ_ADVOCACY`, and `FAQ_SESSION` (The Issue: law, advocacy, session calendar), `FAQ_LEGISLATORS` (legislator brief)
- **Rule:** FAQ answers and sources come only from these. Do not add new Q&A unless the user provides the wording or you add it to content.py and the user approves.

### 6. State table, bills, documents, sources

- **Where:** `content.py` — `BRIEF_STATE_STATUS`, `BRIEF_BILLS_PASSED`, `BRIEF_BILLS_CURRENT`, `BRIEF_SOURCES`, `BRIEF_DOCUMENTS`
- **Rule:** Any state/bill/source list or doc link must come from these. Do not invent states, bill titles, or URLs.

### 7. Session schedule (House/Senate)

- **Where:** `reference/session_schedule.json` — single source of truth for Illinois General Assembly session dates, deadlines, and holidays (104th GA Spring 2026). Loaded at runtime by `src/ilga_graph/session_schedule.py` (`load_schedule()`, `get_all_deadlines()`, `session_label()`, etc.).
- **Used on:** The Issue page (FAQ “Session calendar & deadlines” and key-deadlines list), the **Timeline page** (`/timeline` — "Key session deadlines" section), and any future reminders or date-driven copy.
- **Rule:** All session dates, deadlines, and reminders must be derived from this file. Do not hardcode session or deadline dates elsewhere. When the session calendar changes, update `reference/session_schedule.json`.

## Adding or changing canonical text

- **Strategic message:** Edit `STRATEGIC_MISSION`, `STRATEGIC_VISION`, `STRATEGIC_SUCCESS_MEASURE`, `STRATEGIC_FIVE_POINTS` in content.py.
- **Legislator brief:** Edit `IL_Kei_Vehicle_Registration_Fix_Brief 1.txt` in the repo root. The legislator brief page reads it at request time. Keep the PDF in sync for downloads.
- **Constituent brief:** Edit `Illinois_Kei_Vehicle_Registration_Constituent_Brief.txt` in the repo root. The Issue page narrative reads it at request time. Keep the PDF in sync for downloads.
- **Fact sheet:** Edit `FACT_SHEET_ISSUE`, `FACT_SHEET_POSITION` in content.py. The `/fact-sheet` page renders this; for the sidebar document, place a PDF at `src/ilga_graph/static/advocacy/Kei_Registration_Fact_Sheet.pdf` (e.g. print /fact-sheet to PDF).
- **FAQs:** Edit `FAQ_LAW`, `FAQ_ADVOCACY`, `FAQ_SESSION`, and `FAQ_LEGISLATORS` in content.py.
- **Session schedule:** Edit `reference/session_schedule.json`. Code reads it via `ilga_graph.session_schedule`; do not duplicate dates in content.py or templates.

## For the AI

When asked “do we need a key points section on the leg brief page?”:

1. **Evaluate:** Use the Hardball spec (e.g. Ch 7: framing, making the case) to decide *whether* a key points section is useful (e.g. “yes, for quick scan”).
2. **Content:** If adding it, use **only** STRATEGIC_FIVE_POINTS or short summaries of the existing brief section headings/text from legislator_brief.html. Do not invent new points.
3. **Cite:** Say which canonical source each point comes from (e.g. “From STRATEGIC_FIVE_POINTS” or “From ‘Issue in one sentence’ in legislator_brief.html”).
