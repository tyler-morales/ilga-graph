# Domain glossary

Single source of truth for organizational and domain terms used in the app. These definitions disambiguate overloaded words (e.g. **campaign** = action alert vs overall effort) so code, copy, and docs stay consistent.

**Sources:** `DOMAIN_GLOSSARY`, `KEI_GLOSSARY`, and `SESSION_SCHEDULE_TERMS` in `src/ilga_graph/routers/content.py`. The Python constants are the canonical lists; this doc renders them for humans. The public **[/glossary](/glossary)** page shows domain terms, session schedule terms, and kei vehicle terms. **Inline definitions:** On The Issue, Timeline, and Legislator brief pages, the first occurrence of each term is a clickable control that shows the same definition in a popover (tooltip); definitions are reused from these constants so there is no second copy to maintain. Glossary entries may include optional `source` and `source_url`. **Only entries with `source_url`** (an external link) show a “Source:” line on the glossary page; plain-text sources without a URL are not shown.

---

## Why these distinctions matter

Many terms have more than one meaning in the codebase:

- **Campaign** — A single action alert (DB model) vs the multi-year advocacy initiative.
- **Milestone** — A legislative deadline (session milestone) vs a dated checkpoint on the master timeline vs a bill stage.
- **Phase** — A period on the master timeline (Build, Intro, …) vs the user’s goal phase (district vs broker) vs internal pipeline stages.
- **Timeline** — The master plan on `/timeline` vs the progress checklist on `/updates` vs bill action history.

Using the glossary when writing copy or code avoids mixing these meanings.

---

## Advocacy

| Term | Definition |
|------|------------|
| **Campaign** | A single, time-bound action alert. At most one campaign is active at a time. Has title, message, ask, optional start/end dates; outreach recorded while active is attributed to it. *Not* the overall multi-year advocacy initiative (see Advocacy effort). |
| **Advocacy effort** | The overall multi-year initiative toward the legislative objective (e.g. kei vehicle registration fix). Not a DB object; encompasses all campaigns, updates, and organizing. |
| **Ask** | The specific request to a legislator or constituent (noun). E.g. “Contact your rep” or “Support HB 1234.” Stored in `Campaign.ask` as CTA button text. |
| **Coalition (advocacy)** | Organizations aligned on the issue; building coalition = recruiting orgs and stakeholders. *Not* ML voting coalition (legislators who vote together). |
| **Advocate** | A user who takes outreach action (call, email). May or may not be a constituent of the legislator contacted. |
| **Constituent** | An advocate who lives in a legislator’s district. Stored as a boolean on outreach events; “Constituent Brief” is the canonical document for the public. |
| **Outreach** | A call, email, or no-answer recorded against a legislator. One `OutreachEvent` per action. |
| **Contact** | (Verb) To reach a legislator’s office (call or email). (Noun) The person at the office who answered the call (`OutreachEvent.contact_name`). “Contact period” = campaign duration. |
| **Update** | An email announcement sent to subscribers. Has title, body, type (Major/Minor/Other), optional image. DB: `Update` model. |
| **Brief** | A canonical document: legislator brief (for offices) or constituent brief (for the public). The one-pager PDF is the print version of the legislator brief. |

---

## Legislative

| Term | Definition |
|------|------------|
| **Session** | A day the chamber meets in Springfield, or the full session period (e.g. 104th GA Spring 2026). Session schedule lists session days and deadlines. |
| **Session milestone** | A legislative deadline with a date (e.g. committee deadline, third reading deadline). Used to set campaign end dates in admin; `Campaign.session_milestone_id`. |
| **Bill stage** | Where a bill sits in the process: introduced, committee, floor, passed one chamber, passed both, signed. P(Advance) predicts chance of reaching a positive stage. |
| **Bill action** | A single procedural event on a bill (e.g. “Referred to Assignments”, “Do Pass”). Shown in bill action history. |
| **Voting coalition** | ML-discovered cluster of legislators who vote together. Used in Intelligence. *Not* an advocacy coalition (organizations aligned on the issue). |

---

## Product

| Term | Definition |
|------|------------|
| **Phase** | (Timeline) One of the four periods on the master timeline: Build, Intro, Committee & Floor, Governor. (Goal) “district” or “broker”—which set of outreach steps the user is on. |
| **Master timeline** | The phased plan from now to bill signed, shown on `/timeline` as a waterfall (Gantt-style) with a time axis and phase bars. Source: `TIMELINE_PHASES`. Each phase has a date range and optional milestones. |
| **Progress checklist** | The short ordered list of stages on `/updates` (Outreach → … → Keis be legal). Achieved steps at full opacity, rest at reduced. Source: `PROGRESS_CHECKPOINTS`. Not dated; distinct from master timeline. |
| **Milestone** | A dated checkpoint within a timeline phase (e.g. “Lock lead sponsor(s)”, “Bill introduced”). Shown on `/timeline` under each phase. *Not* session milestone (legislative deadline) or bill stage. |
| **Goal** | The user’s outreach task list: contact district legislators (4 actions), then Power Broker (2 actions). “Your goal” / “This week’s goal” in the sidebar. *Not* the advocacy objective (statutory fix). |
| **Drawer** | The slide-out panel for call scripts and email templates. Opens from “Reach out” on a legislator card. |
| **Funnel** | The user journey from page visit to completed outreach. Measured for conversion (e.g. % who opened drawer and completed at least one call/email). |
| **State of kei** | User self-report: do they have a kei (registered / revoked / denied) or not (would want / would not want). Each submission in `kei_poll_responses`; logged-in also `User.kei_status`. Admin /admin/poll: verified + all-responses (pie + table). Collected via poll: first "Do you have a kei vehicle?" (Yes/No), then either registration status or "Would you want one?". Footer and home poll, /updates?prompt=kei; welcome email links to the poll. |
| **Community story** | A user-submitted photo and short story (name, location, consent) for the home-page marquee. **Share your story** is shown to every user who completes the Kei poll; owners (registered / revoked / denied) open the photo+story modal; non-owners open a text-only **interest statement** modal. Stories are stored as `CommunityStory` (status: pending); admins approve or deny at /admin/stories. Approved items appear in the marquee via `get_marquee_items(db)`; text-only statements at /admin/statements. |


---

## Kei vehicle terms (public glossary)

`KEI_GLOSSARY` in `content.py` defines terms for the public: **Kei vehicle**, **Kei class**, **25-year rule**, **Highway-built**, **625 ILCS 5/3-401(c-1)**, **Shaken**, **One-pager**. Wording is from canonical content (STRATEGIC_*, FAQ_*, briefs); no invented stats.

## Related

- **Inline tooltips** — The Issue, Timeline, and Legislator brief: first occurrence of each term is a button that opens a popover with the same definition; single source in `content.py` (`apply_inline_glossary`, `_inline_glossary_terms`).
- **Session schedule terms** (LRB, committee deadline, third reading, etc.) — `SESSION_SCHEDULE_TERMS` in `content.py`; FAQ, session pill, and `/glossary`.
- **Public glossary page** — `/glossary` shows domain terms, session terms, and kei terms.
- **Metrics glossary** — `metricsGlossary` in GraphQL; Moneyball and empirical metrics.
- **Canonical content** — `docs/canonical/README.md` for where copy comes from.
