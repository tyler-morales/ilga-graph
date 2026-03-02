# White-label decoupling gaps (current state)

This doc lists what is driven by campaign config and what still remains for a true white-label multi-tenant build. It reflects the repo state and should be updated when code or config changes.

**Reference:** [Campaign decoupling (white-label readiness)](campaign-decoupling.md) describes what is already config-driven and how provisioning works.

---

## Done: now driven by campaign.json

- **Poll prompt and redirects:** `poll_prompt_query` — updates page and all redirects use it; welcome email poll URL uses it. Default `"kei"`.
- **Welcome email:** `welcome_email_intro`, `welcome_email_poll_link_text` — body and link text from config; poll URL from `poll_prompt_query`. Fallbacks: `issue_summary`, "tell us your status".
- **Error/404 page:** `error_page_facts` (optional list of `{text, image?, image_alt?, image_credit?}`); when empty, app uses built-in Kei facts. `error_page_fact_label` for alt text and aria-label (default "Kei vehicles").
- **Bill URLs for scraping:** `bill_status_urls` (optional list) in campaign.json; `get_bill_status_urls()` uses it first, then env, then defaults.
- **Mission and attribution:** `strategic_mission`, `mission_attribution` — template globals and content/home views use them; _mission_statement.html and terms.html use config with fallbacks.
- **Terms scope:** terms.html uses `issue_summary|default('Kei vehicle registration')` for "about X".

---

## 1. Already driven by campaign.json (unchanged)

- Hero and advocacy headlines/subheads, `primary_color`, `issue_summary`
- `default_topic`, `brief_pdf_filename`, `brief_pdf_url_path`, `one_pager_points`
- `poll_slug` — used for Poll lookup in admin, updates, and advocacy (fallback `"kei"` when config missing). Current Kei value in config is `"kei-status"`.
- `SITE_NAME` and `META_DESCRIPTION` defaults from config (`config.py`)

---

## 2. Content still in code (not in campaign.json)

All of this lives in [src/ilga_graph/routers/content_constants.py](../src/ilga_graph/routers/content_constants.py) and [src/ilga_graph/constants.py](../src/ilga_graph/constants.py). For white-label, either move into campaign.json or keep as **tenant-specific content** (e.g. per-tenant JSON or env).

| Area | Location | Notes |
|------|----------|--------|
| Brief paths | content_constants.py | `CONSTITUENT_BRIEF_PATH`, `LEGISLATOR_BRIEF_PATH` — hardcoded Kei .txt filenames |
| Strategic copy | content_constants.py | `STRATEGIC_MISSION`, `STRATEGIC_VISION`, `STRATEGIC_FIVE_POINTS`, `WHY_SHOULD_YOU_CARE_*`, `WHY_YOU_CARE_*`, `KEI_POLL_WHY_WE_ASK` |
| Marquee / progress / docs | content_constants.py | `MARQUEE_IMAGES`, `PROGRESS_CHECKPOINTS` ("Keis be legal"), `BRIEF_DOCUMENTS`, `BRIEF_STATE_STATUS`, `FACT_SHEET_PDF_URL` |
| Glossary | content_constants.py | `KEI_GLOSSARY` — Kei vehicle terms |
| FAQs | content_constants.py | `FAQ_LAW`, `FAQ_ADVOCACY`, `FAQ_LEGISLATORS` — Kei-specific Q&A |
| Poll option labels | constants.py | `KEI_STATUS_OPTIONS`, `KEI_POLL_IMPACT_OPTIONS`, `KEI_FIRST_OPTIONS`, etc. — could be campaign-specific option sets |

**Recommendation:** Extend `campaign.json` with optional keys (e.g. `strategic_mission`, `brief_txt_path`, `poll_why_we_ask`, `progress_checkpoints`, `fact_sheet_pdf_url`) and/or support a second artifact (e.g. `campaign_content.json`) for long-form copy and glossary. Keep canonical-sources rule: content from approved sources only; config points at which source.

---

## 3. URL and route coupling (remaining)

- **Done:** Poll prompt and redirects use `poll_prompt_query` from config; welcome email poll URL uses it.
- **Remaining:** Poll form/result routes `/updates/kei-status`, `/updates/kei-status-results`, `/updates/kei-poll-form`, etc. — path segment is literal `kei`. For a second campaign, optionally add a generic route (e.g. `/updates/status`) that reads poll from config.

---

## 4. Welcome email

**Done:** [src/ilga_graph/email_utils.py](../src/ilga_graph/email_utils.py) uses `welcome_email_intro`, `welcome_email_poll_link_text`, and poll URL from `poll_prompt_query`. Opening line ("Whether you daydream...") remains Kei-specific; for full white-label that could be a config key later.

---

## 5. 404 / error page

**Done:** [src/ilga_graph/main.py](../src/ilga_graph/main.py) uses `error_page_facts` from config when non-empty, else built-in Kei facts. `error_page_fact_label` drives alt text and aria-label in 404 template.

---

## 6. Bill URLs for scraping

**Done:** [src/ilga_graph/config.py](../src/ilga_graph/config.py) `get_bill_status_urls()` uses campaign `bill_status_urls` first (when non-empty), then `ILGA_VOTE_BILL_URLS`, then `DEFAULT_BILL_STATUS_URLS`.

---

## 7. Templates and UI copy

**Done:** _mission_statement.html uses `mission_attribution|default(...)`; terms.html uses `issue_summary|default('Kei vehicle registration')`; 404 uses `error_page_fact_label|default('Kei vehicles')`. content.py and home.py pass `strategic_mission` from campaign when set.

**Remaining:** Hardcoded Kei strings still in:

- [fact_sheet.html](../src/ilga_graph/templates/fact_sheet.html): "kei vehicle registration", "sidebar-kei-poll-section"
- [index.html](../src/ilga_graph/templates/index.html): "kei vehicle registration email", `#kei-personalize-flow`, `ConversationUI.initKeiPersonalizeFlow()`
- [_statement_submit_modal.html](../src/ilga_graph/templates/_statement_submit_modal.html): "kei vehicle", "kei vehicle advocacy"
- [_macros.html](../src/ilga_graph/templates/_macros.html): "kei bills", "Keis registrable", kei_poll macro and class names; share text "The Land of Kei", "Kei vehicle registration"
- [_kei_personalize_drawer.html](../src/ilga_graph/templates/_kei_personalize_drawer.html): "Do you have a Kei vehicle?", step labels, id `kei-personalize-flow`

CSS/class names (e.g. `.kei-poll`) can stay unless doing a broader rename to "campaign-poll".

---

## 8. DB and schema (unchanged)

[campaign-decoupling.md](campaign-decoupling.md): User columns `kei_status`, `kei_impact_slug`, and tables like `KeiPollResponse`, `KeiInterestStatement` are Kei-named; for full white-label these would become campaign-scoped. No schema change required for current "set the stage" decoupling.

---

## 9. Account page (current state)

GET/POST `/account` exists ([src/ilga_graph/routers/account.py](../src/ilga_graph/routers/account.py)); campaign-decoupling doc describes it. It shows "Your answers" from `kei_status` / `kei_impact_slug` / `kei_personal_note`. For white-label, those labels and any copy would eventually come from campaign config; no new gap beyond the content-in-code items above.

---

## 10. Config and env naming

- `RATE_LIMIT_KEI_STATUS_ANON_PER_HOUR` in [config.py](../src/ilga_graph/config.py) — Kei-named; behavior is generic. Optional rename later.
- `campaign_config.py` fallback `_default_one_pager_points()` is Kei copy when JSON is missing; doc already notes this.

---

## Summary: next steps

1. **Done:** campaign.json extended with `poll_prompt_query`, `welcome_email_intro`, `welcome_email_poll_link_text`, `strategic_mission`, `mission_attribution`, `error_page_facts`, `error_page_fact_label`, `bill_status_urls`. Updates, email, 404, bill URLs, and mission/terms use them.
2. **Remaining:** Optional brief paths, progress checkpoints, fact sheet URL, glossary/FAQs in config or content loader; poll option labels for tenant-specific polls; generic `/updates/status` route for second campaign.
3. **Remaining:** Fact sheet, index, statement modal, macros, personalize drawer — source visible copy from config where practical.
4. **Optionally:** Rename rate-limit env; later generalize DB column/table names when adding a second tenant.
