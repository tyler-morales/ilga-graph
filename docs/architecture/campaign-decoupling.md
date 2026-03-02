# Campaign decoupling (white-label readiness)

This doc describes how the app is structured for a single campaign today (Kei vehicle registration) and what is in place to support swapping campaigns without rewriting the application.

## Scrapers: generic ILGA

All scrapers operate on generic Illinois General Assembly (ILGA) data:

- **Member/committee scraping** (`scraper.py`) — Members list, detail pages, committees, rosters. No topic or vehicle logic.
- **Bills** (`scrapers/bills.py`) — Bill index from `/Legislation` discovers all doc types (SB, HB, SR, HR, etc.); BillStatus scraping is per-URL. No topic filtering.
- **Votes** (`scrapers/votes.py`) — Parses any ILGA roll-call PDF.
- **Witness slips** (`scrapers/witness_slips.py`) — Fetches slips for any bill status URL.
- **Full text** (`scrapers/full_text.py`) — Full text for any bill.

**Campaign-specific input:** The only campaign hook is the optional list of bill URLs used for focused vote/slip scraping. That list comes from `config.get_bill_status_urls()` (defaults are Kei-related; override via `ILGA_VOTE_BILL_URLS`). The main pipeline (e.g. `scripts/scrape_votes.py`) uses the full bill cache (`bills.json`), so it is already topic-agnostic. For a new campaign, change the env or point bill URL list at a campaign config.

## Database: core generic; a few campaign-specific names

| Area | Status |
|------|--------|
| **OutreachEvent** | Generic: user_id, member_id, kind, campaign_id, notes, support_score. No campaign name in schema. |
| **User** | Campaign-specific columns: `kei_status` (e.g. "registered", "would_want") and `kei_impact_slug` (how it affects you: support_cause, know_someone, civic_duty, other). For white-label these would be campaign-scoped. |
| **Campaign, Update, AuthCode** | Generic. |
| **KeiPollResponse** | Legacy table; new flow uses **Poll** + **PollResponse** (generic). KeiPollResponse kept for backfill/compat. |
| **Polls** | Two seeded for Kei: `kei` (status: have/don't have → 5 options) and `kei_impact` (how it affects you: 4 options). Impact is stored on User and in PollResponse for the kei_impact poll. |
| **KeiInterestStatement** | Kei-named; content is campaign-specific. Concept is generic; name could be generalized later. |

**User profile:** A single read/edit surface at GET/POST `/account` shows and updates User fields (email read-only, ZIP, newsletter toggle, "Your answers" from kei_status/kei_impact_slug/kei_personal_note). Campaign-specific columns are displayed as "your answers" and will be scoped by campaign when we add multi-campaign.

No schema change is required for “set the stage.” When adding a second campaign or a formal campaign entity, consider `campaign_id` + `interest_slug` (or a generic name) for User and for statement-style tables.

## Advocacy router: logic generic; copy and defaults from campaign config

**Generic (already swap-ready):**

- ZIP → district → your legislators + Power Broker (committee/topic-driven).
- Drawer flow: call script, email body, wrap-up, no-answer.
- Script/email building in `advocacy_helpers.py` takes `one_pager_points` and member/zip/chamber as arguments — no campaign name in the algorithm.

**Campaign-driven (from campaign config):**

- Hero headline/subhead (issue and advocacy page).
- Default policy topic (e.g. "Transportation") for Power Broker / category.
- Legislator brief PDF path and download URL.
- One-pager points (strategic bullets for scripts and email).

Campaign config is loaded from `config/campaign.json` (or path in `ILGA_CAMPAIGN_CONFIG`). Swapping campaign = replace or point at a different JSON file; no code change in the advocacy router.

**White-label keys (optional):** `poll_prompt_query` (query param for poll prompt; default `"kei"`), `welcome_email_intro`, `welcome_email_poll_link_text`, `strategic_mission`, `mission_attribution`, `error_page_facts` (list of `{text, image?, image_alt?, image_credit?}`), `error_page_fact_label`, `bill_status_urls` (for scrapers). See `docs/architecture/white-label-decoupling-gaps.md` for full list and remaining gaps.

## Provisioning a tenant (DB + campaign)

To run the app for a different campaign/tenant (e.g. Kei vs. Tenants Union):

1. **Set environment (or use provision script):**
   - `ILGA_DB_PATH` — path to the SQLite DB for this tenant (e.g. `data/kei_prod.db`, `data/tenants_union_prod.db`). Profile defaults: dev → `data/ilga_dev.db`, prod → `data/ilga.db`; override for multi-tenant.
   - `ILGA_CAMPAIGN_CONFIG` — (optional) path to this tenant's `campaign.json`. Default: `config/campaign.json` in repo root.
2. **Run migrations:** Use Alembic or the app startup (which runs `init_db()`). Same schema for all tenants.
3. **Optional:** Use `scripts/provision_tenant.py` to set `ILGA_DB_PATH` and optionally `ILGA_CAMPAIGN_CONFIG` from a tenant id and env (dev/prod), then run the app or migrations.

When `ILGA_DB_PATH` is not set, config derives it from `ILGA_TENANT` (e.g. `ILGA_TENANT=tenants_union` + prod → `data/tenants_union_prod.db`). See `config.py` and the provision script.

## Goal

One config artifact (and optionally env for its path) drives the advocacy “suit of armor.” Scrapers and core DB stay topic-agnostic so the same codebase can serve another campaign (e.g. “Zoning Laws”) by changing config and content, not application logic.
