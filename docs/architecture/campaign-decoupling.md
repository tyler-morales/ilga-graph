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
| **User** | One campaign-specific column: `kei_status` (e.g. "registered", "would_want"). For white-label this would be campaign-scoped (e.g. `campaign_interest` or JSON per campaign). |
| **Campaign, Update, AuthCode** | Generic. |
| **KeiPollResponse** | Legacy table; new flow uses **Poll** + **PollResponse** (generic). KeiPollResponse kept for backfill/compat. |
| **KeiInterestStatement** | Kei-named; content is campaign-specific. Concept is generic; name could be generalized later. |

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
