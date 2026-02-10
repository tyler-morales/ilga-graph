# TODOS

**State of the system:** Modularity Roadmap Steps 1–5 complete. ETL lives in `etl.py`; API can start without scrapers when `ILGA_LOAD_ONLY=1`. Scorecards and Moneyball are cached to `cache/` and reused when member data is unchanged. GraphQL resolvers use request-scoped batch loaders (scorecard, moneyball profile, bill, member). `ILGA_PROFILE=dev|prod`; config; seating; SSR advocacy at `/advocacy`; Moneyball v2 (shell bill filter, institutional bonus, chair-first Power Broker).

---

## Current

- None — Steps 3–5 (ETL split, analytics cache, DataLoaders) are done; use **Next** when you pick up work again.

## Done (this session)

- **Full bill index + scrape-200:** Bill index previously only fetched one range page (100 SB + 100 HB). Implemented pagination in `scrape_bill_index()`: loop over range pages (0001-0100, 0101-0200, ...) until we have enough or a page is empty. `limit=0` means all pages (~9600+ bills). Added **make scrape-200** (200 SB + 200 HB, 2 range pages per type) to test pagination and **make scrape-full** (--sb-limit 0 --hb-limit 0) for full index. README and scrape.py docstring updated. Full scrape is slow (many index pages + one BillStatus request per bill).

- **Scrape → dev/prod pipeline:** Simplified to a clear two-step flow: (1) scrape data into cache (choose size), (2) serve via API. **make scrape** = full scrape for prod (all members, 300 SB + 300 HB). **make scrape-dev** = light scrape for dev (20/chamber, 100 SB + 100 HB, fast). **make dev** / **make dev-full** / **make run** now set **ILGA_LOAD_ONLY=1** so the server only loads from cache (no scraping on startup). Removed **make scrape-fast** in favor of **make scrape-dev**. README and scrape.py docstring updated; .env.example documents ILGA_LOAD_ONLY.

- **Evidence-based script hints & "How we pick these targets":** Script hints were hardcoded generic strings ("high effectiveness, deep network connections") that assumed the user already understood the system. A first-time visitor had no way to know *why* a legislator was chosen or what "effectiveness" meant. Changes: (1) Added `_build_script_hint_*` helpers in `main.py` that inject real numbers (laws passed X of Y, Z% passage rate, W% cross-party) into each card's script hint. Senator/Rep hints include ZIP and district. Power Broker hint explains Chair vs Moneyball selection with stats. Ally hint cites cross-party %. Super Ally merges both with evidence. (2) Template macro `member_card` now reads `member.script_hint` from the card dict instead of receiving a hardcoded 4th argument. (3) Added collapsible "How we pick these targets" section at the bottom of results: plain-language definitions of Senator/Rep (Census ZIP match), Power Broker (committee chair or highest Moneyball), Potential Ally (seatmate with highest cross-party rate), and Super Ally (merged). (4) Nested collapsible "How is the Moneyball score calculated?" with a table of the 6 components, weights, and what each measures. (5) Added CSS for `.how-it-works` collapsible and `.formula-table` in `base.html`. No algorithm changes — only explanation and evidence in the UI.

- **Scorecard / Moneyball clarity:** User wanted empirical stats (bills passed, vetoed, passage rate) front and center and derived metrics (Moneyball, effectiveness) clearly defined to avoid vagueness and bloat. (1) Added `src/ilga_graph/metrics_definitions.py` as single source of truth: empirical metric definitions (laws filed/passed, passage rate, magnet, bridge, pipeline, centrality, etc.) and Moneyball formula (one-liner + component list with weights). (2) Advocacy cards now show empirical first: "Laws passed: X of Y (Z% passage)", "Cross-party co-sponsorship: W%", then "Moneyball: N (composite rank 0–100)" with a "?" tooltip that shows the one-liner. (3) GraphQL query `metricsGlossary` returns full glossary (empirical, effectiveness_score, moneyball_one_liner, moneyball_components) so any client can render tooltips or docs. (4) README section "Metrics: empirical vs derived" documents the approach. No new metrics added; we only clarified and reordered display.

- **Advocacy full-data / seed-mode clarity:** Banner was always showing "Only 40 of 177 legislators" and "~14 demo ZIPs" even when full cache was loaded. Root cause: `make dev` sets `ILGA_SEED_MODE=1`, so (1) members load from cache when present (so user can have 177 after `make scrape`), but (2) ZIP crosswalk always used hardcoded seed (~14 ZIPs) when seed mode ON. Changes: (1) Advocacy banner now uses actual `member_count` and `zip_count` and only shows "Only N of 177" when N < 100; explains how to get full data (`make run` or `ILGA_SEED_MODE=0`). (2) Added `make dev-full` (dev profile + `ILGA_SEED_MODE=0`) to run with full cache + full Census ZCTA for testing. (3) Documented in README ("Testing with full data") and .env.example. To test advocacy with all members and all ZIPs: `make scrape` then `make dev-full` or `make run`.

---

## Next (when you're ready)

- Run `python scripts/scrape.py --fast --force-refresh` to populate `cache/bills.json` with the new format (if needed).
- When shifting to prod: set `ILGA_PROFILE=prod`, `ILGA_CORS_ORIGINS`, and optionally `ILGA_API_KEY`.

---

## Production (checklist for deployment)

- Pre-populate `cache/` (full or incremental scrape) so startup is load-only.
- Set `ILGA_PROFILE=prod` (turns off dev mode + seed mode automatically).
- Set `ILGA_CORS_ORIGINS` to your front-end origin(s) — prod warns if unset.
- Set `ILGA_API_KEY` to protect the GraphQL endpoint — prod warns if empty.
- Use `GET /health` for readiness (`ready` is true when members are loaded).

---

## Backlog / Future

- Full-text search for bill descriptions and member bios.
- (Done: full bill index via `make scrape-full` — pagination in bills.py; 0 = all pages.)
- Optionally derive vote/slip bill list from cache for broader coverage.
- Advocacy frontend: add member photos, richer script text, email links.
- Advocacy frontend: move embedded CSS to `static/style.css` when it grows.
- Advocacy frontend: htmx-powered "drill down" on each card (click to expand full member profile inline).
- Advocacy frontend: interactive map visualization — all senators plotted by ZIP/district, filterable by policy category. Click a district to see its advocacy targets.
- Full Census crosswalk download for prod (currently seed-mode hardcoded ZIPs in dev).

---

## Done (summary)

**Modularity Roadmap Steps 1-2 (config + resilience):** Created `src/ilga_graph/config.py` with `ILGA_PROFILE=dev|prod` (one-knob environment switching) + `.env` support via `python-dotenv`. Profile sets sensible defaults; individual vars override. Prod profile warns at startup if CORS or API_KEY are unconfigured. Updated `bills.py`, `votes.py`, `witness_slips.py`, `scraper.py`, and `main.py` to import from config (removed all hardcoded GaId/SessionId). Added try/except around every lifespan ETL step with stale-cache fallback. Added checkpoint saves (every 50 bills) in `scrape_all_bills`. Makefile uses `ILGA_PROFILE=dev`/`prod` instead of inline flags.

**Cache & seed:** Committee data in `cache/committees.json`; seed data in `mocks/dev/` with `ILGA_SEED_MODE=1` for instant dev; normalized `members.json` + `bills.json` (~70% size reduction); vote events and witness slips cached.

**ETL:** Composable steps — `load_or_scrape_data()`, `compute_analytics()`, `export_vault()`; CLI `scripts/scrape.py` can run each step or full pipeline.

**Resilience & performance:** HTTPAdapter with retries and connection pooling; stale cache warnings; startup timing logs and `.startup_timings.csv`; startup summary table with 7 numbered steps (load, analytics, seating, export, votes, slips, ZIP crosswalk), per-step times and details (counts, cache/seed hints); CSV includes seating_s, slips_s, zip_s, votes, slips, zctas; vote/slip cache eliminates re-scraping on every start.

**Witness slips & GraphQL:** Slips and votes share bill list (`ILGA_VOTE_BILL_URLS`); witness slip summary and paginated summaries; `billSlipAnalytics`, `memberSlipAlignment`; advancement analytics (`billAdvancementAnalyticsSummary`); docs in `graphql/README.md` and main README.

**Bills-first & incremental:** Bills from Legislation pages (single source of truth); bill index to BillStatus detail; `scrapers/bills.py` with `incremental_bill_scrape()`; member-bill linkage from `Bill.sponsor_ids`; cache persisted every run; exporter uses `all_bills` from `bills_lookup` so all cached bills get vault notes.

**Server & CI:** Single app in `main.py` (schema.py types only); ruff lint/format and per-file ignores for schema; tests updated for line-length and duplicate keys.

**Seating Chart / Whisper Network (Verdict 1c):** Added `seat_block_id`, `seat_ring`, `seatmate_names`, `seatmate_affinity` fields to `Member` dataclass. Created `src/ilga_graph/seating.py` with fuzzy name matching (bare last name, "Initial. Lastname", compound last names like "Glowiak Hilton"), the Aisle Rule (neighbors within same section only, no cross-aisle adjacency), and co-sponsorship affinity calculation (overlap % of member's bills with seatmate co-sponsors). Exposed all four fields via GraphQL `MemberType`. Integrated into lifespan (Step 2b) and `run_etl`. Senate-only for now; seating data loaded from `mocks/dev/senate_seats.json`.

**SSR Advocacy Frontend:** Added `jinja2` and `python-multipart` dependencies. Created `src/ilga_graph/zip_crosswalk.py` — Python port of the Google Sheets Census crosswalk. Seed-mode fallback has ~14 ZIPs (6 produce full 4-card results with the 40-member dev dataset). Jinja2 templates with htmx (v2.0.4 CDN) for in-page search. `GET /advocacy` renders the search form; `POST /advocacy/search` returns up to 4 cards: **Your Senator**, **Your Representative**, **Power Broker**, and **Potential Ally**. Each card has a "Why this target" box explaining the analytics reasoning and a role-specific script hint. **Super Ally merge:** when Power Broker and Ally resolve to the same person, they collapse into a single "Super Ally" card with both badges and a combined explanation. **Policy category filter:** optional dropdown maps 12 policy areas (Transportation, Education, etc.) to Senate committee codes; when selected, Power Broker and Ally are restricted to members on those committees (with graceful fallback if the filter eliminates all candidates). Committee rosters loaded from `state.committee_rosters`. Results split into "Your Legislators" and "Strategic Targets" sections.

**Steps 3–5 (ETL, analytics cache, DataLoaders):** **(Step 3)** Created `src/ilga_graph/etl.py` with `load_from_cache()`, `load_or_scrape_data()`, `compute_analytics()`, `export_vault()`, `run_etl()`, and `load_stale_cache_fallback()`. Added `ILGA_LOAD_ONLY`: when set, API startup only loads from cache (no scrape). Lifespan uses load-only path when `LOAD_ONLY=1`; scrape script uses etl and supports `--export-only` with `load_from_cache()`. **(Step 4)** Added `analytics_cache.py`: `save_analytics_cache()` / `load_analytics_cache()` persist scorecards and Moneyball to `cache/scorecards.json` and `cache/moneyball.json`. Staleness: cache is used only if newer than member data (`members.json` mtime). Lifespan and scrape script load when fresh, else compute and save. **(Step 5)** Added `loaders.py` with `ScorecardLoader`, `MoneyballProfileLoader`, `BillLoader`, `MemberLoader` (each with `load()` and `batch_load()`). GraphQL context_getter returns `create_loaders(state)`. Resolvers `member`, `members`, and `moneyball_leaderboard` use `info.context` and batch_load for scorecard/profile so list queries do one batched fetch instead of N lookups.

**Moneyball v2 Recalibration:** Three changes to stop undervaluing legislative leadership. **(1) Shell Bill Filter:** `is_shell_bill()` in `analytics.py` identifies procedural placeholders (description contains "Technical"/"Shell" or < 50 chars) and excludes them from the effectiveness denominator (`law_heat_score`), so leadership who file shell bills aren't penalised for "failures" that were never meant to pass. Applied in `compute_scorecard()`, `compute_all_scorecards()`, and `avg_pipeline_depth()`. **(2) Institutional Power Bonus:** Added `roles: list[str]` to `Member` (aggregates profile title + committee roster titles via `populate_member_roles()`). New `compute_institutional_weight()` returns 1.0 (President/Leader/Speaker), 0.5 (Chair/Spokesperson), or 0.25 (Whip/Caucus Chair). Added to `MoneyballProfile` as `institutional_weight` and to the composite score at 20% weight (existing weights scaled to 80%: effectiveness 24%, pipeline 16%, magnet 16%, bridge 12%, centrality 12%). **(3) Committee-Chair-First Power Broker:** `_find_power_broker()` now prioritises Committee Chairs of the relevant committee when a policy category is selected, falling back to highest Moneyball score only when no Chair is found. Serialization updated in `scraper.py`; test fixtures updated with realistic 50+ char descriptions.
