# TODOS

**State of the system:** Modularity Roadmap Steps 1-2 complete. `ILGA_PROFILE=dev|prod` gives one-knob environment switching; `config.py` centralizes all settings with `.env` support; scrapers are parameterized; lifespan has full try/except resilience with stale-cache fallback. 257/257 tests pass.

---

## Current

- None — Steps 1-2 of Modularity Roadmap are done; use **Next** when you pick up work again.

---

## Next (when you're ready)

- **Step 3 – Separate Collector / Thinker / Publisher:** Extract ETL orchestration from `main.py` into its own module so the API can start without running scrapers.
- **Step 4 – Analytics caching:** Persist scorecards/moneyball to disk so we skip recomputation on startup when data hasn't changed.
- **Step 5 – DataLoader / lazy resolution:** Replace linear scans in GraphQL resolvers with DataLoader-style batching for N+1 efficiency.
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
- Full bill index scrape (~9,600 bills, all range pages).
- Optionally derive vote/slip bill list from cache for broader coverage.

---

## Done (summary)

**Modularity Roadmap Steps 1-2 (config + resilience):** Created `src/ilga_graph/config.py` with `ILGA_PROFILE=dev|prod` (one-knob environment switching) + `.env` support via `python-dotenv`. Profile sets sensible defaults; individual vars override. Prod profile warns at startup if CORS or API_KEY are unconfigured. Updated `bills.py`, `votes.py`, `witness_slips.py`, `scraper.py`, and `main.py` to import from config (removed all hardcoded GaId/SessionId). Added try/except around every lifespan ETL step with stale-cache fallback. Added checkpoint saves (every 50 bills) in `scrape_all_bills`. Makefile uses `ILGA_PROFILE=dev`/`prod` instead of inline flags.

**Cache & seed:** Committee data in `cache/committees.json`; seed data in `mocks/dev/` with `ILGA_SEED_MODE=1` for instant dev; normalized `members.json` + `bills.json` (~70% size reduction); vote events and witness slips cached.

**ETL:** Composable steps — `load_or_scrape_data()`, `compute_analytics()`, `export_vault()`; CLI `scripts/scrape.py` can run each step or full pipeline.

**Resilience & performance:** HTTPAdapter with retries and connection pooling; stale cache warnings; startup timing logs and `.startup_timings.csv`; clean startup summary table; vote/slip cache eliminates re-scraping on every start.

**Witness slips & GraphQL:** Slips and votes share bill list (`ILGA_VOTE_BILL_URLS`); witness slip summary and paginated summaries; `billSlipAnalytics`, `memberSlipAlignment`; advancement analytics (`billAdvancementAnalyticsSummary`); docs in `graphql/README.md` and main README.

**Bills-first & incremental:** Bills from Legislation pages (single source of truth); bill index to BillStatus detail; `scrapers/bills.py` with `incremental_bill_scrape()`; member-bill linkage from `Bill.sponsor_ids`; cache persisted every run; exporter uses `all_bills` from `bills_lookup` so all cached bills get vault notes.

**Server & CI:** Single app in `main.py` (schema.py types only); ruff lint/format and per-file ignores for schema; tests updated for line-length and duplicate keys.
