# TODOS

**State of the system:** Bills-first pipeline and incremental scraping are done. Core data (members, committees, bills, votes, witness slips) is scraped and cached; GraphQL API, Obsidian export, and CI are in place. Ready to level-set for production when you are.

---

## Current

- None — system is in a good spot; use **Next** when you pick up work again.

---

## Next (when you’re ready)

- Run `python scripts/scrape.py --fast --force-refresh` to populate `cache/bills.json` with the new format (if needed).
- Run `python scripts/scrape.py --incremental --fast` on subsequent runs to exercise incremental mode.
- When shifting to prod: pre-populate cache (e.g. scheduled scrape), run app with `ILGA_SEED_MODE=0` and `ILGA_DEV_MODE=0`, restrict `ILGA_CORS_ORIGINS`, and optionally set `ILGA_API_KEY`.

---

## Production (checklist for deployment)

- Pre-populate `cache/` (full or incremental scrape) so startup is load-only.
- Set `ILGA_SEED_MODE=0`, `ILGA_DEV_MODE=0`.
- Set `ILGA_CORS_ORIGINS` to your front-end origin(s).
- Optionally set `ILGA_API_KEY` to protect the GraphQL endpoint.
- Use `GET /health` for readiness (`ready` is true when members are loaded).

---

## Backlog / Future

- Full-text search for bill descriptions and member bios.
- Analytics caching (scorecards/moneyball) to skip recomputation on startup.
- Full bill index scrape (~9,600 bills, all range pages).
- Optionally derive vote/slip bill list from cache for broader coverage.

---

## Done (summary)

**Cache & seed:** Committee data in `cache/committees.json`; seed data in `mocks/dev/` with `ILGA_SEED_MODE=1` for instant dev; normalized `members.json` + `bills.json` (~70% size reduction); vote events and witness slips cached.

**ETL:** Composable steps — `load_or_scrape_data()`, `compute_analytics()`, `export_vault()`; CLI `scripts/scrape.py` can run each step or full pipeline.

**Resilience & performance:** HTTPAdapter with retries and connection pooling; stale cache warnings; startup timing logs and `.startup_timings.csv`; clean startup summary table; vote/slip cache eliminates re-scraping on every start.

**Witness slips & GraphQL:** Slips and votes share bill list (`ILGA_VOTE_BILL_URLS`); witness slip summary and paginated summaries; `billSlipAnalytics`, `memberSlipAlignment`; advancement analytics (`billAdvancementAnalyticsSummary`); docs in `graphql/README.md` and main README.

**Bills-first & incremental:** Bills from Legislation pages (single source of truth); bill index → BillStatus detail; `scrapers/bills.py` with `incremental_bill_scrape()`; member–bill linkage from `Bill.sponsor_ids`; cache persisted every run; exporter uses `all_bills` from `bills_lookup` so all cached bills get vault notes.

**Server & CI:** Single app in `main.py` (schema.py types only); ruff lint/format and per-file ignores for schema; tests updated for line-length and duplicate keys.
