# TODOS

## Done: Committee data caching

Committee data (index, rosters, bills) is now cached in a unified `cache/committees.json` file. Startup loads from cache when present, skipping the slow multi-minute scrape.

## Done: Seed data for instant local dev

Committed dev mock data lives in `mocks/dev/`. When `ILGA_SEED_MODE=1` (default in dev), the scraper falls back to mocks when `cache/` is missing. Run `make dev` for instant startup with no scraping.

## Done: Composable ETL

`run_etl()` is refactored into three independent steps: `load_or_scrape_data()`, `compute_analytics()`, and `export_vault()`. The CLI scripts (`scripts/scrape.py`) can call each step independently.

## Done: Scraper resilience

- HTTPAdapter with 3 retries and connection pooling
- Stale cache warnings (>7 days)
- Progress logging with elapsed time and ETA during scrapes

## Done: Startup performance tracking

- Detailed timing logs for each ETL step (load, analytics, export, votes)
- CSV logging to `.startup_timings.csv` for historical tracking
- Terminal output showing breakdown and total startup time
- `PERFORMANCE.md` documenting bottlenecks and optimization opportunities

## Done: Vote caching & clean startup UI

- Roll-call votes are now cached in `cache/vote_events.json` (no re-scraping on every startup)
- Replaced verbose per-PDF logs with clean color-coded summary table
- Startup summary shows step-by-step breakdown with emoji icons and ANSI colors
- Vote scraping bottleneck eliminated (0.1s from cache vs 8s live scraping)

## Done: Normalized data architecture

- Cache restructured from denormalized (embedded bills per member) to normalized (`members.json` + `bills.json`)
- ~70% reduction in cache size (12.7 MB → ~3.5 MB) by storing each bill once
- Members use disjoint `sponsored_bill_ids` / `co_sponsor_bill_ids` (no overlap)
- `hydrate_members()` resolves IDs to full Bill objects on load
- Legacy per-chamber cache and backward-compat fallbacks removed
- Seed data generation updated to produce normalized format
- GraphQL API exposes `sponsored_bills` and `co_sponsor_bills` on `MemberType`

## Done: Witness slip pipeline

- Witness slips use the same bill list as votes (`DEFAULT_BILL_STATUS_URLS` / `ILGA_VOTE_BILL_URLS`); one source of truth.
- `scrape_all_witness_slips()` in lifespan and `scripts/scrape.py`; cache `cache/witness_slips.json`; seed copy in `generate_seed.py`.
- Amendment slips normalised to parent bill number so `witnessSlips(billNumber)` returns all slips for that bill.
- HB0576 and HB0034 added to defaults; real slip data in cache and mocks (e.g. HB0034: 3,567 slips, SB0008: 7,901).
- GraphQL: `votes(billNumber)` and `witnessSlips(billNumber, limit, offset)` for per-bill votes and slips.

## Done: Witness slip summary (GraphQL)

- `WitnessSlipSummaryType`: `billNumber`, `totalCount`, `proponentCount`, `opponentCount`, `noPositionCount`.
- `witnessSlipSummary(billNumber: String!)`: per-bill counts, no paging.
- `witnessSlipSummaries(limit, offset)`: all bills that have slips, sorted by slip volume descending; paginated. Enables “bills by slip volume” / “high-opposition bills” in one query.

## Done: Bill / member slip analytics (GraphQL)

- `billSlipAnalytics(billNumber: String!)`: returns `BillSlipAnalyticsType` with `controversyScore` (0–1, from `controversial_score()`). Null when the bill has no slips.
- `memberSlipAlignment(memberName: String!)`: returns list of `LobbyistAlignmentEntryType` (`organization`, `proponentCount`) for orgs that file as proponents on that member’s sponsored bills (from `lobbyist_alignment()`). Empty when member not found or no alignment.

## Done: GraphQL query docs

- `graphql/README.md`: Lists query files, documents the recommended “bill + votes + slips” query, and clarifies that `votes` returns a list and `witnessSlips` returns a connection. Also notes `witnessSlipSummaries`, `billSlipAnalytics`, and `memberSlipAlignment`.
- Main `README.md`: Under “Example GraphQL Queries”, added a short note pointing to `graphql/bill_with_votes_and_slips.graphql` and `graphql/README.md`.

## Future

- **Witness slip analytics (volume vs advancement)**: Compare slip volume and position ratio to whether the bill advanced. Expose via GraphQL (e.g. list of “high-volume / stalled” vs “high-volume / passed” bills; ratio field on summary).
- **Incremental scraping**: Only re-scrape members/bills that have changed since last run (use `last_action_date` comparison).
- **Full-text search**: Add a search endpoint to the GraphQL API for bill descriptions and member bios.
- **Analytics caching**: Cache scorecards/moneyball results alongside member data to skip recomputation on startup.

---

## Next steps (lowest-hanging fruit)

- (None listed — doc/query task done.)

## Done: Fix server startup (PR)

- Removed duplicate app block from `schema.py` that referenced undefined names (`GraphQLRouter`, `FastAPI`, `lifespan`, etc.). The app and GraphQL router live only in `main.py`; `schema.py` now only defines types and the Strawberry `schema` object. `make dev` runs without `NameError`.

## Done: PR consistency review (Open Claw / witness-slip-advancement-analytics)

- **Schema vs main:** Confirmed the running app uses `main.py`’s Query and schema only; `schema.py` is the source of types (and a parallel Query/schema that is not mounted). No consumers import `app` or `graphql_app` from schema.
- **Advancement analytics:** `compute_advancement_analytics()` in `analytics.py` returns `high_volume_stalled` / `high_volume_passed`; `BillAdvancementAnalyticsType` and both Query implementations (main + schema) match. `Bill.last_action`, `pipeline_depth`, `classify_bill_status`, `_normalise_bill_number` are all present and used correctly.
- **Type hints in schema.py:** Replaced `Member` / `WitnessSlip` with `MemberModel` / `WitnessSlipModel` in `_sort_key` and `_witness_slip_summary_for_slips` so type hints match the actual imports (avoids undefined-name issues for type checkers).
- **Docs:** Documented `billAdvancementAnalyticsSummary` in `graphql/README.md`.

## Done: Fix GitHub CI lint (ruff)

- **pyproject.toml:** Added `[tool.ruff.lint.per-file-ignores]` so `schema.py` ignores F821 (undefined names `state`, helpers) that are injected by `main` at runtime.
- **Ruff check:** Fixed E501 (line length), F401 (unused imports), F841 (unused variables), W291/W293 (whitespace), F601 (duplicate dict key). Ran `ruff check --fix` and `ruff format`; shortened long lines in exporter, main, schema, analytics, scrapers, tests.
- **Tests:** Shortened SAMPLE_EXPORT / SAMPLE_WITNESS_SLIPS_PAGE in test_witness_slips.py to meet line-length; fixed test_isolate duplicate key in test_moneyball.py; updated assertions for shortened org names.
