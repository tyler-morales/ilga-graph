# TODOS

**State of the system:** Modularity Roadmap Steps 1-2 complete + Seating Chart / Whisper Network (Verdict 1c) + SSR Advocacy Frontend + **Moneyball v2 Recalibration**. `ILGA_PROFILE=dev|prod` gives one-knob environment switching; `config.py` centralizes all settings with `.env` support; scrapers are parameterized; lifespan has full try/except resilience with stale-cache fallback. Seating chart analytics populate `seat_block_id`, `seat_ring`, `seatmate_names`, and `seatmate_affinity` on Senate members. SSR frontend at `/advocacy` provides ZIP-to-district lookup with Senator/Broker/Ally targeting via Jinja2 templates. Moneyball algorithm now filters shell bills from effectiveness denominator, adds institutional-power bonus (20% of composite score), and Power Broker selection prioritises Committee Chairs over raw score.

---

## Current

- None — Moneyball v2 Recalibration is done; use **Next** when you pick up work again.

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

**Moneyball v2 Recalibration:** Three changes to stop undervaluing legislative leadership. **(1) Shell Bill Filter:** `is_shell_bill()` in `analytics.py` identifies procedural placeholders (description contains "Technical"/"Shell" or < 50 chars) and excludes them from the effectiveness denominator (`law_heat_score`), so leadership who file shell bills aren't penalised for "failures" that were never meant to pass. Applied in `compute_scorecard()`, `compute_all_scorecards()`, and `avg_pipeline_depth()`. **(2) Institutional Power Bonus:** Added `roles: list[str]` to `Member` (aggregates profile title + committee roster titles via `populate_member_roles()`). New `compute_institutional_weight()` returns 1.0 (President/Leader/Speaker), 0.5 (Chair/Spokesperson), or 0.25 (Whip/Caucus Chair). Added to `MoneyballProfile` as `institutional_weight` and to the composite score at 20% weight (existing weights scaled to 80%: effectiveness 24%, pipeline 16%, magnet 16%, bridge 12%, centrality 12%). **(3) Committee-Chair-First Power Broker:** `_find_power_broker()` now prioritises Committee Chairs of the relevant committee when a policy category is selected, falling back to highest Moneyball score only when no Chair is found. Serialization updated in `scraper.py`; test fixtures updated with realistic 50+ char descriptions.
