# TODOS

**State of the system:** Modularity Roadmap Steps 1–5 complete. ETL lives in `etl.py`; API can start without scrapers when `ILGA_LOAD_ONLY=1`. Scorecards and Moneyball are cached to `cache/` and reused when member data is unchanged. GraphQL resolvers use request-scoped batch loaders (scorecard, moneyball profile, bill, member). `ILGA_PROFILE=dev|prod`; config; seating; SSR advocacy at `/advocacy`; Moneyball v2 (shell bill filter, institutional bonus, chair-first Power Broker). **Unified GraphQL `search` query** — free-text search across members, bills, and committees with relevance scoring, entity-type filtering, and pagination (`search.py`). **Committee Power Dashboard** — each advocacy card now shows committee assignments with leadership roles, power explanations, and per-committee bill advancement stats. **Institutional Power Badges** — visual hierarchy badges (LEADERSHIP, COMMITTEE CHAIR, TOP 5% INFLUENCE) at the top of each advocacy card with click-to-expand explanations. **Power Card Redesign** — advocacy cards restructured into a consolidated "Power Card" layout matching advocacy professional needs: badge row, inline name with party/district, "Why this person matters" power context box, compact scorecard with caucus comparison, contact row (phone/email/ILGA profile), and expandable detail sections.

---

## Current

- **Power Card Redesign (advocacy card layout overhaul):**
  - **Card layout restructured** to match advocacy professional mockup. Old layout had data scattered across separate expandable sections. New layout consolidates the key "power signals" into a single up-front narrative box.
  - **New card structure (top to bottom):**
    1. **Badge row:** Role label (Your Senator / Power Broker / etc.) + power badges (LEADERSHIP, COMMITTEE CHAIR, TOP 5% INFLUENCE) side by side.
    2. **Name line:** `"Senator Jane Smith (R-Senate District 47)"` — inline party abbreviation and district.
    3. **"Why this person matters" box** (blue-bordered callout) with bulleted power signals:
       - **Committee chair positions** — "CHAIR: Transportation — Decides which bills get hearings — Advanced 29 of 51 bills (57% success rate) - 29 became law"
       - **Ranking** — "#32 of 60 senators" with Top N% highlight when in top 10%
       - **Network** — "Co-sponsors with 174 legislators — 56 Republicans, 118 Democrats — high bipartisan reach" + co-sponsor pull multiplier + co-sponsor passage multiplier
       - **Track record** — "1108 YES / 0 NO across 1144 votes — Votes with party 99.1% — breaks ranks 8x (potentially persuadable)"
       - **Workload** — "Currently managing N active bills"
    4. **Compact scorecard** (always visible, not collapsible): Laws passed with caucus comparison ("2.0x caucus average"), cross-party %, co-sponsor pull with chamber average, Moneyball with rank.
    5. **Contact row:** Phone / Email / ILGA Profile — all inline with icons.
    6. **Why this target + Script Hint** — existing evidence-based content.
    7. **Expandable details:** Voting Record, Committee Assignments, Full Legislative Scorecard (moved to bottom, still collapsible).
  - **`_member_to_card()` (main.py):** Added `rank_chamber`, `chamber_size`, `rank_percentile`, `party_abbr`, `email`, `role`, `active_bills` to card dict. Active bill count computed by checking `last_action` for non-terminal statuses.
  - **`moneyball.py`:** Added `passage_rate_vs_caucus` and `caucus_avg_passage_rate` fields to `MoneyballProfile`. New computation in `compute_moneyball()` calculates party+chamber average passage rate (caucus avg) and each member's multiplier against it. E.g., "2.0x caucus average" means this member passes bills at twice the rate of their party peers.
  - **`analytics_cache.py`:** Updated serialization to include new MoneyballProfile fields.
  - **`_results_partial.html`:** Complete rewrite of the `member_card` macro. Influence network section and old `<dl>` stats section removed from inline — data now consolidated into the power context box and compact scorecard.
  - **`base.html`:** Added CSS for `.card-badge-row` (flex row), `.card-name-meta` (inline party/district), `.power-context` (blue-bordered box with `.power-context-title`, `.power-context-list`, `.power-signal`, `.power-sub`, `.power-highlight`), `.compact-scorecard` (beige bordered box), `.compact-stats`, `.caucus-compare` (green bold for multipliers), `.contact-row` (flex row with icon prefixes for phone/email/profile).

- **Voting Pattern History (Track Record) on advocacy cards:**
  - **`voting_record.py` (new module):** Created `MemberVoteRecord` dataclass (bill_number, bill_description, date, vote=YES/NO/PRESENT/NV, bill_status, vote_type) and `VotingSummary` dataclass (total_floor_votes, yes/no/present/nv counts, yes_rate_pct, party_alignment_pct, party_defection_count, records list). Core functions: `build_member_vote_index()` — iterates all vote events and builds a reverse index (member_name -> VotingSummary) with per-member vote records sorted by date descending; `_compute_all_party_alignment()` — single-pass computation across all floor votes determining each party's majority direction per event then scoring each member's alignment (ties are skipped; PRESENT/NV counted as defections); `build_all_category_bill_sets()` — precomputes bill_number sets for each policy category from committee-bill mappings; `filter_summary_by_category()` — returns a new VotingSummary filtered to only category-relevant bills with recalculated counts.
  - **`main.py` (AppState + lifespan):** Added `state.member_vote_records` (dict[str, VotingSummary]) and `state.category_bill_sets` (dict[str, set[str]]) fields. New Step 4b in lifespan computes both after vote normalization. `_member_to_card()` now accepts optional `category` parameter; when provided, filters voting record to category bills. Returns `voting_record` dict with: records (up to 10 most recent), yes/no/present/nv counts, yes_rate_pct, party_alignment_pct, party_defection_count, is_persuadable flag, and category_label. All `_member_to_card()` calls in `advocacy_search()` pass the selected category.
  - **`_results_partial.html`:** New collapsible "Voting Record" `<details>` section between Legislative Network and Committee Assignments. Shows category-qualified title when filtered (e.g., "Voting Record: Transportation Bills (5 floor votes)"). Each vote row displays: bill number (bold), truncated description, color-coded vote indicator (YES=green, NO=red, PRESENT=amber, NV=gray), bill outcome. Summary bar shows YES/NO pattern with yes rate. Party alignment line when data available. "Potentially persuadable" callout (amber left-border box) when party_defection_count > 0 — the key advocacy signal.
  - **`base.html`:** Added CSS for `.card-voting-record` (collapsible, matching existing pattern), `.vote-list`, `.vote-row` (flexbox layout), `.vote-bill`, `.vote-desc`, `.vote-indicator` with `.vote-yes`/`.vote-no`/`.vote-present`/`.vote-nv` colour variants, `.vote-outcome` with status-based colours, `.vote-summary`, `.vote-pattern`, `.persuadable-callout` (amber left-border on cream background).
  - **Why it matters:** Tells advocacy groups: (1) Is this person persuadable? (votes against party sometimes = yes), (2) Do they support our issue? (voted YES on X of Y related bills). Section only renders when vote data exists, so no empty sections.

- **Institutional Power Badges (visual hierarchy):**
  - **`moneyball.py`:** Added `PowerBadge` dataclass (`label`, `icon`, `explanation`, `css_class`) and `compute_power_badges()` function. Three additive badge types: (1) **LEADERSHIP** — awarded when `institutional_weight >= 0.25`; explanation dynamically built from `member.role` with tier-specific context (top chamber leader / committee leader / party management). (2) **COMMITTEE CHAIR** — awarded for each committee where member is Chair (not Vice-Chair); explanation names the committee(s) and explains gatekeeper power. Multiple chairs consolidated into one badge. (3) **TOP N% INFLUENCE** — awarded when `rank_chamber` is in top 5% of their chamber (`ceil(chamber_size * 0.05)`); explanation shows exact rank and chamber size (e.g., "Ranked #3 of 59 senators").
  - **`main.py` (_member_to_card):** Computes chamber size from `state.moneyball.rankings_house` / `rankings_senate` lengths. Calls `compute_power_badges(mb, committee_roles, chamber_size)` and serialises results as `power_badges` list of dicts in each card. Imported `compute_power_badges` from moneyball module.
  - **`_results_partial.html`:** New `power-badges` div at the top of each card macro, rendered *above* existing role/achievement badges. Each badge uses native `<details>/<summary>` for click-to-expand explanations — no JavaScript required. Icons via CSS `::before` pseudo-elements with Unicode text presentation selectors.
  - **`base.html`:** Added CSS for `.power-badges` (flex row), `.power-badge-wrapper` (inline details), `.power-badge` (summary styled as badge with marker removal for all browsers), three colour variants (`.power-badge-leadership` navy, `.power-badge-chair` gold-brown, `.power-badge-influence` red), icon prefix classes (`.power-icon-leadership/chair/influence` using Unicode shield/lightning/fire), `.power-badge-explanation` (tooltip-like box below badge on expand). Hover state with brightness filter for clickability affordance.

- **Influence Network Visualization (human-readable power picture):**
  - **`moneyball.py` (MoneyballProfile + compute_moneyball):** Added 7 new fields to `MoneyballProfile`: `collaborator_republicans`, `collaborator_democrats`, `collaborator_other` (party breakdown of co-sponsorship peers), `magnet_vs_chamber` (member magnet / chamber avg magnet multiplier, e.g. "1.2x"), `cosponsor_passage_rate` (passage rate of bills this member co-sponsors), `cosponsor_passage_multiplier` (vs **chamber median co-sponsor rate**, not raw chamber passage rate -- avoids selection bias where popular/likely-to-pass bills naturally attract more co-sponsors), `chamber_median_cosponsor_rate` (the baseline, shown in parenthetical for transparency). New Step 3b in `compute_moneyball()` computes all three metrics: iterates adjacency peers for party breakdown, computes per-chamber magnet averages, and calculates co-sponsored bill passage rates vs peer-normalised baselines.
  - **`analytics_cache.py`:** Updated `_profile_to_dict()` to serialize new fields. Deserialization (`MoneyballProfile(**d)`) handles old caches gracefully via defaults.
  - **`main.py` (_member_to_card):** Added `influence_network` dict to card data with all computed metrics. Bipartisan label computed from party balance of network peers but NOT displayed on the card (removed to avoid confusion with the cross-party support metric which measures their own bills). Included in return dict for template access.
  - **`_results_partial.html`:** New "Legislative Network" section between summary stats and Committee Assignments. Three plain-language lines (each guarded by `{% if %}`): (1) "Shares bills with **N legislators** (X Republicans, Y Democrats)", (2) "Their own bills attract **N co-sponsors** on average (Nx the chamber average)", (3) "Bills they join as co-sponsor pass at **Nx** the typical rate (X% vs Y% chamber median)" (only shown when multiplier > 1.0). Wording carefully distinguishes "their own bills" vs "bills they join" to avoid confusion with the top-level cross-party support metric. All numbers shown with transparent baselines — no mystery multipliers. Renamed top-level "Cross-party co-sponsorship" to "Cross-party support: X% of their bills have opposite-party co-sponsors" for clarity. Also replaced "Network centrality: 0.342" in the Moneyball Composite table with "Network reach: N collaborators (0.342)" for contextualised display.
  - **`base.html`:** Added CSS for `.influence-network` (green-left-border box on light green background), `.influence-title`, `.influence-list`, `.influence-detail`. Consistent with existing card section styling.

- **Committee Power Dashboard on advocacy cards:**
  - **`analytics.py`:** Added `CommitteeStats` dataclass (total_bills, advanced_count, passed_count, advancement_rate, pass_rate) and `compute_committee_stats()` function that classifies each committee's bills by pipeline stage and status. Added `build_member_committee_roles()` which builds a reverse index (member_id -> list of committee assignment dicts) with role, leadership flag, committee name, and per-committee bill advancement stats. Roles sorted: Chair > Vice-Chair > Minority Spokesperson > Member.
  - **`main.py` (AppState + lifespan):** Added `state.committee_stats` and `state.member_committee_roles` fields. Computed at startup (Step 3b) after committee data loads. `_member_to_card()` now includes a `committee_roles` list in each card dict, populated from the pre-built reverse index.
  - **`_results_partial.html`:** New collapsible "Committee Assignments (N)" `<details>` block on each card, positioned between summary stats and Legislative Scorecard. Leadership positions (Chair, Vice-Chair, Minority Spokesperson) shown prominently with role badge, plain-language power explanation ("Controls bill hearings and scheduling"), and bill advancement stats ("Advanced 8 of 12 bills (67%) - 3 became law"). Regular memberships listed compactly as comma-separated names.
  - **`base.html`:** Added CSS for `.card-committees`, `.committees-body`, `.committee-row`, `.committee-role-badge` (color-coded: Chair = brown, Vice-Chair = olive, Spokesperson = slate, Member = gray), `.committee-power-note`, `.committee-stats`, `.committee-member-list`. Follows existing collapsible pattern (`.card-scorecard`).
  - **Fix: committee stats showed "0 of N advanced" for all committees.** Two root causes: (1) **Bill number format mismatch** — committee pages list bills without leading zeros (`SB79`) and sometimes with amendment suffixes (`SB228 (SCA1)`), while the bill cache uses zero-padded numbers (`SB0079`). Added `_normalise_committee_bill_number()` to strip suffixes and zero-pad to 4 digits. This fixed 26% of bill lookups (2,658 of 10,105). (2) **ILGA committee bills page only shows currently-pending bills** — bills that already passed through committee are no longer listed. This meant we were only seeing bills stuck at depth 1 ("Assigned to X"). Fix: Added `_build_full_committee_bills()` which scans every bill's `action_history` for "Assigned to"/"Referred to" actions and matches committee names (with fuzzy matching for "X Committee" suffixes and abbreviation variants). This merges historical assignments with the ILGA page data, giving true throughput. Result: Transportation committee went from "0 of 11 (0%)" to "29 of 51 advanced (57%); 29 became law". Overall: 15,182 bills tracked across all committees, 1,063 advanced, 1,057 became law.

- **Scorecard UI added to advocacy cards:**
  - **`_member_to_card()` (main.py):** Now looks up `state.scorecards.get(member.id)` and attaches a `scorecard` dict with template-friendly fields: Lawmaking (laws_filed, laws_passed, law_pass_rate_pct, magnet_score, bridge_pct), Resolutions (resolutions_filed, resolutions_passed, resolution_pass_rate_pct), Overall (total_bills, total_passed, overall_pass_rate_pct, vetoed_count, stuck_count, in_progress_count). Only included when scorecard data exists and the member has at least one bill.
  - **`_results_partial.html`:** Each advocacy card now has a collapsible `<details class="card-scorecard">` block ("Legislative Scorecard") between the summary stats and "Why this target". Three-section table: Lawmaking (HB/SB), Resolutions (HR/SR/HJR/SJR), Overall. Vetoed/stuck/in-progress rows shown only when non-zero. Tooltips on Magnet and Bridge labels.
  - **`base.html`:** Container widened from `max-width: 760px` to `920px` so the main content column (~600px) has enough room for the scorecard table without label wrapping. Added `.card-scorecard`, `.scorecard-body`, `.scorecard-table`, and `.scorecard-section-head` styles (consistent with existing `.how-it-works` / `.formula-table` look).
  - **Width note:** If scorecard tables still wrap on narrow viewports, increase `.container` max-width further or reduce aside width. The 920px value gives ~600px to `.results-main`.

- **Scorecard: Magnet/Bridge definitions, Moneyball breakdown, wider layout:**
  - Magnet and Bridge rows in the scorecard table now have inline definitions: Magnet = "avg co-sponsors per bill -- higher means the legislator attracts more support"; Bridge = "% of bills with at least one cross-party co-sponsor -- measures bipartisan reach".
  - New "Moneyball Composite" section in the scorecard table shows the individual component scores with weights: Passage rate (24%), Pipeline depth (16%), Co-sponsor pull (16%), Cross-party rate (12%), Network centrality (12%), Institutional role (20%). Data passed via new `moneyball` dict in `_member_to_card()`.
  - Container max-width increased from 920px to 1120px for a wider content area.

- **GraphQL unified search query (`search`):**
  - Created `src/ilga_graph/search.py` — in-memory search engine with tiered relevance scoring (exact ID > exact name > prefix > contains name > contains description > secondary fields). Searches Members (name, id, role, party, district, chamber, committees, bio_text), Bills (bill_number, description, synopsis, primary_sponsor, last_action), and Committees (code, name). Returns `SearchHit` dataclasses with `entity_type`, `match_field`, `match_snippet`, `relevance_score`, and the underlying model. No new dependencies.
  - Added to `schema.py`: `SearchEntityType` enum (MEMBER, BILL, COMMITTEE), `SearchResultType` (entity_type, match_field, match_snippet, relevance_score, optional member/bill/committee), `SearchConnection` (items + page_info).
  - New `search(query, entityTypes, offset, limit)` resolver on Query class. Accepts free-text query, optional entity-type filter, and pagination. Member results include scorecard + moneyball via batch loaders.
  - Future: fuzzy matching (Levenshtein), search indexing at startup, vote events and witness slips as searchable entities, autocomplete endpoint.

## Done (this session)

- **Smart tiered index scanning (`scrapers/bills.py`):**
  - **Problem:** Every `make scrape` walked all 125 ILGA index pages (~30 min) even when nothing changed. Wasteful for daily runs.
  - **Tiered strategy:** `incremental_bill_scrape()` now auto-decides scan depth based on `scrape_metadata.json` timestamps:
    - **<24h since last scan → SKIP** index entirely. Only re-scrapes bills with recent activity (from cached `last_action_date`). Takes ~seconds.
    - **<7 days since last full scan → TAIL-ONLY** scan. New `tail_only_bill_indexes()` fetches the ILGA `/Legislation` page once to discover doc types, then only checks range pages at or beyond the highest cached bill number per doc type. E.g., if highest SB is 4052, only page 41 of 41 is fetched (not pages 1-40). Takes ~1-2 min vs 30 min.
    - **>7 days or first run → FULL** scan. All 125 pages as before.
    - **`FULL=1` override:** `make scrape FULL=1` forces a full walk regardless of timestamps.
  - **Metadata tracking:** `scrape_metadata.json` now stores `last_full_scan`, `last_tail_scan`, and `highest_bill_per_type` (e.g., `{"SB": 4052, "HB": 5623, "SR": 614}`). Updated on every scrape. Full scans update both timestamps; tail scans update only `last_tail_scan`.
  - **Helper functions:** `_hours_since()` (timestamp age), `_extract_highest_bill_numbers()` (scan cached bills for max number per doc type), `_bill_number_to_int()` (extract numeric suffix), `tail_only_bill_indexes()` (the tail scan implementation).
  - **Re-scrape logic improved:** Recently active bills (last 30 days) are now re-scraped regardless of scan type — even on "skip" runs. Build entries from cached `Bill.status_url` so no index walk is needed.

- **Pipeline resilience + Makefile overhaul:**
  - **Data loss fix (`scrapers/bills.py`):** `incremental_bill_scrape()` and `scrape_all_bills()` now preserve existing `vote_events` and `witness_slips` when re-scraping bills. Previously, re-running `make scrape` would create fresh `Bill` objects with empty arrays and overwrite the cached bill — destroying hours of vote/slip scraping work. Fix: both functions now carry forward `old.vote_events` and `old.witness_slips` from the existing cached bill before overwriting.
  - **Unified pipeline (`scripts/scrape.py`):** Merged the separate `scripts/scrape.py` (members+bills) and `scripts/scrape_votes.py` (votes+slips) into a single pipeline with 4 phases:
    1. **Phase 1+2:** Members + Bill index + Bill details (incremental, preserves vote/slip data)
    2. **Phase 3:** Votes + witness slips for bills that need them (delegates to `scrape_votes.py` as subprocess)
    3. **Phase 4:** Analytics + Obsidian vault export (optional, `--export`)
  - **Makefile simplified:** Replaced 15+ targets with 8 clear ones:
    - `make scrape` — Smart tiered scan (daily ~2 min, auto-decides). Env vars: `FULL=1` (force full walk), `FRESH=1` (nuke cache), `LIMIT=100` (vote/slip bill limit), `WORKERS=10` (parallel workers), `SKIP_VOTES=1` (phases 1-2 only), `EXPORT=1` (include vault export).
    - `make dev` — Serve from cache (dev mode, auto-reload)
    - `make serve` — Serve from cache (prod mode)
    - `make install` — Install project with dev deps
    - `make test` — Run pytest
    - `make lint` / `make lint-fix` — Ruff check + format
    - `make clean` — Remove cache/ and generated files
  - **No backwards compatibility needed.** Old targets are gone. One command for all data: `make scrape`.

- **Vote PDF tally mismatch fix (`scrapers/votes.py`):**
  - **Root cause:** Single-letter middle initials (E, P, N) in committee vote PDFs were being parsed as vote codes (Excused, Present, Nay). Example: `Y Hastings, Michael E Y Martwick, Robert F` → "E" treated as Excused, so Martwick's Y vote was lost. Always manifested as "expected N yeas, got N-1".
  - **Fix:** Extended the `SMART_A` disambiguation pattern to all single-letter codes that can be middle initials: E, P, N, A. Each code uses a negative lookahead — it's only treated as a vote code when NOT immediately followed by another vote code (e.g., `E(?!\s+(?:NV|Y|N|P|A)\s)`). This means "Michael E Y Martwick" → E followed by Y → middle initial, name becomes "Hastings, Michael E". But "E Smith Y Jones" → E followed by "Smith" → real Excused vote.
  - **PDF artifact fix:** Added post-processing to handle `Y NV Mayfield` extraction artifacts where `NV` gets captured as part of a name. Names starting with a vote code prefix are re-split into the correct code + name.
  - **Tested:** 15 committee + floor vote PDFs, all 15 pass (previously 3+ were failing on every committee PDF with middle initials). Regression tests on floor votes (87Y/28N, 55Y) all pass.

- **Vote/slip scraper robustness overhaul + data recovery (`scrape_votes.py`):**
  - **Root cause:** Progress file (`votes_slips_progress.json`) tracked "done" bills by bill_number but didn't verify actual data existed in `bills.json`. After a cache regeneration, all 11,721 bills were marked "done" in progress but had empty `vote_events`/`witness_slips` arrays. Scraper would report "Nothing to do" while startup showed 0 vote events / 0 slips.
  - **Data recovery:** Merged standalone `cache/vote_events.json` (18 records, 8 bills) and `cache/witness_slips.json` (17,440 records, 5 bills) into `bills.json`. Result: 129 vote events across 62 bills, 17,444 witness slips across 9 bills now load correctly at startup.
  - **Data-verified progress:** Progress validation now checks actual `bill.vote_events` / `bill.witness_slips` arrays, not just the progress file. Stale entries (marked "done" but no data) are automatically dropped. New `--verify` flag rebuilds progress entirely from `bills.json` data.
  - **Dual-track progress:** New `checked_no_data` list tracks bills that were scraped but genuinely had no votes/slips (distinguishes "checked and empty" from "never checked"). Prevents re-scraping known-empty bills.
  - **Heuristic skip:** Bills stalled at intro/assignments (only "Filed with Secretary", "First Reading", "Referred to Assignments" actions) are skipped without HTTP requests — saves ~3,800+ round-trips. These bills can't have votes or witness slips.
  - **Batch saves:** `bills.json` now saved every 25 bills (configurable via `--save-interval`) instead of after every single bill. Eliminates writing ~50MB JSON 11,000+ times. Progress file also batched.
  - **Smart ordering:** SB/HB scraped first (most important), then resolutions, AM last. Previously alphabetical (AM came first, SB/HB never reached).
  - **ETA display:** Each progress line now shows estimated time remaining.
  - **Standalone merge:** Scraper auto-merges `vote_events.json` / `witness_slips.json` into bills at startup if they exist.
  - **Makefile:** Added `make scrape-votes-verify` target.
  - **Estimated full scrape:** ~100 minutes for all ~11,400 remaining bills (5 workers, 0.15s delay). Heuristic skip removes ~255 stalled bills instantly.

- **Lint hardening + cleanup (Ruff E501/W293):**
  - Wrapped long metric/help strings and cleaned blank-line whitespace in:
    `scripts/scrape_votes.py`, `src/ilga_graph/metrics_definitions.py`,
    `src/ilga_graph/schema.py`, `src/ilga_graph/scraper.py`,
    `tests/conftest.py`, and `tests/test_scrape_votes_sample.py`.
  - Added commit-time lint guard: new `.pre-commit-config.yaml` with Ruff check
    (`--fix`) and Ruff format hooks.
  - Added `pre-commit` to dev dependencies and Make targets:
    `make hooks` (install hooks) and `make hooks-run` (run on all files).
  - Fixed exporter regression in `exporter.py`: `_render_member()` now falls back
    to member-attached bills (`sponsored_bills` / `co_sponsor_bills`) when no
    global `bills_lookup` is passed, restoring `[[SBxxxx]]` / `[[HBxxxx]]` links
    in direct rendering tests.
  - Strengthened commit gate: `.pre-commit-config.yaml` now runs full
    `make test` in addition to Ruff, blocking commits on failing tests.

- **Incremental votes/slips scraper (`make scrape-votes`):**
  - Created `scripts/scrape_votes.py` -- standalone incremental scraper for roll-call votes and witness slips. Derives bill list from `cache/bills.json` (all bills with `status_url`), not from hardcoded URLs. Progress tracked in `cache/votes_slips_progress.json` so each run picks up where the last left off.
  - **Resumable:** Progress saved after each bill completes (atomic writes). Ctrl+C at any time -- completed bills are already persisted. Run again to continue from next unscraped bill.
  - **Parallel:** Uses `ThreadPoolExecutor` (default 5 workers) for concurrent bill scraping. `--workers N` to tune.
  - **Real-time output:** Each bill prints a progress line as it completes (`[3/10] HB0034 -- 4 votes, 3217 slips (2.1s)`), plus a summary at end or on interrupt.
  - **CLI:** `--limit N` (default 10, 0 = all remaining), `--workers N` (default 5), `--fast` (shorter delay), `--reset` (wipe progress).
  - **Makefile:** `make scrape-votes` (next 10), `make scrape-votes LIMIT=50`, `make scrape-votes LIMIT=0` (all).
  - **scrape.py cleanup:** Removed inline votes/slips scraping from `scripts/scrape.py`. Bill scrape now preserves existing per-bill `vote_events`/`witness_slips` in cache (no wipe). Log message points to `make scrape-votes`.

- **Roll-call votes / witness slips count verified (not a display bug):**
  - Startup table now shows bill coverage: e.g. "13 vote events (5 bills)" and "17440 slips (5 bills)" so it's clear data is from a subset.

- **Fixed "laws passed: 0 of 0 filed" — all members showing accurate bill stats now.**


- **Fixed "laws passed: 0 of 0 filed" bug (two root causes):**
  - **`is_shell_bill()` false positive** (`analytics.py`): The `len(desc) < 50` threshold was marking **every single bill** as a "shell bill" because ILGA index descriptions are abbreviated titles (max 30 chars, median 25). For example, "CRIM CD-FIREARM SILENCER" (24 chars) was flagged as a shell bill. Fix: Removed the length threshold entirely. Now only matches keyword "Technical", "Shell", or the `-TECH` suffix (ILGA abbreviation for technical/procedural bills like "LOCAL GOVERNMENT-TECH"). This correctly identifies ~1 true shell bill per ~70 substantive bills.
  - **Stale analytics cache** (`analytics_cache.py`): The freshness check (`_member_cache_mtime`) only compared `scorecards.json`/`moneyball.json` against `members.json` mtime. When the full scrape updated `bills.json` (with `sponsor_ids` populated), the analytics cache wasn't invalidated because `members.json` hadn't changed. Fix: Renamed to `_source_data_mtime()` and now checks the **max** mtime of both `members.json` AND `bills.json`. Deleted stale cache files so analytics recompute on next server start.
  - **Verified**: 179 of 180 members now show accurate `laws_filed` / `laws_passed` / `passage_rate`. Totals: 7,126 laws filed, 419 passed across all members. Sample: Neil Anderson 66 filed / 1 passed (1.5%), Omar Aquino 26/2 (7.7%), Christopher Belt 56/4 (7.1%).



- **Discovery-based scraping: parse /Legislation to find ALL doc types and exact ranges:**
  - **New approach** (`scrapers/bills.py`): Added `DocTypeInfo` dataclass and `_discover_doc_types()`. Fetches the ILGA `/Legislation` page once and parses ALL doc type sections (SB, HB, SR, HR, SJR, HJR, SJRCA, HJRCA, EO, JSR, AM) and their exact range page URLs. This eliminates blind pagination, the single-page optimization hack, and the recycled-data guard.
  - **Refactored `scrape_bill_index()`**: Now accepts an optional `range_urls` parameter (from discovery). When provided, iterates over the exact known URLs — no guessing, no infinite loops. Falls back to blind pagination with recycled-data guard only when range_urls is not provided.
  - **Refactored `scrape_all_bill_indexes()`**: Calls `_discover_doc_types()` first, then iterates over all discovered types. SB/HB respect existing `sb_limit`/`hb_limit` CLI flags. All other types are always scraped in full (they're small).
  - **Removed dead code**: `_single_page_list_url()` and `_scrape_bill_index_single_page()` are gone — replaced by the discovery approach.
  - **Chamber derivation fix**: `scrape_bill_status()` now correctly assigns `chamber = "J"` (Joint/Executive) for EO, JSR, and AM doc types, instead of incorrectly defaulting to "H".
  - **Verified**: Full isolated test discovered 11 doc types (125 range pages) and scraped **~15,000+ entries** across all categories:
    - SB: ~4052 | HB: 5623 | SR: 614 | HR: 660 | SJR: 52 | HJR: 54
    - SJRCA: 11 | HJRCA: 25 | EO: 8 | JSR: 1 | AM: 621
  - **Backward compatible**: `sb_limit`/`hb_limit` CLI flags, `etl.py`, `scripts/scrape.py`, and all downstream code unchanged.

- **Previously fixed: infinite loop + 200-bill truncation (now superseded by discovery):**
  - The infinite loop bug (ILGA returning recycled SB0001-SB0100 for out-of-range queries) and the 200-bill truncation bug (single-page optimization returning only 100 per chamber) are both eliminated by the discovery approach. The recycled-data guard is kept only as a safety net in the fallback path.

- **Separation of concerns + cache simplification + single-page index (full plan execution):**
  - **Bill model** (`models.py`): Added `vote_events: list[VoteEvent]` and `witness_slips: list[WitnessSlip]` fields to `Bill` dataclass. Each bill now carries its own vote and slip data.
  - **Single-page SB/HB index** (`scrapers/bills.py`): Added `_single_page_list_url()` and `_scrape_bill_index_single_page()`. For SB and HB, the scraper now fetches the full bill list in **one request per chamber** instead of ~85 paginated requests. Falls back to range-based pagination if the single-page request fails.
  - **Bills cache** (`scrapers/bills.py`): `_bill_from_dict()` and `_bill_to_dict()` now serialize/deserialize `vote_events` and `witness_slips` per bill. `save_bill_cache()` uses atomic writes (temp file + rename).
  - **Unified committees.json** (`scraper.py`): Merged `committee_rosters.json` and `committee_bills.json` into a single `committees.json` where each committee entry includes `roster` and `bill_numbers` inline. Separate cache files no longer written. `_save_unified_committee_cache()` replaces `_save_split_committee_cache()`. Atomic writes used.
  - **GraphQL schema** (`schema.py`): Added `ActionEntryType` (date, chamber, action) and `action_history: list[ActionEntryType]` to `BillType`. **MemberType** now exposes `sponsored_bill_ids: list[str]` and `co_sponsor_bill_ids: list[str]` instead of embedded `sponsored_bills: list[BillType]` / `co_sponsor_bills: list[BillType]`. Separation of concerns enforced at the API level.
  - **Exporter** (`exporter.py`): Bill notes now include an `## Actions` table (date, chamber, action) from `bill.action_history`. Member notes use `member.sponsored_bill_ids` / `member.co_sponsor_bill_ids` + a bills lookup to render `[[bill_number]]` links (IDs only, no embedded bills). Bill set built exclusively from `all_bills` parameter. Co-sponsors resolved by iterating member `co_sponsor_bill_ids`.
  - **ETL + main.py**: `state.vote_events`, `state.vote_lookup`, `state.witness_slips`, `state.witness_slips_lookup` now populated from per-bill data in `state.bills` (no separate vote/slip scraping in the API startup). Resolvers unchanged — they still read from `state.vote_lookup` and `state.witness_slips_lookup`.
  - **Scrape script** (`scripts/scrape.py`): After scraping votes/slips, merges them into the corresponding `Bill` objects in `data.bills_lookup` and re-saves `bills.json` with atomic write. No separate `vote_events.json` or `witness_slips.json` files.
  - **Cache layout** (new minimal set):
    - `members.json` — all members, member details only (IDs for bill references)
    - `committees.json` — all committees with roster + bill_numbers inline
    - `bills.json` — heavyweight: each bill has status, action_history, vote_events, witness_slips
    - `zip_to_district.json` — ZIP → district mapping (unchanged)
    - `scorecards.json` / `moneyball.json` — optional derived cache (recomputed if stale)
    - `scrape_metadata.json` / `bill_index_checkpoint.json` — transient operational files

## Previously done

- **Bill index scraping improvements:** Fixed two issues with bill index scraping: (1) **Clearer progress logs**: Terminal now shows expected bill number ranges using standard convention (e.g., "SB 0001-0100 (page 1)", "SB 0101-0200 (page 2)") and reports actual bills found (e.g., "✓ Parsed 42 bills: SB0001 to SB0042"). This makes it clear which 100-bill range you're on and how many bills actually exist. (2) **Incremental checkpoint saving**: Bill index scraping now saves progress to `cache/bill_index_checkpoint.json` after every page (~100 bills). If interrupted (Ctrl+C), rerunning the scrape will resume from the checkpoint instead of starting over. Checkpoint is cleared automatically when scraping completes successfully. This applies to both the index phase and the detail scraping phase (which already had checkpointing every 50 bills).

- **Chamber-specific associated member fields:** Previously, member associations were generic "Associated Members" for both senators and representatives, which was confusing since senators have 2 representatives and representatives have 1 senator. Changes: (1) Replaced `associated_members` with two chamber-specific fields in GraphQL `MemberType`: `associated_senator` (for Representatives - singular) and `associated_representatives` (for Senators - plural). These are computed from the underlying `associated_members` data based on chamber. (2) Updated `exporter.py` to render chamber-specific section headers in Obsidian markdown: "## Associated Representatives" for senators, "## Associated Senator" for representatives. This makes the API more semantically clear and the UI more intuitive.

- **Full bill index + scrape-200:** Bill index previously only fetched one range page (100 SB + 100 HB). Implemented pagination in `scrape_bill_index()`: loop over range pages (0001-0100, 0101-0200, ...) until we have enough or a page is empty. `limit=0` means all pages (~9600+ bills). Added **make scrape-200** (200 SB + 200 HB, 2 range pages per type) to test pagination and **make scrape-full** (--sb-limit 0 --hb-limit 0) for full index. README and scrape.py docstring updated. Full scrape is slow (many index pages + one BillStatus request per bill).

- **Scrape → dev/prod pipeline:** Simplified to a clear two-step flow: (1) scrape data into cache (choose size), (2) serve via API. **make scrape** = full scrape for prod (all members, 300 SB + 300 HB). **make scrape-dev** = light scrape for dev (20/chamber, 100 SB + 100 HB, fast). **make dev** / **make dev-full** / **make run** now set **ILGA_LOAD_ONLY=1** so the server only loads from cache (no scraping on startup). Removed **make scrape-fast** in favor of **make scrape-dev**. README and scrape.py docstring updated; .env.example documents ILGA_LOAD_ONLY.

- **Evidence-based script hints & "How we pick these targets":** Script hints were hardcoded generic strings ("high effectiveness, deep network connections") that assumed the user already understood the system. A first-time visitor had no way to know *why* a legislator was chosen or what "effectiveness" meant. Changes: (1) Added `_build_script_hint_*` helpers in `main.py` that inject real numbers (laws passed X of Y, Z% passage rate, W% cross-party) into each card's script hint. Senator/Rep hints include ZIP and district. Power Broker hint explains Chair vs Moneyball selection with stats. Ally hint cites cross-party %. Super Ally merges both with evidence. (2) Template macro `member_card` now reads `member.script_hint` from the card dict instead of receiving a hardcoded 4th argument. (3) Added collapsible "How we pick these targets" section at the bottom of results: plain-language definitions of Senator/Rep (Census ZIP match), Power Broker (committee chair or highest Moneyball), Potential Ally (seatmate with highest cross-party rate), and Super Ally (merged). (4) Nested collapsible "How is the Moneyball score calculated?" with a table of the 6 components, weights, and what each measures. (5) Added CSS for `.how-it-works` collapsible and `.formula-table` in `base.html`. No algorithm changes — only explanation and evidence in the UI.

- **Scorecard / Moneyball clarity:** User wanted empirical stats (bills passed, vetoed, passage rate) front and center and derived metrics (Moneyball, effectiveness) clearly defined to avoid vagueness and bloat. (1) Added `src/ilga_graph/metrics_definitions.py` as single source of truth: empirical metric definitions (laws filed/passed, passage rate, magnet, bridge, pipeline, centrality, etc.) and Moneyball formula (one-liner + component list with weights). (2) Advocacy cards now show empirical first: "Laws passed: X of Y (Z% passage)", "Cross-party co-sponsorship: W%", then "Moneyball: N (composite rank 0–100)" with a "?" tooltip that shows the one-liner. (3) GraphQL query `metricsGlossary` returns full glossary (empirical, effectiveness_score, moneyball_one_liner, moneyball_components) so any client can render tooltips or docs. (4) README section "Metrics: empirical vs derived" documents the approach. No new metrics added; we only clarified and reordered display.

- **Advocacy full-data / seed-mode clarity:** Banner was always showing "Only 40 of 177 legislators" and "~14 demo ZIPs" even when full cache was loaded. Root cause: `make dev` sets `ILGA_SEED_MODE=1`, so (1) members load from cache when present (so user can have 177 after `make scrape`), but (2) ZIP crosswalk always used hardcoded seed (~14 ZIPs) when seed mode ON. Changes: (1) Advocacy banner now uses actual `member_count` and `zip_count` and only shows "Only N of 177" when N < 100; explains how to get full data (`make run` or `ILGA_SEED_MODE=0`). (2) Added `make dev-full` (dev profile + `ILGA_SEED_MODE=0`) to run with full cache + full Census ZCTA for testing. (3) Documented in README ("Testing with full data") and .env.example. To test advocacy with all members and all ZIPs: `make scrape` then `make dev-full` or `make run`.

---

## Next (when you're ready)

- **Run full pipeline** (PRIORITY): Pipeline is fixed and unified with smart tiered scanning.
  - `make scrape` — Daily run (~2 min if <24h since last scan, auto-decides tier).
  - `make scrape FULL=1` — Force full index walk (all 125 pages, ~30 min). Use weekly or after major ILGA updates.
  - `make scrape WORKERS=10` — More parallel workers for votes/slips.
  - `make scrape LIMIT=100` — Limit vote/slip phase to 100 bills (for testing).
  - `make scrape FRESH=1` — Nuke cache and re-scrape from scratch.
  - Can Ctrl+C and resume at any time. Batch saves every 25 bills. Vote/slip data is never lost by re-scraping bills.
  
- **Verify GraphQL:** After scrape completes, `make dev` and check that scorecards/Moneyball recompute with vote data. Spot-check 3-4 members' "Laws passed: X of Y" against ilga.gov to verify accuracy. Test advocacy frontend with real ZIPs to see if Power Broker / Ally selections make sense with real vote/slip data.

- When shifting to prod: set `ILGA_PROFILE=prod`, `ILGA_CORS_ORIGINS`, and optionally `ILGA_API_KEY`.

---

## Production (checklist for deployment)

- Pre-populate `cache/` (full or incremental scrape) so startup is load-only.
- Set `ILGA_PROFILE=prod` (turns off dev mode + seed mode automatically).
- Set `ILGA_CORS_ORIGINS` to your front-end origin(s) — prod warns if unset.
- Set `ILGA_API_KEY` to protect the GraphQL endpoint — prod warns if empty.
- Use `GET /health` for readiness (`ready` is true when members are loaded).

---

## ML Pipeline (`feature/ml-pipeline` branch)

**New `src/ilga_graph/ml/` package** — transforms cached legislative JSON data into a normalized Parquet star schema and provides interactive human-in-the-loop ML training.

- **Data Pipeline** (`ml/pipeline.py`, `scripts/ml_pipeline.py`):
  - Flattens `cache/bills.json` (11,722 bills), `cache/members.json` (180 members), `cache/vote_events.json` into 6 normalized Parquet tables in `processed/`:
    - `dim_members.parquet` — 180 rows (member_id PK, name, party, chamber, district, career info)
    - `dim_bills.parquet` — 11,722 rows (bill_id PK, bill_type, synopsis, sponsor FK, dates)
    - `fact_bill_actions.parquet` — 103,332 rows (action history with category classification)
    - `fact_vote_events.parquet` — 6,692 rows (aggregated vote events with outcome)
    - `fact_vote_casts_raw.parquet` — 210,708 rows (individual vote casts before entity resolution)
    - `fact_witness_slips.parquet` — 394,203 rows (slip filings with position, org, testimony type)
  - Runs in ~6.5 seconds via `make ml-pipeline`.

- **Entity Resolution** (`ml/entity_resolution.py`, `ml/active_learner.py`, `scripts/ml_resolve.py`):
  - Maps raw vote PDF names ("Murphy", "Davis,Will", "Loughran Cappel") to canonical member IDs.
  - Multi-strategy resolution: exact variant map (handles compound last names, nicknames, PDF artifacts) -> fuzzy match (rapidfuzz) -> human-in-the-loop.
  - **Auto-resolves 98.7% (380/385 unique names)** out of the box. Remaining 5 names presented interactively.
  - Rich CLI: presents candidates with scores, user picks correct match, choices persist in `processed/entity_gold.json`.
  - Output: `processed/fact_vote_casts.parquet` — resolved with member_id FK.
  - `make ml-resolve` (interactive) or `make ml-resolve AUTO=1` (auto-only).

- **Feature Engineering** (`ml/features.py`):
  - Builds 526-feature matrix per bill: 500 TF-IDF text features + 26 tabular features.
  - **Sponsor features**: party, majority status, historical passage rate (computed without leakage), bill count.
  - **Slip features**: proponent/opponent counts, ratios, org concentration (HHI), written-only ratio.
  - **Temporal features**: intro month, day of year, lame duck flag, chamber origin, bill type.
  - **Early action features**: first-30-days actions only (no outcome leakage).
  - **Time-based train/test split**: 70% train (first by date) / 30% test (latest by date). No random splitting.

- **Bill Outcome Predictor** (`ml/bill_predictor.py`, `scripts/ml_predict.py`):
  - GradientBoosting classifier predicts whether a bill will advance past committee.
  - **ROC-AUC: 0.9477** (no data leakage). Top features: text signals ("technical" = shell bill), sponsor count, witness slip support, intro timing.
  - Interactive active learning: presents least-confident predictions first, user confirms/corrects, model retrains with gold labels.
  - Corrections persist in `processed/bill_labels_gold.json`. Model saved to `processed/bill_predictor.pkl`.
  - `make ml-predict` (interactive) or `make ml-predict TRAIN=1` (train-only).

- **Dependencies**: `polars`, `pyarrow`, `scikit-learn`, `rapidfuzz`, `networkx`, `rich` in `[project.optional-dependencies] ml`.

- **Next (ML backlog)**:
  - Individual vote prediction (recommender system using member embeddings from matrix factorization)
  - Hidden coalition discovery (Node2Vec graph embeddings on co-vote/co-sponsorship network, DBSCAN clustering)
  - Witness slip anomaly detection (Isolation Forest for astroturfing)
  - "Poison pill" detector (semantic similarity between original and amendment synopses)
  - Committee assignment prediction (multi-class text classifier)

---

## Backlog / Future

- (Done: unified GraphQL `search` query — cross-entity free-text search with relevance scoring.)
- Search enhancements: fuzzy matching (Levenshtein), startup index for O(1) lookups, vote events / witness slips as searchable entities, autocomplete / query suggestions endpoint, TF-IDF weighting if data grows.
- (Done: full bill index via `make scrape-full` — pagination in bills.py; 0 = all pages.)
- (Done: `make scrape-votes` derives vote/slip bill list from cache for incremental coverage.)
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

**Resilience & performance:** HTTPAdapter with retries and connection pooling; stale cache warnings; startup timing logs and `.startup_timings.csv`; startup summary table now uses chronological ETL-oriented phases with explicit per-step times/details (core load, analytics, seating, vault export, committee indexes, vote index + normalization, member voting records, witness slips, ZIP crosswalk); CSV includes seating_s, slips_s, zip_s, votes, slips, zctas; vote/slip cache eliminates re-scraping on every start.

- (Done) Add `scripts/startup_timings_report.py`: rich/animated CLI to analyze `.startup_timings.csv` across schema versions and show recent-vs-baseline regressions (`make startup-report`).
- (Done) Clarify full-cache runtime mode: `make dev-full` now sets `ILGA_DEV_MODE=0` so startup does not imply 20/chamber + top-100 caps, and startup logs now explicitly distinguish cache-only vs scrape startup.
- (Done) Rework startup summary table into chronological ETL-oriented phases with clearer per-step timing/details (adds committee-index and member-voting-record steps so timing output maps to actual runtime order).

**Witness slips & GraphQL:** Slips and votes share bill list (`ILGA_VOTE_BILL_URLS`); witness slip summary and paginated summaries; `billSlipAnalytics`, `memberSlipAlignment`; advancement analytics (`billAdvancementAnalyticsSummary`); docs in `graphql/README.md` and main README.

**Bills-first & incremental:** Bills from Legislation pages (single source of truth); bill index to BillStatus detail; `scrapers/bills.py` with `incremental_bill_scrape()`; member-bill linkage from `Bill.sponsor_ids`; cache persisted every run; exporter uses `all_bills` from `bills_lookup` so all cached bills get vault notes.

**Server & CI:** Single app in `main.py` (schema.py types only); ruff lint/format and per-file ignores for schema; tests updated for line-length and duplicate keys.

**Seating Chart / Whisper Network (Verdict 1c):** Added `seat_block_id`, `seat_ring`, `seatmate_names`, `seatmate_affinity` fields to `Member` dataclass. Created `src/ilga_graph/seating.py` with fuzzy name matching (bare last name, "Initial. Lastname", compound last names like "Glowiak Hilton"), the Aisle Rule (neighbors within same section only, no cross-aisle adjacency), and co-sponsorship affinity calculation (overlap % of member's bills with seatmate co-sponsors). Exposed all four fields via GraphQL `MemberType`. Integrated into lifespan (Step 2b) and `run_etl`. Senate-only for now; seating data loaded from `mocks/dev/senate_seats.json`.

**SSR Advocacy Frontend:** Added `jinja2` and `python-multipart` dependencies. Created `src/ilga_graph/zip_crosswalk.py` — Python port of the Google Sheets Census crosswalk. Seed-mode fallback has ~14 ZIPs (6 produce full 4-card results with the 40-member dev dataset). Jinja2 templates with htmx (v2.0.4 CDN) for in-page search. `GET /advocacy` renders the search form; `POST /advocacy/search` returns up to 4 cards: **Your Senator**, **Your Representative**, **Power Broker**, and **Potential Ally**. Each card has a "Why this target" box explaining the analytics reasoning and a role-specific script hint. **Super Ally merge:** when Power Broker and Ally resolve to the same person, they collapse into a single "Super Ally" card with both badges and a combined explanation. **Policy category filter:** optional dropdown maps 12 policy areas (Transportation, Education, etc.) to Senate committee codes; when selected, Power Broker and Ally are restricted to members on those committees (with graceful fallback if the filter eliminates all candidates). Committee rosters loaded from `state.committee_rosters`. Results split into "Your Legislators" and "Strategic Targets" sections.

**Steps 3–5 (ETL, analytics cache, DataLoaders):** **(Step 3)** Created `src/ilga_graph/etl.py` with `load_from_cache()`, `load_or_scrape_data()`, `compute_analytics()`, `export_vault()`, `run_etl()`, and `load_stale_cache_fallback()`. Added `ILGA_LOAD_ONLY`: when set, API startup only loads from cache (no scrape). Lifespan uses load-only path when `LOAD_ONLY=1`; scrape script uses etl and supports `--export-only` with `load_from_cache()`. **(Step 4)** Added `analytics_cache.py`: `save_analytics_cache()` / `load_analytics_cache()` persist scorecards and Moneyball to `cache/scorecards.json` and `cache/moneyball.json`. Staleness: cache is used only if newer than member data (`members.json` mtime). Lifespan and scrape script load when fresh, else compute and save. **(Step 5)** Added `loaders.py` with `ScorecardLoader`, `MoneyballProfileLoader`, `BillLoader`, `MemberLoader` (each with `load()` and `batch_load()`). GraphQL context_getter returns `create_loaders(state)`. Resolvers `member`, `members`, and `moneyball_leaderboard` use `info.context` and batch_load for scorecard/profile so list queries do one batched fetch instead of N lookups.

**Moneyball v2 Recalibration:** Three changes to stop undervaluing legislative leadership. **(1) Shell Bill Filter:** `is_shell_bill()` in `analytics.py` identifies procedural placeholders (description contains "Technical"/"Shell" or ends with "-TECH") and excludes them from the effectiveness denominator (`law_heat_score`), so leadership who file shell bills aren't penalised for "failures" that were never meant to pass. Applied in `compute_scorecard()`, `compute_all_scorecards()`, and `avg_pipeline_depth()`. **(2) Institutional Power Bonus:** Added `roles: list[str]` to `Member` (aggregates profile title + committee roster titles via `populate_member_roles()`). New `compute_institutional_weight()` returns 1.0 (President/Leader/Speaker), 0.5 (Chair/Spokesperson), or 0.25 (Whip/Caucus Chair). Added to `MoneyballProfile` as `institutional_weight` and to the composite score at 20% weight (existing weights scaled to 80%: effectiveness 24%, pipeline 16%, magnet 16%, bridge 12%, centrality 12%). **(3) Committee-Chair-First Power Broker:** `_find_power_broker()` now prioritises Committee Chairs of the relevant committee when a policy category is selected, falling back to highest Moneyball score only when no Chair is found. Serialization updated in `scraper.py`; test fixtures updated with realistic 50+ char descriptions.
