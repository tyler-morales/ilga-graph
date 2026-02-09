# Performance Notes & Startup Timing Analysis

This document tracks performance observations and bottleneck analysis for the ILGA Graph application startup.

## Startup Timing Instrumentation

As of the latest implementation, the application logs detailed timing for each ETL step:

- **Load/Scrape Data** -- Loading from cache/ or mocks/dev or scraping ilga.gov
- **Compute Analytics** -- Building scorecards and moneyball profiles
- **Export Vault** -- Writing markdown files to `ILGA_Graph_Vault/`
- **Vote Scraping** -- Scraping 3 test bills for roll-call votes

All startup timings are logged to `.startup_timings.csv` (gitignored) for historical tracking.

## Typical Startup Times

### Dev Mode with Seed Data (`make dev` after `make clean`)

**Expected breakdown** (20 members from `mocks/dev/`):
- **Load**: 0.04-0.1s (reading JSON from mocks/dev/)
- **Analytics**: 0.04-0.2s (scorecards + moneyball for 20 members)
- **Export**: 0.06-0.3s (writing ~20 member files, ~150 committees, ~200 bills)
- **Votes**: 0.01-0.1s (cached) or 2-8s (first run, scraping 3 bills)
- **Total**: ~0.2-0.7s (cached votes) or ~2-9s (first run)

### Dev Mode with Cache (`make dev` with existing `cache/`)

**Expected breakdown** (60-180 members from `cache/`):
- **Load**: 0.3-0.8s (reading JSON from cache/)
- **Analytics**: 1.5-4.0s (scorecards + moneyball for 60-180 members)
- **Export**: 3.0-8.0s (writing 60-180 members, ~150 committees, ~200 bills)
- **Votes**: 2.0-4.0s (scraping 3 bills)
- **Total**: ~7-17 seconds

### Production Mode with Full Cache (`make run`)

**Expected breakdown** (all members, all bills):
- **Load**: 1.0-2.5s (reading large JSON files)
- **Analytics**: 5.0-15.0s (full legislative cohort)
- **Export**: 10.0-30.0s (writing hundreds of member files, thousands of bills)
- **Votes**: 2.0-4.0s (scraping 3 bills)
- **Total**: ~18-50 seconds

## Known Bottlenecks

### 1. Analytics Computation (Scorecards & Moneyball)

**Bottleneck**: `compute_all_scorecards()` and `compute_moneyball()` are CPU-bound.

**Why it's slow**:
- Builds a co-sponsorship map across all members/bills (O(n*m) where n=members, m=bills)
- Computes network centrality (graph traversal)
- Runs for every member on every startup

**Mitigation**:
- Analytics results could be cached alongside member data in `cache/`
- Currently: Not cached, recomputed every startup

**Impact**: Accounts for 20-40% of startup time with large datasets.

### 2. Vault Export (File I/O)

**Bottleneck**: Writing hundreds of markdown files to `ILGA_Graph_Vault/`.

**Why it's slow**:
- One file per member (60-180 files)
- One file per bill (200-2000+ files)
- One file per committee (~150 files)
- Each file includes frontmatter rendering, wikilink generation, scorecard tables

**Mitigation**:
- Export could be made incremental (only write changed files)
- Currently: Full re-export on every startup

**Impact**: Accounts for 30-50% of startup time, especially with large bill counts.

### 3. Vote Scraping (RESOLVED)

**Was a bottleneck**: Live HTTP scraping of 3 bills from ilga.gov took 2-8 seconds every startup.

**Solution implemented**: Votes are now cached in `cache/vote_events.json`. First run scrapes and caches. Subsequent runs load from cache (~0.1s).

**Impact**: Eliminated fixed 2-8s overhead. Now only pays cost on first run or after `make clean`.

### 4. Seed Data Loading

**Observation**: Even with only 20 members from mocks/dev, startup takes 4-9 seconds.

**Why**:
- Seed data is small (fast load: ~0.2s)
- But analytics still runs (1-2s for 20 members)
- And vault export writes ~200+ files (1-3s)
- And votes are scraped (2-4s)

**This is expected**: The ETL pipeline does real work even with small datasets. The dev mock optimization is about avoiding *scraping*, not avoiding analytics/export.

## Optimization Opportunities

### High Impact
1. **Cache analytics results** -- Save scorecards/moneyball alongside member data to skip recomputation
2. **Incremental vault export** -- Only write changed files (detect by comparing frontmatter hash)

### Medium Impact
3. **Parallel file export** -- Use `ThreadPoolExecutor` to write vault files in parallel
4. **Skip vote scraping in dev** -- Add `ILGA_SKIP_VOTES=1` env var (votes are now cached, but you could skip entirely)

### Low Impact (Already Fast)
5. **Faster JSON loading** -- Use `orjson` instead of `json.load()` (marginal gains)

### Already Implemented
- ✅ **Vote caching** -- Votes are cached in `cache/vote_events.json`
- ✅ **Dev mock fallback** -- Instant startup with `mocks/dev/`
- ✅ **Member/committee caching** -- All scraped data is cached
- ✅ **Normalized cache** -- Bills stored once in `cache/bills.json`, members reference by `leg_id` in `cache/members.json`. ~70% reduction in cache size (from ~12.7 MB down to ~3.5 MB for a full scrape)

## How to Monitor Performance

### View Timing Logs

```bash
# See all historical startup times
cat .startup_timings.csv

# See last 10 startups
tail -10 .startup_timings.csv

# Find slowest startups
sort -t, -k2 -rn .startup_timings.csv | head -10
```

### Analyze Bottlenecks

The CSV columns are:
- `timestamp` -- When the startup occurred
- `total_s` -- Total startup time
- `load_s` -- Time to load/scrape data
- `analytics_s` -- Time to compute scorecards/moneyball
- `export_s` -- Time to write vault files
- `votes_s` -- Time to scrape roll-call votes
- `members` -- Number of members loaded
- `bills` -- Number of unique bills
- `dev_mode` -- True if `ILGA_DEV_MODE=1`
- `seed_mode` -- True if `ILGA_SEED_MODE=1`

### Terminal Output

Each startup now shows a clean color-coded summary table:

```
================================================================================
🚀 Application Startup Complete
================================================================================

Step                     Time  Details                                      
--------------------------------------------------------------------------------
✓ 📦 Load Data           0.04s  20 members, 149 committees (from mocks/dev)
✓ 📊 Analytics           0.04s  20 scorecards, 20 profiles
✓ 📝 Export Vault        0.06s  200 bills exported
✓ 🗳️  Roll-Call Votes    0.01s  13 events (cached)
--------------------------------------------------------------------------------
Total                    0.15s  Dev: True, Seed: True
================================================================================

  🏆 MVP (House, non-leadership): Charles Meier (Score: 27.4)
```

Colors:
- ✓ checkmarks = bright green
- Times = bright green (fast) or yellow (slow)
- Details = white
- Cached indicators = dim gray

## Normalized Cache Architecture

As of the latest refactor, the data cache uses a **normalized** format:

```
cache/
├── members.json      # Member metadata + bill_ids/primary_bill_ids (references only)
├── bills.json        # All unique bills keyed by leg_id (deduplicated)
├── committees.json   # Committees, rosters, and bill assignments
└── vote_events.json  # Cached roll-call vote data
```

**Before normalization**: Each bill appeared once per member who touched it, resulting in 3.7x duplication (43,420 bill records for 11,699 unique bills). Cache size: ~12.7 MB.

**After normalization**: Each bill is stored exactly once in `bills.json`. Members store only `bill_ids` and `primary_bill_ids` (arrays of `leg_id` strings). On load, `hydrate_members()` resolves IDs back to full `Bill` objects. Cache size: ~3.5 MB (70% reduction).

The GraphQL API is unchanged -- members are hydrated in memory before serving.

## Conclusion

**Current performance is acceptable for a POC**, especially with mocks/dev for development. The main bottlenecks are:
1. Analytics computation (not cached)
2. Vault file I/O (not incremental)

These are all solvable with caching/incrementalism if startup time becomes a production concern.
