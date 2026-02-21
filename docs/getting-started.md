# Getting started

Get the ILGA Graph app and docs running locally.

---

## Prerequisites

- **Python 3.10+**
- (Optional) Virtual environment: `python3 -m venv .venv` then `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)

---

## Install

```bash
make install        # pip install -e ".[dev]"
```

Or with docs and ML extras:

```bash
pip install -e ".[dev,docs,ml]"
```

---

## Run the app

Data is **served from cache or mocks** (no scraping on startup when using `make dev` or `make serve`).

| Command | What it does |
|---------|----------------|
| `make dev` | Start the app in **dev mode** at **http://127.0.0.1:8000** (auto-reload). Uses `cache/dev/` if present, else `mocks/dev/`. |
| `make serve` | Start in **prod mode** (cache only, no reload). |

### Where does data come from?

- **Prod:** Legislative data comes from `cache/` only. Run `make scrape` to populate it.
- **Dev:** Legislative data comes from one place: **`cache/dev/`** if it has data (e.g. after `make dev-cache`), otherwise **`mocks/dev/`**. The app never writes to mocks.
- **Outreach** (calls, emails) lives in the SQLite DB. To populate the dev DB with sample outreach, run `make seed-outreach` once (same profile as the app).

Full detail: [Data and environments](development/data-and-environments.md).

### Quick reference

| Directory | Used by | Written by |
|-----------|---------|------------|
| `cache/` | Prod app | `make scrape` |
| `cache/dev/` | Dev app (optional) | `make dev-cache` or manual |
| `mocks/dev/` | Dev app when `cache/dev/` empty | You (commit) or `make snapshot-mocks` |

- **Quick dev (no scrape):** `make dev` — uses `mocks/dev/` if `cache/dev/` is empty.
- **Dev with full cache:** Run `make scrape`, then `make dev-cache`, then `make dev`.
- **Reset dev to mocks:** `make dev-reset` then `make dev`.

---

## Get data (scrape)

Run a scrape to populate `cache/` before or after starting the app:

| Command | What it does |
|---------|----------------|
| `make scrape` | Unified scrape: members + bills + votes + slips in one pass, then ML pipeline. |
| **`make scrape-full`** | **Full reset:** delete `cache/`, then scrape **all** members (~177) + full bill index + ML. Use when your data is wrong or incomplete (e.g. only 20 or 60 members, or missing House). |
| `make scrape FULL=1` | Force full index walk (all pages). |
| `make scrape FRESH=1` | Clear cache and re-scrape from scratch. |
| `make scrape FULLTEXT=1` | Include full text PDF scraping in the same pass. |
| `make scrape WORKERS=20` | More parallel workers (default: 10). |
| `make scrape SKIP_VOTES=1` | Metadata only (no votes/slips). |
| `make scrape-members` | Only members + committees; bills from existing cache. |
| `make scrape-full-members` | Re-fetch full member roster (~177) from ILGA; keeps existing bills. |
| `make scrape-fulltext` | Standalone full text backfill (incremental). |
| `make refresh-photos` | Refresh member photos only (keeps existing cache). |

The unified pipeline scrapes metadata, votes, and witness slips per-bill in a single pass (one BillStatus fetch per bill). Full text is an optional add-on via `FULLTEXT=1`. Progress logs every 20 bills with elapsed time and ETA; checkpoints every 50 bills.

**When your data is wrong or incomplete** (e.g. you only have 20 or 60 members, or House is missing): run **`make scrape-full`**. That wipes `cache/` and re-scrapes everything with **no member limit** (~59 Senate + 118 House) and a full bill index. Takes longer but gives a clean, accurate dataset.

After scraping, restart the app (or rely on auto-reload) to load the new data.

### Snapshot mocks from cache

To refresh the dev seed from your **prod cache** (so mocks stay in sync with schema and a recent subset of real data):

```bash
make snapshot-mocks
```

This reads from `cache/` (run `make scrape` first), takes a **subset** (e.g. ~40 members, ~100 bills with votes/slips, related committees), and writes to `mocks/dev/`. Commit the updated `mocks/dev/` so new contributors get an up-to-date seed without scraping.

- **When to run:** After a full scrape when you want to refresh the committed mocks (e.g. new session, schema change).
- **Not automatic:** Scraping never writes to mocks; you run this explicitly.

---

## Run the docs site

```bash
pip install -e ".[docs]"   # if you didn’t install docs above
make docs-serve
```

Opens at **http://127.0.0.1:8001** (port 8001 so it doesn’t clash with the app on 8000).

- **Build only:** `make docs` → output in `site/`.

---

## Typical workflows

| Goal | Steps |
|------|--------|
| **Quick dev (no scrape)** | `make install` → `make dev` (uses `mocks/dev/` if `cache/dev/` is empty). |
| **Dev with full scraped data** | `make scrape` → `make dev-cache` → `make dev`. |
| **Reset dev to mocks** | `make dev-reset` → `make dev`. |
| **Full local data (prod)** | `make scrape` → `make serve`. |
| **Docs + app** | Terminal 1: `make dev`. Terminal 2: `make docs-serve`. App: 8000, Docs: 8001. |
| **Before a PR** | `make lint` and `make test`. |

---

## Next

- [Data and environments](development/data-and-environments.md) — Where data comes from and make targets.
- [App overview](features/app-overview.md) — What’s in the app (Advocacy, Power Map, Intelligence, GraphQL).
- [CLI reference](reference/cli-make.md) — All `make` targets.
- [Environment variables](reference/environment-variables.md) — Config and profiles.
