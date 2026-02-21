# CLI reference (Make)

All `make` targets. Run `make help` in the repo root to see this list in the terminal.

---

## Server & install

| Target | Description |
|--------|-------------|
| `make install` | Install project: `pip install -e ".[dev]"`. |
| `make dev` | Serve app in **dev mode** (auto-reload, port 8000). Reads from `cache/dev/` if it has data, else `mocks/dev/`. |
| `make serve` | Serve app in **prod mode** (no reload). Reads from `cache/` only. |
| `make dev-reset` | Clear `cache/dev/` so the next `make dev` uses `mocks/dev/`. |
| `make dev-cache` | Copy `cache/` into `cache/dev/` so `make dev` uses full scraped data. Run after `make scrape`. |

---

## Scraping

| Target | Description |
|--------|-------------|
| `make scrape` | Unified scrape: members + bills + votes + slips in one pass, then ML pipeline. |
| **`make scrape-full`** | **Full reset:** delete `cache/`, then scrape all members (~177) + full bill index + ML. Use when data is wrong or incomplete (e.g. only 20/60 members, missing House). |
| `make scrape FULL=1` | Force full index walk (all pages). |
| `make scrape FRESH=1` | Nuke cache and re-scrape. |
| `make scrape FULLTEXT=1` | Include full text PDF scraping in the same pass. |
| `make scrape WORKERS=20` | Use 20 parallel workers (default: 10). |
| `make scrape SKIP_VOTES=1` | Metadata only (no votes/slips). |
| `make scrape EXPORT=1` | Include Obsidian vault export. |
| `make scrape-members` | Only members + committees; load bills from cache. |
| `make scrape-full-members` | Re-fetch **full member roster** from ILGA (~177). Removes `cache/members.json` then runs scrape-members so the scraper does not reuse a small cached roster (e.g. 20). Keeps existing `cache/bills.json`. Use when you see "20 members" and want the full legislature. |
| `make refresh-photos` | Refresh member `photo_url` from ILGA; keep existing cache. |
| `make scrape-fulltext` | Standalone full text backfill (incremental, resumable). |

**Why only 20 members or missing data?** The scraper loads members from `cache/members.json` when present. If that file was ever created with a small set (e.g. 20 from seed or an old run), every later `make scrape` will keep reusing it and never re-fetch from ILGA. Bills are updated by the incremental scrape, but the member list stays small. Fix: run `make scrape-full-members` to replace the roster with the full ~177 members from ILGA while keeping your existing bill cache. For a completely fresh dataset, use `make scrape FRESH=1` (deletes all of `cache/` and re-scrapes everything).

The unified pipeline fetches each bill's BillStatus page **once** and reuses the HTML for metadata, vote tab URL extraction, and witness slip derivation. Stalled bills (intro/assignments only) skip votes/slips/fulltext automatically. Checkpoints every 50 bills; progress logged every 20 bills with `Xm Ys elapsed, ~Wm Zs remaining`.

---

## ML pipeline

| Target | Description |
|--------|-------------|
| `make ml-setup` | Install ML deps: `pip install -e ".[ml]"`. |
| `make ml-run` | Full ML pipeline (scores, coalitions, anomalies). Use `ILGA_ML_SKIP_TUNE=1` to skip tuning. |
| `make ml-pipeline` | Data pipeline only: `cache/*.json` → `processed/*.parquet`. |
| `make ml-resolve` | Entity resolution only. `AUTO=1` for non-interactive; `STATS=1` for stats. |
| `make ml-predict` | Bill outcome prediction only. |
| `make ml-embed` | Node2Vec graph embeddings (co-sponsorship network). |

---

## Quality & docs

| Target | Description |
|--------|-------------|
| `make test` | Run pytest (`PYTHONPATH=src`). |
| `make lint` | Ruff check + format check. |
| `make lint-fix` | Ruff auto-fix + format. |
| `make docs` | Build MkDocs site to `site/`. |
| `make docs-serve` | Serve docs at http://127.0.0.1:8001. |

---

## Utilities

| Target | Description |
|--------|-------------|
| `make snapshot-mocks` | Sample `cache/` into `mocks/dev/` (subset of members, bills, votes, etc.). Run after scrape; commit result to refresh dev seed. |
| `make seed-outreach` | Seed outreach DB: backlog for funky_mama11@gmail.com; in dev only, mock advocates for heat-pill demo. Use same profile as the app. |
| `make db-migrate` | Run Alembic migrations to head. For existing DBs created before Alembic, run once: `alembic stamp head`. |
| `make logs` | Show unified run log (scrape, ml_run, startup). Optional `N=50` for `--tail 50`. |
| `make clean` | Remove `cache/`, `processed/*.parquet`, vault output, `site/`, run logs. |

---

## Help

```bash
make help
```

Prints all targets and their short descriptions.
