.PHONY: scrape scrape-full dev serve dev-reset install test smoke-outreach lint lint-fix clean help ml-setup ml-run ml-pipeline ml-resolve ml-predict ml-embed scrape-fulltext scrape-members scrape-full-members snapshot-mocks logs docs docs-serve

# ── Virtual environment ─────────────────────────────────────────────────────
VENV ?= $(or $(wildcard .venv), $(wildcard venv), $(wildcard src/ilga_graph/.venv))
ifdef VENV
  PYTHON := $(VENV)/bin/python
  BIN    := $(VENV)/bin/
else
  PYTHON := python3
  BIN    :=
endif

# Default target
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ═══════════════════════════════════════════════════════════════════════════════
# One pipeline: make scrape → make dev / make serve
#
# Index strategy (bill list from ILGA):
#   make scrape              Smart/tiered: full walk if no cache or >7d old;
#                            else tail-only (~24h) or skip index (<24h). Then
#                            scrape metadata + votes + slips for new/recent bills.
#   make scrape FULL=1       Force FULL index walk every time (all ~125 pages),
#                            then scrape. Use for new session or "refresh all."
#   make scrape-full-members If you see "20 members" and want the full roster:
#                            re-fetches all members from ILGA (~177). Uses existing
#                            cache/bills.json; run after a normal scrape.
#
# Other flags:
#   make scrape FRESH=1      Nuke cache/ and re-scrape from scratch
#   make scrape FULLTEXT=1  Include full text PDFs in same pass
#   make scrape WORKERS=20  Parallel workers (default 10)
#   make scrape SKIP_VOTES=1 Metadata only (no votes/slips)
#   make scrape EXPORT=1    Include Obsidian vault export
#
#   make dev    Serve from cache (dev mode, auto-reload)
#   make serve  Serve from cache (prod mode)
# ═══════════════════════════════════════════════════════════════════════════════

scrape: ## Unified scrape: members + bills + votes + slips. Smart index (use FULL=1 to force full index walk) + ML
	ILGA_PROFILE=prod $(PYTHON) scripts/scrape.py \
		--fast \
		$(if $(FRESH),--fresh) \
		$(if $(FULL),--full) \
		$(if $(FULLTEXT),--fulltext) \
		$(if $(WORKERS),--workers $(WORKERS)) \
		$(if $(EXPORT),--export) \
		$(if $(SKIP_VOTES),--skip-votes)
	@echo "Running ML pipeline..."
	PYTHONPATH=src $(PYTHON) scripts/ml_run.py || echo "ML pipeline skipped (run make ml-setup first)"

scrape-full: ## Full reset: delete cache/, then scrape all members (~177) + full bill index + ML. Use when data is wrong or incomplete (e.g. only 20 or 60 members).
	$(MAKE) scrape FRESH=1 FULL=1

dev: ## Serve from cache (dev mode, auto-reload)
	ILGA_LOAD_ONLY=1 ILGA_PROFILE=dev $(BIN)uvicorn ilga_graph.main:app --reload --app-dir src

serve: ## Serve from cache (prod mode)
	ILGA_LOAD_ONLY=1 ILGA_PROFILE=prod $(BIN)uvicorn ilga_graph.main:app --app-dir src

dev-reset: ## Clear dev cache (next make dev uses mocks/dev seed data)
	rm -rf cache/dev
	@echo "Dev cache cleared. Next 'make dev' will use mocks/dev/ seed data."

# ── Utilities ──────────────────────────────────────────────────────────────────

install: ## Install project with dev dependencies
	$(BIN)pip install -e ".[dev]"

test: ## Run pytest
	PYTHONPATH=src $(BIN)pytest

smoke-outreach: ## Smoke test: auth + record call/email + visitor-visible stats (temp DB, no server)
	PYTHONPATH=src $(PYTHON) scripts/smoke_test_outreach.py

lint: ## Run ruff check + format check
	$(BIN)ruff check .
	$(BIN)ruff format --check .

lint-fix: ## Auto-fix lint and format
	$(BIN)ruff check --fix .
	$(BIN)ruff format .

# ── ML Pipeline ───────────────────────────────────────────────────────────────

ml-setup: ## Install ML dependencies
	$(BIN)pip install -e ".[ml]"

ml-run: ## Run full ML pipeline (no interaction -- scores, coalitions, anomalies). Use ILGA_ML_SKIP_TUNE=1 to skip hyperparameter tuning (faster).
	PYTHONPATH=src $(PYTHON) scripts/ml_run.py

ml-pipeline: ## Run data pipeline only: cache/*.json -> processed/*.parquet
	PYTHONPATH=src $(PYTHON) scripts/ml_pipeline.py

ml-resolve: ## Entity resolution only (AUTO=1 for no interaction)
	PYTHONPATH=src $(PYTHON) scripts/ml_resolve.py $(if $(AUTO),--auto) $(if $(STATS),--stats)

ml-predict: ## Bill outcome prediction only
	PYTHONPATH=src $(PYTHON) scripts/ml_predict.py

ml-embed: ## Generate Node2Vec graph embeddings (co-sponsorship network)
	PYTHONPATH=src $(PYTHON) -c "from ilga_graph.ml.node_embedder import run_embedding_pipeline; run_embedding_pipeline()"

scrape-members: ## Only fetch members + committees; load bills from cache (no bill/vote scrape). Use after deleting members.json.
	ILGA_PROFILE=prod $(PYTHON) scripts/scrape.py --members-only --fast

scrape-full-members: ## Re-fetch full member roster from ILGA (~177). Removes cache/members.json then runs scrape-members so the scraper does not reuse a small cached roster (e.g. 20).
	rm -f cache/members.json
	$(MAKE) scrape-members

refresh-photos: ## Refresh member photo_url from ILGA detail pages only (requires existing cache/members.json and bills.json).
	ILGA_PROFILE=prod PYTHONPATH=src $(PYTHON) scripts/refresh_member_photos.py

scrape-fulltext: ## Scrape full bill text PDFs (incremental, resumable)
	ILGA_PROFILE=prod PYTHONPATH=src $(PYTHON) scripts/scrape_fulltext.py \
		$(if $(LIMIT),--limit $(LIMIT),--limit 100) \
		$(if $(WORKERS),--workers $(WORKERS)) \
		$(if $(FAST),--fast) \
		$(if $(DELAY),--delay $(DELAY)) \
		$(if $(SAVE_INTERVAL),--save-interval $(SAVE_INTERVAL))

snapshot-mocks: ## Sample cache/ into mocks/dev/ (run after scrape; commit result to refresh dev seed)
	$(PYTHON) scripts/snapshot_mocks.py

# ── Utilities ──────────────────────────────────────────────────────────────────

logs: ## Show unified run log (scrape, ml_run, startup) — terminal dashboard
	PYTHONPATH=src $(PYTHON) scripts/log_dashboard.py $(if $(N),--tail $(N),--tail 20)

# ── Documentation site (MkDocs) ─────────────────────────────────────────────

docs: ## Build the documentation site to site/
	$(BIN)mkdocs build

docs-serve: ## Serve the documentation site at http://127.0.0.1:8001 (port 8001 to avoid clashing with make dev on 8000)
	$(BIN)mkdocs serve -a 127.0.0.1:8001

clean: ## Remove cache/, processed/, and generated vault files
	rm -rf cache/
	rm -rf processed/*.parquet processed/*.pkl
	rm -rf ILGA_Graph_Vault/Bills/ ILGA_Graph_Vault/Committees/ ILGA_Graph_Vault/Members/
	rm -f ILGA_Graph_Vault/*.base
	rm -f ILGA_Graph_Vault/Moneyball\ Report.md
	rm -f .startup_timings.csv
	rm -f .run_log.jsonl
	rm -rf site/
	@echo "Cleaned. Run 'make scrape' then 'make dev'."

seed-outreach: ## Seed outreach DB: real backlog always; mock community data only when ILGA_PROFILE=dev. Dev uses data/ilga_dev.db, prod uses data/ilga.db.
	$(PYTHON) scripts/seed_outreach.py
