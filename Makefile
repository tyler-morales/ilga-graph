.PHONY: scrape dev serve install test lint lint-fix clean help ml-setup ml-run ml-pipeline ml-resolve ml-predict

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
# One pipeline: make scrape → make dev
#
#   make scrape              Smart tiered scan (~2 min daily, auto-decides)
#   make scrape FULL=1       Force full index walk (all 125 pages, ~30 min)
#   make scrape FRESH=1      Nuke cache and re-scrape from scratch
#   make scrape LIMIT=100    Limit vote/slip phase to 100 bills
#   make scrape WORKERS=10   More parallel workers for votes/slips
#   make scrape SKIP_VOTES=1 Skip vote/slip phase
#   make scrape EXPORT=1     Include Obsidian vault export
#
#   make dev                 Serve from cache (dev mode, auto-reload)
#   make serve               Serve from cache (prod mode)
# ═══════════════════════════════════════════════════════════════════════════════

scrape: ## Smart incremental scrape (members + bills + votes + slips + ML)
	$(PYTHON) scripts/scrape.py \
		--fast \
		$(if $(FRESH),--fresh) \
		$(if $(FULL),--full) \
		$(if $(LIMIT),--vote-limit $(LIMIT)) \
		$(if $(WORKERS),--workers $(WORKERS)) \
		$(if $(EXPORT),--export) \
		$(if $(SKIP_VOTES),--skip-votes)
	@echo "Running ML pipeline..."
	PYTHONPATH=src $(PYTHON) scripts/ml_run.py || echo "ML pipeline skipped (run make ml-setup first)"

dev: ## Serve from cache (dev mode, auto-reload)
	ILGA_LOAD_ONLY=1 ILGA_PROFILE=dev $(BIN)uvicorn ilga_graph.main:app --reload --app-dir src

serve: ## Serve from cache (prod mode)
	ILGA_LOAD_ONLY=1 ILGA_PROFILE=prod $(BIN)uvicorn ilga_graph.main:app --app-dir src

# ── Utilities ──────────────────────────────────────────────────────────────────

install: ## Install project with dev dependencies
	$(BIN)pip install -e ".[dev]"

test: ## Run pytest
	PYTHONPATH=src $(BIN)pytest

lint: ## Run ruff check + format check
	$(BIN)ruff check .
	$(BIN)ruff format --check .

lint-fix: ## Auto-fix lint and format
	$(BIN)ruff check --fix .
	$(BIN)ruff format .

# ── ML Pipeline ───────────────────────────────────────────────────────────────

ml-setup: ## Install ML dependencies
	$(BIN)pip install -e ".[ml]"

ml-run: ## Run full ML pipeline (no interaction -- scores, coalitions, anomalies)
	PYTHONPATH=src $(PYTHON) scripts/ml_run.py

ml-pipeline: ## Run data pipeline only: cache/*.json -> processed/*.parquet
	PYTHONPATH=src $(PYTHON) scripts/ml_pipeline.py

ml-resolve: ## Entity resolution only (AUTO=1 for no interaction)
	PYTHONPATH=src $(PYTHON) scripts/ml_resolve.py $(if $(AUTO),--auto) $(if $(STATS),--stats)

ml-predict: ## Bill outcome prediction only
	PYTHONPATH=src $(PYTHON) scripts/ml_predict.py

# ── Utilities ──────────────────────────────────────────────────────────────────

clean: ## Remove cache/, processed/, and generated vault files
	rm -rf cache/
	rm -rf processed/*.parquet processed/*.pkl
	rm -rf ILGA_Graph_Vault/Bills/ ILGA_Graph_Vault/Committees/ ILGA_Graph_Vault/Members/
	rm -f ILGA_Graph_Vault/*.base
	rm -f ILGA_Graph_Vault/Moneyball\ Report.md
	rm -f .startup_timings.csv
	@echo "Cleaned. Run 'make scrape' then 'make dev'."
