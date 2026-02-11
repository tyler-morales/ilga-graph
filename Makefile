.PHONY: install dev dev-full run scrape scrape-200 scrape-full scrape-dev scrape-incremental scrape-votes export seed test lint lint-fix hooks hooks-run clean help
.PHONY: startup-report

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
# Pipeline: scrape (once) → scrape-votes (incremental) → serve.
#   make scrape        → 300 SB + 300 HB (prod-style)
#   make scrape-full   → all ~9600+ bills (full index; slow)
#   make scrape-votes  → next 10 bills for votes + slips (resumable)
#   make dev / run     → serve from cache (LOAD_ONLY=1)
# ═══════════════════════════════════════════════════════════════════════════════

scrape: ## Prod-style: all members, 300 SB + 300 HB, export
	$(PYTHON) scripts/scrape.py --sb-limit 300 --hb-limit 300 --export

scrape-200: ## Test pagination: 200 SB + 200 HB (2 range pages per type), export
	$(PYTHON) scripts/scrape.py --sb-limit 200 --hb-limit 200 --export

scrape-full: ## Full bill index: all ~9600+ bills (slow; complete data)
	$(PYTHON) scripts/scrape.py --sb-limit 0 --hb-limit 0 --export

scrape-dev: ## Light dev: 20/chamber, 100 SB + 100 HB, fast, export
	$(PYTHON) scripts/scrape.py --limit 20 --sb-limit 100 --hb-limit 100 --fast --export

scrape-incremental: ## Incremental: only new/changed bills (keeps existing cache)
	$(PYTHON) scripts/scrape.py --incremental --fast --export

scrape-votes: ## Incremental: scrape votes + witness slips for next N bills (default 10)
	$(PYTHON) scripts/scrape_votes.py --limit $(or $(LIMIT),10) --fast

scrape-votes-sample: ## Sample strategy: scrape every 10th bill first (10% sample, ~75min)
	$(PYTHON) scripts/scrape_votes.py --sample 10 --limit 0 --fast

scrape-votes-gap-fill: ## Gap-fill: complete remaining bills after sample (run without --sample)
	$(PYTHON) scripts/scrape_votes.py --limit 0 --fast

dev: ## Serve from cache in dev mode (run 'make scrape-dev' or 'make scrape' first)
	ILGA_LOAD_ONLY=1 ILGA_PROFILE=dev $(BIN)uvicorn ilga_graph.main:app --reload --app-dir src

dev-full: ## Serve full cache in dev shell (no dev caps, full ZIP crosswalk)
	ILGA_LOAD_ONLY=1 ILGA_PROFILE=dev ILGA_DEV_MODE=0 ILGA_SEED_MODE=0 $(BIN)uvicorn ilga_graph.main:app --reload --app-dir src

run: ## Serve from cache in prod mode (run 'make scrape' first)
	ILGA_LOAD_ONLY=1 ILGA_PROFILE=prod $(BIN)uvicorn ilga_graph.main:app --reload --app-dir src

# ── Other ────────────────────────────────────────────────────────────────────

install: ## Install project with dev dependencies
	$(BIN)pip install -e ".[dev]"

export: ## Re-export vault from cache only (no scrape)
	$(PYTHON) scripts/scrape.py --export-only --fast

seed: ## Regenerate mocks/dev/ from current cache/
	$(PYTHON) scripts/generate_seed.py

test: ## Run pytest
	PYTHONPATH=src $(BIN)pytest

lint: ## Run ruff check + format check
	$(BIN)ruff check .
	$(BIN)ruff format --check .

lint-fix: ## Auto-fix lint and format
	$(BIN)ruff check --fix .
	$(BIN)ruff format .

hooks: ## Install git pre-commit hooks (ruff check + format on commit)
	$(BIN)pre-commit install

hooks-run: ## Run pre-commit checks on all files
	$(BIN)pre-commit run --all-files

clean: ## Remove cache/ and generated vault files
	rm -rf cache/
	rm -rf ILGA_Graph_Vault/Bills/ ILGA_Graph_Vault/Committees/ ILGA_Graph_Vault/Members/
	rm -f ILGA_Graph_Vault/*.base
	rm -f ILGA_Graph_Vault/Moneyball\ Report.md
	rm -f .startup_timings.csv
	@echo "Cleaned. Run 'make scrape' or 'make scrape-dev' then 'make dev' or 'make run'."

startup-report: ## Pretty startup timing report from .startup_timings.csv
	$(PYTHON) scripts/startup_timings_report.py
