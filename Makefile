.PHONY: install dev run scrape scrape-incremental seed export test lint clean help

# ── Virtual environment auto-detection ────────────────────────────────────────
# Searches common venv locations. Override with: make dev VENV=path/to/venv
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
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the project with dev dependencies
	$(BIN)pip install -e ".[dev]"

dev: ## Start server in dev mode with mocks/dev (instant, no scraping)
	ILGA_DEV_MODE=1 ILGA_SEED_MODE=1 $(BIN)uvicorn ilga_graph.main:app --reload --app-dir src

run: ## Start server using cache/ (no mock fallback)
	ILGA_DEV_MODE=0 ILGA_SEED_MODE=0 $(BIN)uvicorn ilga_graph.main:app --reload --app-dir src

scrape: ## Full scrape to cache/ (no server)
	$(PYTHON) scripts/scrape.py --export

scrape-fast: ## Dev scrape (20/chamber, fast delay) to cache/ with export
	$(PYTHON) scripts/scrape.py --limit 20 --fast --export --bill-limit 100

scrape-incremental: ## Incremental scrape (only new/changed bills)
	$(PYTHON) scripts/scrape.py --incremental --fast --export

seed: ## Regenerate mocks/dev/ from current cache/
	$(PYTHON) scripts/generate_seed.py

export: ## Re-export vault from cached data (no scraping, no server)
	$(PYTHON) scripts/scrape.py --export-only --fast

test: ## Run pytest
	PYTHONPATH=src $(BIN)pytest

lint: ## Run ruff linter and formatter check
	$(BIN)ruff check .
	$(BIN)ruff format --check .

lint-fix: ## Auto-fix lint issues
	$(BIN)ruff check --fix .
	$(BIN)ruff format .

clean: ## Remove cache/ and generated vault files
	rm -rf cache/
	rm -rf ILGA_Graph_Vault/Bills/ ILGA_Graph_Vault/Committees/ ILGA_Graph_Vault/Members/
	rm -f ILGA_Graph_Vault/*.base
	rm -f ILGA_Graph_Vault/Moneyball\ Report.md
	rm -f .startup_timings.csv
	@echo "Cleaned. Run 'make dev' or 'make scrape' to regenerate."
