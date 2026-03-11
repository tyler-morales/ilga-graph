#!/usr/bin/env bash
# Run on the Vultr (or any) server to perform incremental scrape and restart the app.
# Usage: from project root, ./scripts/scrape-on-server.sh
# Cron: 0 3 * * * /home/USER/ilga-graph/scripts/scrape-on-server.sh >> /home/USER/ilga-graph/logs/scrape.log 2>&1
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Use project venv
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "No .venv found in $ROOT" >&2
  exit 1
fi

export ILGA_PROFILE=prod

# Incremental scrape (no --fresh): members + bills + votes + slips
python scripts/scrape.py --fast || true

# Optional: run ML pipeline so intelligence features stay current
if command -v make >/dev/null 2>&1; then
  (PYTHONPATH=src make ml-run 2>/dev/null) || true
fi

# Reload app so it reads updated cache (requires passwordless sudo for this command)
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl restart ilga-graph 2>/dev/null || echo "Could not restart ilga-graph (sudo systemctl restart ilga-graph)" >&2
fi
