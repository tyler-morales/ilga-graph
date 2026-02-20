#!/usr/bin/env bash
# Start uvicorn for production or PaaS. Uses PORT if set (Railway, Render), else 8000.
set -e
cd "$(dirname "$0")/.."
exec uvicorn ilga_graph.main:app --app-dir src --host 0.0.0.0 --port "${PORT:-8000}"
