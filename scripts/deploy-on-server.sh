#!/usr/bin/env bash
# Run from repo root (e.g. cd ~/ilga-graph && bash scripts/deploy-on-server.sh).
# Used by CI deploy job and for manual deploys: pull, install deps, restart service.
set -e

git pull origin main
source .venv/bin/activate
pip install -e . --quiet
sudo systemctl restart ilga-graph
