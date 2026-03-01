#!/usr/bin/env python3
"""Set ILGA_DB_PATH (and optionally ILGA_CAMPAIGN_CONFIG) for a tenant, then run app or migrations.

Usage:
  python -m scripts.provision_tenant TENANT_ID [dev|prod] [run|migrate]

Examples:
  python -m scripts.provision_tenant tenants_union prod run
  python -m scripts.provision_tenant kei prod migrate

Sets:
  ILGA_DB_PATH=data/{tenant_id}_{env}.db
  ILGA_PROFILE={env}
  ILGA_CAMPAIGN_CONFIG=config/{tenant_id}.json (if that file exists)

Then runs:
  run     → uvicorn ilga_graph.main:app (default)
  migrate → alembic upgrade head
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    tenant_id = sys.argv[1].strip().lower().replace(" ", "_")
    env = (sys.argv[2].strip().lower() if len(sys.argv) > 2 else "dev") or "dev"
    if env not in ("dev", "prod"):
        env = "dev"
    action = (sys.argv[3].strip().lower() if len(sys.argv) > 3 else "run") or "run"
    if action not in ("run", "migrate"):
        action = "run"

    db_path = f"data/{tenant_id}_{env}.db"
    os.environ["ILGA_DB_PATH"] = db_path
    os.environ["ILGA_PROFILE"] = env

    config_path = ROOT / "config" / f"{tenant_id}.json"
    if config_path.exists():
        os.environ["ILGA_CAMPAIGN_CONFIG"] = str(config_path)

    print(f"ILGA_DB_PATH={db_path} ILGA_PROFILE={env}", file=sys.stderr)
    if config_path.exists():
        print(f"ILGA_CAMPAIGN_CONFIG={config_path}", file=sys.stderr)

    if action == "migrate":
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT,
            env=os.environ,
        )
        return result.returncode
    # run
    import uvicorn

    uvicorn.run("ilga_graph.main:app", host="0.0.0.0", port=8000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
