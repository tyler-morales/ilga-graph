#!/usr/bin/env python3
"""Deactivate all outreach campaigns so the 500 poll campaign banner is active.

Uses the same ILGA_DB_PATH / ILGA_PROFILE as the app. Run from repo root:

    PYTHONPATH=src python scripts/deactivate_all_campaigns.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


async def _main() -> None:
    from ilga_graph import config as cfg
    from ilga_graph.campaign_helpers import deactivate_all_campaigns
    from ilga_graph.db import DB_PATH, async_session_factory, init_db

    await init_db()
    print(f"Database: {DB_PATH} (ILGA_PROFILE={cfg.PROFILE})")

    async with async_session_factory() as session:
        n = await deactivate_all_campaigns(session)
        await session.commit()
    print(f"Deactivated {n} outreach campaign(s). The 500 poll campaign banner will now show.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
