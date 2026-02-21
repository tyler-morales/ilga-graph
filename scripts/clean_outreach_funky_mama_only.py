#!/usr/bin/env python3
"""One-time cleanup: remove all outreach_events not from funky_mama11@gmail.com.

Leaves only outreach data you know you made (dev fresh start). Uses the same
ILGA_DB_PATH / ILGA_PROFILE as the app. Run from repo root:

    PYTHONPATH=src python scripts/clean_outreach_funky_mama_only.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")


async def _main() -> None:
    from sqlalchemy import delete

    from ilga_graph import config as cfg
    from ilga_graph.db import DB_PATH, async_session_factory, init_db
    from ilga_graph.db_models import OutreachEvent

    await init_db()
    print(f"Database: {DB_PATH} (ILGA_PROFILE={cfg.PROFILE})")

    keep_email = "funky_mama11@gmail.com"
    async with async_session_factory() as session:
        result = await session.execute(
            delete(OutreachEvent).where(OutreachEvent.user_email != keep_email)
        )
        removed = result.rowcount
        await session.commit()
    print(f"Removed {removed} outreach events (kept only {keep_email}).")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
