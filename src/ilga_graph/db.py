"""Async SQLite database engine and session factory.

Uses SQLAlchemy async with aiosqlite driver.  The DB file lives at
``data/ilga.db`` (configurable via ``ILGA_DB_PATH`` env var).

Call ``init_db()`` once during app lifespan to create tables, then use
``get_db()`` as a FastAPI dependency to get an ``AsyncSession``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from . import config as cfg

LOGGER = logging.getLogger(__name__)

DB_PATH: Path = Path(cfg._env("ILGA_DB_PATH", "data/ilga.db"))

_engine = create_async_engine(
    f"sqlite+aiosqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)

async_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def init_db() -> None:
    """Create all tables if they don't exist.  Safe to call on every startup."""
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from .db_models import Base

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Add columns to outreach_events if table exists but columns are missing (existing DBs)
        for col_sql in (
            "ALTER TABLE outreach_events ADD COLUMN contact_name VARCHAR(128)",
            "ALTER TABLE outreach_events ADD COLUMN support_score INTEGER",
            "ALTER TABLE outreach_events ADD COLUMN constituent BOOLEAN",
        ):
            try:
                await conn.execute(text(col_sql))
            except OperationalError:
                pass
    LOGGER.info("Database ready at %s", DB_PATH)


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency — yields an ``AsyncSession``, auto-commits on success."""
    async with async_session_factory() as session:
        yield session  # type: ignore[misc]
