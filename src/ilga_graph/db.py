"""Async SQLite database engine and session factory.

Uses SQLAlchemy async with aiosqlite driver.  The DB file lives at
``data/ilga.db`` (configurable via ``ILGA_DB_PATH`` env var).

Call ``init_db()`` once during app lifespan to run migrations (Alembic),
then use ``get_db()`` as a FastAPI dependency to get an ``AsyncSession``.
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


def _run_alembic_upgrade() -> bool:
    """Run Alembic migrations to head if available. Returns True if ran, False to use fallback."""
    try:
        from alembic.config import Config

        from alembic import command
    except ImportError:
        return False
    db_path_abs = DB_PATH.resolve()
    root = db_path_abs.parent
    while root != root.parent and not (root / "alembic.ini").exists():
        root = root.parent
    if not (root / "alembic.ini").exists():
        return False
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path_abs}")
    command.upgrade(config, "head")
    return True


def _is_duplicate_column_error(exc: BaseException) -> bool:
    """True if the error is SQLite 'duplicate column name' (safe to ignore)."""
    msg = str(exc).lower()
    return "duplicate column name" in msg or "already exists" in msg


async def _init_db_fallback() -> None:
    """Fallback when Alembic not available (e.g. temp DB in tests): create_all + ALTERs."""
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from .db_models import Base

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for col_sql in (
            "ALTER TABLE users ADD COLUMN zip_code VARCHAR(10)",
            "ALTER TABLE users ADD COLUMN wants_updates BOOLEAN DEFAULT 1",
            "ALTER TABLE outreach_events ADD COLUMN contact_name VARCHAR(128)",
            "ALTER TABLE outreach_events ADD COLUMN support_score INTEGER",
            "ALTER TABLE outreach_events ADD COLUMN constituent BOOLEAN",
            "ALTER TABLE bug_reports ADD COLUMN attachment_paths TEXT",
        ):
            try:
                await conn.execute(text(col_sql))
            except OperationalError as e:
                if not _is_duplicate_column_error(e):
                    raise
    LOGGER.debug("Database fallback init (create_all + ALTERs) at %s", DB_PATH)


async def init_db() -> None:
    """Run Alembic migrations to head; fallback to create_all + ALTERs if Alembic unavailable."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    import asyncio

    loop = asyncio.get_event_loop()
    ran = await loop.run_in_executor(None, _run_alembic_upgrade)
    if not ran:
        await _init_db_fallback()
    LOGGER.info("Database ready at %s", DB_PATH)


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency — yields an ``AsyncSession``, auto-commits on success."""
    async with async_session_factory() as session:
        yield session  # type: ignore[misc]
