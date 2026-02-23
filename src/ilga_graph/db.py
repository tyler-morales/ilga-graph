"""Async SQLite database engine and session factory.

Uses SQLAlchemy async with aiosqlite driver.  The DB file lives at
``data/ilga.db`` (configurable via ``ILGA_DB_PATH`` env var).

Call ``init_db()`` once during app lifespan to run migrations (Alembic),
then use ``get_db()`` as a FastAPI dependency to get an ``AsyncSession``.

Migrations run in a subprocess to avoid in-process Alembic (logging takeover
or exit behavior that can kill the server). Set ``ILGA_SKIP_MIGRATIONS=1`` to
skip migrations on startup (run ``alembic upgrade head`` manually first).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from . import config as cfg

LOGGER = logging.getLogger(__name__)

# Resolve DB path against project root so it's consistent under uvicorn --app-dir src
_project_root = Path(__file__).resolve().parent.parent.parent
_db_path_cfg = Path(cfg._env("ILGA_DB_PATH", "data/ilga.db"))
DB_PATH: Path = _project_root / _db_path_cfg if not _db_path_cfg.is_absolute() else _db_path_cfg

_engine = create_async_engine(
    f"sqlite+aiosqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)

async_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


def _alembic_root() -> Path | None:
    """Return project root (where alembic.ini lives), or None."""
    root = DB_PATH.resolve().parent
    while root != root.parent and not (root / "alembic.ini").exists():
        root = root.parent
    return root if (root / "alembic.ini").exists() else None


def _run_alembic_upgrade() -> bool:
    """Run Alembic migrations to head via subprocess to avoid in-process exit/logging issues."""
    if os.environ.get("ILGA_SKIP_MIGRATIONS") == "1":
        LOGGER.info("Skipping migrations (ILGA_SKIP_MIGRATIONS=1)")
        return False
    root = _alembic_root()
    if root is None:
        return False
    alembic_ini = root / "alembic.ini"
    env = os.environ.copy()
    env["ILGA_DB_PATH"] = str(DB_PATH.resolve())
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(alembic_ini), "upgrade", "head"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        msg = f"Alembic upgrade exited {result.returncode}"
        if err:
            msg += f"\nstderr:\n{err}"
        if out:
            msg += f"\nstdout:\n{out}"
        LOGGER.error("%s", msg)
        raise RuntimeError(msg)
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
