"""Tests for auth + outreach SQLite DB: init_db, migrations, schema.

Uses a temp DB path so the real data/ilga.db is never touched.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Import after patching env so DB_PATH is the test path
import ilga_graph.config as cfg_mod
import ilga_graph.db as db_mod


async def _init_db_under_env(db_path: Path) -> None:
    with patch.dict(os.environ, {"ILGA_DB_PATH": str(db_path)}, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        await db_mod.init_db()


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_ilga.db"


@pytest.fixture
def patched_db(test_db_path: Path):
    """Set ILGA_DB_PATH to temp file and reload db module."""
    with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        yield test_db_path
    # Teardown: restore so other tests aren't affected
    if "ILGA_DB_PATH" in os.environ:
        del os.environ["ILGA_DB_PATH"]
    importlib.reload(cfg_mod)
    importlib.reload(db_mod)


class TestInitDb:
    """init_db() creates tables and is idempotent."""

    def test_creates_tables(self, test_db_path: Path) -> None:
        _run(_init_db_under_env(test_db_path))
        assert test_db_path.exists()

        async def check():
            from sqlalchemy import text

            async with db_mod._engine.connect() as conn:
                r = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = {row[0] for row in r}
            assert "users" in tables
            assert "auth_codes" in tables
            assert "outreach_events" in tables

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            _run(check())

    def test_idempotent_no_raise(self, test_db_path: Path) -> None:
        _run(_init_db_under_env(test_db_path))
        # Second run must not raise (migrations may hit "column already exists")
        _run(_init_db_under_env(test_db_path))


class TestOutreachEventsSchema:
    """outreach_events has columns needed for aggregation."""

    def test_has_expected_columns(self, test_db_path: Path) -> None:
        _run(_init_db_under_env(test_db_path))

        async def check():
            from sqlalchemy import text

            async with db_mod._engine.connect() as conn:
                r = await conn.execute(
                    text("PRAGMA table_info(outreach_events)"),
                )
                rows = r.fetchall()
            # sqlite PRAGMA returns (cid, name, type, notnull, dflt_value, pk)
            names = {row[1] for row in rows}
            assert "member_id" in names
            assert "kind" in names
            assert "zip_code" in names
            assert "notes" in names
            assert "contact_name" in names
            assert "support_score" in names
            assert "constituent" in names
            assert "created_at" in names
            assert "user_id" in names
            assert "user_email" in names
            assert "outcome" in names

        with patch.dict(os.environ, {"ILGA_DB_PATH": str(test_db_path)}, clear=False):
            importlib.reload(cfg_mod)
            importlib.reload(db_mod)
            _run(check())


class TestSessionPersistence:
    """Session factory commits and data is visible in a new session."""

    def test_session_commits_and_read_back(self, patched_db: Path) -> None:
        _run(_init_db_under_env(patched_db))

        async def insert_then_read():
            from sqlalchemy import select

            from ilga_graph.db_models import User

            async with db_mod.async_session_factory() as session:
                user = User(email="test_session@example.com")
                session.add(user)
                await session.commit()
            async with db_mod.async_session_factory() as session:
                r = await session.execute(select(User).where(User.email == "test_session@example.com"))
                u = r.scalar_one_or_none()
                assert u is not None
                assert u.email == "test_session@example.com"

        _run(insert_then_read())
