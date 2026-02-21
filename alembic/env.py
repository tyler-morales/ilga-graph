"""Alembic env: use ILGA_DB_PATH / ILGA_PROFILE for DB URL (same as app)."""

from __future__ import annotations

import sys
from pathlib import Path

# Project root and src on path so we can import ilga_graph
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config
if config.config_file_name is not None:
    from logging.config import fileConfig

    fileConfig(config.config_file_name)

from ilga_graph.config import _env
from ilga_graph.db_models import Base

# Same path as app (profile default or ILGA_DB_PATH)
_db_path = Path(_env("ILGA_DB_PATH", "data/ilga.db"))
if not _db_path.is_absolute():
    _db_path = (ROOT / _db_path).resolve()
config.set_main_option("sqlalchemy.url", f"sqlite:///{_db_path}")

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
