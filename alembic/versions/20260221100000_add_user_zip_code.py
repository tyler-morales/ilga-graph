"""Add zip_code to users for pre-fill and remembered district.

Revision ID: 20260221100000
Revises: 20260221000000
Create Date: 2026-02-21

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260221100000"
down_revision: str | None = "20260221000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("zip_code", sa.String(length=10), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "zip_code")
