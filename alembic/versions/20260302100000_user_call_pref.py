"""Add call_pref to users (advocacy: I'll call vs email only).

Revision ID: 20260302100000
Revises: 20260301100000
Create Date: 2026-03-02

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260302100000"
down_revision: str | None = "20260301100000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("call_pref", sa.String(length=8), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "call_pref")
