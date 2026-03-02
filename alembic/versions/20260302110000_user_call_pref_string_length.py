"""Widen users.call_pref for decision-tree values (call_only, elevator).

Revision ID: 20260302110000
Revises: 20260302100000
Create Date: 2026-03-02

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260302110000"
down_revision: str | None = "20260302100000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.alter_column(
            "call_pref",
            existing_type=sa.String(length=8),
            type_=sa.String(length=16),
            existing_nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.alter_column(
            "call_pref",
            existing_type=sa.String(length=16),
            type_=sa.String(length=8),
            existing_nullable=True,
        )
