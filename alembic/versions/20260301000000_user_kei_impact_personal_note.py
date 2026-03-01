"""Add kei_impact_slug and kei_personal_note to users.

Revision ID: 20260301000000
Revises: 20260228150000
Create Date: 2026-03-01

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260301000000"
down_revision: str | None = "20260228150000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("kei_impact_slug", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("kei_personal_note", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "kei_personal_note")
    op.drop_column("users", "kei_impact_slug")
