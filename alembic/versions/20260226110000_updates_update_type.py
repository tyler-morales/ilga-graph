"""Add update_type to updates table.

Revision ID: 20260226110000
Revises: 20260226100000
Create Date: 2026-02-26

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260226110000"
down_revision: str | None = "20260226100000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "updates",
        sa.Column(
            "update_type",
            sa.String(length=32),
            nullable=False,
            server_default="other",
        ),
    )


def downgrade() -> None:
    op.drop_column("updates", "update_type")
