"""Add session_milestone_id to campaigns.

Revision ID: 20260227100000
Revises: 20260227000000
Create Date: 2026-02-27

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260227100000"
down_revision: str | None = "20260227000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "campaigns",
        sa.Column("session_milestone_id", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("campaigns", "session_milestone_id")
