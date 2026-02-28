"""Add kei_status and welcome_email_sent_at to users.

Revision ID: 20260228100000
Revises: 20260227100000
Create Date: 2026-02-28

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260228100000"
down_revision: str | None = "20260227100000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("kei_status", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("welcome_email_sent_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "welcome_email_sent_at")
    op.drop_column("users", "kei_status")
