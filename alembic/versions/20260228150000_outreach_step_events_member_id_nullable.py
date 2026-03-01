"""Make outreach_step_events.member_id nullable for WYC funnel (no legislator).

Revision ID: 20260228150000
Revises: 20260228140000
Create Date: 2026-02-28

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260228150000"
down_revision: str | None = "20260228140000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("outreach_step_events") as batch_op:
        batch_op.alter_column(
            "member_id",
            existing_type=sa.String(length=32),
            nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("outreach_step_events") as batch_op:
        batch_op.alter_column(
            "member_id",
            existing_type=sa.String(length=32),
            nullable=False,
        )
