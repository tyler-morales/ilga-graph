"""Add session_id to outreach_step_events for anonymous funnel tracking.

Revision ID: 20260223110000
Revises: 20260223100000
Create Date: 2026-02-23

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260223110000"
down_revision: str | None = "20260223100000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "outreach_step_events",
        sa.Column("session_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        op.f("ix_outreach_step_events_session_id"),
        "outreach_step_events",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "ix_outreach_step_events_session_type_time",
        "outreach_step_events",
        ["session_id", "outreach_type", "reached_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_outreach_step_events_session_type_time",
        table_name="outreach_step_events",
    )
    op.drop_index(
        op.f("ix_outreach_step_events_session_id"),
        table_name="outreach_step_events",
    )
    op.drop_column("outreach_step_events", "session_id")
