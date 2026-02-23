"""Add outreach_step_events for funnel/checkpoint tracking.

Revision ID: 20260223100000
Revises: 20260223000000
Create Date: 2026-02-23

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260223100000"
down_revision: str | None = "20260223000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "outreach_step_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("member_id", sa.String(length=32), nullable=False),
        sa.Column("outreach_type", sa.String(length=16), nullable=False),
        sa.Column("step_slug", sa.String(length=64), nullable=False),
        sa.Column("reached_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_outreach_step_events_member_id"),
        "outreach_step_events",
        ["member_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_outreach_step_events_outreach_type"),
        "outreach_step_events",
        ["outreach_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_outreach_step_events_step_slug"),
        "outreach_step_events",
        ["step_slug"],
        unique=False,
    )
    op.create_index(
        op.f("ix_outreach_step_events_user_id"),
        "outreach_step_events",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_outreach_step_events_type_slug_time",
        "outreach_step_events",
        ["outreach_type", "step_slug", "reached_at"],
        unique=False,
    )
    op.create_index(
        "ix_outreach_step_events_user_member_type",
        "outreach_step_events",
        ["user_id", "member_id", "outreach_type"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_outreach_step_events_user_member_type",
        table_name="outreach_step_events",
    )
    op.drop_index(
        "ix_outreach_step_events_type_slug_time",
        table_name="outreach_step_events",
    )
    op.drop_index(
        op.f("ix_outreach_step_events_user_id"),
        table_name="outreach_step_events",
    )
    op.drop_index(
        op.f("ix_outreach_step_events_step_slug"),
        table_name="outreach_step_events",
    )
    op.drop_index(
        op.f("ix_outreach_step_events_outreach_type"),
        table_name="outreach_step_events",
    )
    op.drop_index(
        op.f("ix_outreach_step_events_member_id"),
        table_name="outreach_step_events",
    )
    op.drop_table("outreach_step_events")
