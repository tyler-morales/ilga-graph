"""Add kei_poll_responses table.

Revision ID: 20260228110000
Revises: 20260228100000
Create Date: 2026-02-28

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260228110000"
down_revision: str | None = "20260228100000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "kei_poll_responses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("kei_status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_kei_poll_responses_user_id"), "kei_poll_responses", ["user_id"])
    op.create_index(op.f("ix_kei_poll_responses_session_id"), "kei_poll_responses", ["session_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_kei_poll_responses_session_id"), table_name="kei_poll_responses")
    op.drop_index(op.f("ix_kei_poll_responses_user_id"), table_name="kei_poll_responses")
    op.drop_table("kei_poll_responses")
