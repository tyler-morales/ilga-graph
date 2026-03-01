"""Add kei_interest_statements table (text-only marquee submissions from non-owners).

Revision ID: 20260228140000
Revises: 20260228130000
Create Date: 2026-02-28

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260228140000"
down_revision: str | None = "20260228130000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "kei_interest_statements",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("location", sa.String(length=100), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("consent", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("admin_message", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_kei_interest_statements_user_id"),
        "kei_interest_statements",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_kei_interest_statements_status"),
        "kei_interest_statements",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_kei_interest_statements_status"),
        table_name="kei_interest_statements",
    )
    op.drop_index(
        op.f("ix_kei_interest_statements_user_id"),
        table_name="kei_interest_statements",
    )
    op.drop_table("kei_interest_statements")
