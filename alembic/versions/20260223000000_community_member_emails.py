"""Add community_member_emails for community-sourced legislator emails.

Revision ID: 20260223000000
Revises: 20260221100000
Create Date: 2026-02-23

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260223000000"
down_revision: str | None = "20260221100000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "community_member_emails",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("member_id", sa.String(length=32), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_community_member_emails_member_id"),
        "community_member_emails",
        ["member_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_community_member_emails_member_user_email"),
        "community_member_emails",
        ["member_id", "email", "user_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_community_member_emails_user_id"),
        "community_member_emails",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_community_member_emails_user_id"),
        table_name="community_member_emails",
    )
    op.drop_index(
        op.f("ix_community_member_emails_member_user_email"),
        table_name="community_member_emails",
    )
    op.drop_index(
        op.f("ix_community_member_emails_member_id"),
        table_name="community_member_emails",
    )
    op.drop_table("community_member_emails")
