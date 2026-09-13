"""Add money_intel_leads table for lobbyist / buyer waitlist.

Revision ID: 20260913140000
Revises: 20260913100000
Create Date: 2026-09-13

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260913140000"
down_revision: str | None = "20260913100000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "money_intel_leads",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=True),
        sa.Column("org", sa.String(length=200), nullable=True),
        sa.Column("role", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("email", name="uq_money_intel_leads_email"),
    )
    op.create_index(
        "ix_money_intel_leads_created_at",
        "money_intel_leads",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_money_intel_leads_created_at", table_name="money_intel_leads")
    op.drop_table("money_intel_leads")
