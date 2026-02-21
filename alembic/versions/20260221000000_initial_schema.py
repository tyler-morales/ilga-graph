"""Initial schema: users, auth_codes, outreach_events, bug_reports.

Revision ID: 20260221000000
Revises:
Create Date: 2026-02-21

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260221000000"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )

    op.create_table(
        "auth_codes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("code_hash", sa.String(length=128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_auth_codes_email"), "auth_codes", ["email"], unique=False)

    op.create_table(
        "outreach_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("user_email", sa.String(length=320), nullable=True),
        sa.Column("member_id", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("zip_code", sa.String(length=10), nullable=True),
        sa.Column("outcome", sa.String(length=64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("contact_name", sa.String(length=128), nullable=True),
        sa.Column("support_score", sa.Integer(), nullable=True),
        sa.Column("constituent", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_outreach_events_member_id"), "outreach_events", ["member_id"], unique=False
    )
    op.create_index(
        op.f("ix_outreach_events_user_id"), "outreach_events", ["user_id"], unique=False
    )
    op.create_index(
        "ix_outreach_member_kind", "outreach_events", ["member_id", "kind"], unique=False
    )

    op.create_table(
        "bug_reports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("reporter_email", sa.String(length=320), nullable=True),
        sa.Column("page_url", sa.String(length=2048), nullable=True),
        sa.Column("attachment_paths", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("bug_reports")
    op.drop_index("ix_outreach_member_kind", table_name="outreach_events")
    op.drop_index(op.f("ix_outreach_events_user_id"), table_name="outreach_events")
    op.drop_index(op.f("ix_outreach_events_member_id"), table_name="outreach_events")
    op.drop_table("outreach_events")
    op.drop_index(op.f("ix_auth_codes_email"), table_name="auth_codes")
    op.drop_table("auth_codes")
    op.drop_table("users")
