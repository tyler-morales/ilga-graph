"""Add polls, poll_options, poll_responses; seed kei poll and backfill from kei_poll_responses.

Revision ID: 20260228130000
Revises: 20260228120000
Create Date: 2026-02-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import text

from alembic import op

revision: str = "20260228130000"
down_revision: str | None = "20260228120000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

KEI_OPTIONS = [
    ("registered", "I have a kei (registered)", 0),
    ("revoked", "I had a kei; registration was revoked", 1),
    ("denied", "I was denied registration", 2),
    ("would_want", "I don't have a kei but would want one", 3),
    ("would_not_want", "I don't have a kei and wouldn't want one", 4),
]


def upgrade() -> None:
    op.create_table(
        "polls",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=True, server_default="1"),
        sa.Column("placement", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_polls_slug"), "polls", ["slug"], unique=True)
    op.create_index(op.f("ix_polls_is_active"), "polls", ["is_active"])

    op.create_table(
        "poll_options",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("poll_id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=256), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=True, server_default="0"),
        sa.ForeignKeyConstraint(["poll_id"], ["polls.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_poll_options_poll_id"), "poll_options", ["poll_id"])
    op.create_index(
        op.f("ix_poll_options_poll_slug"), "poll_options", ["poll_id", "slug"], unique=True
    )

    op.create_table(
        "poll_responses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("poll_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("option_slug", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["poll_id"], ["polls.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_poll_responses_poll_id"), "poll_responses", ["poll_id"])
    op.create_index(op.f("ix_poll_responses_user_id"), "poll_responses", ["user_id"])
    op.create_index(op.f("ix_poll_responses_session_id"), "poll_responses", ["session_id"])

    # Seed kei poll and options
    conn = op.get_bind()
    conn.execute(
        text(
            "INSERT INTO polls (id, slug, title, is_active, placement, created_at) "
            "VALUES (1, 'kei', 'State of kei', 1, NULL, datetime('now'))"
        )
    )
    for i, (slug, label, sort_order) in enumerate(KEI_OPTIONS):
        conn.execute(
            text(
                "INSERT INTO poll_options (id, poll_id, slug, label, sort_order) "
                "VALUES (:id, 1, :slug, :label, :sort_order)"
            ),
            {"id": i + 1, "slug": slug, "label": label, "sort_order": sort_order},
        )

    # Backfill from kei_poll_responses (option_slug = kei_status)
    conn.execute(
        text(
            "INSERT INTO poll_responses (poll_id, user_id, session_id, option_slug, created_at) "
            "SELECT 1, user_id, session_id, kei_status, created_at FROM kei_poll_responses"
        )
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_poll_responses_session_id"), table_name="poll_responses")
    op.drop_index(op.f("ix_poll_responses_user_id"), table_name="poll_responses")
    op.drop_index(op.f("ix_poll_responses_poll_id"), table_name="poll_responses")
    op.drop_table("poll_responses")
    op.drop_index(op.f("ix_poll_options_poll_slug"), table_name="poll_options")
    op.drop_index(op.f("ix_poll_options_poll_id"), table_name="poll_options")
    op.drop_table("poll_options")
    op.drop_index(op.f("ix_polls_is_active"), table_name="polls")
    op.drop_index(op.f("ix_polls_slug"), table_name="polls")
    op.drop_table("polls")
