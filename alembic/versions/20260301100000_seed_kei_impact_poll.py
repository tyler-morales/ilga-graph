"""Seed kei_impact poll and options (Q3: How does this affect you?).

Revision ID: 20260301100000
Revises: 20260301000000
Create Date: 2026-03-01

"""

from collections.abc import Sequence

from sqlalchemy import text

from alembic import op

revision: str = "20260301100000"
down_revision: str | None = "20260301000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

KEI_IMPACT_OPTIONS = [
    ("support_cause", "I support the cause", 0),
    ("know_someone", "I know someone affected", 1),
    ("civic_duty", "Civic duty", 2),
    ("other", "Other", 3),
]


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        text(
            "INSERT INTO polls (id, slug, title, is_active, placement, created_at) "
            "VALUES (2, 'kei_impact', 'How does this affect you?', 1, NULL, datetime('now'))"
        )
    )
    for i, (slug, label, sort_order) in enumerate(KEI_IMPACT_OPTIONS):
        conn.execute(
            text(
                "INSERT INTO poll_options (id, poll_id, slug, label, sort_order) "
                "VALUES (:id, 2, :slug, :label, :sort_order)"
            ),
            {"id": 6 + i, "slug": slug, "label": label, "sort_order": sort_order},
        )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(text("DELETE FROM poll_options WHERE poll_id = 2"))
    conn.execute(text("DELETE FROM poll_responses WHERE poll_id = 2"))
    conn.execute(text("DELETE FROM polls WHERE id = 2"))
