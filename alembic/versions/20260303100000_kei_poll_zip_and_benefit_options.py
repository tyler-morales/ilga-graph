"""Add zip_code to kei_poll_responses; update kei_impact poll to benefit-oriented Q3 options.

Revision ID: 20260303100000
Revises: 20260302110000
Create Date: 2026-03-03

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import text

from alembic import op

revision: str = "20260303100000"
down_revision: str | None = "20260302110000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# New kei_impact options: (id offset 6, slug, label, sort_order)
KEI_IMPACT_OPTIONS_NEW = [
    (6, "direct_owner", "I could keep (or get) my kei legally registered", 0),
    (7, "want_to_buy", "I could buy a kei I've been wanting", 1),
    (8, "know_someone", "It would help someone I know", 2),
    (9, "support_cause", "I support clearer rules for kei vehicles", 3),
]


def upgrade() -> None:
    op.add_column(
        "kei_poll_responses",
        sa.Column("zip_code", sa.String(length=10), nullable=True),
    )

    conn = op.get_bind()
    conn.execute(text("UPDATE polls SET title = 'What would this fix mean for you?' WHERE id = 2"))
    for opt_id, slug, label, sort_order in KEI_IMPACT_OPTIONS_NEW:
        conn.execute(
            text(
                "UPDATE poll_options SET slug = :slug, label = :label, sort_order = :sort_order "
                "WHERE id = :id AND poll_id = 2"
            ),
            {"id": opt_id, "slug": slug, "label": label, "sort_order": sort_order},
        )
    conn.execute(
        text(
            "UPDATE poll_responses SET option_slug = 'support_cause' "
            "WHERE poll_id = 2 AND option_slug IN ('civic_duty', 'other')"
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    # Restore original poll_options slugs/labels (existing poll_responses keep their option_slug)
    OLD_OPTIONS = [
        (6, "support_cause", "I support the cause", 0),
        (7, "know_someone", "I know someone affected", 1),
        (8, "civic_duty", "Civic duty", 2),
        (9, "other", "Other", 3),
    ]
    for opt_id, slug, label, sort_order in OLD_OPTIONS:
        conn.execute(
            text(
                "UPDATE poll_options SET slug = :slug, label = :label, sort_order = :sort_order "
                "WHERE id = :id AND poll_id = 2"
            ),
            {"id": opt_id, "slug": slug, "label": label, "sort_order": sort_order},
        )
    conn.execute(text("UPDATE polls SET title = 'How does this affect you?' WHERE id = 2"))

    op.drop_column("kei_poll_responses", "zip_code")
