"""Campaigns table and outreach_events.campaign_id.

Revision ID: 20260227000000
Revises: 20260226120000
Create Date: 2026-02-27

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260227000000"
down_revision: str | None = "20260226120000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Idempotent: safe when campaigns or outreach_events.campaign_id already exist."""
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "campaigns" not in insp.get_table_names():
        op.create_table(
            "campaigns",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("title", sa.String(length=200), nullable=False),
            sa.Column("message", sa.Text(), nullable=False),
            sa.Column("ask", sa.String(length=100), nullable=False),
            sa.Column("target_type", sa.String(length=16), nullable=False, server_default="all"),
            sa.Column("target_member_ids", sa.Text(), nullable=True),
            sa.Column("target_district_ids", sa.Text(), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=True),
            sa.Column("start_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("end_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
    if "campaigns" in insp.get_table_names():
        existing = [idx["name"] for idx in insp.get_indexes("campaigns")]
        if "ix_campaigns_is_active" not in existing:
            op.create_index(
                op.f("ix_campaigns_is_active"),
                "campaigns",
                ["is_active"],
                unique=False,
            )

    if "outreach_events" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("outreach_events")]
        if "campaign_id" not in cols:
            with op.batch_alter_table("outreach_events", schema=None) as batch_op:
                batch_op.add_column(sa.Column("campaign_id", sa.Integer(), nullable=True))
                batch_op.create_foreign_key(
                    "fk_outreach_events_campaign_id_campaigns",
                    "campaigns",
                    ["campaign_id"],
                    ["id"],
                )
                batch_op.create_index(
                    op.f("ix_outreach_events_campaign_id"),
                    ["campaign_id"],
                    unique=False,
                )
        else:
            fks = [fk["name"] for fk in insp.get_foreign_keys("outreach_events")]
            existing_ix = [idx["name"] for idx in insp.get_indexes("outreach_events")]
            need_fk = "fk_outreach_events_campaign_id_campaigns" not in fks
            need_ix = "ix_outreach_events_campaign_id" not in existing_ix
            if need_fk or need_ix:
                with op.batch_alter_table("outreach_events", schema=None) as batch_op:
                    if "fk_outreach_events_campaign_id_campaigns" not in fks:
                        batch_op.create_foreign_key(
                            "fk_outreach_events_campaign_id_campaigns",
                            "campaigns",
                            ["campaign_id"],
                            ["id"],
                        )
                    if "ix_outreach_events_campaign_id" not in existing_ix:
                        batch_op.create_index(
                            op.f("ix_outreach_events_campaign_id"),
                            ["campaign_id"],
                            unique=False,
                        )


def downgrade() -> None:
    op.drop_index(op.f("ix_outreach_events_campaign_id"), table_name="outreach_events")
    op.drop_constraint(
        "fk_outreach_events_campaign_id_campaigns",
        "outreach_events",
        type_="foreignkey",
    )
    op.drop_column("outreach_events", "campaign_id")
    op.drop_index(op.f("ix_campaigns_is_active"), table_name="campaigns")
    op.drop_table("campaigns")
