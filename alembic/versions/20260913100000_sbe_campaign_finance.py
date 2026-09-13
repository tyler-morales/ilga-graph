"""SBE campaign-finance tables: committees, receipts, member matches.

Revision ID: 20260913100000
Revises: 20260303100000
Create Date: 2026-09-13

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260913100000"
down_revision: str | None = "20260303100000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sbe_committees",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("type_of_committee", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=8), nullable=True),
        sa.Column("party", sa.String(length=64), nullable=True),
        sa.Column("purpose", sa.Text(), nullable=True),
        sa.Column("city", sa.String(length=128), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=True),
        sa.Column("refer_name", sa.String(length=128), nullable=True),
    )
    op.create_table(
        "sbe_receipts",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("committee_id", sa.String(length=32), nullable=False),
        sa.Column("received_date", sa.String(length=16), nullable=True),
        sa.Column("amount", sa.Float(), nullable=False, server_default="0"),
        sa.Column("last_only_name", sa.String(length=256), nullable=True),
        sa.Column("first_name", sa.String(length=128), nullable=True),
        sa.Column("occupation", sa.String(length=256), nullable=True),
        sa.Column("employer", sa.String(length=256), nullable=True),
        sa.Column("city", sa.String(length=128), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=True),
        sa.Column("d2_part", sa.String(length=16), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.ForeignKeyConstraint(["committee_id"], ["sbe_committees.id"]),
    )
    op.create_index("ix_sbe_receipts_committee_id", "sbe_receipts", ["committee_id"])
    op.create_index("ix_sbe_receipts_received_date", "sbe_receipts", ["received_date"])
    op.create_index(
        "ix_sbe_receipts_committee_date",
        "sbe_receipts",
        ["committee_id", "received_date"],
    )
    op.create_table(
        "sbe_member_matches",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("committee_id", sa.String(length=32), nullable=False),
        sa.Column("member_id", sa.String(length=32), nullable=True),
        sa.Column("match_method", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("candidate_id", sa.String(length=32), nullable=True),
        sa.Column("candidate_name", sa.String(length=256), nullable=True),
        sa.Column("office", sa.String(length=64), nullable=True),
        sa.Column("district", sa.String(length=16), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["committee_id"], ["sbe_committees.id"]),
    )
    op.create_index("ix_sbe_member_matches_committee_id", "sbe_member_matches", ["committee_id"])
    op.create_index("ix_sbe_member_matches_member_id", "sbe_member_matches", ["member_id"])
    op.create_index("ix_sbe_member_matches_status", "sbe_member_matches", ["status"])
    op.create_table(
        "sbe_ingest_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_url", sa.String(length=512), nullable=True),
        sa.Column("window_start", sa.String(length=16), nullable=True),
        sa.Column("committees_loaded", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("receipts_loaded", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("members_matched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("match_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("notes", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("sbe_ingest_runs")
    op.drop_index("ix_sbe_member_matches_status", table_name="sbe_member_matches")
    op.drop_index("ix_sbe_member_matches_member_id", table_name="sbe_member_matches")
    op.drop_index("ix_sbe_member_matches_committee_id", table_name="sbe_member_matches")
    op.drop_table("sbe_member_matches")
    op.drop_index("ix_sbe_receipts_committee_date", table_name="sbe_receipts")
    op.drop_index("ix_sbe_receipts_received_date", table_name="sbe_receipts")
    op.drop_index("ix_sbe_receipts_committee_id", table_name="sbe_receipts")
    op.drop_table("sbe_receipts")
    op.drop_table("sbe_committees")
