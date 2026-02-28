"""SQLAlchemy ORM models for user accounts and outreach tracking."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    zip_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    wants_updates: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Update(Base):
    """Campaign update or announcement. sent_at is set when email blast is sent."""

    __tablename__ = "updates"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    body_plain: Mapped[str] = mapped_column(Text, nullable=False)
    body_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    update_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="other", server_default="other"
    )
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sent_count: Mapped[int] = mapped_column(Integer, default=0)


class Campaign(Base):
    """Targeted legislator contact period (action alert). Only one should be is_active at a time."""

    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    ask: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. "contact your rep"
    target_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="all", server_default="all"
    )  # all | by_district
    target_member_ids: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    target_district_ids: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    session_milestone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AuthCode(Base):
    """Short-lived 6-digit email verification codes."""

    __tablename__ = "auth_codes"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    code_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class OutreachEvent(Base):
    """One row per call / email / no-answer logged by an advocate."""

    __tablename__ = "outreach_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(nullable=True, index=True)
    user_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    member_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # call | email | no_answer
    zip_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # person who picked up / was contacted
    contact_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # 1=opposed, 2=skeptical, 3=neutral, 4=interested, 5=champion
    support_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # was advocate a constituent of this rep?
    constituent: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    campaign_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("campaigns.id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (Index("ix_outreach_member_kind", "member_id", "kind"),)


class OutreachStepEvent(Base):
    """One row per checkpoint reached in call/email flow (funnel analytics).

    outreach_type is 'call' or 'email'. step_slug comes from outreach_steps.py.
    For anonymous funnel tracking: user_id can be NULL when session_id is set.
    No unique constraint: we allow multiple reached_at per (user/session, member, type, step)
    for repeat sessions; analytics can take max(reached_at) or count as needed.
    """

    __tablename__ = "outreach_step_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    member_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    outreach_type: Mapped[str] = mapped_column(String(16), nullable=False)  # call | email
    step_slug: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    reached_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("ix_outreach_step_events_user_member_type", "user_id", "member_id", "outreach_type"),
        Index("ix_outreach_step_events_type_slug_time", "outreach_type", "step_slug", "reached_at"),
        Index(
            "ix_outreach_step_events_session_type_time",
            "session_id",
            "outreach_type",
            "reached_at",
        ),
    )


class CommunityMemberEmail(Base):
    """Community-sourced legislator email: submitted by callers when member has no public email.

    One row per (member_id, email, user_id). Same user resubmitting same email is idempotent.
    Best email for a member = email with largest distinct submitter count; tie = most recent.
    """

    __tablename__ = "community_member_emails"

    id: Mapped[int] = mapped_column(primary_key=True)
    member_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    user_id: Mapped[int | None] = mapped_column(nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index(
            "ix_community_member_emails_member_user_email",
            "member_id",
            "email",
            "user_id",
            unique=True,
        ),
    )


class BugReport(Base):
    """In-app bug report from the beta banner (no GitHub/email required)."""

    __tablename__ = "bug_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    reporter_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    page_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    # JSON array of stored image filenames (under BUG_REPORT_UPLOAD_DIR).
    attachment_paths: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
