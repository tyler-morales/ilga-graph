"""SQLAlchemy ORM models for user accounts and outreach tracking."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
    kei_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    kei_impact_slug: Mapped[str | None] = mapped_column(String(32), nullable=True)
    kei_personal_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    # no | yes | call_only | elevator
    call_pref: Mapped[str | None] = mapped_column(String(16), nullable=True)
    welcome_email_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
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
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    __table_args__ = (Index("ix_outreach_member_kind", "member_id", "kind"),)


class OutreachStepEvent(Base):
    """One row per checkpoint reached in call/email/WYC flow (funnel analytics).

    outreach_type is 'call', 'email', or 'wyc'. step_slug comes from outreach_steps.py.
    For anonymous funnel tracking: user_id can be NULL when session_id is set.
    For WYC (why-you-care) steps: member_id is NULL (no legislator in that flow).
    No unique constraint: we allow multiple reached_at per (user/session, member, type, step)
    for repeat sessions; analytics can take max(reached_at) or count as needed.
    """

    __tablename__ = "outreach_step_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    member_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    outreach_type: Mapped[str] = mapped_column(String(16), nullable=False)  # call | email | wyc
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


class Poll(Base):
    """Admin-created poll. placement controls where it appears (home, sidebar, updates)."""

    __tablename__ = "polls"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    placement: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # home | sidebar | updates
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    options: Mapped[list[PollOption]] = relationship(
        "PollOption", back_populates="poll", order_by="PollOption.sort_order", lazy="selectin"
    )


class PollOption(Base):
    """One choice for a poll. Unique (poll_id, slug)."""

    __tablename__ = "poll_options"

    id: Mapped[int] = mapped_column(primary_key=True)
    poll_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("polls.id"), nullable=False, index=True
    )
    slug: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    poll: Mapped[Poll] = relationship("Poll", back_populates="options")

    __table_args__ = (Index("ix_poll_options_poll_slug", "poll_id", "slug", unique=True),)


class PollResponse(Base):
    """One row per poll submission. Logged-in: user_id set; anon: session_id optional."""

    __tablename__ = "poll_responses"

    id: Mapped[int] = mapped_column(primary_key=True)
    poll_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("polls.id"), nullable=False, index=True
    )
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True, index=True
    )
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    option_slug: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class KeiPollResponse(Base):
    """One row per kei poll. Logged-in: user_id set; anon: user_id NULL, session_id optional.
    Legacy; new data uses PollResponse. Kept for migration backfill and backward compat."""

    __tablename__ = "kei_poll_responses"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True, index=True
    )
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    kei_status: Mapped[str] = mapped_column(String(32), nullable=False)
    zip_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class CommunityStory(Base):
    """User-submitted photo + story for the home page marquee. Requires admin approval."""

    __tablename__ = "community_stories"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    location: Mapped[str] = mapped_column(String(100), nullable=False)
    story: Mapped[str] = mapped_column(Text, nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    consent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    admin_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class KeiInterestStatement(Base):
    """User-submitted text-only statement (e.g. would buy if legal) for marquee. Non-owners only."""

    __tablename__ = "kei_interest_statements"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    location: Mapped[str] = mapped_column(String(100), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    consent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    admin_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SbeCommittee(Base):
    """Illinois SBE campaign committee (not an ILGA legislative committee)."""

    __tablename__ = "sbe_committees"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    type_of_committee: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str | None] = mapped_column(String(8), nullable=True)
    party: Mapped[str | None] = mapped_column(String(64), nullable=True)
    purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    refer_name: Mapped[str | None] = mapped_column(String(128), nullable=True)


class SbeReceipt(Base):
    """One SBE campaign disclosure receipt (contribution) in the ingest window."""

    __tablename__ = "sbe_receipts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    committee_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("sbe_committees.id"), nullable=False, index=True
    )
    received_date: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_only_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    occupation: Mapped[str | None] = mapped_column(String(256), nullable=True)
    employer: Mapped[str | None] = mapped_column(String(256), nullable=True)
    city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    d2_part: Mapped[str | None] = mapped_column(String(16), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_sbe_receipts_committee_date", "committee_id", "received_date"),)


class SbeMemberMatch(Base):
    """Committee → sitting ILGA member match (accepted or review)."""

    __tablename__ = "sbe_member_matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    committee_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("sbe_committees.id"), nullable=False, index=True
    )
    member_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    match_method: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    candidate_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    candidate_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    office: Mapped[str | None] = mapped_column(String(64), nullable=True)
    district: Mapped[str | None] = mapped_column(String(16), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class SbeIngestRun(Base):
    """One SBE money-layer ingest run (audit / match-rate history)."""

    __tablename__ = "sbe_ingest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    window_start: Mapped[str | None] = mapped_column(String(16), nullable=True)
    committees_loaded: Mapped[int] = mapped_column(Integer, default=0)
    receipts_loaded: Mapped[int] = mapped_column(Integer, default=0)
    members_matched: Mapped[int] = mapped_column(Integer, default=0)
    match_rate: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class MoneyIntelLead(Base):
    """Waitlist lead for follow-the-money intel (lobbyist / adjacent buyers).

    Not a User account and not the campaign-update subscriber list.
    """

    __tablename__ = "money_intel_leads"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    org: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # comma-joined allowlist: lobbyist, lawyer, nonprofit
    role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
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
