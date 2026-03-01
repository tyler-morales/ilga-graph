# Auth + outreach DB: implementation scan and test coverage

This doc summarizes the DB implementation, potential issues, and how tests verify behavior.

---

## Components

| Component | Role |
|-----------|------|
| **db.py** | Async SQLite engine (`ILGA_DB_PATH`), `init_db()` (Alembic upgrade head, or fallback create_all + ALTERs), `get_db()` FastAPI dependency |
| **alembic/** | Versioned migrations; `alembic upgrade head` creates/updates schema. Schema version stored in `alembic_version` table. |
| **db_models.py** | `User` (id, email, **zip_code**, **kei_status**, **welcome_email_sent_at**, created_at, last_login_at), `AuthCode`, **Campaign** (title, message, ask, target_type, target_member_ids/target_district_ids, is_active, start_at/end_at), `OutreachEvent` (**campaign_id** nullable FK), **OutreachStepEvent** (funnel checkpoints; **session_id** nullable for anonymous), `CommunityMemberEmail` (community-sourced legislator emails), **Poll** (id, slug, title, is_active, placement, created_at), **PollOption** (id, poll_id, slug, label, sort_order), **PollResponse** (id, poll_id, user_id nullable, session_id nullable, option_slug, created_at), **KeiPollResponse** (legacy; one row per kei poll submission; dual-write with PollResponse for kei) — SQLAlchemy ORM |
| **outreach_steps.py** | Canonical step slugs for call (answered + no-answer) and email flows; `is_valid_step()` for validation |
| **routers/auth.py** | Request code, verify code (optional anon_session_id for attribution), logout, `/me` |
| **routers/outreach.py** | Record event, **record step** (POST /outreach/step, accepts anonymous session_id), stats by member, my-history |
| **dependencies.py** | Session token (itsdangerous), `get_current_user_optional`, `require_user` |

---

## Data flow: how the DB gets full and how it's read

**Single SQLite file** per process. Path: `ILGA_DB_PATH` (profile default: dev → `data/ilga_dev.db`, prod → `data/ilga.db`). One async engine in `db.py`; all app code uses `get_db()`. The **kei poll** uses this same DB: table `kei_poll_responses` (migration `20260228110000`); **polls** (admin-created polls), **poll_options**, **poll_responses** (migration `20260228130000`). Kei poll is seeded as Poll slug=kei; submissions dual-write to KeiPollResponse and PollResponse. On app startup, `init_db()` runs Alembic to head (or fallback `create_all`), so the table exists in both dev and prod as long as the app has run once or migrations have been run manually (e.g. `make dev` runs `alembic upgrade head` before uvicorn).

### Write paths (how the DB gets data)

| Source | Tables | When |
|--------|--------|------|
| **Auth** | `auth_codes`, then `users` (and `auth_codes.used`); on first sign-in, welcome email sent and `users.welcome_email_sent_at` set | POST /auth/request-code → POST /auth/verify-code |
| **Outreach (in-app)** | `outreach_events` | Logged-in user records call/email/no_answer via POST /outreach/record; optional **campaign_id** (validated against active campaign) stored for attribution |
| **Outreach steps (funnel)** | `outreach_step_events` | Logged-in user: client POST /outreach/step (member_id, outreach_type, step_slug); server inserts with user_id. **Anonymous:** same endpoint with optional session_id; when unauthenticated and session_id valid, server inserts with user_id=NULL, session_id set. Server also inserts step when recording call/email/no_answer (call_recorded, email_recorded, no_answer_recorded). Step slugs in `outreach_steps.py`. Column `session_id` (String(64), nullable) added for anonymous funnel; attribution on sign-in backfills session_id → user_id. |
| **Community emails** | `community_member_emails` | When recording a call (POST /outreach/record, kind=call) with optional `legislator_email`, and member has no public email, one row per (member_id, email, user_id); used to pre-fill drawer for next constituent |
| **State of kei** | `users.kei_status`, `kei_poll_responses`, `poll_responses` | Two-step poll: (1) “Do you have a kei vehicle?” Yes/No; (2) if Yes → registration status (registered/revoked/denied), if No → would/wouldn’t want one. POST /updates/kei-status: **every** submission inserts one row into `kei_poll_responses` and one into `poll_responses` (poll_id = kei). **Logged-in:** also set `users.kei_status`. **Anonymous:** no user row created/updated; response shows thanks, aggregate results (verified only), nudge + inline sign-in to “Sign in to add your vote.” **Poll results count only verified users** (last_login_at IS NOT NULL). GET /updates/kei-status-results returns counts by kei_status for verified users. **Admin:** GET /admin/poll redirects to kei poll results or /admin/polls. GET /admin/polls lists all polls (create, edit, results); GET /admin/polls/{id}/results shows verified + all-responses (pie + table). Dashboard shows "Current poll(s)" stat. Poll on home, sidebar, /updates?prompt=kei. |
| **Seed script** | `users`, `outreach_events` | `make seed-outreach` or `python scripts/seed_outreach.py` |

- **Auth:** request-code inserts one `AuthCode`; verify-code finds it, marks used, and gets-or-creates `User`, updates `last_login_at`.
- **Outreach:** POST /outreach/record (requires auth) inserts one `OutreachEvent` per call/email/no_answer (zip on the event is stored for analytics only; it does not update `User.zip_code`). When an active **Campaign** exists and the client sends `campaign_id`, the server validates it and stores it on the event for per-campaign reporting. In the call drawer, the **call is recorded when the user selects an interest level** (1–5) in the post-call poll, not when they open the follow-up email draft. The **voicemail / no-answer path**: when the user clicks "End call" in the voicemail section, the client records via POST /outreach/record with `kind=no_answer` (when signed in), then POST /advocacy/call/{id}/no-answer returns the no-answer outcome partial and the drawer body is replaced so the user sees "No problem — offices get busy" and CTAs (Send email instead / Close). Anonymous users see the same no-answer screen but no event is persisted. The server validates that `member_id` refers to a legislator in app state (`find_member_by_id`); if not, it returns 400 so the DB never stores invalid or slug-style ids.
- **Seed:** `scripts/seed_outreach.py` uses the same profile/env as the app; runs `init_db()`, then gets-or-creates user for the backlog email: **prod** → `moratyle@gmail.com`, **dev** → `funky_mama11@gmail.com`. Inserts `OutreachEvent` rows only when the rep name **resolves to a canonical member id** (from cache/mocks members.json). Unresolved names are skipped (no slug fallback), so the DB stays consistent. Re-running seed deletes existing outreach events for that user before inserting, so it is idempotent. Use the same `ILGA_PROFILE` (or same `ILGA_DB_PATH`) for app and seed. In dev, the seed also inserts **"this week"** and mock-advocate events with **hardcoded numeric member_ids** so the landing ticker and heat pills work; all use canonical ids.

### Read paths (where DB data is used)

- **Auth:** verify-code and GET /auth/me read `auth_codes` and `users`. GET /auth/me returns `kei_status`. On first sign-in (welcome_email_sent_at is None), a welcome email is sent and `welcome_email_sent_at` is set.
- **Admin dashboard (ED / coalition view):** GET /admin is the single “status of the advocacy effort” view (Hardball-style for staff and board). It shows: list size (users, subscribers), outreach totals and mix (last 90d), **outreach trend** (last 7d and 30d calls/emails), conversion link (Stats → /admin/outreach), **last update sent** (date and title of most recent sent update), **active campaign** (title, end date, action count), and **top legislators by contact volume** (top 5 member_ids from OutreachEvent). **Polls:** GET /admin/polls (nav "Polls") lists all polls; create, edit, per-poll results (verified + all pie/table). GET /admin/poll redirects to kei results or list. Dashboard shows "Current poll(s)" stat. No new public routes; reuse existing admin auth.
- **Outreach:** GET /outreach/stats/{member_id}, GET /outreach/interest-poll/{member_id} (public), GET /outreach/my-stats and GET /outreach/my-history (auth required). **Funnel:** `outreach_step_events` is written by POST /outreach/step (and by record when recording call/email/no_answer). **Conversion report:** GET /admin/outreach/conversion (require_admin; available in prod) returns denominator (distinct identities who opened drawer in last 90 days), numerator (distinct users who completed at least one call/email in same window), and conversion_pct. See *Anonymous funnel and conversion* below.
- **Community emails:** Advocacy drawer and wrap-up (GET /advocacy/drawer, POST /advocacy/call/{id}/wrapup) call `get_effective_email_for_member()` which reads `community_member_emails` to pre-fill recipient when member has no public email; best email = most submitters, then most recent.
- **Advocacy:** When a logged-in user has a saved `User.zip_code` (valid and in district data), the advocacy page pre-fills the hero ZIP and runs the search without requiring a URL param. **User ZIP source of truth:** The hero zip input is search-only and never persists. The stored zip is set or updated **only** when the user clicks the zip under the sign-in strip (inline edit) or uses "Use location" and commits (PATCH /advocacy/api/me/zip). Visiting with `?zip=` or recording outreach does **not** overwrite `User.zip_code`. Drawer checks whether the current user has called this member (count from `outreach_events`). Results page builds `user_called_member_ids` / `user_emailed_member_ids` (for "Reached out" pill) and **outreach_heat** (count of distinct users who reached out per member) for the **fire pill** on each card. The **landing hero ticker** shows one number: **total outreach actions** (all time) = count of all call/email events. Copy: "Add your voice. X+ outreach actions already made."

### Outreach checkpoint steps (outreach_steps.py)

- **Call (answered):** drawer_opened, phone_clicked, staffer_name_captured, office_email_captured, end_call_clicked, interest_selected, call_recorded, wrapup_draft_clicked, wrapup_skipped.
- **Call (no-answer):** drawer_opened, voicemail_toggled, end_call_clicked_vm, no_answer_recorded.
- **Email:** drawer_opened, signed_in, subject_confirmed, details_filled, pdf_grabbed, send_clicked, email_recorded.
- **Why-you-care (WYC):** wyc_poll_submitted, wyc_branch_viewed, wyc_clicked_to_advocacy, wyc_clicked_to_the_issue, wyc_share_story_clicked, wyc_change_answer_clicked. Outcome-focused (to advocacy, to the-issue, share story). Aligned with Hardball Ch7 (making the case to the base) and Ch6 (mobilizing grassroots support). POST /outreach/step with `outreach_type=wyc` accepts optional `member_id`; events are stored with `member_id=NULL`.

### Anonymous funnel tracking and conversion

**Goal:** Track outreach funnel steps for anonymous users (e.g. drawer opened, phone clicked) so we can report conversion as: “Of everyone who opened the advocacy drawer (including before sign-in), X% completed at least one call or email.”

- **Schema:** `outreach_step_events` has nullable `session_id` (String(64)) and nullable `member_id` (for WYC steps). Rows can have `user_id` set (authenticated) or `session_id` set (anonymous). WYC rows use `member_id=NULL`. Index on `session_id` and composite `(session_id, outreach_type, reached_at)` for conversion queries.
- **Client:** A stable anonymous session id is stored in `sessionStorage` under key `ilga_anon_sid` (UUID or 32-char hex). It is created once per tab/session and reused. The client sends it as `session_id` on every POST /outreach/step when the user is **not** signed in (at least for drawer_opened and phone_clicked). When signed in, step requests do not include `session_id` so the server stores only `user_id`.
- **Server:** POST /outreach/step accepts optional `session_id`. If the request is unauthenticated and `session_id` is present and valid (1–64 chars, alphanumeric + hyphen), the server inserts a row with `user_id=NULL` and `session_id` set. Invalid or missing `session_id` when unauthenticated returns 401 (or 400 for bad format).
- **Attribution:** When the user signs in (POST /auth/verify-code), the client may send `anon_session_id` (the same value as `ilga_anon_sid`). If present and valid, the server **backfills** all `outreach_step_events` rows with that `session_id`: sets `user_id` to the signed-in user and clears `session_id` (sets to NULL). So one identity is not double-counted: after sign-in, that user’s earlier anonymous steps are attributed to them.
- **Conversion report:** GET /admin/outreach/conversion (require_admin; available in prod) returns a shared 90-day window and:
  - **conversions:** Object keyed by slug, each with `denominator`, `numerator`, `conversion_pct`. Minimum set: `drawer_to_outreach` (main pitch), `phone_to_call` (call completion), `drawer_to_email`, `signed_in_to_outreach` (users who signed in in window and took action).
  - **volumes:** Object keyed by slug: `identities_opened_drawer`, `users_completed_outreach`, `total_calls`, `total_emails`, `total_outreach_actions`, `identities_clicked_phone`.
  - When denominator is 0, conversion_pct is 0.0 (no divide-by-zero). Multiple tabs each have their own session id (sessionStorage); denominator may be slightly conservative.
- **Definition (main pitch):** “Conversion = % of distinct identities (user or anonymous session) who opened the advocacy drawer in the last 90 days and completed at least one call or email in the same window.” No double-counting: a user who opened the drawer anonymously then signed in counts once in the denominator and, if they completed outreach, once in the numerator.
- **Privacy:** Tracking is first-party only (our DB, our domain). No sale or third-party sharing. Session identifier is in sessionStorage; disclosed in the privacy policy; step/conversion data retained for 12 months then deleted or anonymized. No consent banner required for this minimal use; disclosure in the policy is the minimum.

### Fire pill on member cards (data-driven)

The **fire pill** (🔥 N) in the corner of each advocacy member card is **data-driven** from the database. It shows *N unique outreach advocate(s)* — the count of distinct users who have recorded a call or email for that legislator. Data comes from `outreach_events`: the advocacy results endpoint queries `COUNT(DISTINCT user_id)` per `member_id` and passes `outreach_heat` into the template. The pill is rendered only when `heat_count > 0` (see `_results_partial.html` macro `member_card` and class `heat-score-pill`). Seeded or live outreach events are what make the number appear; if app and seed use different DB paths (e.g. dev vs prod), the pill will not show seeded data.

---

## Double-check checklist (verify DB setup and pull/push)

1. **Startup log** — Run the app (e.g. `make dev`). In logs you should see: `Database ready at data/ilga_dev.db` (or your path). Confirms `init_db()` ran and path is correct.

2. **Tables exist** — `sqlite3 data/ilga_dev.db` then `SELECT name FROM sqlite_master WHERE type='table';` — expect `users`, `auth_codes`, `outreach_events`, `outreach_step_events`, `community_member_emails`, **`kei_poll_responses`**, **`polls`**, **`poll_options`**, **`poll_responses`**.

3. **Auth write + read** — Request a code (POST /auth/request-code), then in DB: `SELECT * FROM auth_codes ORDER BY id DESC LIMIT 1;` — one new row. Verify code (POST /auth/verify-code); then `SELECT * FROM users;` — one user; that `auth_codes.used` = 1. GET /auth/me returns that email.

4. **Outreach write + read** — While logged in, record one call (POST /outreach/record). Then `SELECT * FROM outreach_events ORDER BY id DESC LIMIT 1;` — one new row. GET /outreach/stats/{that_member_id} shows calls: 1. GET /outreach/my-history shows the event.

5. **Seed and app same DB** — Run app and seed with the same profile (e.g. both `ILGA_PROFILE=dev`). Open advocacy results for a ZIP that has seeded data; the **fire pill** (🔥 N) and "Reached out" state on cards should reflect seeded events. If you seed prod DB but run app in dev (different files), the pill won’t show seeded data.

6. **Poll record + display** — **Dev:** `make dev` runs `alembic upgrade head` before starting (so `kei_poll_responses`, `polls`, `poll_options`, `poll_responses` exist). Submit the kei poll (home or /updates?prompt=kei); then `SELECT * FROM kei_poll_responses ORDER BY id DESC LIMIT 5;` and `SELECT * FROM poll_responses WHERE poll_id=1;` — new row(s). Public results and GET /updates/kei-status-results show counts; admin GET /admin/poll redirects to kei results or GET /admin/polls lists/create/edit/results. **Prod:** Use the same DB as the app (`ILGA_DB_PATH`); run migrations once before first deploy (`PYTHONPATH=src ILGA_PROFILE=prod $(PYTHON) -m alembic upgrade head` from project root) or rely on app startup (`init_db()` runs Alembic when `alembic.ini` exists). Same path for dev (`data/ilga_dev.db`) and prod (`data/ilga.db`) is set by profile when `ILGA_DB_PATH` is unset.

7. **Automated tests** — `pytest tests/test_db.py tests/test_auth_outreach.py tests/test_updates.py -v` — temp DB, init_db, auth and outreach flows, schema, kei poll record/display and admin poll.

8. **Smoke test** — `make smoke-outreach` — in-process app, temp DB, sign-in, record call + email, assert GET /outreach/stats and GET /outreach/my-history.

**Dev mode: advocacy ZIP codes** — In dev/seed mode the app uses `mocks/dev/zip_to_district.json`, which only contains ZIPs that map to the 40 mock members’ districts (~148 ZIPs). If you enter a ZIP that isn’t in that file (e.g. 60608), you’ll see “ZIP code not found”. The error message in dev suggests sample ZIPs to try (e.g. 60007, 60104, 60107). Use any of those or run `make snapshot-mocks` after a scrape to refresh the mock ZIP list.

---

## Scan: potential issues and mitigations

1. **Schema migrations (Alembic)**  
   On startup, `init_db()` runs Alembic migrations to head when `alembic.ini` exists (project root); otherwise it falls back to `create_all` + ALTERs (e.g. temp DB in tests). New schema changes go in new migration files under `alembic/versions/`. **Existing DBs** created before Alembic: run once from project root: `alembic stamp head` (or `alembic stamp 20260221000000`), so the DB is marked at the current revision and future `upgrade head` does nothing until you add new migrations.

2. **ALTER fallback (duplicate column only)**  
   When using the fallback (no Alembic), `init_db()` runs ALTERs for legacy columns. Only `OperationalError` with "duplicate column name" (or "already exists") is ignored; other failures (e.g. disk full) are re-raised.

3. **member_id consistency**
   `OutreachEvent.member_id` must be a **canonical member id** (same as `Member.id` from members data). The record endpoint rejects unknown ids (400) and truncates to 32 chars. The seed script only inserts events when the rep name resolves to an id (no slug fallback). **Mitigation:** Record validates; seed skips unresolved names.

4. **Verify-code race**  
   Two concurrent requests with the same valid code could both pass the `used == False` check before either commits. **Risk:** Same code used twice. **Mitigation:** Low likelihood for 6-digit codes; tests assert that after one successful verify the code is marked used (sequential).

5. **Session handling**  
   `get_db()` yields a session; FastAPI closes it after the request. Uncommitted changes are rolled back on close. **Risk:** None identified. **Mitigation:** Tests assert commit by reading back after record.

6. **Support score / constituent parsing**  
   `_parse_support_score` accepts 1–5 only; `_parse_constituent` accepts 1/0, true/false, yes/no. **Risk:** Invalid values stored as NULL; acceptable. **Mitigation:** Unit tests for parsing and API round-trip.

7. **Stats with no events**  
   `GET /outreach/stats/{member_id}` with no rows returns `calls: 0, emails: 0, no_answers: 0, total: 0`. **Mitigation:** Test covers empty stats.

8. **my-history limit**  
   History is capped at 100 events, newest first. **Mitigation:** Documented; test can assert limit and ordering.

9. **SQL injection**  
   All queries use SQLAlchemy ORM or `select()` with parameters. No raw user input in SQL. **Risk:** None. **Mitigation:** N/A.

10. **Timezone**  
   `created_at` and `AuthCode.expires_at` use `DateTime(timezone=True)` and `_utcnow()`. **Mitigation:** Tests assert events have `created_at` and history returns ISO format.

10. **Boolean in SQLite**  
    SQLAlchemy maps Python `bool` to SQLite integer 0/1. **Mitigation:** Test stores `constituent=True/False` and asserts round-trip.

---

## Test coverage

- **tests/test_db.py** — `init_db()` (tables + fallback idempotent when no Alembic), `get_db()` yields session, schema (columns exist).
- **tests/test_auth_outreach.py** — Auth flow (request-code → verify-code → /me, logout), outreach record (unauthenticated 401, valid record, support_score/constituent/contact_name/notes), stats (empty and with data), my-history (401 when anonymous, list and ordering), parsing helpers for support_score and constituent.

Run: `make test` or `pytest tests/test_db.py tests/test_auth_outreach.py -v`.

---

## Smoke test (terminal, no server)

**scripts/smoke_test_outreach.py** runs an automated end-to-end flow in the terminal using a temp DB:

1. Sign in (verify with a pre-seeded auth code).
2. Record one call and one email.
3. Assert public **GET /outreach/stats/{member_id}** (no auth) shows the new counts — i.e. a visitor would see the outreach.
4. Assert **GET /outreach/my-history** returns both events.

No Brevo or running server required. Use before deploy to confirm the outreach DB path works.

```bash
make smoke-outreach
# or
PYTHONPATH=src python scripts/smoke_test_outreach.py
```
