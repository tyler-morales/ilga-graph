# Auth + outreach DB: implementation scan and test coverage

This doc summarizes the DB implementation, potential issues, and how tests verify behavior.

---

## Components

| Component | Role |
|-----------|------|
| **db.py** | Async SQLite engine (`ILGA_DB_PATH`), `init_db()` (create tables + migrate old DBs), `get_db()` FastAPI dependency |
| **db_models.py** | `User`, `AuthCode`, `OutreachEvent` (SQLAlchemy ORM) |
| **routers/auth.py** | Request code, verify code, logout, `/me` |
| **routers/outreach.py** | Record event, stats by member, my-history |
| **dependencies.py** | Session token (itsdangerous), `get_current_user_optional`, `require_user` |

---

## Data flow: how the DB gets full and how it's read

**Single SQLite file** per process. Path: `ILGA_DB_PATH` (profile default: dev → `data/ilga_dev.db`, prod → `data/ilga.db`). One async engine in `db.py`; all app code uses `get_db()`.

### Write paths (how the DB gets data)

| Source | Tables | When |
|--------|--------|------|
| **Auth** | `auth_codes`, then `users` (and `auth_codes.used`) | POST /auth/request-code → POST /auth/verify-code |
| **Outreach (in-app)** | `outreach_events` | Logged-in user records call/email/no_answer via POST /outreach/record |
| **Seed script** | `users`, `outreach_events` | `make seed-outreach` or `python scripts/seed_outreach.py` |

- **Auth:** request-code inserts one `AuthCode`; verify-code finds it, marks used, and gets-or-creates `User`, updates `last_login_at`.
- **Outreach:** POST /outreach/record (requires auth) inserts one `OutreachEvent` per call/email/no_answer.
- **Seed:** `scripts/seed_outreach.py` uses the same profile/env as the app; runs `init_db()`, then gets-or-creates user for the backlog email and inserts many `OutreachEvent` rows (real backlog + in dev, mock advocate data). Use the same `ILGA_PROFILE` (or same `ILGA_DB_PATH`) for app and seed so they point at the same file. In dev, the seed also inserts additional **"this week"** events with relative `created_at` (e.g. now − 0..6 days) so the landing ticker always shows a realistic "Over N calls made to Springfield this week", and includes a spread of **support_score** (1–5) for at least one member so the call-drawer interest poll bar chart shows all segments (Opposed → Champion).

### Read paths (where DB data is used)

- **Auth:** verify-code and GET /auth/me read `auth_codes` and `users`.
- **Outreach:** GET /outreach/stats/{member_id}, GET /outreach/interest-poll/{member_id} (public), GET /outreach/my-history (auth required).
- **Advocacy:** Drawer checks whether the current user has called this member (count from `outreach_events`). Results page builds `user_called_member_ids` / `user_emailed_member_ids` (for "Reached out" pill) and **outreach_heat** (count of distinct users who reached out per member) for the **fire pill** on each card.

### Fire pill on member cards (data-driven)

The **fire pill** (🔥 N) in the corner of each advocacy member card is **data-driven** from the database. It shows *N unique outreach advocate(s)* — the count of distinct users who have recorded a call or email for that legislator. Data comes from `outreach_events`: the advocacy results endpoint queries `COUNT(DISTINCT user_id)` per `member_id` and passes `outreach_heat` into the template. The pill is rendered only when `heat_count > 0` (see `_results_partial.html` macro `member_card` and class `heat-score-pill`). Seeded or live outreach events are what make the number appear; if app and seed use different DB paths (e.g. dev vs prod), the pill will not show seeded data.

---

## Double-check checklist (verify DB setup and pull/push)

1. **Startup log** — Run the app (e.g. `make dev`). In logs you should see: `Database ready at data/ilga_dev.db` (or your path). Confirms `init_db()` ran and path is correct.

2. **Tables exist** — `sqlite3 data/ilga_dev.db` then `SELECT name FROM sqlite_master WHERE type='table';` — expect `users`, `auth_codes`, `outreach_events`.

3. **Auth write + read** — Request a code (POST /auth/request-code), then in DB: `SELECT * FROM auth_codes ORDER BY id DESC LIMIT 1;` — one new row. Verify code (POST /auth/verify-code); then `SELECT * FROM users;` — one user; that `auth_codes.used` = 1. GET /auth/me returns that email.

4. **Outreach write + read** — While logged in, record one call (POST /outreach/record). Then `SELECT * FROM outreach_events ORDER BY id DESC LIMIT 1;` — one new row. GET /outreach/stats/{that_member_id} shows calls: 1. GET /outreach/my-history shows the event.

5. **Seed and app same DB** — Run app and seed with the same profile (e.g. both `ILGA_PROFILE=dev`). Open advocacy results for a ZIP that has seeded data; the **fire pill** (🔥 N) and "Reached out" state on cards should reflect seeded events. If you seed prod DB but run app in dev (different files), the pill won’t show seeded data.

6. **Automated tests** — `pytest tests/test_db.py tests/test_auth_outreach.py -v` — temp DB, init_db, auth and outreach flows, schema.

7. **Smoke test** — `make smoke-outreach` — in-process app, temp DB, sign-in, record call + email, assert GET /outreach/stats and GET /outreach/my-history.

**Dev mode: advocacy ZIP codes** — In dev/seed mode the app uses `mocks/dev/zip_to_district.json`, which only contains ZIPs that map to the 40 mock members’ districts (~148 ZIPs). If you enter a ZIP that isn’t in that file (e.g. 60608), you’ll see “ZIP code not found”. The error message in dev suggests sample ZIPs to try (e.g. 60007, 60104, 60107). Use any of those or run `make snapshot-mocks` after a scrape to refresh the mock ZIP list.

---

## Scan: potential issues and mitigations

1. **Migration idempotency**  
   `init_db()` runs `ALTER TABLE outreach_events ADD COLUMN ...` for `contact_name`, `support_score`, `constituent`. Failures (e.g. column already exists) are caught and ignored. **Risk:** If a future ALTER fails for a different reason (e.g. disk full), we swallow it. **Mitigation:** Tests run `init_db()` twice and assert tables/columns exist; no crash on second run.

2. **member_id length**  
   `OutreachEvent.member_id` is `String(32)`. The router does not truncate. SQLite stores arbitrary length; the 32-char limit is schema/documentation. **Risk:** Overlong IDs could affect downstream reporting. **Mitigation:** Tests use valid-length IDs; optional follow-up: truncate in router to 32.

3. **Verify-code race**  
   Two concurrent requests with the same valid code could both pass the `used == False` check before either commits. **Risk:** Same code used twice. **Mitigation:** Low likelihood for 6-digit codes; tests assert that after one successful verify the code is marked used (sequential).

4. **Session handling**  
   `get_db()` yields a session; FastAPI closes it after the request. Uncommitted changes are rolled back on close. **Risk:** None identified. **Mitigation:** Tests assert commit by reading back after record.

5. **Support score / constituent parsing**  
   `_parse_support_score` accepts 1–5 only; `_parse_constituent` accepts 1/0, true/false, yes/no. **Risk:** Invalid values stored as NULL; acceptable. **Mitigation:** Unit tests for parsing and API round-trip.

6. **Stats with no events**  
   `GET /outreach/stats/{member_id}` with no rows returns `calls: 0, emails: 0, no_answers: 0, total: 0`. **Mitigation:** Test covers empty stats.

7. **my-history limit**  
   History is capped at 100 events, newest first. **Mitigation:** Documented; test can assert limit and ordering.

8. **SQL injection**  
   All queries use SQLAlchemy ORM or `select()` with parameters. No raw user input in SQL. **Risk:** None. **Mitigation:** N/A.

9. **Timezone**  
   `created_at` and `AuthCode.expires_at` use `DateTime(timezone=True)` and `_utcnow()`. **Mitigation:** Tests assert events have `created_at` and history returns ISO format.

10. **Boolean in SQLite**  
    SQLAlchemy maps Python `bool` to SQLite integer 0/1. **Mitigation:** Test stores `constituent=True/False` and asserts round-trip.

---

## Test coverage

- **tests/test_db.py** — `init_db()` (tables + migrations idempotent), `get_db()` yields session, schema (columns exist).
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
