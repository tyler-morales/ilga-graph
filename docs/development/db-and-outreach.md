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
