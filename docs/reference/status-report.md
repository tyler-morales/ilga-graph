# Status report

Snapshot of what’s broken, buggy, missing, and what to do before deployment.

---

## Summary

| Area | Status | Notes |
|------|--------|--------|
| **Tests** | ✅ Pass | 301 tests pass; 1 deprecation warning (aiosqlite) |
| **Lint** | ⚠️ Fails | 69 ruff errors (fixable with `make lint-fix` or manual fixes) |
| **Docs build** | ✅ Builds | One page not in nav: `development/db-and-outreach.md` |
| **Deployment readiness** | ✅ Documented | See [Deployment](deployment.md); checklist below |
| **Procfile** | ⚠️ Incomplete | No `--port $PORT`; override on Railway/Render |

---

## Broken / buggy

- **Lint (ruff):** 69 errors across the repo. Examples:
  - **tests:** Unused imports in `test_auth_outreach.py` (`timedelta`, `timezone`, `AuthCode`); line too long in `test_db.py`.
  - **scripts:** `refresh_member_photos.py` — E402 (imports not at top), E501 (line length). `seed_outreach.py` — E402, E501 on long seed data lines.
  - Running `make lint-fix` will auto-fix 21 of them; the rest need manual edits (e.g. breaking long strings in seed data).
- **Docs nav:** `docs/development/db-and-outreach.md` exists but is not listed in `mkdocs.yml` nav, so it doesn’t appear in the doc site.

Nothing in the app code is obviously *broken* at runtime; tests pass and the app starts. The main gaps are lint and doc nav.

---

## Missing / incomplete

1. **Procfile for PaaS:** Current Procfile is:
   ```text
   web: uvicorn ilga_graph.main:app --app-dir src --host 0.0.0.0
   ```
   It does **not** set `ILGA_LOAD_ONLY=1`, `ILGA_PROFILE=prod`, or `--port $PORT`. For Railway/Render you must **override the start command** in the dashboard (see [Deployment](deployment.md)) with something like:
   ```bash
   ILGA_LOAD_ONLY=1 ILGA_PROFILE=prod uvicorn ilga_graph.main:app --app-dir src --host 0.0.0.0 --port $PORT
   ```
   So: Procfile is a minimal default; production env and port must be set by the platform.

2. **Doc page not in nav:** Add `development/db-and-outreach.md` to the Development section in `mkdocs.yml` if you want it visible in the doc site.

3. **Optional letter template:** Deployment doc says you can add `static/advocacy/letter-template.pdf`; if missing, the app returns a clear message and the rest of the flow works.

---

## What to do to get closer to deployment

### 1. Fix lint (recommended before deploy)

```bash
make lint-fix   # fixes 21 automatically
# Then manually fix remaining E501 (long lines) and E402 in scripts if you care about clean lint
```

### 2. Production checklist (from [Deployment](deployment.md))

1. Set **`ILGA_PROFILE=prod`** and **`ILGA_LOAD_ONLY=1`** (e.g. in platform env or start command).
2. Set **`ILGA_CORS_ORIGINS`** to your public URL(s).
3. Set **`ILGA_AUTH_SECRET`** to a new random value.
4. (Recommended) Set **`ILGA_API_KEY`** so `/graphql` and programmatic API routes are protected.
5. Populate **cache/** (or `ILGA_CACHE_DIR`) with full or seed data (e.g. run `make scrape` locally, then upload or rsync to the server / persistent volume).
6. Ensure **data/** (or `ILGA_DB_PATH`’s directory) is writable and, on PaaS, on a **persistent volume**.
7. Configure **SMTP** if you want email verification for advocates (otherwise codes are console-only).
8. Use **HTTPS** in production (reverse proxy or platform-managed TLS).

### 3. Platform-specific

- **Railway / Render:** Override start command to include `--port $PORT` and the env vars above. Attach a persistent volume for `data/` (and optionally `cache/`).
- **Health check:** Use **GET /health**. `ready: true` when `len(state.members) > 0`; otherwise app is up but not “ready” (e.g. empty cache).

### 4. Optional polish

- Add `development/db-and-outreach.md` to `mkdocs.yml` nav.
- Fix remaining ruff errors in scripts (E402, E501) so `make lint` passes.
- Resolve the aiosqlite deprecation warning in tests (low priority).

---

## No known runtime bugs

- **Health:** `/health` returns counts and `ready`; exempt from API key.
- **Auth/outreach:** `/auth`, `/outreach` exempt; DB created via `init_db()` on startup.
- **SSR pages:** `/advocacy`, `/explore`, `/intelligence` exempt when `ILGA_API_KEY` is set.
- **GraphQL:** `/graphql` correctly requires `X-API-Key` when `ILGA_API_KEY` is set.

No open FIXME/TODO in the codebase that clearly block deployment; the main blockers are **lint**, **Procfile/env/port** for your chosen platform, and **cache + persistent data directory**.
