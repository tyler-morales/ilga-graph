# Status report

Snapshot of what’s broken, buggy, missing, and what to do before deployment.

---

## Summary

| Area | Status | Notes |
|------|--------|--------|
| **Tests** | ✅ Pass | 301 tests pass; 1 deprecation warning (aiosqlite) |
| **Lint** | ✅ Pass | `make lint` passes (Ruff check + format) |
| **Docs build** | ✅ Builds | All pages in nav (status-report, db-and-outreach) |
| **Deployment readiness** | ✅ Documented | See [Deployment](deployment.md); checklist below |
| **Procfile** | ✅ Complete | Uses `scripts/start_web.sh`; honors `$PORT` (default 8000) |

---

## Broken / buggy

- **Nothing known.** Lint passes; docs nav includes status-report and db-and-outreach. Tests pass; app starts.

---

## Missing / incomplete

1. **Procfile:** Uses `scripts/start_web.sh`, which binds to `$PORT` (default 8000). Set **env vars** in your platform (e.g. `ILGA_LOAD_ONLY=1`, `ILGA_PROFILE=prod`, `ILGA_CORS_ORIGINS`, `ILGA_AUTH_SECRET`) — see [Deployment](deployment.md).

2. **Optional letter template:** Add `static/advocacy/letter-template.pdf` if desired; if missing, the app returns a clear message and the rest of the flow works.

---

## What to do to get closer to deployment

### 1. Lint

`make lint` passes. Run before deploy if you’ve made local changes.

### 2. Production checklist (from [Deployment](deployment.md))

1. Set **`ILGA_PROFILE=prod`** and **`ILGA_LOAD_ONLY=1`** (e.g. in platform env or start command).
2. Set **`ILGA_CORS_ORIGINS`** to your public URL(s) (e.g. `https://landofkei.org`).
3. Set **`ILGA_AUTH_SECRET`** to a new random value.
4. (Recommended) Set **`ILGA_API_KEY`** so `/graphql` and programmatic API routes are protected.
5. Populate **cache/** (or `ILGA_CACHE_DIR`) with full or seed data (e.g. run `make scrape` locally, then upload or rsync to the server / persistent volume).
6. Ensure **data/** (or `ILGA_DB_PATH`’s directory) is writable and, on PaaS, on a **persistent volume**.
7. Configure **SMTP** if you want email verification for advocates (otherwise codes are console-only).
8. Use **HTTPS** in production (reverse proxy or platform-managed TLS).

### 3. Platform-specific

- **Railway / Render:** Set env vars in the dashboard; the default Procfile uses `scripts/start_web.sh` (honors `$PORT`). Attach a persistent volume for `data/` (and optionally `cache/`).
- **Health check:** Use **GET /health**. `ready: true` when `len(state.members) > 0`; otherwise app is up but not “ready” (e.g. empty cache).

### 4. Optional polish

- Resolve the aiosqlite deprecation warning in tests (low priority).

---

## No known runtime bugs

- **Health:** `/health` returns counts and `ready`; exempt from API key.
- **Auth/outreach:** `/auth`, `/outreach` exempt; DB created via `init_db()` on startup.
- **SSR pages:** `/advocacy`, `/explore`, `/intelligence` exempt when `ILGA_API_KEY` is set.
- **GraphQL:** `/graphql` correctly requires `X-API-Key` when `ILGA_API_KEY` is set.

No open FIXME/TODO that clearly block deployment. Before go-live: set **env vars** (see checklist above), **populate cache/**, and use a **persistent data/** (and optionally cache/) volume on PaaS.
