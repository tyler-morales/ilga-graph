# Deployment

How to run the ILGA Graph app in production so you can share a link (VPS, Railway, Render, etc.).

---

## What you need

- **One long-running process** — the FastAPI app (uvicorn). No separate database server; the app uses SQLite in a file.
- **Cache directory** — populated with scraped data (~200 MB). The app reads from `cache/` at startup (or `ILGA_CACHE_DIR`).
- **Writable directory** — for `data/ilga.db` (auth + outreach). Default path: `data/ilga.db` (or `ILGA_DB_PATH`).
- **Working directory** — the process must run from the **project root** (the directory that contains `cache/`, `data/`, `src/`). Paths are relative by default.

---

## Start command

From the **project root**:

```bash
ILGA_LOAD_ONLY=1 ILGA_PROFILE=prod uvicorn ilga_graph.main:app --app-dir src --host 0.0.0.0
```

- `ILGA_LOAD_ONLY=1` — load from cache only (no scraping on startup). Required for production.
- `ILGA_PROFILE=prod` — production profile (no dev bar, no seed fallback).
- `--app-dir src` — so uvicorn finds the `ilga_graph` package.
- `--host 0.0.0.0` — bind to all interfaces (needed when the host is remote or in a container).

**Port:** The repo **Procfile** runs `scripts/start_web.sh`, which uses `$PORT` when set (Railway, Render) and defaults to 8000 otherwise. No need to override the start command for port. For a custom start command, use:

```bash
ILGA_LOAD_ONLY=1 ILGA_PROFILE=prod uvicorn ilga_graph.main:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}
```

---

## Required environment variables (production)

| Variable | Purpose |
|----------|---------|
| `ILGA_PROFILE` | `prod` |
| `ILGA_LOAD_ONLY` | `1` |
| `ILGA_CORS_ORIGINS` | Your front-end origin(s), e.g. `https://landofkei.org` (comma-separated if multiple). The app warns at startup if this is missing in prod. |
| `ILGA_AUTH_SECRET` | A strong random string for signing session cookies. Change from the default in production. |
| `ILGA_API_KEY` | (Recommended.) If set, non-exempt API routes require the `X-API-Key` header. The app warns if empty in prod. |

**Optional but recommended for full functionality:**

| Variable | Purpose |
|----------|---------|
| `ILGA_APP_BASE_URL` | Public URL of the app (e.g. `https://landofkei.org`). When set, the startup banner and logs show this URL instead of `http://127.0.0.1:8000`. |
| `ILGA_DB_PATH` | Path to SQLite DB (default: `data/ilga.db`). Use an absolute path if the process cwd is not the project root. |
| `ILGA_CACHE_DIR` | Path to cache directory (default: `cache`). Use an absolute path if needed. |
| `ILGA_SMTP_*` | SMTP settings so verification codes are emailed to users. **If unset, codes are logged to the terminal only** (no email sent) — fine for dev; set these in production. See [Email (Brevo)](email-brevo.md) for setup. |

See [Environment variables](environment-variables.md) for the full list.

---

## Cache and data layout

- **Cache:** Put your scraped data (e.g. from `make scrape` locally) in the `cache/` directory at project root — or at the path given by `ILGA_CACHE_DIR`. The app expects files such as `members.json`, `bills.json`, `committees.json`, `zip_to_district.json`, `scorecards.json`, `moneyball.json`, etc. Without a populated cache, the app will start but `state.members` will be empty and `/health` will report `ready: false`.
- **Database:** The app creates `data/ilga.db` (or `ILGA_DB_PATH`) on first run via `init_db()`. Ensure the process can write to that directory. On PaaS, use a **persistent volume** mounted at `data/` (and optionally at `cache/`) so the DB and cache survive redeploys.

---

## Health check

- **GET /health** — Returns JSON with `status`, `ready` (true when `len(state.members) > 0`), and counts (members, bills, committees, vote_events). Use this for platform health checks or load balancers.
- **GET /health** is exempt from API key middleware.

---

## Quick reference by platform

| Platform | Notes |
|----------|--------|
| **VPS** | SSH in, clone repo, install deps (`pip install -e .` or use a venv), upload or rsync `cache/`, set env (e.g. in `.env` or systemd), run the start command. Use Nginx or Caddy as reverse proxy with HTTPS. Run the app with systemd so it restarts on reboot. |
| **Vultr (Ubuntu)** | Step-by-step with common pitfalls: [Vultr deployment guide](vultr-deployment-guide.md) (SSH user, Python version, venv activation, startup wait, UFW before Certbot). |
| **Railway / Render** | Connect the repo, set env vars, set **root directory** to the project root (or ensure start command runs from it). Add a **persistent volume** for `data/` (and optionally `cache/`), then upload your cache into the volume or copy it in via a one-off job. Set the start command to include `--port $PORT` if the platform sets `PORT`. |
| **Fly.io** | Use a Dockerfile or `fly launch`, attach a **Fly Volume** for `data/` (and optionally `cache/`). Run the same uvicorn command; ensure the container’s working directory is the project root. |

---

## Advocacy systems (ready for deployment)

The advocacy flow is self-contained and works in production with the same env as above:

- **Routes:** `/` redirects to `/advocacy`. Advocacy page, drawer (call/email/no-answer), search, and wrap-up are all served server-side.
- **Auth:** `/auth` and `/outreach` are exempt from API-key middleware so browser users can sign in and record outreach.
- **SSR pages:** `/advocacy`, `/explore`, and `/intelligence` (and sub-pages) are exempt so the full site works when `ILGA_API_KEY` is set; the key then only protects `/graphql` and programmatic API routes.
- **DB:** `init_db()` runs on startup and creates `data/ilga.db` (or `ILGA_DB_PATH`) and the directory if needed. No manual migrations.
- **Letter template:** Optional. Add `static/advocacy/letter-template.pdf` to serve it at `/advocacy/letter-template.pdf`; if missing, the app returns a clear message and the rest of the flow still works.

---

## Checklist before go-live

1. Set `ILGA_PROFILE=prod`, `ILGA_LOAD_ONLY=1`.
2. Set `ILGA_CORS_ORIGINS` to your public URL(s) (e.g. `https://landofkei.org`).
3. Set `ILGA_AUTH_SECRET` to a new random value.
4. (Recommended) Set `ILGA_API_KEY` and protect non-exempt routes.
5. Populate `cache/` (or `ILGA_CACHE_DIR`) with full or seed data.
6. Ensure `data/` (or `ILGA_DB_PATH`’s directory) is writable and, on PaaS, on a persistent volume.
7. Configure SMTP if you want email verification for advocates.
8. Use HTTPS in production (reverse proxy or platform-managed TLS).
