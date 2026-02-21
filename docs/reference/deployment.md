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
| `ILGA_APP_BASE_URL` | Public URL of the app (e.g. `https://landofkei.org`). Used for the startup banner, logs, canonical URLs, and Open Graph share cards. **Use HTTPS in production** and set this to your **https://** URL so canonical URLs, share cards, and bug-report links are correct. |
| `ILGA_DB_PATH` | Path to SQLite DB (default: `data/ilga.db`). Use an absolute path if the process cwd is not the project root. |
| `ILGA_CACHE_DIR` | Path to cache directory (default: `cache`). Use an absolute path if needed. |
| `ILGA_SMTP_*` | SMTP settings so verification codes are emailed to users. **If unset, codes are logged to the terminal only** (no email sent) — fine for dev; set these in production. See [Email (Brevo)](email-brevo.md) for setup. |
| `ILGA_UMAMI_WEBSITE_ID` | Optional. When set, the base template injects the [Umami](https://umami.is) analytics script (Cloud or self-hosted). Sign up at umami.is → Add website → copy the website ID into this variable. See [Environment variables](environment-variables.md). |

See [Environment variables](environment-variables.md) for the full list.

---

## Cache and data layout

- **Cache:** Put your scraped data (e.g. from `make scrape` locally) in the `cache/` directory at project root — or at the path given by `ILGA_CACHE_DIR`. The app expects files such as `members.json`, `bills.json`, `committees.json`, `zip_to_district.json`, `scorecards.json`, `moneyball.json`, etc. Without a populated cache, the app will start but `state.members` will be empty and `/health` will report `ready: false`.
- **Database:** The app creates `data/ilga.db` (or `ILGA_DB_PATH`) on first run via `init_db()`. Ensure the process can write to that directory. On PaaS, use a **persistent volume** mounted at `data/` (and optionally at `cache/`) so the DB and cache survive redeploys.

---

## Performance and resilience

Static assets under `/static` are served with `Cache-Control: public, max-age=3600` so repeat visits are faster. When HTMX or network requests fail, the app shows a dismissible “We're having trouble” banner; when the user goes offline, a “You're offline” indicator is shown and hidden automatically. See [Performance and resilience](performance-resilience.md) for details.

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

## Connecting Umami (analytics)

To enable lightweight analytics with [Umami Cloud](https://umami.is) (free tier: 100k events/month, 3 websites, 6 months retention):

1. Sign up at **https://umami.is** and log in.
2. In the dashboard, go to **Websites** → **Add website**. Enter your site name and domain (e.g. `landofkei.org`).
3. Copy the **website ID** (a UUID) from the setup instructions or the website settings.
4. Set in your production environment (e.g. in `.env` on the server): `ILGA_UMAMI_WEBSITE_ID=your-website-id-here`
5. Redeploy or restart the app. The base template injects the script only when `ILGA_UMAMI_WEBSITE_ID` is set, so dev stays untracked unless you set it there too.

The tracker script URL defaults to `https://cloud.umami.is/script.js`. For self-hosted Umami, set `ILGA_UMAMI_SCRIPT_URL` to your instance's script URL.

---

## Troubleshooting

### OG image "invalid or unreachable" when sharing links

Social crawlers (Facebook, Twitter, LinkedIn, etc.) fetch the `og:image` URL from your page. The app builds that URL as `ILGA_APP_BASE_URL/static/og-image.png`. If `ILGA_APP_BASE_URL` is not set in production, it defaults to `http://127.0.0.1:8000`, so the meta tag points to localhost and crawlers cannot reach it.

**Fix:** In your production environment (e.g. `.env` on the server or platform env vars), set:

```bash
ILGA_APP_BASE_URL=https://landofkei.org
```

Use your actual public HTTPS URL (no trailing slash). Restart the app, then re-run your share-preview tool or clear the platform’s link cache so it refetches the image.

If the validator shows **127.0.0.1** in the "Open Graph tags found" list, the app is still using the default base URL—confirm `ILGA_APP_BASE_URL` is set where the app runs (and that the process was restarted after changing env).

---

## Checklist before go-live

1. Set `ILGA_PROFILE=prod`, `ILGA_LOAD_ONLY=1`.
2. Set `ILGA_CORS_ORIGINS` to your public URL(s) (e.g. `https://landofkei.org`).
3. Set `ILGA_APP_BASE_URL` to your public URL (e.g. `https://landofkei.org`) so canonical URLs, Open Graph share cards, and the sitemap/robots.txt URLs use the correct domain.
4. Set `ILGA_AUTH_SECRET` to a new random value.
5. (Recommended) Set `ILGA_API_KEY` and protect non-exempt routes.
6. Populate `cache/` (or `ILGA_CACHE_DIR`) with full or seed data.
7. Ensure `data/` (or `ILGA_DB_PATH`’s directory) is writable and, on PaaS, on a persistent volume.
8. Configure SMTP if you want email verification for advocates.
9. (Optional) Set `ILGA_UMAMI_WEBSITE_ID` for analytics; see **Connecting Umami** above.
10. **Use HTTPS in production** (reverse proxy or platform-managed TLS) and set **`ILGA_APP_BASE_URL`** to your public **https://** URL (e.g. `https://landofkei.org`) so canonical URLs, Open Graph share cards, and bug-report links are correct. The app logs a warning at startup if `ILGA_PROFILE=prod` and `ILGA_APP_BASE_URL` is not HTTPS.

**Security headers:** The app sends `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and a **Content-Security-Policy-Report-Only** (or enforcing CSP when `ILGA_CSP_ENFORCE=1`). For HTTPS deployments, set **`ILGA_HSTS_ENABLED=1`** so the app sends `Strict-Transport-Security` (or configure HSTS in your reverse proxy instead). See [Environment variables](environment-variables.md) for `ILGA_CSP_REPORT_URI`, `ILGA_CSP_ENFORCE`, and `ILGA_HSTS_ENABLED`.
