# Environment variables

The app loads `.env` from the project root via `python-dotenv`. Copy `.env.example` to `.env` and adjust.

---

## Profiles

`ILGA_PROFILE` sets defaults for the flags below. Any variable can override the profile.

| Profile | `ILGA_DEV_MODE` | `ILGA_SEED_MODE` | `ILGA_CORS_ORIGINS` | Notes |
|---------|------------------|-------------------|----------------------|--------|
| `dev` | `1` | `1` | `*` | Lighter scrape, seed fallback when cache missing. |
| `prod` | `0` | `0` | *(must set)* | No seed; CORS and API key should be set. |

---

## Full reference

| Variable | Default | Description |
|----------|---------|-------------|
| **`ILGA_PROFILE`** | `dev` | `dev` or `prod`. |
| `ILGA_GA_ID` | `18` | General Assembly ID (e.g. 104th GA). |
| `ILGA_SESSION_ID` | `114` | Session ID. |
| `ILGA_BASE_URL` | `https://www.ilga.gov/` | ILGA site base URL. |
| `ILGA_CACHE_DIR` | `cache` | Directory for scraped JSON cache. |
| `ILGA_MOCK_DIR` | `mocks/dev` | Seed/mock data directory. |
| `ILGA_DEV_MODE` | *profile* | `1` = lighter scrape; `0` = production. |
| `ILGA_SEED_MODE` | *profile* | `1` = use seed when cache missing; `0` = require cache. |
| `ILGA_INCREMENTAL` | `0` | `1` = incremental bill scrape (new/changed only). |
| `ILGA_LOAD_ONLY` | `0` | `1` = API only loads from cache (no scrape on startup). `make dev` / `make run` set this. |
| `ILGA_MEMBER_LIMIT` | `0` | Max members per chamber (0 = all). |
| `ILGA_TEST_MEMBER_URL` | *(empty)* | Optional single member URL for testing. |
| `ILGA_TEST_MEMBER_CHAMBER` | `Senate` | Chamber for the test member URL. |
| `ILGA_CORS_ORIGINS` | *profile* | Comma-separated CORS origins. |
| `ILGA_API_KEY` | *(empty)* | If set, non-exempt routes require `X-API-Key` header. |
| `ILGA_APP_BASE_URL` | `http://127.0.0.1:8000` | Public URL of this app; used in the startup banner, logs, canonical URL, and Open Graph tags. Set in production (e.g. `https://landofkei.org`). |
| `ILGA_DOCS_BASE_URL` | *(empty)* | Optional docs site URL for the startup banner when different from the app. |
| `ILGA_SITE_NAME` | `Kei Truck Freedom` | Site name for Open Graph and Twitter cards. |
| `ILGA_META_DESCRIPTION` | *(short default)* | Default meta description and OG/Twitter description. |
| `ILGA_OG_IMAGE_URL` | *(empty)* | Optional absolute URL for share card image (1200×630). If unset, defaults to `APP_BASE_URL/static/og-image.png`. |
| `ILGA_UMAMI_WEBSITE_ID` | *(empty)* | When set, the base template injects the Umami analytics script. Get the ID from [Umami Cloud](https://umami.is) → Add website. |
| `ILGA_UMAMI_SCRIPT_URL` | `https://cloud.umami.is/script.js` | Tracker script URL; override for self-hosted Umami. |
| `ILGA_VOTE_BILL_URLS` | *(built-in)* | Comma-separated bill status URLs for votes/slips. |
| `ILGA_SMTP_HOST` | *(empty)* | SMTP server (e.g. `smtp-relay.brevo.com`). **If unset, verification codes are logged to the terminal only** — no email is sent. See [Email (Brevo)](email-brevo.md). |
| `ILGA_SMTP_PORT` | `587` | SMTP port. |
| `ILGA_SMTP_USER` | *(empty)* | SMTP login (Brevo: use the SMTP login from the SMTP tab, not your account email). |
| `ILGA_SMTP_PASS` | *(empty)* | SMTP key or password. |
| `ILGA_SMTP_FROM` | *(empty)* | Sender address for verification emails. |
| `ILGA_SMTP_TLS` | `1` | Use TLS (1) or not (0). |

---

## Feature flags

Client-side UX toggles (e.g. ZIP search loading animation) use a **single registry** in `src/ilga_graph/config.py` (`_FEATURE_REGISTRY`). Each flag has:

- **Profile default** — In `dev`, flags default on where it makes sense; in `prod`, they default off so production stays conservative until you opt in.
- **Env override** — Set `ILGA_FEATURE_<NAME>=1` or `0` in `.env` to override the profile default.

The app exposes only **client-facing** flags (those with `expose_to_client: True`) to templates as `features`; the advocacy index passes them to JS as `window.__ILGA_FEATURES`. Adding a new flag = one entry in the registry (no need to wire it in each route).

| Variable | Dev default | Prod default | Description |
|----------|-------------|--------------|-------------|
| `ILGA_FEATURE_ZIP_LOADING_ANIMATION` | `1` | `0` | ZIP search: show truck loading animation before member cards. `1` = on, `0` = off. |

See `config.py` for the full registry and `get_client_features()`.

---

## Production checklist

```bash
ILGA_PROFILE=prod
ILGA_CORS_ORIGINS=https://landofkei.org
ILGA_API_KEY=your-secret-key
```

The app warns at startup if CORS or API key are missing in prod.

---

## PaaS port

The app does not read a `PORT` environment variable. Uvicorn defaults to port 8000. On platforms that assign a port (e.g. Railway, Render), set the **start command** to pass it explicitly, for example:

```bash
uvicorn ilga_graph.main:app --app-dir src --host 0.0.0.0 --port $PORT
```

See [Deployment](deployment.md) for the full production start command.
