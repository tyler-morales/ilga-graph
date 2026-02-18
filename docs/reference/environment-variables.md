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
| `ILGA_VOTE_BILL_URLS` | *(built-in)* | Comma-separated bill status URLs for votes/slips. |

---

## Production checklist

```bash
ILGA_PROFILE=prod
ILGA_CORS_ORIGINS=https://your-app.example.com
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
