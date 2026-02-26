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
| `ILGA_APP_BASE_URL` | `http://127.0.0.1:8000` | Public URL of this app; used in the startup banner, logs, canonical URL, and Open Graph tags. **Required for share cards:** set in production to your public HTTPS URL (e.g. `https://landofkei.org`) so `og:image` is reachable by social crawlers; otherwise the default points to localhost and the image will be reported as invalid/unreachable. In production the site should be served over HTTPS and this value **must** use `https://`. |
| `ILGA_DOCS_BASE_URL` | *(empty)* | Optional docs site URL for the startup banner when different from the app. |
| `ILGA_SITE_NAME` | `The Land of Kei` | Advocacy group / site brand used in the browser tab title, UI, footer, Open Graph and Twitter cards, and verification email subject. |
| `ILGA_META_DESCRIPTION` | Cause-tailored default (advocate with The Land of Kei, Kei vehicle registration, 625 ILCS 5/3-401(c-1)) | Default meta description and OG/Twitter description. |
| `ILGA_OG_IMAGE_URL` | *(empty)* | Optional absolute URL for share card image (1200×630). If unset, defaults to `APP_BASE_URL/static/og-image.png`. Set `ILGA_APP_BASE_URL` in production so this default is a public HTTPS URL; or set this explicitly to a full URL (e.g. CDN) if you host the image elsewhere. |
| `ILGA_UMAMI_WEBSITE_ID` | *(empty)* | When set, the base template injects the Umami analytics script. Get the ID from [Umami Cloud](https://umami.is) → Add website. |
| `ILGA_UMAMI_SCRIPT_URL` | `https://cloud.umami.is/script.js` | Tracker script URL; override for self-hosted Umami. |
| `ILGA_BETA_BANNER` | `0` | Set to `1` to show the site-wide beta notice (“new site, may have bugs” + Report a bug link). Set to `0` to hide. Banner is dismissible per session. When on, "Report a bug" also appears in the advocacy drawer header and at the end of the call and email flows. |
| `ILGA_BETA_BANNER_FEEDBACK_URL` | *(empty)* | If set, “Report a bug” goes to this URL (e.g. Google Form, GitHub). **Default:** unset → link goes to the **in-app form** at `/report-bug` (no GitHub or external service). |
| `ILGA_BETA_BANNER_EMAIL` | *(empty)* | When someone submits the in-app bug report form, a notification is sent to this address **if SMTP is configured**. Email includes: timestamp, email, issue, page URL, screenshot (inline image and/or link to `/report-bug/attachments/…`, or “No image sent”), and metadata (IP, User-Agent). Set `ILGA_APP_BASE_URL` in production so screenshot links work. Example: `feedback@landofkei.org`. |
| `ILGA_BUG_REPORT_UPLOAD_DIR` | `data/bug_report_uploads` | Directory for bug-report screenshot uploads. Empty = image upload disabled. Created on first upload. |
| `ILGA_BUG_REPORT_MAX_IMAGE_MB` | `5` | Max size per image (MB). Larger uploads are rejected. |
| `ILGA_RATE_LIMIT_BUG_REPORT_PER_HOUR` | `10` | Max bug report submissions per client IP per hour. In-memory; resets on restart. |
| `ILGA_RATE_LIMIT_REQUEST_CODE_PER_15MIN` | `3` | Max auth “request code” attempts per IP and per email per 15 minutes. |
| `ILGA_RATE_LIMIT_VERIFY_CODE_PER_15MIN` | `10` | Max auth “verify code” attempts per IP per 15 minutes. |
| `ILGA_TURNSTILE_SITE_KEY` | *(empty)* | Cloudflare Turnstile **site key** (public). When set with `ILGA_TURNSTILE_SECRET_KEY`, the bug report form shows the CAPTCHA widget and the server verifies the token. Free tier: 1M requests/month. [Dashboard](https://dash.cloudflare.com/?to=/:account/turnstile). |
| `ILGA_TURNSTILE_SECRET_KEY` | *(empty)* | Cloudflare Turnstile **secret key** (server-only). Required for server-side verification when using Turnstile. |
| `ILGA_VOTE_BILL_URLS` | *(built-in)* | Comma-separated bill status URLs for votes/slips. |
| `ILGA_AUTH_COOKIE_MAX_AGE` | `2592000` (30 days) | Session cookie max-age in seconds. |
| `ILGA_AUTH_SECRET` | `dev-secret-change-me` | Secret for signing session and CSRF tokens. **Set a strong random value in production.** |
| `ILGA_ADMIN_EMAILS` | *(empty)* | Comma-separated email addresses allowed to access the admin area (`/admin`, `/admin/updates`, `/admin/users`, `/admin/outreach`). Sign in at `/admin/login` (same email-code flow as the rest of the site). |
| `ILGA_SMTP_HOST` | *(empty)* | SMTP server (e.g. `smtp-relay.brevo.com`). **If unset, verification codes are logged to the terminal only** — no email is sent. See [Email (Brevo)](email-brevo.md). |
| `ILGA_SMTP_PORT` | `587` | SMTP port. |
| `ILGA_SMTP_USER` | *(empty)* | SMTP login (Brevo: use the SMTP login from the SMTP tab, not your account email). |
| `ILGA_SMTP_PASS` | *(empty)* | SMTP key or password. |
| `ILGA_SMTP_FROM` | *(empty)* | Sender address for verification emails. |
| `ILGA_SMTP_TLS` | `1` | Use TLS (1) or not (0). |
| `ILGA_CSP_REPORT_URI` | *(empty)* | Optional endpoint for CSP violation reports. When set, the CSP header includes `report-uri <value>`. Omit in dev to rely on console reporting. |
| `ILGA_CSP_ENFORCE` | `0` | When `1`, the app sends **Content-Security-Policy** (enforcing). When `0` (default), it sends **Content-Security-Policy-Report-Only** so violations are reported but not blocked. Use report-only first, then switch to enforce once the policy is validated. |
| `ILGA_HSTS_ENABLED` | `0` | When `1` (and `ILGA_PROFILE=prod`), the app sends `Strict-Transport-Security: max-age=31536000; includeSubDomains`. **Only enable when the entire site is served over HTTPS** (e.g. behind a reverse proxy that terminates TLS). Do not enable for local dev or HTTP-only deployments. |

**Sitemap and robots:** The app serves `/sitemap.xml` (key pages: `/`, `/advocacy`, `/intelligence`, `/explore`) and `/robots.txt` (allow all, with a `Sitemap:` line). Both use `ILGA_APP_BASE_URL` for absolute URLs, so set it in production for correct discovery by search engines.

---

## Feature flags

Client-side UX toggles (e.g. ZIP search loading animation) use a **single registry** in `src/ilga_graph/config.py` (`_FEATURE_REGISTRY`). Each flag has:

- **Profile default** — In `dev`, flags default on where it makes sense; in `prod`, they default off so production stays conservative until you opt in.
- **Env override** — Set `ILGA_FEATURE_<NAME>=1` or `0` in `.env` to override the profile default.

The app exposes only **client-facing** flags (those with `expose_to_client: True`) to templates as `features`; the advocacy index passes them to JS as `window.__ILGA_FEATURES`. Adding a new flag = one entry in the registry (no need to wire it in each route). The truck loading animation was moved to the bug report success view (`/report-bug?submitted=1`); ZIP search now uses a direct HTMX swap with no loading UI.

See `config.py` for the full registry and `get_client_features()`.

---

## Security (forms and submissions)

**Headers:** The app sends `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: strict-origin-when-cross-origin`. It also sends a **Content-Security-Policy** (report-only by default; set `ILGA_CSP_ENFORCE=1` to enforce). The CSP allows scripts/styles from `'self'` and trusted CDNs: `unpkg.com`, `cdn.jsdelivr.net`, `cloud.umami.is`, `challenges.cloudflare.com`, `d3js.org`. Optional `ILGA_CSP_REPORT_URI` sends violation reports to your endpoint. For HTTPS-only deployments, set `ILGA_HSTS_ENABLED=1` to send HSTS (or configure HSTS in your reverse proxy). **HTTPS and base URL:** In production, serve the site over HTTPS and set `ILGA_APP_BASE_URL` to your public `https://` URL so cookies, canonical URLs, and share cards behave correctly; the app warns at startup if prod and base URL is not HTTPS.

- **CSRF:** State-changing POSTs (bug report, auth request-code/verify-code, outreach record) require a valid CSRF token. The app sets an `XSRF-TOKEN` cookie (not HttpOnly) and expects the same value in the request body (`csrf_token`) so JS can send it with `fetch()`. Tokens are signed with `ILGA_AUTH_SECRET` and expire after 1 hour.
- **Rate limiting:** Bug report, request-code, and verify-code are rate-limited per IP (and per email for request-code) to reduce spam and brute-force. Limits are configurable (see table above); storage is in-memory and resets on process restart.
- **Input:** Bug report `page_url` is accepted only if it starts with `http://` or `https://` and is under 2048 characters. Description has a minimum length; optional reporter email is validated (format: local@domain with a dot in domain, length limits). Content is escaped in notification emails.
- **CAPTCHA (optional):** When `ILGA_TURNSTILE_SITE_KEY` and `ILGA_TURNSTILE_SECRET_KEY` are set, the bug report form shows a [Cloudflare Turnstile](https://www.cloudflare.com/products/turnstile/) widget. The server validates the token with Cloudflare’s siteverify API before accepting the report. Turnstile is free (1M requests/month) and supports invisible/managed modes.

---

## Production checklist

```bash
ILGA_PROFILE=prod
ILGA_CORS_ORIGINS=https://landofkei.org
ILGA_API_KEY=your-secret-key
```

The app warns at startup if CORS or API key are missing in prod.

---

## CI/CD deploy secrets (GitHub Actions only)

Automated deploy to Vultr (on push to `main`) uses **GitHub Actions secrets**, not `.env`. These are configured in the repo under Settings → Secrets and variables → Actions: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`. The app never reads them. See [Vultr deployment guide](vultr-deployment-guide.md#automated-deploy-cicd) for one-time server setup.

---

## PaaS port

The app does not read a `PORT` environment variable. Uvicorn defaults to port 8000. On platforms that assign a port (e.g. Railway, Render), set the **start command** to pass it explicitly, for example:

```bash
uvicorn ilga_graph.main:app --app-dir src --host 0.0.0.0 --port $PORT
```

See [Deployment](deployment.md) for the full production start command.
