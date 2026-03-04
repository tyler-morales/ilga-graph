# Performance and resilience

How the app improves perceived performance and handles offline or failed requests.

---

## Static asset caching

- **Path:** `/static` (CSS, JS, images).
- **Implementation:** The app mounts a custom `StaticFilesWithCache` (in `main.py`):
  - **Minified assets** (`.min.css`, `.min.js`): `Cache-Control: public, max-age=31536000, immutable` (1 year). Used in production when `ILGA_PROFILE=prod`.
  - **All other static files:** `Cache-Control: public, max-age=3600` (1 hour).
- **Effect:** Browsers and CDNs cache assets for repeat visits; minified production assets are cached long-term.

## Response compression

- **Implementation:** `GZipMiddleware` (Starlette) is registered in `middleware.py`. Responses larger than 500 bytes that the client accepts with `Accept-Encoding: gzip` are compressed.
- **Effect:** Smaller transfer size for HTML, CSS, and JS, improving FCP and LCP on slow networks.
- **Production:** A reverse proxy (e.g. Nginx, Caddy) can also enable gzip or Brotli; the app’s middleware ensures compression even when no proxy is used.

## Minified CSS and JS (production)

- **Build:** Run `make minify` (or `npm run minify`) from the project root to generate `.min.css` and `.min.js` under `src/ilga_graph/static/`. Requires Node.js and installs `clean-css` and `terser`.
- **Serving:** When `ILGA_PROFILE=prod`, the app links to the `.min` assets in `base.html`; when the profile is not prod, it links to the unminified files. Ensure minified files exist before deploying production (run `make minify` in CI or before release).
- **Effect:** Smaller payloads and faster parsing on the client, improving Lighthouse performance scores.

---

## Offline and failure messaging

When critical requests fail (network error, server error, or the user goes offline), the app shows clear, friendly messages instead of leaving the user with a stuck spinner or no feedback.

### Connection error banner

- **When:** Any HTMX request fails (e.g. `htmx:responseError` or `htmx:sendError`). This covers intelligence tabs, advocacy search, bill explanation, and other HTMX-driven content.
- **Where:** Site-wide. The banner lives in `base.html` and appears below the beta banner (if present) and any current action banner.
- **Message:** “We're having trouble. Check your connection and try again.”
- **Behavior:** Dismissible (× button). Uses `role="alert"` and `aria-live="polite"` so screen readers announce it. Keyboard-accessible (dismiss button is focusable, visible focus ring).
- **Optional use from fetch:** The global script exposes `window.showConnectionError()`. You can call it from `fetch().catch()` in critical flows if you want the same banner instead of (or in addition to) in-context error text.

### Site banners (robust component)

- **Slim bar (beta + current action):** Same layout for both — left emoji + pill (BETA / ACTION), **centered** message text, dismiss (×) button **top-right**. Max-width 1120px; padding 10px 32px (room for dismiss). Beta: warm gradient + orange accent; action: teal gradient + teal accent. Dismissible per session; action bar uses `ilga_current_action_banner_dismissed` (top) or `ilga_current_action_banner_inline_dismissed` (inline slot). Templates: `base.html` (beta), `_current_action_banner.html` (action; slot `top` default, `inline` when set).
- **Priority callout (Updates):** Used on **Updates** under "Where we are" when `active_campaign` is set. In-content callout (not a full-width banner): teal palette and left-accent like the current action banner, eyebrow + "PRIORITY" pill, title, message, underlined link CTA. Template `_priority_callout.html`; CSS `.priority-callout` in base.css. Reusable callout component; campaign banner (`_campaign_banner.html`, `.campaign-banner` in advocacy-form.css) retained for potential Advocacy index use.
- **Placements:** (1) **Top bar** — beta (if `ILGA_BETA_BANNER`) then current action slim bar in `base.html`. (2) **Updates page** — priority callout only (no slim bar inline); `updates.html` overrides `current_action_banner` block empty and includes `_priority_callout.html` in main content.
- **Behavior:** Slim bars: dismissible per session; `role="region"`, `aria-label`; keyboard-accessible dismiss with visible focus ring.

### Offline indicator

- **When:** The browser reports that the user is offline (`navigator.onLine === false`), via the `offline` event. It is hidden again on the `online` event.
- **Message:** “You're offline.”
- **Behavior:** No dismiss button; it auto-hides when the user is back online. Uses `role="status"` and `aria-live="polite"`.

### In-context errors (unchanged)

The advocacy drawer, hero sign-in, report-bug form, and explore graph already show their own error messages (e.g. “Couldn't load. Try again.” in the drawer, “Network error — try again” for sign-in). The global connection-error banner does not replace those; it covers HTMX and any other flows that did not previously show a clear message.

---

## Related

- [Deployment](deployment.md) — production checklist and env vars.
- [Environment variables](environment-variables.md) — full list including optional tuning.
