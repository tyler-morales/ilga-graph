# Performance and resilience

How the app improves perceived performance and handles offline or failed requests.

---

## Static asset caching

- **Path:** `/static` (CSS, JS, images).
- **Implementation:** The app mounts a custom `StaticFilesWithCache` (in `main.py`) that sets `Cache-Control: public, max-age=3600` (1 hour) on all static file responses.
- **Effect:** Browsers and CDNs can cache assets for repeat visits; no explicit cache headers were set before, so this improves repeat-load performance.
- **Future:** If you introduce versioned or hashed asset URLs (e.g. `?v=RELEASE` or hashed filenames), you can extend the subclass to use a longer `max-age` (e.g. 1 year) for those paths and keep a shorter one for unversioned files.

---

## Offline and failure messaging

When critical requests fail (network error, server error, or the user goes offline), the app shows clear, friendly messages instead of leaving the user with a stuck spinner or no feedback.

### Connection error banner

- **When:** Any HTMX request fails (e.g. `htmx:responseError` or `htmx:sendError`). This covers intelligence tabs, advocacy search, bill explanation, and other HTMX-driven content.
- **Where:** Site-wide. The banner lives in `base.html` and appears below the beta banner (if present).
- **Message:** “We're having trouble. Check your connection and try again.”
- **Behavior:** Dismissible (× button). Uses `role="alert"` and `aria-live="polite"` so screen readers announce it. Keyboard-accessible (dismiss button is focusable, visible focus ring).
- **Optional use from fetch:** The global script exposes `window.showConnectionError()`. You can call it from `fetch().catch()` in critical flows if you want the same banner instead of (or in addition to) in-context error text.

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
