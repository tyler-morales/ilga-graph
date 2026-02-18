# Dev Bar — How It Works (Internals)

This document describes how the Dev Bar is implemented: URL contract, backend injection, template rendering, and client-side behavior.

---

## Overview

The dev bar is a global floating toolbar that activates when `?dev` is in the URL. It provides context-aware quick actions for testing features on any page without clicking through the normal UI.

**Entry points:**

1. **Any page with `?dev`** — Activates the dev bar globally. Persists via `sessionStorage`.
2. **Test page** — `GET /advocacy/test` — Standalone page with form and quick links (now uses `?dev` instead of `?test=1`).
3. **Keyboard shortcut** — `Ctrl+Shift+D` toggles the bar.

---

## Architecture

```
User hits /advocacy?dev
    │
    ├─ Server: base.html renders #dev-bar HTML (gated by dev_available Jinja2 global)
    │
    ├─ Client JS (IIFE in base.html):
    │   ├─ Detects ?dev in URL → stores in sessionStorage
    │   ├─ Shows #dev-bar, sets body padding
    │   ├─ Detects current path → shows context panel (advocacy or intelligence)
    │   ├─ If advocacy: fetches /api/dev/members for dropdown
    │   └─ If URL has member_id + view: auto-opens drawer via openAdvocacyDrawer() after 500ms
    │
    └─ Dev bar actions:
        ├─ Open Call / Open Email → calls openAdvocacyDrawer() directly
        ├─ Search ZIP → triggers advocacy form submit
        ├─ Intel Go → navigates to /intelligence/member/{id} or /intelligence/bill/{id}
        └─ Close → removes sessionStorage, hides bar
```

---

## URL contract

### Activation

| Param | Effect |
|-------|--------|
| `?dev` | Activates dev bar, stores in `sessionStorage` |
| `?dev=off` | Deactivates dev bar, removes `sessionStorage` |

### Advocacy deep-link (auto-open drawer)

| Query param | Required | Description |
|-------------|----------|-------------|
| `dev` | Yes | Activates dev mode |
| `zip` | No | 5-digit ZIP; default `60601`. Pre-fills hero input and drawer. |
| `member_id` | Yes (for auto-open) | Legislator ID (e.g. `S123`). |
| `view` | Yes (for auto-open) | `call` or `email`. Determines which drawer to open. |

**Example:** `/advocacy?dev&zip=60601&member_id=S123&view=call`

---

## Backend

### Config guard

In `main.py`, the Jinja2 template environment gets a global:

```python
templates.env.globals["dev_available"] = DEV_MODE
```

`DEV_MODE` comes from `config.py` and is `True` when `ILGA_DEV_MODE=1` (default in dev profile, `False` in prod). This means `base.html` only renders the dev bar markup when running in dev mode.

### Routes

**`GET /advocacy`** — Accepts optional `zip`, `member_id`, `view` query params. The `zip` param is passed to the template context to pre-fill the hero ZIP input. The `member_id` and `view` params are accepted by the route signature so FastAPI doesn't reject them; the dev bar JS reads them from the URL directly.

**`GET /api/dev/members`** — Returns the first 20 members as JSON (`[{"id": "...", "name": "..."}]`). Returns 404 when `DEV_MODE` is `False`. Exempt from API key middleware. Used by the dev bar to populate the member dropdown without bloating every page's template context.

**`GET /advocacy/test`** — Unchanged route handler. Template updated to use `?dev` instead of `?test=1`.

---

## Templates

### `base.html`

The dev bar is rendered at the bottom of `<body>`, inside `{% if dev_available %}`:

- **HTML:** `#dev-bar` div with header (badge, nav, shortcut, close button) and two context panels (`#dev-panel-advocacy`, `#dev-panel-intelligence`).
- **CSS:** Scoped styles for the dark floating bar, inputs, buttons, chips.
- **JS (IIFE):** Runs immediately:
  1. Checks URL for `?dev` → stores `__ilga_dev_mode` in `sessionStorage`.
  2. If `?dev=off` → removes `sessionStorage` key, exits.
  3. If `sessionStorage` has the key → shows the bar, adds body padding.
  4. Detects current path → shows appropriate context panel.
  5. On advocacy pages: fetches `/api/dev/members` to populate dropdown.
  6. If URL has `view` + `member_id` on advocacy pages: registers a `DOMContentLoaded` listener that calls `openAdvocacyDrawer()` after 500ms.
  7. Registers `Ctrl+Shift+D` keydown handler for toggle.
  8. Exposes `window.__devBar` with methods: `close()`, `openDrawer(view)`, `submitZip()`, `goIntelMember()`, `goIntelBill()`.

### `advocacy_test.html`

- Form action is `GET /advocacy` with hidden input `name="dev" value=""`.
- Quick links use `?dev&zip=...&member_id=...&view=call|email`.
- Added notice box about the new dev bar.
- Added "Other pages" section with links to intelligence/explore with `?dev`.

### `index.html`

No changes needed. The `openAdvocacyDrawer` function is already defined in the content block. The dev bar's auto-open logic (in base.html) calls it after 500ms via `DOMContentLoaded`.

---

## Client-side flow (auto-open)

1. User opens `/advocacy?dev&zip=60601&member_id=S123&view=call`.
2. Server renders advocacy page. `base.html` renders `#dev-bar` (because `dev_available` is `True`).
3. Dev bar IIFE runs: detects `?dev`, stores `sessionStorage`, shows bar, detects `/advocacy` path, shows advocacy panel, fetches members for dropdown.
4. IIFE sees `view=call` and `member_id=S123` in URL params. Registers `DOMContentLoaded` handler.
5. After DOM ready + 500ms, calls `openAdvocacyDrawer('call', 'S123', '60601', '')`.
6. `openAdvocacyDrawer` (defined in `index.html`) opens the drawer, fetches `/advocacy/drawer?view=call&member_id=S123&zip=60601`, injects the response.

---

## Persistence

Dev mode uses `sessionStorage` (key: `__ilga_dev_mode`). This means:

- Persists across page navigations within the same tab.
- Does not persist across tabs or after closing the browser tab.
- No server-side state or cookies involved.

---

## Summary table

| Piece | Location | Purpose |
|-------|----------|---------|
| Config guard | `main.py` (`templates.env.globals`) | Gates dev bar rendering to dev profile only. |
| Dev bar HTML/CSS/JS | `base.html` (bottom of body) | Floating toolbar with context panels. |
| Members API | `GET /api/dev/members` | JSON list for dev bar member dropdown. |
| Advocacy route | `GET /advocacy` | Accepts `zip`, `member_id`, `view` params. |
| Test page | `GET /advocacy/test`, `advocacy_test.html` | Form + quick links using `?dev`. |
| Auto-open | Dev bar JS (IIFE) | Reads URL params, calls `openAdvocacyDrawer()` after 500ms. |
| Persistence | `sessionStorage` (`__ilga_dev_mode`) | Keeps dev bar active across navigation. |
