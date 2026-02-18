# Dev Bar — Fast Feature Testing

The **Dev Bar** is a floating toolbar that appears at the bottom of any page when you append `?dev` to the URL. It gives you instant access to test any feature — open call scripts, email drawers, jump to intelligence sub-pages, navigate member/bill detail — without clicking through the full UI.

> **Replaces the old `?test=1` system.** The old test mode params no longer work. Use `?dev` instead.

---

## Quick start

1. Go to any page with `?dev` appended: `/advocacy?dev`, `/intelligence?dev`, `/explore?dev`
2. The dark dev toolbar appears at the bottom of the viewport.
3. Use the quick actions to test features. The bar persists across page navigations.

---

## Activating and deactivating

| Action | How |
|--------|-----|
| Turn on | Append `?dev` to any URL (e.g. `/advocacy?dev`) |
| Turn off | Click the **X** button, press **Ctrl+Shift+D**, or navigate to `?dev=off` |
| Toggle | Press **Ctrl+Shift+D** on any page |

Once activated, the dev bar persists across normal navigation (stored in `sessionStorage`). You don't need to add `?dev` to every link — just once to start a session.

---

## What's in the dev bar

### Header (all pages)

- **DEV** badge and current page path
- **Nav links** — Advocacy, Intelligence, Explore, GraphQL, Logs (all with `?dev`)
- **Keyboard shortcut** hint (`Ctrl+Shift+D`)

### Advocacy panel (on `/advocacy*` pages)

- **ZIP** input (default `60601`, editable)
- **Member** dropdown (first 20 legislators, loaded from `/api/dev/members`)
- **Open Call** — Opens the call script drawer for the selected member and ZIP
- **Open Email** — Opens the email drawer for the selected member and ZIP
- **Search ZIP** — Submits the advocacy search form with the dev bar's ZIP value

### Intelligence panel (on `/intelligence*` pages)

- **Chip links** to every sub-page: Summary, Raw, Predictions, Coalitions, Anomalies, Influence, Recruitment, Committees, Accuracy, Witness Slips
- **Member ID** input + Go button — Jump to `/intelligence/member/{id}`
- **Bill ID** input + Go button — Jump to `/intelligence/bill/{id}`

---

## Deep-link URLs (bookmarkable)

Open the advocacy page with a specific drawer pre-opened:

**Call script:**

```
/advocacy?dev&zip=60601&member_id=YOUR_ID&view=call
```

**Email drawer:**

```
/advocacy?dev&zip=60601&member_id=YOUR_ID&view=email
```

Replace `YOUR_ID` with the legislator's ID (e.g. from the dev bar member dropdown). Bookmark these for one-click access.

---

## Test page

The legacy test page at `/advocacy/test` still works and has been updated to use `?dev` links. It provides the same form and quick links as before, plus links to intelligence and explore pages with dev mode enabled.

---

## Production safety

The dev bar is **only available when `ILGA_DEV_MODE=1`** (default in `ILGA_PROFILE=dev`). In production (`ILGA_PROFILE=prod`), the dev bar HTML, CSS, and JS are never rendered. Appending `?dev` to a production URL has no effect.

---

## Summary

| Goal | What to do |
|------|------------|
| Open call script for a member | `/advocacy?dev&zip=60601&member_id=ID&view=call` or use dev bar dropdown |
| Open email drawer for a member | Same with `view=email` |
| Jump to intelligence sub-page | Open `/intelligence?dev` and click a chip, or go directly (e.g. `/intelligence/predictions?dev`) |
| View a specific member's detail | Use dev bar Member ID input on intelligence pages, or go to `/intelligence/member/ID?dev` |
| Toggle dev bar | Press `Ctrl+Shift+D` |
| Leave dev mode | Click X, press `Ctrl+Shift+D`, or go to `?dev=off` |
