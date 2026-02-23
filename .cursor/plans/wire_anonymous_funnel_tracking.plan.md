# Wire Anonymous Funnel, Conversion Report, and Privacy-Compliant Tracking

## Overview

Integrate anonymous funnel tracking end-to-end: client-side anon session ID generation and wiring, server-side attribution on verify-code, conversion report (with expanded metric definitions below), privacy policy updates, and docs/TODOS updates.

---

## Metric Definitions for Conversion Report

Use the same time window (e.g. last 90 days) for all so they’re comparable.

### 1. Funnel conversion (denominator → numerator)

| # | Name / slug | Denominator | Numerator | Why it's useful |
|---|-------------|-------------|-----------|------------------|
| 1 | **drawer_to_outreach** | Distinct identities who had drawer_opened in window | Distinct users who recorded ≥1 call or email in window | Main pitch: "Of everyone who opened the flow, X% took action." |
| 2 | **phone_to_call** | Distinct identities who had phone_clicked (call flow) in window | Distinct users who recorded a call or no_answer in window | "Of everyone who started a call, X% finished (recorded outcome)." |
| 3 | **drawer_to_call** | Distinct identities who had drawer_opened in window | Distinct users who recorded a call (or no_answer) in window | "Of everyone who opened the flow, X% made a call." |
| 4 | **drawer_to_email** | Distinct identities who had drawer_opened in window | Distinct users who recorded an email in window | "Of everyone who opened the flow, X% sent an email." |
| 5 | **signed_in_to_outreach** | Distinct users who signed in (e.g. last_login in window or proxy: users with ≥1 step event in window) | Distinct users who recorded ≥1 call or email in window | "Of signed-up users, X% took at least one action." |
| 6 | **email_flow_start_to_send** | Distinct identities who had signed_in (email flow step) in window | Distinct users who had email_recorded in window | "Of people who started the email flow (signed in step), X% sent." |

### 2. Volume / engagement (counts, not rates)

| # | Name / slug | What it is | Why it's useful |
|---|-------------|------------|------------------|
| 7 | **identities_opened_drawer** | Count of distinct identities with drawer_opened in window | "X people opened the advocacy flow." |
| 8 | **users_completed_outreach** | Count of distinct users with ≥1 call or email in window | "X people took action." |
| 9 | **total_calls** | Count of outreach_events with kind = call in window | "X calls made." |
| 10 | **total_emails** | Count of outreach_events with kind = email in window | "X emails sent." |
| 11 | **total_outreach_actions** | Count of call + email events in window | "X total actions." |
| 12 | **identities_clicked_phone** | Count of distinct identities with phone_clicked in window | "X people started a call." |

### 3. Optional: step-through rates (for drop-off analysis)

| # | Name / slug | Denominator | Numerator | Why it's useful |
|---|-------------|-------------|-----------|------------------|
| 13 | **drawer_to_phone_click** | Identities with drawer_opened | Identities with phone_clicked | "Of people who opened the drawer, X% clicked to call." |
| 14 | **phone_click_to_call_recorded** | Identities with phone_clicked | Users with call or no_answer recorded | Same as #2; can expose as step-through for dashboards. |

### Suggested minimum set for "range of impactful stats"

**Conversions (rates):**

- **drawer_to_outreach** (main pitch)
- **phone_to_call** (call completion)
- **drawer_to_email** (email slice)
- **signed_in_to_outreach** (activated users)

**Volumes (counts):**

- **identities_opened_drawer**
- **users_completed_outreach**
- **total_calls**
- **total_emails**
- **total_outreach_actions**

That gives: several conversion definitions for different angles; clear volume stats to back them up ("X people opened, Y took action, Z% conversion").

### Response shape (for API / dashboard)

Return one object per definition, with a shared window:

```json
{
  "window_days": 90,
  "window_start": "...",
  "window_end": "...",
  "conversions": {
    "drawer_to_outreach":    { "denominator": 200, "numerator": 45, "conversion_pct": 22.5 },
    "phone_to_call":         { "denominator": 120, "numerator": 38, "conversion_pct": 31.67 },
    "drawer_to_email":        { "denominator": 200, "numerator": 30, "conversion_pct": 15.0 },
    "signed_in_to_outreach":  { "denominator": 80,  "numerator": 40, "conversion_pct": 50.0 }
  },
  "volumes": {
    "identities_opened_drawer": 200,
    "users_completed_outreach": 45,
    "total_calls": 60,
    "total_emails": 30,
    "total_outreach_actions": 90
  }
}
```

**Implementation note:** The current conversion endpoint returns a single metric. Extend `GET /admin/outreach/conversion` (or add a new endpoint) to compute and return this full shape: `conversions` (object keyed by slug) and `volumes` (object keyed by slug). For **signed_in_to_outreach**, denominator can use users with `last_login_at` in window, or as proxy: distinct `user_id` in `outreach_step_events` in window (active in funnel).

---

## Original implementation plan (summary)

### Current state

- `OutreachStepEvent.session_id` and POST /outreach/step anonymous path exist; client still bails when not signed in.
- No client `ilga_anon_sid`; no verify-code attribution; conversion report is single-metric only.
- Privacy policy has no funnel/usage disclosure.

### 1. Client → server wiring

- **hero-auth-strip.js:** Generate and store `ilga_anon_sid` in sessionStorage (UUID); expose `window._ilgaAnonSid`; clear/refresh on sign-in/sign-out.
- **index.html:** `recordOutreachStep()` — remove auth guard; when anon, send `session_id`; when signed in, do not send session_id.
- **Verify-code (both paths):** Send `anon_session_id` in form body; after success, clear anon sid from storage.

### 2. Attribution

- **auth.py:** Accept `anon_session_id`; on successful verify-code, backfill `outreach_step_events` (set user_id, clear session_id). Use shared `_validate_anon_session_id` (e.g. in security.py).

### 3. Conversion report

- Extend endpoint to return the **response shape** above: multiple conversions (drawer_to_outreach, phone_to_call, drawer_to_email, signed_in_to_outreach) and volumes (identities_opened_drawer, users_completed_outreach, total_calls, total_emails, total_outreach_actions).
- Keep single 90-day window; document denominator-zero and multi-tab behavior.

### 4. Privacy and compliance

- **privacy.html:** Add "Usage and conversion measurement" (first-party session id, link on sign-in, retention e.g. 12 months, no sale/sharing).
- Optional: one-line soft notice in footer.

### 5. Docs and TODOS

- **db-and-outreach.md:** Anonymous funnel (session_id, client ilga_anon_sid), unauthenticated step API, attribution at verify-code, conversion definitions and endpoint, privacy (first-party, disclosed, retention).
- **TODOS.md:** Entry for anonymous funnel + conversion wired; tracking privacy-compliant.

### 6. Edge cases

- Denominator zero: return 0 or null; no divide-by-zero.
- Multiple tabs: sessionStorage per tab; document that denominator may be slightly conservative.

### Files to modify (original list)

- `hero-auth-strip.js`, `index.html` — anon sid + step + verify-code body
- `auth.py` — anon_session_id + backfill
- `security.py` (or session_utils) — shared validator; `outreach.py` import it
- `admin.py` — extend conversion endpoint to full response shape (conversions + volumes)
- `privacy.html`, optionally `base.html`
- `docs/development/db-and-outreach.md`, `TODOS.md`
