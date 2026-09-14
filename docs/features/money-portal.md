# Illinois Influence buyer portal

Separate public product shell for the money-making campaign-finance product. Buyers land on `/money`, not the Land of Kei advocacy site or the Intelligence dashboard.

Per Hardball Ch 3 (`docs/hardball-spec/04-ch3-decision-making.md`), campaign money is the medium of access. The portal is a **private desk**, not a data dashboard: one promise, one email field, one quiet demo path.

## Routes

| Path | Role |
|------|------|
| `GET /money` | Product landing. One line (“Illinois money trails, matched to sitting members”) plus a single email field. CTA **Request access**. |
| `GET /money/demo` | One member (Harmon `3268`) and one bill (`SB0341`). Watch / Ask / Flag in plain English. Member/bill lookup and the funded table live behind **Browse**. `?bill=SB0341` stays on the demo; `?member=3268` redirects to the member view. |
| `GET /money/member/{id}` | Member receipt trail (same `member_money_trail_view` data as Intelligence). Seed: `3268` Harmon. |
| `GET /money/bill/{bill}` | Bill money context (same `bill_money_context_view` data). Seed: `SB0341`. Receipts are **not earmarked to the bill**. |
| `GET/POST /money/signup` | Same one-field access form as the landing (shared `_money_portal_signup_form.html`). Same `money_intel_leads` store, CSRF, and per-IP rate limit as before. Optional name/org/role still accepted on POST (engine CTA). |
| `GET/POST /intelligence/money/signup` | Redirects to `/money/signup` (GET 302, POST 307). |

The research engine at `/intelligence/money` is unchanged (`intelligence.py` not modified). `/money` is exempt in `api_key_middleware` (same prefix pattern as `/intelligence`) so HTMX on the access form does not 401.

## Design

Own layout: `templates/money_portal_base.html` + `static/css/money-portal.css`. No advocacy nav, no Intelligence chrome, no Kei marketing. Brand is **Illinois Influence**. Footer trust line: Illinois State Board of Elections filings. Honest caveats (not earmarked; Moneyball is separate) stay short and quiet.

## Honest scale

`is_sample_scale_finance()` (in `intelligence_helpers.py`) treats an index as sample-scale when matched members or indexed receipts sit below a statewide threshold. The demo and detail pages show a sample-scale banner when that is true, and omit it when the loaded index is statewide.

## What this is not

- Not Moneyball (legislative-effectiveness scoring)
- Not lobbyist-registration filings
- Not expenditures, independent expenditures, 527s, or federal overlays
- Receipts are not earmarked to bills

Ingest, match, GraphQL money resolvers, and `/admin/money-leads` are not modified. No new GraphQL field. No shared-route rename. See [Campaign finance](campaign-finance.md) and [Money intel waitlist](money-intel-signup.md).
