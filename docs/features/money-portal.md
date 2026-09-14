# Illinois Influence buyer portal

Separate public product shell for the money-making campaign-finance product. Buyers land on `/money`, not the Land of Kei advocacy site or the Intelligence dashboard.

Per Hardball Ch 3 (`docs/hardball-spec/04-ch3-decision-making.md`), campaign money is the medium of access. The portal is a **private desk**: one Speed-1 promise, one email field, one quiet submit, optional link into a real trail (Harmon). Not a waitlist launch, not a data dashboard.

## Routes

| Path | Role |
|------|------|
| `GET /money` | First screen only: “Sitting-member money trails, matched to ILGA legislators” + email + Submit. Whisper link to Harmon `3268`. |
| `GET /money/demo` | Redirects to the Harmon trail (`/money/member/3268`). `?member=` and `?bill=` redirect to the matching trail. |
| `GET /money/member/{id}` | Member receipt trail (same `member_money_trail_view` data as Intelligence). Seed: `3268` Harmon. |
| `GET /money/bill/{bill}` | Bill money context (same `bill_money_context_view` data). Seed: `SB0341`. Receipts are **not earmarked to the bill**. |
| `GET/POST /money/signup` | Same first-screen form as the landing. Same `money_intel_leads` store, CSRF, and per-IP rate limit. Optional name/org/role still accepted on POST (engine CTA). |
| `GET/POST /intelligence/money/signup` | Redirects to `/money/signup` (GET 302, POST 307). |

The research engine at `/intelligence/money` is unchanged (`intelligence.py` not modified). `/money` is exempt in `api_key_middleware` (same prefix pattern as `/intelligence`) so HTMX on the access form does not 401.

## Design

Own layout: `templates/money_portal_base.html` + `static/css/money-portal.css`. No advocacy nav, no Intelligence chrome, no Kei marketing, no Demo nav, no feature grid. Brand is **Illinois Influence**. Footer trust line: Illinois State Board of Elections filings. Honest caveats (not earmarked; Moneyball) live on the trail pages, not the first screen.

## Honest scale

`is_sample_scale_finance()` (in `intelligence_helpers.py`) treats an index as sample-scale when matched members or indexed receipts sit below a statewide threshold. Member and bill pages show a sample-scale banner when that is true, and omit it when the loaded index is statewide.

## What this is not

- Not Moneyball (legislative-effectiveness scoring)
- Not lobbyist-registration filings
- Not expenditures, independent expenditures, 527s, or federal overlays
- Receipts are not earmarked to bills

Ingest, match, GraphQL money resolvers, and `/admin/money-leads` are not modified. No new GraphQL field. No shared-route rename. See [Campaign finance](campaign-finance.md) and [Money intel waitlist](money-intel-signup.md).
