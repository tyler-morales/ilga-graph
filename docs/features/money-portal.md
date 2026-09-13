# Illinois Influence buyer portal

Separate public product shell for the money-making campaign-finance product. Buyers land on `/money`, not the Land of Kei advocacy site or the Intelligence dashboard.

Per Hardball Ch 3 (`docs/hardball-spec/04-ch3-decision-making.md`), campaign money is the medium of access. The portal explains the **job** (a decision under a deadline), not a donor table to admire.

## Routes

| Path | Role |
|------|------|
| `GET /money` | Product landing. Promise, job, honest caveats. Primary CTA → demo; secondary → waitlist. |
| `GET /money/demo` | Buyer spine: live SBE KPIs, member lookup / top-funded shortlist, bill lookup, Watch / Ask / Flag prompts. `?bill=SB0341` stays on the demo; `?member=3268` redirects to the member view. |
| `GET /money/member/{id}` | Member receipt trail (same `member_money_trail_view` data as Intelligence). Seed: `3268` Harmon. |
| `GET /money/bill/{bill}` | Bill money context (same `bill_money_context_view` data). Seed: `SB0341`. Receipts are **not earmarked to the bill**. |
| `GET/POST /money/signup` | Canonical waitlist. Same `money_intel_leads` store, CSRF, and per-IP rate limit as before. |
| `GET/POST /intelligence/money/signup` | Redirects to `/money/signup` (GET 302, POST 307). |

The research engine at `/intelligence/money` is unchanged. It soft-links **Buyer portal → /money**.

## Design

Own layout: `templates/money_portal_base.html` + `static/css/money-portal.css`. No advocacy nav, no Intelligence chrome, no Kei marketing. Brand is **Illinois Influence / Follow the money**.

## Honest scale

`is_sample_scale_finance()` (in `intelligence_helpers.py`) treats an index as sample-scale when matched members or indexed receipts sit below a statewide threshold. The demo and detail pages show a sample-scale banner when that is true, and omit it when the loaded index is statewide.

## What this is not

- Not Moneyball (legislative-effectiveness scoring)
- Not lobbyist-registration filings
- Not expenditures, independent expenditures, 527s, or federal overlays
- Receipts are not earmarked to bills

Ingest, match, and GraphQL money resolvers are not modified. See [Campaign finance](campaign-finance.md) and [Money intel waitlist](money-intel-signup.md).
