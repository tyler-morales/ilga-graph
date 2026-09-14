# Money intel waitlist (lobbyist email signup)

Marketing / waitlist capture for Illinois lobbyists and adjacent buyers (lawyers, nonprofits) who want **follow-the-money intel**: member contribution trails and bill sponsor funding context.

The Follow-the-money **engine** stays at `/intelligence/money` (SBE KPIs, bill lookup, top-funded members). The **buyer waitlist** lives at `/money/signup` (Illinois Influence portal). `/intelligence/money/signup` redirects there. The engine page still has a CTA that posts to the same `money_intel_leads` store.

This is **not** the SOS/lobbyist disclosure data join, not an expenditure feed, and not Moneyball-as-finance. Moneyball on this site remains a legislative-power score.

Fixture / dev-scale lookup-seed copy (Harmon / SB0341) is shown only when the loaded campaign-finance index is sample-scale. A statewide ingest hides that banner.

Aligns with Hardball Ch 7 (listservs created by website subscribe) and Founding Sales Ch 6 (short inbound form, light qualification). Single opt-in for MVP: the address is stored when the form is submitted.

## How to view

| Environment | Engine + CTA | Buyer portal / waitlist |
|-------------|--------------|-------------------------|
| **Local** (`make dev`) | http://127.0.0.1:8000/intelligence/money | http://127.0.0.1:8000/money · `/money/demo` · `/money/signup` |
| **Prod** (landofkei.org) | https://landofkei.org/intelligence/money | https://landofkei.org/money · `/money/demo` · `/money/signup` |

Form fields: **portal** (`/money`, `/money/signup`) is email only (CTA **Request access**). **Engine CTA** on `/intelligence/money` still has optional name, organization/firm, and role checkboxes (lobbyist / lawyer / nonprofit). Both POST to the same `money_intel_leads` store.

States: success, already on the list, invalid email, CSRF, rate limit.

## How to export leads

Leads live in SQLite table `money_intel_leads` (not `users`). They are **not** campaign-update subscribers.

### Admin UI (signed in as an `ILGA_ADMIN_EMAILS` user)

1. Open `/admin/money-leads` (local: http://127.0.0.1:8000/admin/money-leads ; prod: https://landofkei.org/admin/money-leads).
2. Click **Download CSV** or go directly to `/admin/money-leads.csv`.

CSV columns: `email`, `name`, `org`, `role`, `created_at`.

### sqlite3 (same machine as the DB)

Dev default DB is `data/ilga_dev.db`; prod is `data/ilga.db` (or `ILGA_DB_PATH`).

```bash
sqlite3 data/ilga_dev.db -header -csv \
  "SELECT email, name, org, role, created_at FROM money_intel_leads ORDER BY created_at;"
```

On production, point that command at the live DB path.

## Spam protection

Same pattern as `/updates/subscribe-email`: CSRF double-submit cookie plus per-IP rate limit (`ILGA_RATE_LIMIT_SUBSCRIBE_EMAIL_PER_HOUR`, default 10). Turnstile is not on this form (it is used on the poll and bug-report forms, not on email subscribe).

## Schema

Alembic revision `20260913140000` (after `20260913100000` SBE campaign finance). Model: `MoneyIntelLead` in `db_models.py`.
