# Money intel waitlist (lobbyist email signup)

Marketing / waitlist capture for Illinois lobbyists and adjacent buyers (lawyers, nonprofits) who want **follow-the-money intel**: member contribution trails and bill sponsor funding context.

This is **not** the SOS/lobbyist disclosure data join, not an expenditure feed, and not Moneyball-as-finance. Moneyball on this site remains a legislative-power score.

Aligns with Hardball Ch 7 (listservs created by website subscribe) and Founding Sales Ch 6 (short inbound form, light qualification). Single opt-in for MVP: the address is stored when the form is submitted.

## How to view

| Environment | Landing (value prop + demo seeds + form) | Signup-only page |
|-------------|------------------------------------------|------------------|
| **Local** (`make dev`) | http://127.0.0.1:8000/intelligence/money | http://127.0.0.1:8000/intelligence/money/signup |
| **Prod** (landofkei.org) | https://landofkei.org/intelligence/money | https://landofkei.org/intelligence/money/signup |

Demo seeds on the landing page:

- Member trail: [Don Harmon (3268)](/intelligence/member/3268)
- Bill sponsor funding context: [SB0341](/intelligence/bill/SB0341)

Form fields: required email; optional name, organization/firm, role checkboxes (lobbyist / lawyer / nonprofit).

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

Alembic revision `20260913140000`. Model: `MoneyIntelLead` in `db_models.py`.
