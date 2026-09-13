# Campaign finance (SBE money layer)

MVP slice: official Illinois State Board of Elections **Committees** + **Receipts** joined onto sitting ILGA **Member IDs**. GraphQL can answer “who funded this member?” and “who funded the people on this bill?”

This is the monetization money join — not Moneyball (which is legislative-effectiveness scoring). Per Hardball Ch 3 (`docs/hardball-spec/04-ch3-decision-making.md`), campaign money is the medium of access. Lobbyist/entity join is **out of scope** (v1.1). Expenditures are not ingested.

---

## Source

Official bulk files from [`https://downloads.elections.il.gov/`](https://downloads.elections.il.gov/) (same dataset as [Download CD Data Files](https://elections.il.gov/CampaignDisclosure/DownloadCDDataFiles.aspx)):

| File | Role |
|------|------|
| `Candidates.txt` | Candidate name, office, district |
| `CmteCandidateLinks.txt` | Committee ↔ candidate |
| `Committees.txt` | Committee name, type, status |
| `Receipts.txt` | Contributions (tab-delimited; ~1GB full history) |

`Receipts.txt` starts in the 1990s. Ingest binary-searches by `RcvDate` and downloads only the recent tail (default **2025-01-01+**, ~50MB). Archived (superseded) receipts are dropped.

Unit tests use a **documented real extract** in `tests/fixtures/sbe/` (captured 2026-09-13).

---

## How to run ingest

```bash
# From official SBE files (chunked HTTP Range downloads)
PYTHONPATH=src python scripts/ingest_sbe_money.py

# Or: parse a local directory of .txt files (no download)
PYTHONPATH=src python scripts/ingest_sbe_money.py --from-dir tests/fixtures/sbe

# Also drop the index where the running app will see it (mocks/dev or cache/)
PYTHONPATH=src python scripts/ingest_sbe_money.py --from-dir tests/fixtures/sbe --also-write-data-dir
```

Make target: `make ingest-sbe-money` (optional `SINCE=2024-01-01`, `FROM_DIR=...`).

Requires `members.json` in the current data dir (`cache/` in prod, `mocks/dev` in a clean dev tree).

**Writes**

- SQLite tables on `ILGA_DB_PATH`: `sbe_committees`, `sbe_receipts`, `sbe_member_matches`, `sbe_ingest_runs` (Alembic `20260913100000`)
- `processed/campaign_finance/index.json` — compact index loaded at startup
- `processed/campaign_finance/unmatched.json` — review queue

Startup looks for `campaign_finance.json` in the data dir, then `processed/campaign_finance/index.json`.

---

## Member matching

Only **Active Candidate** committees linked to `State Senator` / `State Representative` rows are considered. Dissolved/final historical committees are skipped so the review list is current.

| Method | Confidence | Rule |
|--------|------------|------|
| `gold` | 1.00 | `docs/canonical/sbe_committee_member_gold.json` override |
| `office_district_name` | 0.85–0.95 | Chamber + district + last name (first-name overlap preferred) |
| `name_chamber` | 0.80 | Unique first+last in that chamber when district is missing or stale |
| `unresolved` | 0 | Written to `unmatched.json` for review |

**Known limits**

- Districts in SBE files have trailing spaces and historical rows (e.g. Syverson 34 vs 35). We prefer the row that matches the **sitting** member.
- Ambiguous last names (Harris, Jones, Anderson) require first name or district.
- Nicknames (`Emanuel "Chris" Welch`) and suffixes (`Emil Jones, III`) are normalized.
- Match rate is for *active legislative candidate committees*, not all 34k SBE committees or dissolved history. A 2025+ run against the 50-member `mocks/dev` roster matched **45/50 sitting members** and indexed thousands of real receipts. Unmatched review rows are mostly former members or district changes (e.g. Regan Deering 88 vs sitting 95).
- Contributor → member is best-effort (self-receipts / transfers) and will miss most individual donors.
- Money is **not earmarked to a bill**. `billMoneyContext` is “who funded these sponsors/voters,” not “who paid for this bill.”

---

## GraphQL

| Query | Use |
|-------|-----|
| `memberMoneyTrail(memberId, limit)` | Committees, totals, top donors, recent receipts |
| `billMoneyContext(billNumber, limit)` | Sponsor trails + overlapping donors |
| `campaignFinanceSummary` | Window, match rate, counts |

Examples: `graphql/member_money_trail.graphql`, `graphql/bill_money_context.graphql`.

---

## Intelligence UI

SSR pages (same Jinja2 + HTMX stack as the rest of Intelligence):

| Path | What it shows |
|------|----------------|
| `/intelligence/money` | Follow-the-money hub: window, match rate, top-funded members, bill lookup |
| `/intelligence/member/{id}` | Member money trail (committees, top donors, recent receipts) |
| `/intelligence/bill/{number-or-id}` | Bill money context (sponsor/voter trails + overlapping donors) |
| `/intelligence/` | Summary teaser with window / match-rate KPIs |

**Local (`make dev`):** http://127.0.0.1:8000/intelligence/money — uses `mocks/dev/campaign_finance.json`. Try member `3268` (Don Harmon) and bill `SB0341`.

**Production (landofkei.org):** https://landofkei.org/intelligence/money — same paths after deploy, once the prod data dir has `campaign_finance.json` or `processed/campaign_finance/index.json`.

Copy on these pages states that receipts are **not earmarked to a bill**, and that this layer is **not Moneyball** (effectiveness scoring). Per Hardball Ch 3 (`docs/hardball-spec/04-ch3-decision-making.md`), campaign money is the medium of access.

Lobbyist email signup is **not** on this surface.

---

## Gaps (not this slice)

- Expenditures
- Lobbyist / SOS entity join
- Independent expenditure / 527 / federal overlays
