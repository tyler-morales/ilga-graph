# Testing the bills-first pipeline and server

## What to expect after a scrape + export

1. **Cache**
   - `cache/bills.json` — all bills (e.g. 200 if you used `--sb-limit 100 --hb-limit 100`, or 3490+ if using existing cache). Each entry has `bill_number`, `leg_id`, `description`, `chamber`, `last_action`, `last_action_date`, `primary_sponsor`, and (after a fresh bill scrape) `synopsis`, `status_url`, `sponsor_ids`, `house_sponsor_ids`, `action_history`.
   - `cache/scrape_metadata.json` — `last_bill_scrape_at`, `bill_index_count`.

2. **Vault export**
   - `ILGA_Graph_Vault/Members/` — one `.md` per member.
   - `ILGA_Graph_Vault/Committees/` — one `.md` per committee.
   - `ILGA_Graph_Vault/Bills/` — one `.md` per bill (up to `bill_export_limit` when set; e.g. 100 in dev). You should see **bill files here** (e.g. `SB0001.md`, `HB0054.md`), not "Exported 0 bill files."
   - `ILGA_Graph_Vault/Moneyball Report.md`.

3. **Server**
   - `make dev` or `make run` loads the same cache and exposes it at `/graphql`. Bills and members are linked via `Bill.sponsor_ids` → member IDs.

---

## GraphQL query to test on the server

With the server running at **http://localhost:8000**, open **http://localhost:8000/graphql** and run:

```graphql
query TestBillsFirst {
  bills(sortBy: LAST_ACTION_DATE, sortOrder: DESC, limit: 5) {
    items {
      billNumber
      legId
      description
      chamber
      lastAction
      lastActionDate
      primarySponsor
      synopsis
      statusUrl
      sponsorIds
      houseSponsorIds
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
    }
  }
}
```

Note: `bills` returns a **connection** (`BillConnection`) with `items` (the bill list) and `pageInfo`, not the bill fields at the top level.

**What you should see**
- A list of up to 5 bills, with the new fields (`synopsis`, `statusUrl`, `sponsorIds`, `houseSponsorIds`). If your cache was from the old format, `synopsis` / `statusUrl` / `sponsorIds` may be empty; after a fresh bill scrape they will be populated.
- No errors; the query returns `{ "data": { "bills": [ ... ] } }`.

**Optional: member → bills (sponsor linkage)**

```graphql
query TestMemberBills {
  member(name: "Neil Anderson") {
    name
    sponsoredBills {
      billNumber
      description
      lastActionDate
    }
    coSponsorBills {
      billNumber
      description
    }
  }
}
```

After a scrape that has `sponsor_ids` on bills and members loaded, sponsored/co-sponsored bills should appear. With old cache (no `sponsor_ids`), these lists may be empty.
