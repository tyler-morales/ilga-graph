# GraphQL API

The app exposes a **Strawberry GraphQL** API at **`POST /graphql`**. Use the built-in playground at **/graphql** in the browser.

---

## Endpoint

- **URL:** `http://127.0.0.1:8000/graphql` (when the app is running).
- **Method:** `POST`.
- **Body:** JSON with `query` and optional `variables`, e.g. `{ "query": "...", "variables": { "billNumber": "HB0034" } }`.

---

## Example queries (from the repo)

The **`graphql/`** folder in the repo contains ready-to-use `.graphql` files:

| File | Purpose |
|------|---------|
| `bill_with_votes_and_slips.graphql` | **Recommended** — One bill’s votes, witness slip summary, and paginated witness slips. |
| `bill_vote_timeline.graphql` | Full vote timeline (committee → floor) and member journeys per chamber. |
| `paginated_queries.graphql` | Members, bills, committees, witness slips, vote events with pagination. |
| `votes_with_counts.graphql` | Vote events with counts. |

Copy queries into the playground or send them via your client with variables as JSON.

---

## Key operations

- **Bill + votes + slips:** Use the `BillWithVotesAndSlips` query in `bill_with_votes_and_slips.graphql`. Variables: `{ "billNumber": "HB0034" }`.
  - `votes(billNumber)` → **list** of vote events (no `items` / `pageInfo`).
  - `witnessSlipSummary(billNumber)` → counts (total, proponent, opponent, no position).
  - `witnessSlips(billNumber, limit, offset)` → **connection** with `items` and `pageInfo`.

- **Metrics glossary:** `metricsGlossary` — definitions of Moneyball and other metrics for tooltips and “how is this calculated?” UI.

- **Members / bills / leaderboard:** `member(name)`, `members(...)`, `bills(...)`, `moneyballLeaderboard(...)` with sort and filter options.

- **Search:** Free-text search across members, bills, and committees with relevance and entity-type filtering.

---

## Schema notes

- **Votes** for a bill: use `votes(billNumber)` (returns a list).
- **Witness slips** for a bill: use `witnessSlips(billNumber, limit, offset)` (returns a connection with `items` and `pageInfo`).
- There is no root-level `voteEvents` field; use `votes(billNumber)` or the documented queries.

For full query examples and variable shapes, see the **`graphql/README.md`** file in the repository.
