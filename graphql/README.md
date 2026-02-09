# GraphQL query examples

Example queries for the ILGA Graph API at `POST /graphql`. Copy into the GraphQL playground or send as JSON `{ "query": "...", "variables": { ... } }`.

## Query files

| File | Purpose |
|------|---------|
| `bill_with_votes_and_slips.graphql` | **Recommended** — One bill’s votes, witness slip summary, and paginated witness slips. |
| `bill_vote_timeline.graphql` | Full vote timeline (committee → floor) and member journeys per chamber. |
| `paginated_queries.graphql` | Members, bills, committees, witness slips, vote events with pagination. |
| `votes_with_counts.graphql` | Vote events with counts. |

## Bill + votes + witness slips (recommended)

Use the **`BillWithVotesAndSlips`** query in `bill_with_votes_and_slips.graphql` to get everything for one bill in one request:

- **`votes(billNumber)`** — Returns a **list** of vote events (no `items` / `pageInfo`). Empty list if the bill has no votes yet.
- **`witnessSlipSummary(billNumber)`** — Per-bill counts: `totalCount`, `proponentCount`, `opponentCount`, `noPositionCount`.
- **`witnessSlips(billNumber, limit, offset)`** — Returns a **connection**: `items` (list of slips) and `pageInfo` (`totalCount`, `hasNextPage`, etc.).

**Important:** The schema uses `votes` (list) for vote events per bill and `witnessSlips` (connection) for paginated slips. There is no `voteEvents` root field.

**Variables:** `{ "billNumber": "HB0034" }` (or `"HB0576"`, `"SB0852"`, `"SB0008"`, `"SB0009"`).

## Other useful queries

- **All bills by slip volume:** `witnessSlipSummaries(limit: 20, offset: 0)` — items sorted by `totalCount` descending.
- **Controversy score for a bill:** `billSlipAnalytics(billNumber: "HB0034")` → `controversyScore` (0–1).
- **Lobbyist alignment for a member:** `memberSlipAlignment(memberName: "Chris Welch")` → list of `{ organization, proponentCount }`.
