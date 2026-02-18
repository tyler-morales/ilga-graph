---
name: graphql-resolver-and-loaders
description: Implements or updates Strawberry GraphQL resolvers using the project's batch loaders and state-from-context pattern; documents new API in docs. Use when adding or changing GraphQL resolvers, queries, batch loaders, or API reference docs.
---

# GraphQL Resolver and Loaders

## When to use

When adding or changing Strawberry resolvers, queries, or data loading.

## Instructions

1. **State**: Resolvers get `state` from the GraphQL context (injected by main). Use it for members, bills, committees, scorecards — do not introduce new global state.
2. **Loaders**: Use existing batch loaders (scorecard, moneyball, bill, member) to avoid N+1 queries. Resolvers should be thin: resolve IDs via loaders and return typed data.
3. **Schema**: Add or change types and queries in `src/ilga_graph/schema.py`. Respect per-file-ignores in `pyproject.toml` (e.g. F821 for schema.py if state/helpers are injected at runtime).
4. **Docs**: Document new or changed queries in `docs/reference/graphql.md` (signature, arguments, return shape, example).
5. After shipping, run the "Update TODOS and docs" workflow (update TODOS.md and any affected docs).
