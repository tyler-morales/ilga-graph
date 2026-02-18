# ILGA Graph Documentation

Documentation for the **ILGA Graph** proof-of-concept: Illinois General Assembly data, advocacy tools, and ML intelligence.

---

## Getting started

- [**Getting started**](getting-started.md) — Install, run the app, scrape data, run the docs site. Typical workflows.

---

## Features & user guides

- [**App overview**](features/app-overview.md) — What’s in the app: Advocacy, Power Map, Intelligence, GraphQL. Paths and one-liners.
- [**Dev Bar**](user-guide/advocacy-test-mode.md) — Floating dev toolbar activated by `?dev` on any URL. Quick-access to call scripts, email drawers, intelligence sub-pages, and deep-link bookmarks.

---

## Reference

- [**CLI (Make)**](reference/cli-make.md) — All `make` targets: server, scrape, ML, test, lint, docs.
- [**Environment variables**](reference/environment-variables.md) — Profiles and full variable list.
- [**GraphQL API**](reference/graphql.md) — Endpoint, example query files, key operations.
- [**Project structure**](reference/project-structure.md) — Where to find code, cache, and docs.

---

## Development & internals

- [**Dev Bar — How it works**](development/advocacy-test-mode-internals.md) — URL contract, config guard, backend injection, template rendering, client-side persistence.
- [**Bills-first pipeline testing**](BILLS_FIRST_TESTING.md) — What to expect after scrape + export; GraphQL queries to test the server.

---

## Quick links (app paths)

| What | Path |
|------|------|
| Advocacy (main flow) | `/advocacy` |
| Dev Bar (activate on any page) | Any URL + `?dev` |
| Test page (dev jump links) | `/advocacy/test` |
| Power Map (graph) | `/explore` |
| ML Intelligence | `/intelligence` |
| GraphQL playground | `/graphql` |
