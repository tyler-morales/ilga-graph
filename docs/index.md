# ILGA Graph Documentation

Documentation for the **ILGA Graph** proof-of-concept: Illinois General Assembly data, advocacy tools, and ML intelligence.

---

## Getting started

- [**Getting started**](getting-started.md) — Install, run the app, scrape data, run the docs site. Typical workflows.

---

## Features & user guides

- [**App overview**](features/app-overview.md) — What’s in the app: Advocacy, Power Map, Intelligence, GraphQL. Paths and one-liners.
- [**Campaign finance (SBE money layer)**](features/campaign-finance.md) — Committees + Receipts ingest, member matching, `memberMoneyTrail` / `billMoneyContext`.
- [**Buyer portal (Illinois Influence)**](features/money-portal.md) — Separate `/money` product shell: one-line landing + email access, quiet demo, member/bill views.
- [**Money intel waitlist**](features/money-intel-signup.md) — Lobbyist/buyer email signup at `/money/signup` (old `/intelligence/money/signup` redirects); how to export leads.
- [**Pitch one-pagers**](pitch/) — Audience-specific one-pagers: [advocacy/nonprofit](pitch/advocacy-nonprofit.md), [lobbyist](pitch/lobbyist.md), [candidate](pitch/candidate.md), [investor](pitch/investor.md).
- [**Dev Bar**](user-guide/advocacy-test-mode.md) — Floating dev toolbar activated by `?dev` on any URL. Quick-access to call scripts, email drawers, intelligence sub-pages, and deep-link bookmarks.

---

## Reference

- [**CLI (Make)**](reference/cli-make.md) — All `make` targets: server, scrape, ML, test, lint, docs.
- [**Environment variables**](reference/environment-variables.md) — Profiles and full variable list.
- [**Fact sheet best practices**](reference/fact-sheet-analysis.txt) — Analysis of sample advocacy fact sheets and recommendations; used to update the Kei vehicle fact sheet.
- [**GraphQL API**](reference/graphql.md) — Endpoint, example query files, key operations.
- [**Project structure**](reference/project-structure.md) — Where to find code, cache, and docs.

---

## Development & internals

- [**Dev Bar — How it works**](development/advocacy-test-mode-internals.md) — URL contract, config guard, backend injection, template rendering, client-side persistence.
- [**Component playground**](development/component-playground.md) — Dev-only `/dev/playground` to isolate UI components (truck animation, drawer, etc.); how to add scenes.
- [**Bills-first pipeline testing**](development/bills-first-testing.md) — What to expect after scrape + export; GraphQL queries to test the server.

---

## Quick links (app paths)

| What | Path |
|------|------|
| Advocacy (main flow) | `/advocacy` |
| Dev Bar (activate on any page) | Any URL + `?dev` |
| Test page (dev jump links) | `/advocacy/test` |
| Component playground (dev) | `/dev/playground` |
| Power Map (graph) | `/explore` |
| ML Intelligence | `/intelligence` |
| Follow the money (research engine) | `/intelligence/money` |
| Illinois Influence buyer portal | `/money`, `/money/demo`, `/money/signup` |
| Money intel waitlist | `/money/signup` (canonical); `/intelligence/money/signup` redirects |
| GraphQL playground | `/graphql` |
