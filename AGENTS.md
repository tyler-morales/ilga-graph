# Hardball — Agent Instructions

This project is **The Land of Kei** (ILGA Graph): a full-stack advocacy app for Illinois residents to contact legislators about Kei vehicle registration (625 ILCS 5/3-401(c-1)). Stack: Python 3.11+, FastAPI, Strawberry GraphQL, Jinja2 + HTMX, SQLite/Postgres, ML analytics pipeline.

## Core Principles

1. **Ontology is the foundational semantic layer** — Legislative and advocacy data are modeled as objects (nouns), links (relationships), actions (verbs), and logic (ML/rules). The application plane consumes the **Ontology SDK (OSDK)** in `src/ilga_graph/ontology/`; prefer the OSDK over raw `state.member_lookup` / `state.bills_lookup` for object and link semantics. See `docs/architecture/legislative-ontology.md`.
2. **Hardball spec is source of truth** — Advocacy, lobbying, and legislator outreach behavior come from `docs/hardball-spec/`. Read the relevant chunk before implementing or changing advocacy features.
3. **No hallucination of content** — Substantive copy, key points, FAQs, and brief text come only from canonical sources: `content.py`, legislator/constituent brief `.txt` files, and approved templates. See `.cursor/rules/no-hallucination-content.mdc`.
4. **Agent-first** — Delegate to specialized agents for domain tasks (planning, backend, frontend, tests, a11y).
5. **Test-driven** — Write tests before implementation; 80%+ coverage for logic. Use existing fixtures and `tests/async_helpers.run_async()` when in an event loop.
6. **Simplify and delete** — Prefer removing or consolidating code over adding. After adding a feature, remove duplicate logic, dead code, and redundant UI.

## Stack and Layout

| Area | Details |
|------|---------|
| **Ontology** | Foundational semantic layer in `src/ilga_graph/ontology/`: objects, links, actions, mapping, logic, OSDK. Application and GraphQL consume `state.ontology_sdk` for object/link/action semantics. See `docs/architecture/legislative-ontology.md`. |
| **Backend** | FastAPI in `main.py`, routers in `routers/`, state via `app_state` (no new globals). GraphQL in `schema.py`; resolvers use batch loaders (scorecard, moneyball, bill, member) and ontology queries. |
| **Frontend** | Jinja2 templates in `templates/`, HTMX partials; call `htmx.process(container)` after injecting HTML via JS. Styles in `base.html` or `static/css/`; use `.gmail-*`, `.drawer-*` naming. |
| **Data** | ETL in `etl.py`; scrapers in `src/ilga_graph/scrapers/`; ML/analytics in `analytics.py`, `ml/`. Data plane feeds the ontology mapping layer. |
| **Canonical content** | `content_constants.py` (STRATEGIC_*, FAQ_*, BRIEF_*); briefs in `docs/canonical/briefs/` (legislator-brief.txt, constituent-brief.txt); strategy books in `docs/canonical/books/` (hardball.txt, founding-sales.txt). |

## Project Agents (`.cursor/agents/`)

Use these for hardball-specific work:

| Agent | Purpose |
|-------|---------|
| **accessibility-agent** | Lighthouse/pa11y, WCAG 2.1 AA, ARIA/keyboard/focus. Use for a11y audits and fixes. |
| **backend** | FastAPI, GraphQL resolvers, ETL, ML; respects `app_state` and batch loaders. |
| **frontend** | Jinja2, HTMX, CSS/JS, drawer/card patterns, `htmx.process()` after dynamic HTML. |
| **tests** | pytest + httpx in `tests/`, fixtures, Make/CLI test targets. |

## General-Purpose Agents (`~/.cursor/agents/` or ECC)

Delegate to these for cross-cutting tasks:

| Agent | When to Use |
|-------|-------------|
| **planner** | New feature or refactor; implementation blueprint and phases. |
| **architect** | System design, scalability, schema/API shape. |
| **tdd-guide** | Enforce write-tests-first on a feature or bug fix. |
| **code-reviewer** | After writing or modifying code; quality and security. |
| **python-reviewer** | Python-specific review (PEP 8, type hints, security). |
| **security-reviewer** | Before touching auth, secrets, or sensitive data. |
| **build-error-resolver** | When the build or test run fails. |
| **refactor-cleaner** | Dead code removal, consolidation. |
| **doc-updater** | Keep README, API docs, and docs site in sync. |
| **database-reviewer** | Schema design, query optimization, migrations. |
| **e2e-runner** | Playwright E2E for critical user flows. |

## Commands (`.cursor/commands/`)

- `/plan` — Implementation plan (planner agent).
- `/tdd` — TDD workflow (tdd-guide).
- `/code-review` — Quality review (code-reviewer).
- `/python-review` — Python-focused review.
- `/build-fix` — Fix build errors (build-error-resolver).
- `/refactor-clean` — Dead code cleanup.
- `/update-docs` — Update documentation.
- `/test-coverage` — Coverage analysis.
- `/learn`, `/checkpoint`, `/verify` — Session learning and verification.

## Skills (project + global)

**Project (`.cursor/skills/`):** canonical-content-sources, extract-route-group-to-router, god-tier-debugging, graphql-resolver-and-loaders, hardball-spec-reference, update-pr-from-commits. Use when editing site copy, refactoring routes, implementing GraphQL, or aligning with the Hardball spec.

**Global (`~/.cursor/skills/`):** backend-patterns, api-design, tdd-workflow, python-patterns, python-testing, security-review, verification-loop, strategic-compact, search-first, and others. Invoke when the task matches the skill’s scope.

## Rules

- **hardball-spec-source-of-truth** — Before advocacy/lobbying/legislator features, read the relevant `docs/hardball-spec/` chunk and cite it.
- **no-hallucination-content** — Do not invent key points, FAQs, or brief text; use only content.py and canonical brief/template sources.
- ECC common and Python rules in `~/.cursor/rules/` (or project `.cursor/rules/`) apply: coding style, git workflow, testing, security, performance.

## TODOS.md

Track work and future-looking items in `TODOS.md`. After refactors or deletions, add a Refactor row or "Deleted/consolidated" line. Keep this file updated as you implement.
