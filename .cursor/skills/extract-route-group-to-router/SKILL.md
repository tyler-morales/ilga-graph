---
name: extract-route-group-to-router
description: Extracts a group of FastAPI routes from main.py into a dedicated APIRouter in routers/, preserving state and dependency patterns, then updates main and TODOS. Use when refactoring main.py, moving routes to a router, or when the user asks to extract routes, split routers, or modularize FastAPI routes.
---

# Extract Route Group to Router

## When to use

When refactoring main.py by moving a set of related routes (e.g. `/intelligence/*`, `/explore`, `/api/*`) into a router.

## Instructions

1. Create or choose a module under `src/ilga_graph/routers/` (e.g. `intelligence.py`, `explore.py`).
2. Define an `APIRouter()` and move the route handlers from main into it.
3. Preserve dependencies: handlers that need app state should receive `state` the same way main does (injected or from request/app state). Do not introduce new globals.
4. In `main.py`: mount the router with `app.include_router(router, prefix="...")` and remove the moved route definitions.
5. Update `TODOS.md` under Refactor (e.g. "Intelligence routes extracted to `routers/intelligence.py` (2026-02-18).").
6. If any docs reference the URLs or structure, update them.
