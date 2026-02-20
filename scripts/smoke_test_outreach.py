#!/usr/bin/env python3
"""Smoke test: auth + record call + record email + verify data saved + visitor-visible stats.

Runs in the terminal with a temp DB. No server or Brevo required.
Usage: make smoke-outreach  or  PYTHONPATH=src python scripts/smoke_test_outreach.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

# Run from repo root so paths and imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import ilga_graph.config as cfg_mod
import ilga_graph.db as db_mod
import ilga_graph.dependencies as deps_mod
from ilga_graph.db_models import AuthCode
from ilga_graph.routers import auth as auth_router_mod
from ilga_graph.routers import outreach as outreach_router_mod


def _make_app(db_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db_mod.init_db()
        yield

    app = FastAPI(title="Smoke Auth+Outreach", lifespan=lifespan)
    app.include_router(auth_router_mod.router)
    app.include_router(outreach_router_mod.router)
    return app


async def _add_auth_code(email: str, plain_code: str) -> None:
    async with db_mod.async_session_factory() as session:
        session.add(
            AuthCode(
                email=email,
                code_hash=hashlib.sha256(plain_code.encode()).hexdigest(),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
            )
        )
        await session.commit()


def _run() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="ilga_smoke_outreach_"))
    db_path = tmp_dir / "smoke_ilga.db"

    env = {
        "ILGA_DB_PATH": str(db_path),
        "ILGA_AUTH_SECRET": "smoke-test-secret",
    }
    with patch.dict(os.environ, env, clear=False):
        importlib.reload(cfg_mod)
        importlib.reload(db_mod)
        importlib.reload(deps_mod)
        importlib.reload(auth_router_mod)
        importlib.reload(outreach_router_mod)

        async def _setup_db() -> None:
            await db_mod.init_db()
            await _add_auth_code("smoke@example.com", "999000")

        asyncio.run(_setup_db())

        app = _make_app(db_path)
        with TestClient(app, raise_server_exceptions=True) as client:
            # Trigger lifespan (tables already exist; lifespan runs init_db() again safely)
            client.get("/auth/me")

            # 1) Sign in (simulate user verifying code)
            r = client.post(
                "/auth/verify-code",
                data={"email": "smoke@example.com", "code": "999000"},
            )
            assert r.status_code == 200, f"verify-code: {r.status_code} {r.text}"
            data = r.json()
            assert data.get("ok") is True, f"verify-code body: {data}"
            assert "ilga_session" in r.cookies, "Session cookie not set"

            # 2) Record a call
            r = client.post(
                "/outreach/record",
                data={
                    "member_id": "smoke_call_member",
                    "kind": "call",
                    "zip_code": "60601",
                    "support_score": "4",
                    "constituent": "yes",
                },
            )
            assert r.status_code == 200, f"record call: {r.status_code} {r.text}"
            assert r.json().get("ok") is True and "event_id" in r.json()

            # 3) Record an email
            r = client.post(
                "/outreach/record",
                data={
                    "member_id": "smoke_email_member",
                    "kind": "email",
                    "zip_code": "60602",
                },
            )
            assert r.status_code == 200, f"record email: {r.status_code} {r.text}"
            assert r.json().get("ok") is True and "event_id" in r.json()

            # 4) Visitor view: public stats (no auth) — "someone who visits would see the outreach"
            visitor = TestClient(app, raise_server_exceptions=True)
            visitor.get("/auth/me")  # trigger lifespan

            r = visitor.get("/outreach/stats/smoke_call_member")
            assert r.status_code == 200, f"stats call: {r.status_code}"
            stats_call = r.json()
            assert stats_call["calls"] == 1 and stats_call["total"] == 1, (
                f"Visitor should see 1 call for smoke_call_member: {stats_call}"
            )

            r = visitor.get("/outreach/stats/smoke_email_member")
            assert r.status_code == 200, f"stats email: {r.status_code}"
            stats_email = r.json()
            assert stats_email["emails"] == 1 and stats_email["total"] == 1, (
                f"Visitor should see 1 email for smoke_email_member: {stats_email}"
            )

            # 5) Authenticated user: my-history shows both events
            r = client.get("/outreach/my-history")
            assert r.status_code == 200, f"my-history: {r.status_code} {r.text}"
            history = r.json()
            events = history.get("events", [])
            assert len(events) == 2, f"Expected 2 events in my-history, got {len(events)}"
            kinds = {e["kind"] for e in events}
            assert kinds == {"call", "email"}, f"Expected call+email in history, got {kinds}"

    print(
        "Smoke test passed: sign-in -> record call -> record email -> "
        "data saved -> visitor sees stats."
    )


if __name__ == "__main__":
    try:
        _run()
    except AssertionError as e:
        print("Smoke test failed:", e, file=sys.stderr)
        sys.exit(1)
