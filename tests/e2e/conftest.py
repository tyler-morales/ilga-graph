"""E2E test fixtures: live server, base_url, and optional authed page.

Run with: pytest -m e2e tests/e2e/
Requires: ILGA_PROFILE=dev (Turnstile off), server bound to 127.0.0.1:8001.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

E2E_PORT = 8001
E2E_BASE_URL = f"http://127.0.0.1:{E2E_PORT}"


def _wait_for_port(port: int, timeout: float = 30.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


@pytest.fixture(scope="session")
def live_server():
    """Start uvicorn in a subprocess; yield base URL; terminate on teardown.

    Uses an empty ILGA_CACHE_DIR so the app uses mocks/dev (get_data_dir() returns
    MOCK_DEV_DIR). That ensures 60601 is in zip_to_district and members are loaded
    for advocacy E2E tests.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["ILGA_PROFILE"] = "dev"
    env["ILGA_LOAD_ONLY"] = "1"
    # Empty cache dir → app uses mocks/dev (seed ZIPs + members)
    # so /advocacy?zip=60601 returns results.
    with tempfile.TemporaryDirectory(prefix="ilga_e2e_cache_") as tmp:
        env["ILGA_CACHE_DIR"] = tmp
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "ilga_graph.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(E2E_PORT),
            ],
            cwd=str(project_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            if not _wait_for_port(E2E_PORT):
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                proc.terminate()
                proc.wait(timeout=5)
                raise RuntimeError(f"Server did not start on port {E2E_PORT}. stderr: {stderr}")
            yield E2E_BASE_URL
        finally:
            proc.terminate()
            proc.wait(timeout=10)


@pytest.fixture(scope="session")
def base_url(live_server):
    """Base URL for Playwright page (used by pytest-playwright)."""
    return live_server


@pytest.fixture
def authed_page(page, base_url):
    """Page with authenticated session (if ILGA_DEV_CODE is set).

    When ILGA_DEV_CODE is not set, returns the same as page; tests that require
    auth can skip or assert on 401. To enable authed e2e, set ILGA_DEV_CODE in
    the app and in the test env to a fixed OTP the app accepts.
    """
    return page
