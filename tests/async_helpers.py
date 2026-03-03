"""Run coroutines from sync test code when an event loop may already be running (e.g. TestClient)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync test code.

    If no loop is running, uses asyncio.run(). If a loop is already running, runs the
    coroutine in a separate thread with its own loop to avoid RuntimeError from nested
    asyncio.run().
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[return-value]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()  # type: ignore[return-value]
