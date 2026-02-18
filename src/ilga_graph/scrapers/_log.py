"""Shared logging helpers for the scraper pipeline."""

from __future__ import annotations

import logging


def fmt_duration(seconds: float) -> str:
    """Format seconds as ``Xm YYs`` (e.g. ``2m 05s``, ``0m 30s``)."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def log_progress(
    logger: logging.Logger,
    completed: int,
    total: int,
    bill_number: str,
    elapsed: float,
    eta: float,
) -> None:
    """Log a standardized progress line.

    Format: ``[  N/total] BILL_NUMBER — Xm Ys elapsed, ~Wm Zs remaining``
    """
    logger.info(
        "  [%*d/%d] %-10s — %s elapsed, ~%s remaining",
        len(str(total)),
        completed,
        total,
        bill_number,
        fmt_duration(elapsed),
        fmt_duration(eta),
    )


def log_phase(logger: logging.Logger, label: str, elapsed: float, detail: str = "") -> None:
    """Log a phase-complete summary line."""
    suffix = f" — {detail}" if detail else ""
    logger.info("%s complete in %s%s", label, fmt_duration(elapsed), suffix)
