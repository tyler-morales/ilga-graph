"""Unified run log for scraping, ML pipeline, and startup.

Append-only JSONL store so any task can record: task name, start/end,
per-phase durations, and status. Used by the terminal dashboard
(scripts/log_dashboard.py) and the web /logs page to analyze
bottlenecks and recent runs.

Usage:
    from ilga_graph.run_log import RunLogger

    with RunLogger("ml_run") as log:
        with log.phase("Backtest"):
            ...  # do work
        with log.phase("Data Pipeline", detail="1234 rows"):
            ...
    # On exit, run is appended to .run_log.jsonl

Or manual:
    log = RunLogger("scrape")
    log.start()
    log.phase("Members + Bills", duration_s=12.3, detail="40 members")
    log.end(status="ok")
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Append-only; one JSON object per line.
DEFAULT_LOG_PATH = Path(".run_log.jsonl")


@dataclass
class PhaseRecord:
    name: str
    duration_s: float
    detail: str | None = None


@dataclass
class RunRecord:
    """One line in the run log."""

    run_id: str
    task: str
    started_at: str  # ISO
    ended_at: str | None = None
    duration_s: float | None = None
    status: str = "running"  # ok | error | running
    phases: list[dict] = field(default_factory=list)  # [{name, duration_s, detail}]
    error: str | None = None
    meta: dict = field(default_factory=dict)  # task-specific (e.g. bills_count)

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "task": self.task,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "duration_s": self.duration_s,
                "status": self.status,
                "phases": self.phases,
                "error": self.error,
                "meta": self.meta,
            }
        )

    @classmethod
    def from_json_line(cls, line: str) -> RunRecord | None:
        line = line.strip()
        if not line:
            return None
        try:
            d = json.loads(line)
            return cls(
                run_id=d.get("run_id", ""),
                task=d.get("task", ""),
                started_at=d.get("started_at", ""),
                ended_at=d.get("ended_at"),
                duration_s=d.get("duration_s"),
                status=d.get("status", "ok"),
                phases=d.get("phases", []),
                error=d.get("error"),
                meta=d.get("meta", {}),
            )
        except (json.JSONDecodeError, TypeError):
            return None


class RunLogger:
    """Context manager and programmatic API for logging a single run."""

    def __init__(
        self,
        task: str,
        *,
        log_path: Path | None = None,
        meta: dict | None = None,
    ):
        self.task = task
        self.log_path = log_path if log_path is not None else get_log_path()
        self.meta = dict(meta or {})
        self.run_id = str(uuid.uuid4())[:8]
        self._started_at: str | None = None
        self._start_time: float | None = None
        self._phases: list[dict] = []
        self._status = "ok"
        self._error: str | None = None

    def start(self) -> None:
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._start_time = time.perf_counter()
        self._phases = []
        self._status = "ok"
        self._error = None

    def phase(
        self, name: str, *, duration_s: float | None = None, detail: str | None = None
    ) -> None:
        """Record a phase. If duration_s is None, you must use @phase() context instead."""
        if duration_s is not None:
            self._phases.append(
                {"name": name, "duration_s": round(duration_s, 2), "detail": detail}
            )

    @contextmanager
    def phase_ctx(self, name: str, detail: str | None = None):
        """Context manager to time a phase."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            duration_s = time.perf_counter() - t0
            self._phases.append(
                {"name": name, "duration_s": round(duration_s, 2), "detail": detail}
            )

    def end(self, status: str = "ok", error: str | None = None) -> None:
        self._status = status
        self._error = error
        self._write()

    def _write(self) -> None:
        if self._start_time is None:
            return
        ended_at = datetime.now(timezone.utc).isoformat()
        duration_s = round(time.perf_counter() - self._start_time, 2)
        record = RunRecord(
            run_id=self.run_id,
            task=self.task,
            started_at=self._started_at or ended_at,
            ended_at=ended_at,
            duration_s=duration_s,
            status=self._status,
            phases=self._phases,
            error=self._error,
            meta=self.meta,
        )
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(record.to_json_line() + "\n")
        except OSError as e:
            LOGGER.warning("Run log append failed: %s", e)

    def __enter__(self) -> RunLogger:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self._status = "error"
            self._error = f"{exc_type.__name__}: {exc_val}" if exc_val else exc_type.__name__
        self.end(self._status, self._error)
        return None  # do not suppress


def load_recent_runs(
    n: int = 100,
    *,
    task: str | None = None,
    log_path: Path | None = None,
) -> list[RunRecord]:
    """Load the last n runs (newest first). Optionally filter by task."""
    path = log_path or DEFAULT_LOG_PATH
    if not path.exists():
        return []
    records: list[RunRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = RunRecord.from_json_line(line)
            if rec is None:
                continue
            if task is None or rec.task == task:
                records.append(rec)
    records.reverse()
    return records[-n:][::-1]


def get_log_path() -> Path:
    """Path to the run log file (for dashboard and CLI)."""
    return Path(os.environ.get("ILGA_RUN_LOG", str(DEFAULT_LOG_PATH)))


def append_startup_run(
    total_s: float,
    load_s: float,
    analytics_s: float,
    seating_s: float,
    export_s: float,
    votes_s: float,
    slips_s: float,
    zip_s: float,
    member_count: int,
    bill_count: int,
    vote_count: int,
    slip_count: int,
    zcta_count: int,
    dev_mode: bool,
    seed_mode: bool,
    *,
    log_path: Path | None = None,
) -> None:
    """Append a startup run to the run log (called from main.py lifespan)."""
    path = log_path if log_path is not None else get_log_path()
    now = datetime.now(timezone.utc).isoformat()
    phases = [
        {
            "name": "Load",
            "duration_s": round(load_s, 2),
            "detail": f"{member_count} members, {bill_count} bills",
        },
        {"name": "Analytics", "duration_s": round(analytics_s, 2), "detail": None},
        {"name": "Seating", "duration_s": round(seating_s, 2), "detail": None},
        {"name": "Export", "duration_s": round(export_s, 2), "detail": None},
        {"name": "Votes", "duration_s": round(votes_s, 2), "detail": f"{vote_count} events"},
        {"name": "Slips", "duration_s": round(slips_s, 2), "detail": f"{slip_count} slips"},
        {"name": "ZIP", "duration_s": round(zip_s, 2), "detail": f"{zcta_count} ZCTAs"},
    ]
    record = RunRecord(
        run_id=str(uuid.uuid4())[:8],
        task="startup",
        started_at=now,
        ended_at=now,
        duration_s=round(total_s, 2),
        status="ok",
        phases=phases,
        error=None,
        meta={
            "members": member_count,
            "bills": bill_count,
            "dev_mode": dev_mode,
            "seed_mode": seed_mode,
        },
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(record.to_json_line() + "\n")
    except OSError as e:
        LOGGER.warning("Startup run log append failed: %s", e)
