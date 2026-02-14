#!/usr/bin/env python3
"""Terminal view of the unified run log — scraping, ML, startup.

2000s-hacker style: minimal, monospace, green/amber. Use for quick
bottleneck checks and "how things are looking."

Usage:
    make logs              # last 20 runs
    make logs N=50         # last 50 runs
    python scripts/log_dashboard.py --tail 30
    python scripts/log_dashboard.py --task ml_run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ilga_graph.run_log import get_log_path, load_recent_runs  # noqa: E402

# ANSI (strip if not TTY for pipes)
G = "\033[92m"  # green
Y = "\033[33m"  # amber
R = "\033[91m"  # red
D = "\033[90m"  # dim
B = "\033[1m"  # bold
X = "\033[0m"  # reset


def _t(s: str) -> str:
    """Shorten timestamp to local time."""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%m/%d %H:%M")
    except Exception:
        return s[:16] if len(s) >= 16 else s


def _fmt_dur(s: float | None) -> str:
    if s is None:
        return "—"
    if s >= 60:
        return f"{s / 60:.1f}m"
    return f"{s:.1f}s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="View unified run log (scrape, ml_run, startup).",
    )
    parser.add_argument(
        "--tail",
        "-n",
        type=int,
        default=20,
        help="Number of recent runs to show (default: 20).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter by task name (e.g. ml_run, scrape, startup).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color.",
    )
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()
    if not use_color:
        global G, Y, R, D, B, X
        G = Y = R = D = B = X = ""

    path = get_log_path()
    if not path.exists():
        print(f"{D}Run log empty or missing: {path}{X}")
        print(f"{D}Run 'make dev', 'make ml-run', or 'make scrape' to generate entries.{X}")
        return 0

    runs = load_recent_runs(n=args.tail, task=args.task)
    if not runs:
        print(f"{D}No runs found (task={args.task or 'any'}).{X}")
        return 0

    # Header
    pad = 36 - len(str(args.task or ""))
    task_str = f"task={args.task}" if args.task else ""
    print(f"{B}{G}╔══════════════════════════════════════════════════════════════╗{X}")
    print(f"{B}{G}║  RUN LOG  tail={len(runs)}  {task_str}{' ' * pad}║{X}")
    print(f"{B}{G}╚══════════════════════════════════════════════════════════════╝{X}")
    print()

    # Bottleneck hint: which phase is usually slowest per task
    task_phases: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        if r.task not in task_phases:
            task_phases[r.task] = {}
        for p in r.phases:
            name = p.get("name", "?")
            if name not in task_phases[r.task]:
                task_phases[r.task][name] = []
            task_phases[r.task][name].append(p.get("duration_s", 0) or 0)

    for r in runs:
        status_color = G if r.status == "ok" else R
        task_color = Y
        print(
            f"  {D}{_t(r.started_at)}{X}  {task_color}{r.task:12}{X}  "
            f"{_fmt_dur(r.duration_s):>6}  {status_color}{r.status}{X}  {D}#{r.run_id}{X}"
        )
        # Top 2 slowest phases for this run
        if r.phases:

            def _phase_dur(p):
                return p.get("duration_s", 0) or 0

            sorted_phases = sorted(r.phases, key=_phase_dur, reverse=True)
            for p in sorted_phases[:2]:
                d = p.get("duration_s")
                detail = p.get("detail") or ""
                if d is not None and d > 0:
                    tail = f"  {detail}" if detail else ""
                    print(f"       {D}└ {p.get('name', '?')}: {_fmt_dur(d)}{tail}{X}")
        print()

    # Summary: bottleneck by task
    print(f"{B}{G}── Bottleneck (avg phase time) ──{X}")
    for task, phases in task_phases.items():
        by_name: dict[str, float] = {}
        for name, durs in phases.items():
            by_name[name] = sum(durs) / len(durs) if durs else 0
        top = sorted(by_name.items(), key=lambda x: x[1], reverse=True)[:3]
        parts = [f"{name}: {_fmt_dur(avg)}" for name, avg in top]
        print(f"  {Y}{task:12}{X}  {', '.join(parts)}")
    print(f"{D}Log file: {path}{X}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
