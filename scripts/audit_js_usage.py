#!/usr/bin/env python3
"""
Audit JS usage: find function definitions in templates (inline script) and static/js,
then report those that appear never to be called (candidate dead code).

Usage:
  python scripts/audit_js_usage.py [--templates-dir ...] [--js-dir ...] [--report]

Output: Prints candidate-dead function names. Use --report to write audit_js_report.txt.
Conservative: only when zero references; callback/string refs may be missed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Entry points (HTML onclick, library, IIFE) — never flag as dead.
SAFELIST = frozenset(
    {
        "closeAdvocacyDrawer",
        "openDrawer",
        "confetti",
        "tippy",
        "htmx",
        "requestAnimationFrame",
        "setTimeout",
        "addEventListener",
        "dispatchEvent",
        "fetch",
        "replaceState",
        "scrollIntoView",
        "getCsrfToken",
        "updateAuthStrip",
        "refreshAuthStripProgress",
        "getAnonSessionId",
        "updateGoalBlockFromStorage",
        "mergeLocalStorageGoalToServer",
        "logGoalListDiagnostic",
        "initAuthStrip",
        "initHeroInlineSignin",
    }
)


def extract_script_sources(templates_dir: Path, js_dir: Path) -> list[tuple[str, str]]:
    """Return [(source_name, content), ...] for all inline scripts and external JS files."""
    sources: list[tuple[str, str]] = []
    for path in templates_dir.rglob("*.html"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", text):
            if "src=" in m.group(0)[:50]:
                continue
            content = m.group(1).strip()
            if content and "application/json" not in m.group(0):
                sources.append((str(path), content))
    for path in js_dir.glob("*.js"):
        sources.append((str(path), path.read_text(encoding="utf-8", errors="replace")))
    return sources


def find_definitions(content: str) -> set[str]:
    """Return set of function names defined in content (declarations and var/const = function)."""
    names: set[str] = set()
    # function name( or function name (
    for m in re.finditer(r"\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(", content):
        names.add(m.group(1))
    # var name = function or const name = function
    pat = r"\b(?:var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*function\s*\("
    for m in re.finditer(pat, content):
        names.add(m.group(1))
    return names


def count_definitions(content: str, name: str) -> int:
    """Count how many times name is defined (function name( or var name = function)."""
    pattern = re.compile(
        r"\bfunction\s+"
        + re.escape(name)
        + r"\s*\(|\b(?:var|let|const)\s+"
        + re.escape(name)
        + r"\s*=\s*function\s*\("
    )
    return len(pattern.findall(content))


def count_references(content: str, name: str) -> int:
    """Count uses: name(, .name(, window.name, typeof name, or callback refs."""
    escaped = re.escape(name)
    patterns = [
        r"(?<![a-zA-Z0-9_])\b" + escaped + r"\s*\(",  # name(
        r"\.\s*" + escaped + r"\s*\(",  # .name(
        r"window\.\s*" + escaped + r"\b",
        r"typeof\s+" + escaped,
        r"=\s*" + escaped + r"\s*[;,\)]",  # = name; or = name, or = name)
        r",\s*" + escaped + r"\s*[\)\]}]",  # , name) or , name]
        r"\(\s*" + escaped + r"\s*\)",  # ( name )
        r"\.(?:onclick|onload|onerror)\s*=\s*" + escaped,  # .onclick = name
    ]
    total = 0
    for p in patterns:
        total += len(re.findall(p, content))
    return total


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    templates_dir = repo_root / "src/ilga_graph/templates"
    js_dir = repo_root / "src/ilga_graph/static/js"
    report_path = repo_root / "audit_js_report.txt"
    write_report = "--report" in sys.argv

    sources = extract_script_sources(templates_dir, js_dir)
    all_definitions: set[str] = set()
    for _source_name, content in sources:
        all_definitions |= find_definitions(content)

    dead: list[str] = []
    for name in sorted(all_definitions):
        if name in SAFELIST:
            continue
        total_defs = sum(count_definitions(c, name) for _, c in sources)
        total_refs = sum(count_references(c, name) for _, c in sources)
        if total_defs >= 1 and total_refs <= total_defs:
            dead.append(name)
            print(name)

    if write_report:
        report_path.write_text("\n".join(dead) + "\n", encoding="utf-8")
        print(f"\nWrote {len(dead)} candidate-dead function names to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
