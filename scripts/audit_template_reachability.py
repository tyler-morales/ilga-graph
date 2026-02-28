#!/usr/bin/env python3
"""
Audit template reachability: find all templates that are either rendered directly
(TemplateResponse / get_template) or included/extended, and report any template file
that is never referenced (candidate for removal after manual confirmation).

Usage:
  python scripts/audit_template_reachability.py [--templates-dir ...] [--report]

Scans: Python TemplateResponse/get_template; HTML {% include %} / {% extends %}.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def collect_rendered_templates(repo_root: Path) -> set[str]:
    """From Python files, collect template names passed to TemplateResponse or get_template."""
    names: set[str] = set()
    for path in (repo_root / "src/ilga_graph").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r'TemplateResponse\s*\(\s*["\']([^"\']+\.html)["\']', text):
            names.add(m.group(1))
        for m in re.finditer(
            r'TemplateResponse\s*\(\s*request\s*,\s*["\']([^"\']+\.html)["\']', text
        ):
            names.add(m.group(1))
        for m in re.finditer(r'get_template\s*\(\s*["\']([^"\']+)["\']', text):
            n = m.group(1)
            if not n.endswith(".html"):
                n += ".html"
            names.add(n)
        for m in re.finditer(r'["\']([a-zA-Z_][a-zA-Z0-9_/]*\.html)["\']', text):
            names.add(m.group(1))
        for m in re.finditer(r'"template":\s*["\']([^"\']+\.html)["\']', text):
            names.add(m.group(1))
    return names


def collect_includes_extends(templates_dir: Path) -> set[str]:
    """From HTML templates, collect names in {% include "..." %} and {% extends "..." %}."""
    names: set[str] = set()
    for path in templates_dir.rglob("*.html"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r'{%\s*include\s+["\']([^"\']+)["\']', text):
            n = m.group(1)
            if not n.endswith(".html"):
                n += ".html"
            names.add(n)
        for m in re.finditer(r'{%\s*extends\s+["\']([^"\']+)["\']', text):
            n = m.group(1)
            if not n.endswith(".html"):
                n += ".html"
            names.add(n)
    return names


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    templates_dir = repo_root / "src/ilga_graph/templates"
    report_path = repo_root / "audit_template_reachability_report.txt"
    write_report = "--report" in sys.argv

    rendered = collect_rendered_templates(repo_root)
    included = collect_includes_extends(templates_dir)
    referenced = rendered | included

    template_files = {p.name for p in templates_dir.rglob("*.html")}
    # Also names in subdirs (e.g. dev_playground/_scene_truck.html)
    template_files |= {str(p.relative_to(templates_dir)) for p in templates_dir.rglob("*.html")}

    unreachable = []
    for p in sorted(templates_dir.rglob("*.html")):
        rel = str(p.relative_to(templates_dir))
        if rel not in referenced:
            unreachable.append(rel)

    if unreachable:
        print("Templates not referenced by TemplateResponse, get_template, include, or extends:")
        for u in unreachable:
            print(f"  {u}")
    else:
        print("All templates referenced (TemplateResponse/get_template or include/extends).")

    if write_report:
        report_path.write_text("\n".join(unreachable) + "\n", encoding="utf-8")
        print(f"\nWrote report to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
