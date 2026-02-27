#!/usr/bin/env python3
"""
Audit CSS usage: extract class/id from templates, compare to CSS selectors,
report candidate dead rules.

Usage:
  python scripts/audit_css_usage.py [--css-dir ...] [--templates-dir ...] [--report]

Output: Prints used counts, then candidate-dead (file:line). Use --report to write
audit_css_report.txt. Safelist prefixes/literals keep dynamic/third-party selectors.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Selectors matching these prefixes are never flagged as dead (third-party or JS-injected).
SAFELIST_PREFIXES = (
    "tippy-",
    "htmx-",
    "drawer-",
    "reminder-modal",
    "power-badge-",  # backend pb.css_class values
    "phone-",
    "tooltip-",
    "signal-chip",
    "js-",
    "iconify-",
    "popper-",
)

# Exact class/id names that are dynamic or always keep.
SAFELIST_LITERALS = frozenset(
    {
        "power-badge",
        "open",
        "active",
        "hidden",
        "offline-indicator",
        "current-action-banner--dismissed",
        "beta-banner--dismissed",
        "session-pill--expanded",
    }
)

# Element selectors we never flag (no class/id or global).
GLOBAL_SELECTORS = frozenset(
    {
        "*",
        "html",
        "body",
        "a",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "ul",
        "li",
        "span",
        "div",
        "button",
        "input",
        "label",
        "nav",
        "main",
        "footer",
        "header",
        "section",
        "article",
        "svg",
        "path",
    }
)


def extract_used_from_templates(templates_dir: Path) -> tuple[set[str], set[str]]:
    """Extract class and id tokens from all HTML templates and inline script references."""
    used_classes: set[str] = set()
    used_ids: set[str] = set()
    templates_dir = Path(templates_dir)
    if not templates_dir.is_dir():
        return used_classes, used_ids

    for path in templates_dir.rglob("*.html"):
        text = path.read_text(encoding="utf-8", errors="replace")
        # {% block body_class %}page-home{% endblock %} etc.
        for m in re.finditer(r"{%\s*block\s+body_class\s*%}\s*([^%{}\s]+)\s*{%", text):
            used_classes.add(m.group(1).strip())
        # class="..." and class='...'
        for m in re.finditer(r'\bclass\s*=\s*["\']([^"\']*)["\']', text):
            for token in re.split(r"\s+", m.group(1).strip()):
                token = token.strip()
                if token and "{{" not in token:
                    used_classes.add(token)
                # Jinja class like "power-badge {{ pb.css_class }}" -> keep power-badge
                for part in re.split(r"\s+", m.group(1)):
                    if part.startswith(".") or (part and "{" not in part):
                        used_classes.add(part.strip(".\"'"))
        for m in re.finditer(r'\bid\s*=\s*["\']([^"\']*)["\']', text):
            tid = m.group(1).strip()
            if tid and "{{" not in tid:
                used_ids.add(tid)
        # Script references: getElementById('x'), querySelector('.x'), querySelectorAll('.x')
        for m in re.finditer(r"getElementById\s*\(\s*['\"]([^'\"]+)['\"]", text):
            used_ids.add(m.group(1))
        for m in re.finditer(r"querySelector(?:All)?\s*\(\s*['\"]([^'\"]+)['\"]", text):
            sel = m.group(1).strip()
            if sel.startswith("."):
                for token in re.split(r"[\s>+~.#\[]+", sel[1:]):
                    if token and not token.startswith("["):
                        used_classes.add(token)
            elif sel.startswith("#"):
                used_ids.add(sel[1:].split("[")[0])
            else:
                # e.g. "div.foo" -> add foo
                for part in re.split(r"[\s>+~,.#\[]+", sel):
                    if part and part not in GLOBAL_SELECTORS and not part.startswith("["):
                        if part.startswith("#"):
                            used_ids.add(part[1:].split("[")[0])
                        else:
                            used_classes.add(part)
        for m in re.finditer(r"getElementsByClassName\s*\(\s*['\"]([^'\"]+)['\"]", text):
            used_classes.add(m.group(1))
        for m in re.finditer(r"closest\s*\(\s*['\"]([^'\"]+)['\"]", text):
            sel = m.group(1)
            if sel.startswith("."):
                used_classes.add(sel[1:].split()[0].split("[")[0])
            elif sel.startswith("#"):
                used_ids.add(sel[1:].split("[")[0])
    return used_classes, used_ids


def is_safelisted(token: str, is_id: bool) -> bool:
    if token in SAFELIST_LITERALS:
        return True
    if is_id:
        for p in SAFELIST_PREFIXES:
            if p.strip("-") in token or token.startswith(p.replace("-", "")):
                return True
        return False
    for p in SAFELIST_PREFIXES:
        if token.startswith(p) or token == p.rstrip("-"):
            return True
    return False


def extract_leading_class_or_id(selector: str) -> list[tuple[str, bool]]:
    """From a single selector (no comma), return [(name, is_id), ...] for classes and ids."""
    # Normalize: take the rightmost segment (last part after space or >) for specificity.
    parts = re.split(r"\s+|\s*>\s*", selector.strip())
    last = parts[-1] if parts else ""
    tokens: list[tuple[str, bool]] = []
    # Match #id and .class in last segment
    for m in re.finditer(r"#([a-zA-Z0-9_-]+)", last):
        tokens.append((m.group(1), True))
    for m in re.finditer(r"\.([a-zA-Z0-9_-]+)", last):
        tokens.append((m.group(1), False))
    if not tokens and last and not re.match(r"^[a-zA-Z]*$", last):
        # compound like .a.b
        for m in re.finditer(r"\.([a-zA-Z0-9_-]+)", selector):
            tokens.append((m.group(1), False))
        for m in re.finditer(r"#([a-zA-Z0-9_-]+)", selector):
            tokens.append((m.group(1), True))
    return tokens


def _parse_css_flat(text: str, start_pos: int, start_line: int) -> list[tuple[int, str, str]]:
    """Parse CSS and return (line, selector_block, body); flatten @media by recursing into body."""
    rules: list[tuple[int, str, str]] = []
    pos = start_pos
    line = start_line
    n = len(text)
    while pos < n:
        open_ = text.find("{", pos)
        if open_ == -1:
            break
        line = start_line + text.count("\n", 0, open_)
        selector_block = text[pos:open_].strip()
        depth = 1
        close = open_ + 1
        while close < n and depth:
            if text[close] == "{":
                depth += 1
            elif text[close] == "}":
                depth -= 1
            close += 1
        body = text[open_ + 1 : close - 1]
        if selector_block.strip().startswith("@"):
            # Recurse into @media / @supports body to collect inner rules
            inner_start_line = line + 1
            rules.extend(_parse_css_flat(body, 0, inner_start_line))
        else:
            rules.append((line, selector_block, body))
        pos = close
    return rules


def parse_css_rules_simple(css_path: Path) -> list[tuple[int, str, str]]:
    """Parse CSS into (line_no, selector_block, body); @media flattened."""
    text = css_path.read_text(encoding="utf-8", errors="replace")
    return _parse_css_flat(text, 0, 1)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    css_dir = repo_root / "src/ilga_graph/static/css"
    templates_dir = repo_root / "src/ilga_graph/templates"
    report_path = repo_root / "audit_css_report.txt"
    write_report = "--report" in sys.argv

    used_classes, used_ids = extract_used_from_templates(templates_dir)
    print(f"Used classes: {len(used_classes)}")
    print(f"Used ids: {len(used_ids)}")

    dead_report: list[str] = []
    for css_file in sorted(css_dir.glob("*.css")):
        try:
            rules = parse_css_rules_simple(css_file)
        except Exception as e:
            print(f"Warning: could not parse {css_file}: {e}", file=sys.stderr)
            continue
        rel = css_file.relative_to(repo_root)
        for line_no, selector_block, _body in rules:
            # Strip CSS comments so "/* ... */ .foo" becomes ".foo"
            selector_block = re.sub(r"/\*[\s\S]*?\*/", " ", selector_block)
            for sel in re.split(r"\s*,\s*", selector_block):
                sel = sel.strip()
                if not sel or sel.startswith("@"):
                    continue
                # Skip pure element selectors
                if re.match(r"^[a-zA-Z]+$", sel) and sel in GLOBAL_SELECTORS:
                    continue
                if sel.startswith("*") or sel == "html" or sel == "body":
                    continue
                tokens = extract_leading_class_or_id(sel)
                if not tokens:
                    continue
                is_dead = True
                for name, is_id in tokens:
                    if is_id and name in used_ids:
                        is_dead = False
                        break
                    if not is_id and name in used_classes:
                        is_dead = False
                        break
                    if is_safelisted(name, is_id):
                        is_dead = False
                        break
                if is_dead:
                    line = f"{rel}:{line_no} {sel}"
                    dead_report.append(line)
                    print(line)

    if write_report:
        report_path.write_text("\n".join(dead_report) + "\n", encoding="utf-8")
        print(f"\nWrote {len(dead_report)} candidate-dead rules to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
