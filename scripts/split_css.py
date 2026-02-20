#!/usr/bin/env python3
"""Split monolithic base.css into base, advocacy, intelligence. Strips leading indent.

Note: The app now uses split advocacy (form, drawer, email, cards) and intelligence
(dashboard, tables) CSS files; this script outputs single advocacy.css and
intelligence.css. To regenerate the split files, run this script then split
advocacy.css and intelligence.css by section (see base.html link order).
"""

from pathlib import Path

STATIC_CSS = Path(__file__).resolve().parent.parent / "src" / "ilga_graph" / "static" / "css"
BASE = STATIC_CSS / "base.css"


def strip_indent(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        if line.strip():
            out.append(line.lstrip())
        else:
            out.append("")
    return out


def main() -> None:
    raw = BASE.read_text()
    lines = raw.splitlines(keepends=True)
    # Normalize: strip leading spaces that were in the original <style> block
    normalized = strip_indent(lines)

    def get_range(start: int, end: int) -> str:
        return "".join(normalized[start:end])

    # 1-based line ranges (inclusive) -> 0-based slice
    # base: reset+body+container+typography (1-56), error+footer+htmx (5457-5504),
    #       section headers + tables + score bars (5620-5801), responsive (7422-7697)
    # advocacy: 57-5456
    # intelligence: intel dashboard/tabs (5505-5619), committee/action (5802-7421)
    base_parts = [
        get_range(0, 56),  # 1-56
        get_range(5456, 5504),  # 5457-5504
        get_range(5619, 5801),  # 5620-5801
        get_range(7421, len(normalized)),  # 7422-end
    ]
    advocacy_css = get_range(56, 5456)  # 57-5456
    intel_parts = [
        get_range(5504, 5619),  # 5505-5619
        get_range(5801, 7421),  # 5802-7421
    ]
    intelligence_css = "".join(intel_parts)

    (STATIC_CSS / "base.css").write_text("".join(base_parts))
    (STATIC_CSS / "advocacy.css").write_text(advocacy_css)
    (STATIC_CSS / "intelligence.css").write_text(intelligence_css)
    print("Wrote base.css, advocacy.css, intelligence.css")


if __name__ == "__main__":
    main()
