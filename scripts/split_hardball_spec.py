#!/usr/bin/env python3
"""
Split hardball.txt into chunked markdown files under docs/hardball-spec/.
Run from repo root: python scripts/split_hardball_spec.py
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "hardball.txt"
OUT_DIR = REPO_ROOT / "docs" / "hardball-spec"

# (start_line_1based, end_line_inclusive, filename, title)
# Line numbers from grep on hardball.txt (1-based)
SECTIONS = [
    (138, 182, "00-foreword.md", "Foreword"),
    (184, 218, "01-introduction.md", "Introduction: The Value of Nonprofit Lobbying to Democracy"),
    (221, 317, "02-ch1-framing-context.md", "Ch 1: Framing the Context for a New Approach"),
    (319, 379, "03-ch2-new-paradigm.md", "Ch 2: Toward a New Paradigm for Nonprofit Advocacy/Lobbying"),
    (381, 700, "04-ch3-decision-making.md", "Ch 3: The Decision-Making Process"),
    (703, 874, "05-ch4-advocacy-law.md", "Ch 4: Advocacy, Lobbying, and the Law"),
    (876, 1089, "06-ch5-advocacy-foundation.md", "Ch 5: Building an Advocacy Foundation"),
    (1091, 1579, "07-ch6-managing-lobbying.md", "Ch 6: Managing the Lobbying Effort/Organization"),
    (1581, 2326, "08-ch7-influencing-process.md", "Ch 7: Influencing the Decision-Making Process"),
    (2328, 2371, "09-ch8-postmortem.md", "Ch 8: Postmortem"),
    (2372, 2540, "10-bibliography.md", "Bibliography & Resources"),
]


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for start_1, end_1, filename, title in SECTIONS:
        start_idx = start_1 - 1
        end_idx = min(end_1, len(lines))
        section_lines = lines[start_idx:end_idx]
        content = "\n".join(section_lines)
        body = f"# {title}\n\nSource: hardball.txt (Barry Hessenius, *Hardball Lobbying for Nonprofits*)\n\n---\n\n{content}"
        (OUT_DIR / filename).write_text(body, encoding="utf-8")
        print(f"Wrote {filename} ({len(section_lines)} lines)")

    print(f"Done. Chunks in {OUT_DIR}")


if __name__ == "__main__":
    main()
