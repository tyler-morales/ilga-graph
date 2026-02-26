#!/usr/bin/env python3
"""Detect UI changes, capture screenshot, and update a PR comment."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MARKER = "<!-- pr-ui-screenshot-update -->"
DEFAULT_PATTERNS = [
    r"(^|/)src/.+\\.(html|css|js|mjs|cjs|ts|tsx)$",
    r"(^|/)src/.+/templates/.+\\.html$",
    r"(^|/)src/.+/static/.+\\.(css|js|png|jpg|jpeg|webp|svg|gif)$",
    r"(^|/)templates/.+\\.html$",
    r"(^|/)frontend/.+",
]


@dataclass
class Config:
    base_ref: str
    head_ref: str
    url: str | None
    output_dir: Path
    repo: str | None
    pr_number: int | None
    ui_patterns: list[str]
    wait_ms: int
    commit_screenshot: bool
    push: bool
    dry_run: bool


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def run_lines(cmd: list[str]) -> list[str]:
    result = run(cmd)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def resolve_repo(explicit_repo: str | None) -> str:
    if explicit_repo:
        return explicit_repo
    env_repo = os.getenv("GITHUB_REPOSITORY")
    if env_repo:
        return env_repo
    result = run(["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    repo = result.stdout.strip()
    if not repo:
        raise RuntimeError("Could not resolve repository. Pass --repo OWNER/REPO.")
    return repo


def resolve_pr_number(explicit_pr: int | None, repo: str) -> int:
    if explicit_pr:
        return explicit_pr
    result = run(["gh", "pr", "view", "--repo", repo, "--json", "number", "--jq", ".number"])
    value = result.stdout.strip()
    if not value:
        raise RuntimeError("Could not resolve PR number. Pass --pr-number.")
    return int(value)


def changed_files(base_ref: str, head_ref: str) -> list[str]:
    return run_lines(["git", "diff", "--name-only", f"{base_ref}...{head_ref}"])


def has_ui_changes(files: list[str], patterns: list[str]) -> list[str]:
    regexes = [re.compile(pattern) for pattern in patterns]
    matches: list[str] = []
    for path in files:
        if any(regex.search(path) for regex in regexes):
            matches.append(path)
    return matches


def capture_screenshot(url: str, output_path: Path, wait_ms: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "npx",
        "--yes",
        "playwright",
        "screenshot",
        "--wait-for-timeout",
        str(wait_ms),
        url,
        str(output_path),
    ]
    run(cmd)


def current_sha() -> str:
    return run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()


def current_branch() -> str:
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def commit_and_optionally_push(path: Path, push: bool) -> None:
    run(["git", "add", str(path)])
    status = run(["git", "status", "--porcelain", "--", str(path)]).stdout.strip()
    if not status:
        return
    sha = current_sha()
    run(["git", "commit", "-m", f"chore: add PR UI screenshot for {sha}"])
    if push:
        run(["git", "push"])


def github_raw_url(repo: str, branch: str, path: Path) -> str:
    quoted = str(path).replace(" ", "%20")
    return f"https://github.com/{repo}/blob/{branch}/{quoted}?raw=1"


def build_comment(
    sha: str,
    base_ref: str,
    head_ref: str,
    changed_ui: list[str],
    screenshot_path: Path | None,
    screenshot_url: str | None,
) -> str:
    lines = [
        MARKER,
        "## PR UI Screenshot Update",
        f"- Commit: `{sha}`",
        f"- Diff: `{base_ref}...{head_ref}`",
        f"- UI files changed: `{len(changed_ui)}`",
    ]

    if changed_ui:
        lines.append("\n### Changed UI Files")
        for item in changed_ui[:20]:
            lines.append(f"- `{item}`")
        if len(changed_ui) > 20:
            lines.append(f"- `... {len(changed_ui) - 20} more`")

    if screenshot_url:
        lines.extend(["\n### Screenshot", f"![Updated UI]({screenshot_url})"])
    elif screenshot_path:
        lines.extend(["\n### Screenshot", f"Generated at `{screenshot_path}`"])
    else:
        lines.append("\nNo screenshot generated because no UI file changes were detected.")

    return "\n".join(lines)


def find_existing_comment_id(repo: str, pr_number: int) -> int | None:
    result = run(
        ["gh", "api", f"repos/{repo}/issues/{pr_number}/comments", "--paginate"],
        check=True,
    )
    comments = json.loads(result.stdout)
    for comment in comments:
        body = comment.get("body") or ""
        if MARKER in body:
            return int(comment["id"])
    return None


def upsert_comment(repo: str, pr_number: int, body: str) -> None:
    comment_id = find_existing_comment_id(repo, pr_number)
    if comment_id is None:
        run(
            [
                "gh",
                "api",
                f"repos/{repo}/issues/{pr_number}/comments",
                "--method",
                "POST",
                "-f",
                f"body={body}",
            ]
        )
        return

    run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/comments/{comment_id}",
            "--method",
            "PATCH",
            "-f",
            f"body={body}",
        ]
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--url", help="URL to screenshot when UI changed")
    parser.add_argument("--output-dir", default="tmp/pr-ui-screenshots")
    parser.add_argument("--repo", help="GitHub repo in OWNER/REPO format")
    parser.add_argument("--pr-number", type=int, help="Pull request number")
    parser.add_argument(
        "--ui-pattern",
        action="append",
        default=[],
        help="Regex pattern for UI file detection (repeatable)",
    )
    parser.add_argument("--wait-ms", type=int, default=2000)
    parser.add_argument("--commit-screenshot", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.push and not args.commit_screenshot:
        parser.error("--push requires --commit-screenshot")

    return Config(
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        url=args.url,
        output_dir=Path(args.output_dir),
        repo=args.repo,
        pr_number=args.pr_number,
        ui_patterns=args.ui_pattern or DEFAULT_PATTERNS,
        wait_ms=args.wait_ms,
        commit_screenshot=args.commit_screenshot,
        push=args.push,
        dry_run=args.dry_run,
    )


def main() -> int:
    cfg = parse_args()

    files = changed_files(cfg.base_ref, cfg.head_ref)
    changed_ui = has_ui_changes(files, cfg.ui_patterns)

    screenshot_path: Path | None = None
    screenshot_url: str | None = None
    sha = current_sha()

    if cfg.dry_run:
        body = build_comment(
            sha=sha,
            base_ref=cfg.base_ref,
            head_ref=cfg.head_ref,
            changed_ui=changed_ui,
            screenshot_path=None,
            screenshot_url=None,
        )
        print(body)
        return 0

    if changed_ui:
        if not cfg.url:
            raise RuntimeError("UI changes detected but no --url provided for screenshot capture.")
        screenshot_path = cfg.output_dir / f"pr-ui-{sha}.png"
        capture_screenshot(cfg.url, screenshot_path, cfg.wait_ms)

    repo = resolve_repo(cfg.repo)
    pr_number = resolve_pr_number(cfg.pr_number, repo)

    if screenshot_path and cfg.commit_screenshot:
        commit_and_optionally_push(screenshot_path, cfg.push)
        screenshot_url = github_raw_url(repo, current_branch(), screenshot_path)

    body = build_comment(
        sha=sha,
        base_ref=cfg.base_ref,
        head_ref=cfg.head_ref,
        changed_ui=changed_ui,
        screenshot_path=screenshot_path,
        screenshot_url=screenshot_url,
    )

    upsert_comment(repo, pr_number, body)
    print(f"Updated PR #{pr_number} in {repo}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if exc.stderr else str(exc)
        print(f"ERROR: command failed: {message}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
