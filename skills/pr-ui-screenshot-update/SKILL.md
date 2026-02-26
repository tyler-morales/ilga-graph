---
name: pr-ui-screenshot-update
description: Detect UI-facing file changes on new commits, capture updated UI screenshots, and post or update a pull-request comment with results. Use when a user asks to refresh PR visuals after a push, verify whether a commit changed UI, or keep PRs updated with latest screenshots during review.
---

# PR UI Screenshot Update

Run a repeatable flow that checks whether a push changed UI files, captures a screenshot when needed, and updates the PR with a single status comment.

## Workflow

1. Confirm repository state and target PR.
- Resolve PR number from `--pr-number` or infer it from current branch with `gh pr view --json number`.
- Resolve repo from `--repo` or infer it with `gh repo view --json nameWithOwner`.

2. Detect whether the new commit changed UI-relevant files.
- Compare `--base-ref` and `--head-ref` using `git diff --name-only <base>...<head>`.
- Match files against UI patterns: HTML/CSS/JS/TSX, templates, or static frontend assets.
- Skip screenshot capture when no UI files changed.

3. Capture screenshot when UI changes exist.
- Require `--url` for the page to capture.
- Use `npx --yes playwright screenshot --wait-for-timeout <ms> <url> <output.png>`.
- Store artifacts under `--output-dir` (default `tmp/pr-ui-screenshots`).

4. Optionally commit screenshot for inline PR rendering.
- Use `--commit-screenshot` to add screenshot to git.
- Use `--push` (with `--commit-screenshot`) to push the commit.
- When committed, include a GitHub raw image URL in the PR comment.

5. Update PR comment.
- Maintain one sticky comment with marker `<!-- pr-ui-screenshot-update -->`.
- Create a comment if missing; patch existing marker comment if present.
- Include commit SHA, UI change summary, and screenshot link/path.

## Command

```bash
python3 skills/pr-ui-screenshot-update/scripts/update_pr_ui_screenshot.py \
  --base-ref origin/main \
  --head-ref HEAD \
  --url http://127.0.0.1:8000 \
  --pr-number 123
```

## Options

- `--base-ref`: merge base side for diff (default `origin/main`)
- `--head-ref`: head side for diff (default `HEAD`)
- `--url`: page URL to screenshot (required when UI changed)
- `--output-dir`: screenshot output directory
- `--repo`: `OWNER/REPO`; inferred if omitted
- `--pr-number`: PR number; inferred if omitted
- `--ui-pattern`: custom regex for UI file detection (repeatable)
- `--wait-ms`: Playwright wait before capture (default `2000`)
- `--commit-screenshot`: commit generated screenshot
- `--push`: push the screenshot commit
- `--dry-run`: print actions without writing comments

## Operational Notes

- Start the app server before running this workflow.
- Keep screenshot path stable in CI to make artifact collection predictable.
- Use [references/github-actions-example.md](references/github-actions-example.md) for an end-to-end pull-request automation template.
