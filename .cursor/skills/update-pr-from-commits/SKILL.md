---
name: update-pr-from-commits
description: Updates GitHub PR title and body from the current branch's commits and TODOS.md. Use when the user asks to update the PR, refresh the PR description, sync PR with latest commits, or after pushing new commits to a feature branch and the PR is out of date.
---

# Update PR from branch commits

Sync a GitHub PR's title and body with the current branch: derive both from `git log main..HEAD`, optional diff stat, and TODOS.md; create the PR if none exists, otherwise edit the existing one via `gh pr create` / `gh pr edit`.

## When to use

- User says "update the PR", "refresh the PR description", or "sync PR with latest commits".
- After pushing new commits to a feature branch (invoked by rule or user).
- When the user says the PR is out of date with the branch.

## Workflow

### 1. Identify branch and PR

- Current branch: `git branch --show-current`.
- List PR for this branch: `gh pr list --head <branch> --json number,url`.
- If no PR exists: create one with generated title and body (steps 2–5, then `gh pr create`).
- If PR exists: get details with `gh pr view <number>`; then update title and body (steps 2–5, then `gh pr edit`).

### 2. Gather context for title and body

- Commits: `git log main..HEAD --oneline` (or `master..HEAD` if `main` does not exist).
- Optional scope: `git diff main..HEAD --stat` (or `master..HEAD`).
- Read **TODOS.md** (first ~80–120 lines) for current state and recent work to summarize.

### 3. Generate title

- One line, conventional style: `type(scope): short description` (e.g. `feat(advocacy): unified drawer and email flow`).
- Choose the dominant theme from commits/TODOS: `feat`, `fix`, `docs`, `refactor`, `chore`, etc., and a short scope if obvious.

### 4. Generate body

Write the body to a temp file (e.g. `.pr-body.md`). Structure:

```markdown
## Summary
1–2 sentences on what this branch does.

## Highlights
- Main change 1 (feature, refactor, docs, deploy)
- Main change 2
- ...

## Testing
One line (e.g. lint, tests, manual checks).
```

Use commits, diff stat, and TODOS.md to fill Summary and Highlights.

### 5. Apply to GitHub

- **No PR:**  
  `gh pr create --base main --head <branch> --title "..." --body-file .pr-body.md`  
  (Use `--base master` if the repo default branch is `master`.)
- **Existing PR:**  
  `gh pr edit <number> --title "..." --body-file .pr-body.md`
- Delete `.pr-body.md` after success.

### 6. Confirm

Tell the user the PR URL and that the title and body were updated (or that the PR was created).

## Notes

- Default base branch is `main`; use `master` only if `main` does not exist.
- Keep the title under ~72 characters; keep the body scannable (short bullets, one-line Testing).
