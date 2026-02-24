---
name: god-tier-debugging
description: Enforces a three-step debugging workflow—repro first (failing test), isolate (bisect logic path), root cause (fix data/schema not symptoms). Use when the user mentions bug, fix, error, or issue.
---

# God-Tier Debugging Workflow

Apply this workflow whenever debugging a bug, error, or issue. Do not suggest a fix until all three steps are satisfied.

## 1. Repro First

**Do not suggest a fix until you have a failing test case that reproduces the bug.**

- Add or identify a test (unit, integration, or minimal script) that fails in the presence of the bug.
- The test must be deterministic and clearly demonstrate the wrong behavior.
- If the user has not provided steps to reproduce, ask for them or infer from the error and add a test that encodes those steps.

## 2. Isolate

**Narrow down the cause by bisecting the logic path.**

- Identify the minimal code path that triggers the failure.
- Use binary search on the call stack or data flow: comment out or stub half the path, re-run; repeat until the failing region is as small as possible.
- Distinguish "this code path is involved" from "this is the actual cause."

## 3. Root Cause

**Fix the systemic issue (data/schema/contract), not just the symptom (edge case).**

- Prefer fixing the **source**: wrong data shape, missing validation, incorrect schema, or broken invariant.
- Avoid one-off guards or special cases unless they are the only correct fix (e.g. a genuine edge case in the spec).
- After the fix, the repro test from step 1 must pass, and no new tests should fail.

## Checklist

Before proposing a fix:

- [ ] Failing test (or equivalent repro) exists and is committed or documented.
- [ ] Cause isolated to a minimal logic path or component.
- [ ] Fix targets root cause (data/schema/contract) rather than symptom where possible.
- [ ] Repro test passes after the fix.

## Anti-patterns

- **Suggesting a fix before repro:** Always write or run a failing test first.
- **Fixing the symptom only:** e.g. adding `if x is None` without fixing why `x` can be wrong.
- **Skipping isolation:** Guessing at the cause instead of bisecting to the real failure point.
