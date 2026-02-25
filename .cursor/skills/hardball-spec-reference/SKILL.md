---
name: hardball-spec-reference
description: Before implementing advocacy/lobbying features, planning product behavior, or fixing advocacy-related bugs, reads the relevant chunk(s) from docs/hardball-spec/ and cites specific spec text when proposing or implementing. Use when the user asks for features tied to lobbying, legislator outreach, coalitions, making the case, or when aligning code with the Hardball book.
---

# Hardball Spec Reference

When planning features, fixing bugs, or writing code that touches **advocacy, lobbying, legislator outreach, coalitions, or “making the case”** for this project, treat the Hardball spec as source of truth and cite it explicitly.

## Quick steps

1. **Identify the topic** (e.g. framing the issue, 501(c)(3) limits, who to influence, volunteer coordination, post-campaign evaluation).
2. **Choose the chunk** using `docs/hardball-spec/README.md` (table: “When to use which chunk”). Open that file (and adjacent chunks if the topic spans sections).
3. **Read the relevant section** and extract the definitions, process, or constraints that apply.
4. **Propose or implement** in line with the spec. In your response or in code comments, **cite the chunk** and, when it affects behavior, a short quote or summary (e.g. “Per docs/hardball-spec/08-ch7-influencing-process.md: objective must be specific and focused…”).

## Chunk → topic quick map

| Need to know… | Chunk |
|---------------|--------|
| Why lobbying is core; 501c(4), PAC, 527 | 00-foreword.md |
| Advocacy vs lobbying; coalitions; hardball vs softball | 02-ch1-framing-context.md |
| How decisions get made; money; pay-to-play; who to influence | 04-ch3-decision-making.md |
| 501(c)(3), 501(h), 501(c)(4), PAC, 527; reporting | 05-ch4-advocacy-law.md |
| Building foundation; structure; strategic plan | 06-ch5-advocacy-foundation.md |
| Roles (ED, development, comms, volunteers, media, research) | 07-ch6-managing-lobbying.md |
| Objectives, strategy, framing, timeline, making the case, research | 08-ch7-influencing-process.md |
| Post-campaign evaluation; dealing with defeat | 09-ch8-postmortem.md |

## Output

- When a design or code decision comes from the spec, say so and point to the chunk (and optionally a brief quote).
- If the spec is ambiguous or silent, say that and propose a reasonable interpretation rather than inventing behavior that might contradict the book.
