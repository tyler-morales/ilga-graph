# Hardball Spec — Chunked Reference

**Source of truth:** *Hardball Lobbying for Nonprofits* (Barry Hessenius). The full text lives at `docs/canonical/books/hardball.txt`. This folder holds the same content split into **bite-sized chunks** so you (and the AI) can look up specific topics without reading the whole book.

## When to use which chunk

| Chunk | Use when you need… |
|-------|---------------------|
| **00-foreword.md** | Big picture: why this book exists, 501c(4)/PAC/527, lobbying as core management function. |
| **01-introduction.md** | Rationale for nonprofit lobbying, democracy, public benefit, balancing private-sector influence. |
| **02-ch1-framing-context.md** | Advocacy vs lobbying, why nonprofits don’t lobby, “softball” vs “hardball,” coalitions. |
| **03-ch2-new-paradigm.md** | New paradigm, “in-your-face” advocacy, political threat, operational status of parties/campaigns. |
| **04-ch3-decision-making.md** | How decisions get made: money, pay-to-play, insider game, life of elected officials, who to influence. |
| **05-ch4-advocacy-law.md** | Legal framework: 501(c)(3), 501(h), 501(c)(4), PACs, 527s; what’s allowed and reporting. |
| **06-ch5-advocacy-foundation.md** | Building the foundation: structure, coalitions, strategic plan, buy-in, funding the effort. |
| **07-ch6-managing-lobbying.md** | Running the operation: roles (ED, development, communications, volunteer coord, media, research), hierarchy, staffing. |
| **08-ch7-influencing-process.md** | Tactics: setting objectives, strategy, framing the issue, timeline, making the case, research, contacting officials. |
| **09-ch8-postmortem.md** | After a campaign: evaluation, dealing with defeat, surveys, focus groups, improving for next time. |
| **10-bibliography.md** | Further reading and key URLs (Urban Institute, Alliance for Justice, CLPI, opensecrets, etc.). |

## For the AI

When planning features, fixing bugs, or writing code that touches advocacy, lobbying, legislator outreach, or coalitions:

1. **Treat this spec as source of truth.** Prefer definitions and processes from these chunks over generic assumptions.
2. **Cite the spec.** When a decision comes from Hardball, name the chunk and (if helpful) a short quote or summary.
3. **Use the right chunk.** Open the chunk that matches the topic (e.g. “how do we frame the issue?” → `08-ch7-influencing-process.md`; “what can a 501(c)(3) do?” → `05-ch4-advocacy-law.md`).

## Regenerating chunks

If `docs/canonical/books/hardball.txt` is updated (e.g. re-exported from PDF), regenerate the chunks:

```bash
python3 scripts/split_hardball_spec.py
```

Line ranges are defined in `scripts/split_hardball_spec.py`.
