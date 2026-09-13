# SBE campaign disclosure fixture

Real Illinois State Board of Elections extract used by unit tests.

**Source:** official bulk files at `https://downloads.elections.il.gov/`
(same files linked from
[Download CD Data Files](https://elections.il.gov/CampaignDisclosure/DownloadCDDataFiles.aspx)).

**Captured:** 2026-09-13. Tab-delimited, official headers.

| File | What it is |
|------|------------|
| `Candidates.txt` | Legislative candidate rows for current ILGA members in `mocks/dev` (plus linked office history). |
| `Committees.txt` | Those candidates' **Active Candidate** committees. |
| `CmteCandidateLinks.txt` | Committee ↔ candidate links. |
| `Receipts.txt` | Non-archived receipts from 2025–2026 for those committees (capped at 10 per committee). |

This is a **documented sample subset** of official SBE files, not invented money data.
Full ingest downloads the current files and keeps a recent window (default 2025-01-01+).
