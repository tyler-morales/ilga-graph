# App overview

The ILGA Graph app exposes Illinois General Assembly data and ML-backed analytics through a web UI and a GraphQL API. The deployed site is branded as **The Land of Kei** (advocacy group / nonprofit); the name appears in the browser title, footer, share cards, and verification emails.

---

## Main areas

| Area | Path | Description |
|------|------|-------------|
| **Advocacy** | `/advocacy` | Landing is built for high conversion: threat-focused copy, one-field ZIP form, and social proof (ticker of calls to Springfield, trust badges). Hero image is configurable via `static/images/hero-kei.jpg` (fallback if missing). ZIP-based lookup: find your legislators and recommended targets (senator, rep, Power Broker). Each card has a single **Reach out** button that opens the advocacy drawer. The drawer has a **Call** and **Email** step strip (text links) so you can switch between the call script and the email template at any time; the recommended path is **call first, then email** (follow-up). Opening "Reach out" defaults to the Call step, or to Email if you've already called that legislator. Call script and email drawer use the same prefilled templates; the email view supports **Follow-up** (after a call) or **First outreach**. **ZIP and constituent status** are shared—edit in either step and the other updates when you switch. In the email step, changing the **City or ZIP** mad lib to a different 5-digit ZIP updates the **I'm a constituent** checkbox automatically. |
| **Test mode** | `/advocacy/test` | Dev back door: jump straight to the call script or email drawer for any member without going through the full flow. See [Advocacy Test Mode](../user-guide/advocacy-test-mode.md). |
| **Power Map** | `/explore` | Interactive D3 force-directed graph of legislators (nodes by influence, party-colored, linked by co-sponsorship). ZIP and topic filters. |
| **Intelligence** | `/intelligence` | ML dashboard: bills to watch, power movers, coalition landscape, anomaly alerts. Deep dives: `/intelligence/member/{id}`, `/intelligence/bill/{id}`. Raw tables at `/intelligence/raw`. |
| **GraphQL** | `/graphql` | Playground and API for members, bills, committees, votes, witness slips, metrics glossary, search. |

---

## Data flow (high level)

1. **Scrape** — Scripts hit ilga.gov; results cached in `cache/` (members, bills, committees, votes, etc.).
2. **Load** — On startup the app loads cache (and optionally runs ML pipeline); in dev, can fall back to `mocks/dev/`.
3. **Serve** — FastAPI serves HTML (Jinja2) and GraphQL (Strawberry). No scraping on request.

---

## Key concepts

- **Moneyball score** — 0–100 composite ranking legislators (passage, pipeline, co-sponsor pull, cross-party, network, institutional). Used for Power Broker and leaderboards; see `metricsGlossary` in GraphQL.
- **Advocacy targets** — For a given ZIP: your senator, your rep, and a “Power Broker” (high influence senator). Optional policy filter by committee.
- **Advocacy copy** — In-app text in the call script, email drawer, no-answer drawer, and result hints is written in a conversational, human tone (e.g. "No public email — we'll guide you to get it on the call") so advocates feel guided, not instructed.
- **Loading facts** — When the ZIP search loading animation is enabled (feature flag), the "finding targets" state shows one random kei truck fact per search (e.g. "Kei means light in Japanese", "Suzuki Carry, Honda Acty, Daihatsu Hijet — classic kei trucks", 660cc engine cap, 25-year import rule) for light gameification while results load.

- **Share cards and analytics** — The base template sets Open Graph and Twitter Card meta so shared links show a title, description, and image. Canonical URL and default meta description come from `ILGA_APP_BASE_URL` and config; the site name (default **The Land of Kei**) is used for `og:site_name`, page title, and footer. The default description is cause-tailored for Kei vehicle registration advocacy (625 ILCS 5/3-401(c-1)). Optional [Umami](https://umami.is) analytics: set `ILGA_UMAMI_WEBSITE_ID` (e.g. in prod) to inject the tracking script; no cookies, 100k events/month on the free tier. For search engine discovery, the app serves `/sitemap.xml` and `/robots.txt` (both use `ILGA_APP_BASE_URL`).
- **Legal and trust** — The footer links to a [Privacy policy](/privacy) (what we collect: sign-in email, optional bug-report email, Umami analytics; how we use it; retention; cookies) and [Terms of use](/terms) (disclaimer, not the ILGA, acceptable use). These pages are at `/privacy` and `/terms`.
- **Outreach data** — When signed in, the app records each call, email, or no-answer in a local SQLite DB (`data/ilga.db`). Each event stores: **time/date** (`created_at`), **legislator** (`member_id`), **kind** (call / email / no_answer), **ZIP**, optional **outcome**, **notes**, **contact name** (person who picked up), **support score** (1–5: opposed → champion), and **constituent** (whether the advocate was a constituent). The call drawer asks "How interested did they seem?" in a **Quick poll** bubble: tap Opposed → Champion (red to gold pills). You then see how many other advocates reported for that office (or "You're the first to report!" if none yet); your vote is included in the count. On the advocacy cards, a single **Reach out** button opens the drawer; when you've both called and emailed a legislator, it shows **Reached out** (muted) so you can see your progress at a glance. **Email outreach** is recorded only when you click **I sent** in the email drawer (after using **Open in email client** and actually sending); verifying your From address does not record outreach. The hero **Sign in** and the email drawer **From** field use the same magic-link flow and UI copy (Send code → 6-digit code → Confirm; same loading states and error messages) for consistency. If you are not signed in, clicking **I sent** does not save; the button shows **Sign in to save this outreach** and the drawer scrolls to the From/verify step so you can sign in and have your next action recorded. See `GET /outreach/my-history` and `POST /outreach/record`. Run `make seed-outreach` to prefill from a seed table.
- **Obsidian vault** — Optional export: `ILGA_Graph_Vault/` with markdown notes and Bases views (generated by exporter, not by the web app by default).

For full architecture and module breakdown, see the main **README** in the repo root.
