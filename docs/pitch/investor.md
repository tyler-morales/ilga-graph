# Lobbying operations platform: constituent contact at scale, with data that proves it

**For:** Angels, impact investors, and institutional investors evaluating a civic/advocacy tech opportunity.

---

## The problem

Advocacy and lobbying depend on constituent contact and message discipline. Most tools are generic—petitions, blast email—and don’t target the right members or produce credible conversion and tracking. Coalitions, lobbyists, and campaigns need a system that does three things: **find** the right legislators for every supporter, **guide** them through one call and one email with one message, and **record** what happened. Today that’s ad hoc. We built the platform that does it.

## Why this works

- **Find.** One ZIP → district senator, representative, and one high-leverage target (the **Power Broker**), derived from legislative data and ML-style analytics (Moneyball, influence). No manual lookup; no wrong targets.
- **Guide.** Guided call/email drawer: script and template from the coalition’s one-pager. Same message, right targets. Call-first-then-email flow; optional “how interested did they seem?” for office-level signal.
- **Record.** Outreach and funnel metrics: drawer opens, calls, emails, conversion (e.g. drawer-to-outreach). Success metrics align with what coalitions control: constituent contacts, co-sponsors, witness slips, coalition readiness. Methodology from *Hardball Lobbying for Nonprofits*—category alignment with how lobbying actually works.

Built on official legislative data (Illinois General Assembly) and a full pipeline: scrapers, ETL, analytics, GraphQL API. Not a thin wrapper; real technical depth.

## What you get (product)

- **Web app:** Advocacy flow (ZIP → targets → drawer), legislator brief page, The Issue page, campaign updates (Major/Minor) and email to subscribers, admin dashboard (overview, send updates, users, outreach stats). Optional Intelligence (bills, members, coalitions) and Power Map (influence graph).
- **GraphQL API:** Members, bills, committees, votes, witness slips, search, metrics glossary—for integration or analytics.
- **Outreach and conversion metrics:** Stored per user (signed-in) or per session; aggregated for funnel and conversion reporting.

## Traction and scalability

- **Proof of concept:** Illinois, one issue (Kei vehicle registration), deployed as The Land of Kei. Real users, real outreach, real funnel metrics.
- **Expansion:** Same stack, new state data and content. Add other state legislatures (scrape/ETL pattern already proven); multiple issues and clients (coalitions, lobbyists, campaigns). Platform is issue-agnostic; swap copy and one-pager.
- **Revenue:** B2B2C. Advocacy groups, lobbyists, and campaigns pay (subscriptions or campaign fees). Their constituents use the product free. Optional: analytics/Intelligence upsell, multi-tenant or white-label for firms or large coalitions.

## Proof

- **Real deployment.** The Land of Kei is live; methodology and metrics are in use.
- **Real methodology.** *Hardball* is the standard for nonprofit lobbying; we built to its playbook (strategic message, 3–5 conclusions, constituent contact, measurable success).
- **Real metrics.** Contacts, co-sponsors, witness slips—not vanity metrics. Technical depth: scrapers, ETL, ML pipeline, GraphQL, admin and outreach persistence.

We don’t claim we pass bills. We claim we turn supporters into disciplined constituent contact and give coalitions the numbers to prove it. That’s the category we’re building.

## Next step

**See the product and the funnel metrics.** Demo the advocacy flow and admin/outreach stats. Discuss expansion path: states, issues, and revenue model.
