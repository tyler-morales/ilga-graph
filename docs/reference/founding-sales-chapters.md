# Founding Sales Reference: Chapter Breakdown and Advocacy Use

Use this doc when drafting sales or advocacy copy, scripts, or flows. **Source of truth:** [reference/founding-sales.txt](../../reference/founding-sales.txt) (~4,035 lines).

**Rule:** When using Founding Sales for advocacy or sales copy, cite chapter and (optionally) line range from `reference/founding-sales.txt`.

---

## 1. What the file contains (from TOC)

**Founding Sales** (Peter Kazanjy, 2020) is a tactical guide for founders and first-time sellers in B2B direct sales (especially B2B SaaS). It is structured in two stages:

- **Part I — Experimentation mode** (pre–product-market fit): evangelical sales, doing the work yourself, “doesn’t scale” activities, tight feedback loops.
- **Part II — Scaling mode** (post–product-market fit): hiring, onboarding, management, repeatable process.

### Table of contents summary

| Section          | Content                                                                                                                                                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Introduction** | Why the book exists; “two stages” (figuring it out vs scaling); who it’s for (founders/first-time sellers, B2B direct sales).                                                                                                     |
| **Ch 1**         | Mindset: plenty not scarcity, activity over perfection, be direct, many shallow relationships, assume the sale, expect to win/unfazed by rejection, record everything, be expert, glass-house transparency, sales is math.        |
| **Ch 2**         | Narrative and product marketing: problem–solution–specifics framing; who has the problem; costs; current solutions and why they fail; what changed; how the new solution works; qualitative/quantitative proof; pricing.          |
| **Ch 3**         | Sales materials: deck structure (problem, cost, alternatives, what changed, solution, proof, pricing, appendix); outreach email templates; phone/voicemail scripts; demo scripts; video (MVP overview, explainer); customization. |
| **Ch 4**         | Early prospecting: ideal customer profile (ICP); who has the pain; account sourcing; point-of-contact discovery (who will care).                                                                                                  |
| **Ch 5**         | Prospect outreach and demos: stages of the cycle; email; calling; cadence; referral; inbound capture.                                                                                                                             |
| **Ch 6**         | Inbound lead capture and response: qualification, forms, response, lightweight discovery, follow-up.                                                                                                                              |
| **Ch 7**         | Pitching: goal of pitch; persuasion formula; prep; materials; intro; presentation/demo/ask; objections; follow-up; practice.                                                                                                      |
| **Ch 8**         | Down-funnel: negotiation, closing, pipeline management.                                                                                                                                                                           |
| **Ch 9**         | Customer success: implementation, ongoing success, renewals; support; proactive monitoring; QBRs; learning from CS.                                                                                                               |
| **Ch 10**        | Early sales management and scaling: anti-patterns; abstraction and roles; manager role.                                                                                                                                           |
| **Ch 11**        | Sales hiring: when to scale; specialization; sources; screening and closing.                                                                                                                                                      |
| **Ch 12**        | Onboarding and training: why it matters; 101; tools and process; drilling and shadowing; ongoing learning.                                                                                                                        |
| **Ch 13**        | Conclusion; when to hire a sales leader; further reading.                                                                                                                                                                         |

---

## 2. Chapter line ranges (quick reference)

Use these to cite or re-read a specific chapter in `reference/founding-sales.txt`:

| Chapter            | Lines (approx.) | Title                                                       |
| ------------------ | --------------- | ----------------------------------------------------------- |
| Front matter / TOC | 1–138           | Praise, copyright, **Contents**                             |
| Introduction       | 139–262         | How a PM/PMktg guy became sales leader; who for; two stages |
| Part I header      | 263–268         | “PART I EXPERIMENTATION MODE”                               |
| Ch 1               | 238–345         | Mindset changes in first-time sales professionals           |
| Ch 2               | 348–564         | Baking your narrative and product marketing basics          |
| Ch 3               | 566–1160        | Sales materials basics                                      |
| Ch 4               | 1162–1392       | Early prospecting: finding your first customers             |
| Ch 5               | 1394–1665       | Prospect outreach and demo appointment setting              |
| Ch 6               | 1667–1758       | Early inbound lead capture and response                     |
| Ch 7               | 1760–2285       | Pitching: preparation, presentation, demos, objections      |
| Ch 8               | 2287–2449       | Down-funnel selling: negotiation, closing, pipeline         |
| Ch 9               | 2451–2885       | Customer success basics                                     |
| Part II header     | 2887–2895       | “PART II SCALING MODE”                                      |
| Ch 10              | 2897–3322       | Early sales management and scaling                          |
| Ch 11              | 3324–3707       | High-impact sales hiring                                    |
| Ch 12              | 3709–3979       | High-impact sales onboarding and training                   |
| Ch 13              | 3981–4021       | Where do you go from here? / Conclusion                     |
| Acknowledgments    | 4023–4035       | Acknowledgments                                             |

---

## 3. How we use this

### Sales-side (revenue / growth)

- **Narrative (Ch 2)** — Problem–solution–specifics and “who has the problem” for product description (e.g. campaign platform, value to orgs). Use for pitch docs, investor/lobbyist/candidate one-pagers, “why this exists” copy.
- **Materials (Ch 3)** — Deck structure (problem → cost → alternatives → what changed → solution → proof → pricing); “deck for presenting vs deck for sending.” Email templates and cadence for donor/partner outreach.
- **Prospecting (Ch 4)** — ICP and “who has the pain” for targeting orgs and roles (ED, comms, volunteer coordinators). Point-of-contact discovery = who will care (e.g. staff vs member).
- **Outreach and cadence (Ch 5)** — Multi-touch email + call, cadence, “click targets” (links to collateral) for outbound to partners, donors, pilot customers.
- **Pitching and objections (Ch 7)** — Pre-call planning, objection handling, “ask for the sale” for sales and partnership meetings.
- **Pipeline and close (Ch 8)** — Pipeline stages and close/negotiation if we add a formal sales or partnership pipeline.

### Advocacy-side (constituent / legislator outreach)

- **Narrative (Ch 2)** — **Direct fit.** Problem (e.g. Kei registration fix), who has it (constituents, legislators, staff), costs of inaction, current “solutions” and gaps, what changed (e.g. other states), how our fix works, proof. Use when refining legislator brief, the-issue narrative, and call script structure. Aligns with Hardball Ch7 “making the case” and framing. See [docs/canonical/README.md](../canonical/README.md).
- **Materials (Ch 3)**  
  - **One-pager / brief as “deck.”** Problem → cost → existing solutions and gaps → what changed → our solution → proof → “ask.” Audit legislator and constituent briefs against this; add missing beats (e.g. “what changed” / “why now”).  
  - **Call script as “demo script.”** Customization (district, member name, committee, contact count), clear sections (intro → problem → legal why → ask → easy yes close), pushbacks = objection handling. Reference when editing `src/ilga_graph/advocacy_helpers.py` and `src/ilga_graph/templates/_advocacy_drawer_call.html`.  
  - **Email templates.** Short/long variants, clear CTA, “click targets” (e.g. one-pager link), personalization (district, name). Matches email drawer and `src/ilga_graph/email_utils.py`.
- **Prospecting (Ch 4)** — Ideal “customer” = legislator or staff. “Who has the problem?” = who can move the bill. “Who will be excited?” = point-of-contact: district, committee, power, broker status. Use for targeting (legislator cards, “call your rep” vs “call these key members”) and prioritization in docs.
- **Outreach and cadence (Ch 5)** — Call + email together, cadence, “voicemail as audio email.” Aligns with call + email drawer and future multi-touch flows (e.g. reminder to call again).
- **Pitching (Ch 7)** — **Call flow = pitch.** Intro (who I am, why I’m calling), discovery (“Ever heard of Kei vehicles?”), problem, proof, ask, objection handling (“We need more info” → three points), easy yes close. Use when adding or refining script sections and pushback responses in `src/ilga_graph/advocacy_helpers.py`.
- **Mindset (Ch 1)** — **Volunteer/constituent side.** “Plenty not scarcity” and “activity above all” for many touches; “be direct” and “assume the sale” for a clear, confident ask (“be aware,” “review the one-pager”); “record everything” for outreach events and funnel tracking. See `src/ilga_graph/outreach_steps.py`, [docs/development/db-and-outreach.md](../development/db-and-outreach.md).
- **Customer success (Ch 9)** — **Constituent as “customer.”** Implementation = first successful call/email (script, one-pager). “Rediscovery” = why they care (Kei poll, personalize). Follow-up and “no summer break” = reminders. QBR-style value recap → “here’s what your outreach did” for signed-in users (e.g. “You’ve contacted N offices; here’s the bill status”).

---

## 4. Narrative audit checklist (Ch 2)

When editing the legislator brief, The Issue, or call script, verify the flow matches Ch 2 (lines 348–564):

1. **Problem** — What is the business/constituent pain? (Issue in one sentence.)
2. **Who has it** — Constituents, legislators, staff.
3. **Cost** — Impact of inaction (denials, revocations, cost to residents).
4. **Alternatives / gaps** — Current “solution” (SOS interpretation) and why it fails.
5. **What changed** — Why now (other states, federal pathway, trend).
6. **Solution** — Narrow statutory fix / our ask.
7. **Proof** — Other states, one-pager, state table.
8. **Ask** — Clear CTA (be aware, review one-pager, endorse, sponsor).

Reference this checklist for future brief/the_issue edits.

---

## 5. Materials checklist (Ch 3)

- **Brief (deck for sending):** Problem → cost → alternatives/gaps → what changed → solution → proof → ask. Ensure one-pager/summary link in sidebar (Documents). See line ranges 596–798 in reference/founding-sales.txt.
- **Call script (pitch flow):** Intro → discovery (Kei explainer) → problem → legal why → ask → easy yes close; pushbacks = objection handling. `advocacy_helpers.py`, `_advocacy_drawer_call.html`.
- **Email:** One clear **click target** (e.g. “Download one-pager (PDF)”) so staff can act immediately; subject/body personalization (district, name). `_advocacy_drawer_email.html`, `email_utils.py`.

---

## 6. Advocacy targeting (Ch 4)

- **Ideal “customer” (ICP):** Legislator or staff who can move the bill.
- **Who has the problem:** Offices that can advance a statutory fix (district rep/senator, committee chair, high-influence broker).
- **Who will be excited (point-of-contact):** District = constituent leverage; committee = gatekeeper; Power Broker = influence. Use for legislator cards, “call your rep” vs “call these key members,” and prioritization in docs.

---

## 7. Outreach and cadence (Ch 5)

- **Target (Ch 5, lines 1394–1665):** Call + email together; multi-touch cadence (email → call → voicemail → follow-up); “selling the conversation” (get the read / get the meeting).
- **Current implementation:** Single touch — one call and/or one email per drawer session. No multi-touch cadence yet.
- **Future:** Optional “remind to follow up” or “Mark as done / Schedule follow-up” for power users. Document in product backlog when adding.

---

## 8. Objection handling (Ch 7)

Map Ch 7 objection types (lines 1760–2285) to call script pushbacks:

| They say | You say (script / drawer) |
|----------|---------------------------|
| “We need more info.” | “No problem — I’ll email a one-pager with the statute and which states have already fixed their Kei bans.” |
| “Just send me something.” | “I’ll send the one-pager to your email — what’s the best address? It has the statute, the fix, and which states have already done this.” |
| “Call back later.” | “Sure — I’ll follow up with an email now so you have the one-pager, and I can call again next week. What’s the best email?” |
| “They’re not highway vehicles…” | “I get that — that’s why we’re asking for a state clarification with restrictions. Federal import rules are one thing; we’re just asking Illinois to set clear rules for registration.” |
| “We can’t commit.” | “That’s totally fine — could you point me to the right staffer for transportation or vehicle policy, and the best email to send the one-pager?” |

Keep copy consistent with canonical content (no new policy claims).

---

## 9. Mindset (Ch 1) and value recap (Ch 9)

- **Ch 1 (lines 238–345):** Activity above all, be direct, assume the sale. Volunteer-facing copy: hero/CTA should reinforce “your call/email matters” or “quick action” (one line). No new UI components.
- **Ch 9 (lines 2451–2885):** Value recap for constituent = “here’s what your outreach did.” **Future enhancement:** For signed-in users with an account/summary view, add “You’ve contacted N office(s)” and/or “Bill status: …” from campaign/bill data. Document in TODOS when building.

---

## 10. Summary

- **File:** ~4,035 lines; B2B direct sales playbook in two stages (experimentation → scaling).
- **Chapters:** 13 + intro and two part headers; use the line ranges above to jump in the .txt.
- **Advocacy (highest leverage):** Ch 2 (narrative), Ch 3 (materials + script), Ch 4 (targeting), Ch 5 (cadence), Ch 7 (pitch/objections), Ch 1 (mindset), Ch 9 (follow-up/value recap).
- **Sales:** Same chapters support pitch decks, partner/donor outreach, and pipeline when we add that motion.
