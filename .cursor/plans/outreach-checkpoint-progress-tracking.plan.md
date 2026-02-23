# Outreach checkpoint and progress tracking (updated)

## Plan updates (from user feedback)

### Call (answered path) — checkpoint changes

- **Add: phone number clicked** — User taps/clicks the phone number (tel: link) to initiate the call. This is a strong intent signal and should be a checkpoint (e.g. `phone_clicked`).
- **Add: optional “checkmarks” for call form inputs** — Track when the user captures:
  - **Staffer name** — when they fill and/or “save” the staffer/contact name (e.g. `staffer_name_captured`).
  - **Office email** — when they fill and/or “save” the office email in the call script (e.g. `office_email_captured`).

These are optional in the sense that not every user will fill them; the funnel can still show how many reached each step. Implementation: emit step events when the name/email inputs lose focus with non-empty value, or when the existing “saved” pill/state is shown (if you have an explicit save), whichever matches current UX.

### Email — steps unchanged; progression UX “cleaner and smoother”

- **Keep** the 6 steps (Sign in, Subject, Fill details, Grab PDF, Send, Confirm).
- **Improve** the way the UI guides the user through steps: the current **blue (and green) focus** treatment feels heavy. Make the process cleaner and smoother.

**Current behavior (what to change):**

- [advocacy-email.css](src/ilga_graph/static/css/advocacy-email.css): `.guide-pulse` / `.guide-pulse-green` apply an **animated 2s infinite outline** (blue/green) to the current step’s container.
- [index.html](src/ilga_graph/templates/index.html) `EmailGuide.setStep()`: adds those classes, calls `scrollIntoView`, and in several places **programmatically focuses** the next input (From, subject button, mad lib blanks, PDF link, etc.), which adds the browser’s default focus ring on top of the guide pulse.

**Directions for a cleaner, smoother experience:**

1. **Softer “you are here” indicator**  
   Replace or supplement the strong pulsing outline with one or more of:
   - A **one-time** gentle pulse or fade-in when the step changes, then a **static** subtle indicator (e.g. light border or soft background tint) so the eye isn’t constantly pulled by animation.
   - A **background highlight** (e.g. light tint) instead of a thick outline, so the step area is clear without a bright frame.
   - Rely more on the **step counter + dots** (“Step 2 of 6” and active/done dots) as the primary “where you are” signal, and use a very subtle container treatment (e.g. 1px neutral border or faint background) instead of a bold blue/green ring.

2. **Reduce aggressive programmatic focus**  
   Where it’s not strictly necessary for a11y (e.g. after sign-in we may still want to focus From once), consider **not** moving focus on every step change. That way the blue focus ring doesn’t keep jumping; the user can Tab through at their own pace. Keep **keyboard flow** working (Tab order, focus trap in drawer) and use **focus-visible** for a consistent, visible-but-not-jarring ring when the user actually tabs.

3. **Unify focus styling**  
   Use a single, **subtle** focus style (e.g. `focus-visible` with a thin outline or soft box-shadow in a neutral or brand-muted color) so that when we do focus an element, it doesn’t feel like a second, competing “highlight” on top of the guide.

4. **Optional: prefers-reduced-motion**  
   If the guide ever keeps a light animation, respect `prefers-reduced-motion: reduce` (no pulse, instant step change).

Implementation can live in [advocacy-email.css](src/ilga_graph/static/css/advocacy-email.css) (guide-pulse, step container, focus-visible) and in [index.html](src/ilga_graph/templates/index.html) (where `setStep` adds classes and calls `.focus()` — trim or conditionalize focus calls where safe).

---

## Revised call (answered) checkpoints

| Order | Step slug | Description |
|-------|-----------|-------------|
| 1 | `drawer_opened` | User opened the call drawer for this member |
| 2 | `phone_clicked` | User clicked the phone number (tel: link) to start the call |
| 3 | `staffer_name_captured` | (Optional) User filled/saved the staffer/contact name |
| 4 | `office_email_captured` | (Optional) User filled/saved the office email |
| 5 | `end_call_clicked` | User clicked "End call" (script path) |
| 6 | `interest_selected` | User selected interest level (1–5) in the poll |
| 7 | `call_recorded` | Server recorded the call (POST /outreach/record kind=call) |
| 8 | `wrapup_draft_clicked` | User clicked "Draft your follow-up email" (optional) |
| (alt) | `wrapup_skipped` | User clicked "I'll do it later" (optional) |

Call (no-answer) and Email step slugs stay as in the original plan; email steps are unchanged, with the UX improvements above applied to how the guide and focus behave.

---

## Original plan summary (for reference)

- **OutreachEvent** stays as the record of completed actions (call/email/no_answer).
- New **`outreach_step_events`** table: `user_id`, `member_id`, `outreach_type`, `step_slug`, `reached_at` (and optional unique constraint for idempotency).
- Step definitions in code (e.g. `outreach_steps.py`) with allowed slugs per type; `POST /outreach/step` to record a step (validates member_id + step_slug).
- Server emits `call_recorded` / `email_recorded` / `no_answer_recorded` from POST /outreach/record; client (and server where applicable) emits the rest (drawer_opened, phone_clicked, staffer_name_captured, office_email_captured, end_call_clicked, interest_selected, wrapup_*, etc.).
- Email: same 6 steps; implementation work is on **smoother, cleaner** step progression and focus (see above).
