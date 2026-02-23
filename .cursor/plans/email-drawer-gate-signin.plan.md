# Email drawer: gate "I sent it!" on sign-in (Option 1)

No localStorage. Anonymous users can use the drawer and mailto; "I sent it!" requires sign-in. When an anonymous user clicks "I sent it!", direct them to sign in above (scroll to From, show nudge) instead of calling the API or saving anywhere.

## Current behavior

- **[index.html](src/ilga_graph/templates/index.html)**  
  - `recordOutreach()` returns `Promise.resolve()` when `!_ilgaUserEmail` (line 1278), so no request is sent.  
  - The "I sent it!" handler (lines 1882–1916) calls `recordOutreach(memberId, 'email', zip).then(...)`. For anon, the promise resolves with `undefined`, so the handler hits the `else` branch and shows "Couldn't save. Try again." (or similar); it never shows the auth nudge or moves the guide to Step 0.  
- The email drawer already has **Step 0 = "Sign in"** and a From field with Send code / Confirm ([_advocacy_drawer_email.html](src/ilga_graph/templates/_advocacy_drawer_email.html)). The guide’s step for sign-in targets `#gmail-from-row` (index.html ~1514).

## Target behavior

1. **Anonymous**  
   - User can open the drawer, fill subject/body, click "Open in your email app", and send from their client.  
   - When they click **"I sent it!"**:  
     - Do **not** call `recordOutreach` (no API call).  
     - Show the same sign-in nudge as today: button text "Sign in to save this outreach", class `sent-signin-nudge`, revert after 3s.  
     - Call `guide.setStep(0)` so the step indicator shows "Sign in".  
     - Scroll `#gmail-from-row` into view (and optionally focus the From input `#gmail-from-input`) so the user is directed to sign in above.  
   - No persistence and no localStorage.

2. **Authenticated**  
   - Unchanged: `recordOutreach` runs, POST to `/outreach/record`, then `setOutreachDone`, `refreshAuthStripProgress`, `refreshAdvocacyResults`. On success, drawer shows "Sent!" and guide completes.

## Implementation steps

1. **"I sent it!" click handler**  
   In `initDrawerEmailActions` (index.html), in the `sentBtn.onclick` handler:  
   - If `!_ilgaUserEmail`:  
     - Do not call `recordOutreach`.  
     - Set button span text to "Sign in to save this outreach", add `sent-signin-nudge`, call `guide.setStep(0)`.  
     - Scroll the From row into view: get the row with `wrap.querySelector('#gmail-from-row')` and call `scrollIntoView({ behavior: 'smooth', block: 'nearest' })`. Optionally focus `#gmail-from-input` if the sign-in state is the input (not code/verified).  
     - After 3s: remove `sent-signin-nudge`, restore span text to "I sent it!".  
     - Return (no `.then` on recordOutreach).  
   - Else (signed in): keep current logic — call `recordOutreach(memberId, 'email', zip).then(...)` and handle ok / auth error / other error as today.  
   - Optional: when signed in, pass `constituent` from the drawer’s `data-drawer-constituent` into `recordOutreach` so the API can store it (outreach.py already accepts it); this is a small improvement, not required for Option 1.

2. **recordOutreach**  
   No change required. It already returns early for anon; the only change is the click handler branching on `_ilgaUserEmail` before calling it.

3. **Docs / TODOS**  
   - In [TODOS.md](TODOS.md): add a completed or in-progress line for "Email drawer: gate I sent it on sign-in (no localStorage); anon gets sign-in nudge + scroll to From."  
   - If you have an app-overview or user-facing doc that describes anonymous vs signed-in behavior for the email drawer, note that "I sent it!" only saves when signed in and that anon users are directed to sign in in the drawer.

## Files to touch

| File | Change |
|------|--------|
| [src/ilga_graph/templates/index.html](src/ilga_graph/templates/index.html) | In the "I sent it!" handler inside `initDrawerEmailActions`: if `!_ilgaUserEmail`, run the sign-in nudge + `guide.setStep(0)` + scroll to `#gmail-from-row` (and optional focus); else keep existing `recordOutreach(...).then(...)`. |
| [TODOS.md](TODOS.md) | Log the feature. |
| Docs (optional) | Short note in app-overview or similar that anon can use mailto but must sign in to save. |

## Edge cases

- **User signs in after nudge**: They can click "I sent it!" again; this time `recordOutreach` runs and saves. No need to re-send the email.  
- **From row not in DOM**: If the drawer body was swapped and `#gmail-from-row` is missing, `scrollIntoView` is a no-op; the nudge and step change still apply.  
- **Accessibility**: Ensure the sign-in nudge text is visible to screen readers (it’s the button’s text content). After scrolling, focus the From input when in input state so keyboard users land in the right place.
