# TODOS

**State of the system**

- Modularity roadmap 1–5 done: ETL in `etl.py`, `ILGA_LOAD_ONLY=1`, scorecards/Moneyball cached, GraphQL batch loaders, `ILGA_PROFILE=dev|prod`
- SSR advocacy at `/advocacy`; Moneyball v2 (shell bill filter, institutional bonus, chair-first Power Broker)
- GraphQL `search` query (members, bills, committees); Committee Power Dashboard; Institutional Power Badges; Power Card redesign
- True Influence Engine (`influence.py`); prediction table v5; ML pipeline bug audit; Intelligence story redesign; Legislative Power Map at `/explore`

---

## Refactor (current & backlog)

| Date       | Area                    | Summary |
|------------|-------------------------|--------|
| 2026-02-18 | main.py split           | app_state, constants, startup_banner, date_parse, member_lookup; routers (advocacy 02-19, intelligence + bills + explore 02-20); base.html CSS → static/css; Cursor skills extract-route-group, graphql-resolver-and-loaders. main.py ~4780 → ~1800 lines. |
| 2026-02-18 | Advocacy helpers        | All advocacy logic in `advocacy_helpers.py`; state as first arg; member lookup in `member_lookup.py`, re-exported. |
| 2026-02-19 | Unified outreach drawer | Single "Reach out" button; step strip (Call · Email); drawer close-then-open; files: _results_partial, index, base. |
| 2026-02-19 | UI consistency          | Member card grid fix, error display single place, drawer naming, recommendation chips from Python, legislator_drawer_context. |
| 2026-02-19 | Snapshot mocks          | `make snapshot-mocks` samples cache JSON types from mocks; zip_to_district subset for 40 mock members. |
| 2026-02-20 | Refactor completed      | CSS → static/css (variables, base, advocacy×4, intelligence×2); intelligence → routers/intelligence.py; explore → routers/explore.py; SHAP → routers/bills.py. |
| 2026-02-18/20 | Legacy purge        | Removed intelligence/explore from main, inline CSS; POST /advocacy/drawer/after-call; dead ML fulltext; test mode/deep link; unused CSS. |
| 2026-02-21 | main.py refactor        | Lifespan → startup.py; site routes (/, /advocacy, favicon, sitemap, robots) → routers/site.py; /logs, /health, /api/dev/members → routers/admin.py; GraphQL Query + ML types → graphql_query.py; CORS/API key/CSRF/security/request-logging → middleware.py. main.py ~1921 → ~277 lines. Tests pass; _member_career_start re-exported from main for test compat. |

---

## Current

| Date       | Area                         | Summary |
|------------|------------------------------|--------|
| 2026-02-20 | Error experience             | Custom 404/500 (and 422) HTML pages (same chrome as site); global exception handlers in main.py for HTTPException, RequestValidationError, and uncaught Exception; _wants_html(request) for HTML vs JSON; catch-all route for unmatched paths; server-side logging for 500s (no stack trace in response). Templates: 404.html, 500.html, 422.html; base.css .error-page. |
| 2026-02-21 | 404 Kei vehicle facts        | 404 page no longer shows Home/Advocacy links; shows a random Kei vehicle fun fact (KEI_VEHICLE_FACTS in main.py), hint to refresh for another; _404_context(request), .error-page-fact, .error-page-hint in base.css. |
| 2026-02-21 | 404 fun fact images          | Facts are dicts with text + optional image, image_alt, image_credit; 4 facts have CC-licensed Wikimedia Commons photos (N-BOX, N-BOX Custom, Hijet truck, Jimny). 404.html shows figure + img + figcaption when present; base.css .error-page-fact-figure, .error-page-fact-img, .error-page-fact-credit, .error-page-fact-text. |
| 2026-02-21 | Branch sync & PR #21         | Merged main into feature/refactor-data-source-playground-2026-02-21 (PR #20: CSRF reuse, async Turnstile, explore/intelligence template globals, rate-limiter deque). No conflicts. PR #21 title/body updated to describe branch changes and pulled-from-main fixes. |
| 2026-02-21 | Dev component playground     | Dev-only GET /dev/playground (and /dev/playground/{scene_id}) to isolate UI components (truck animation, drawer call/email) without full flows. Scene registry in dev_playground_scenes.py; truck partial _report_bug_truck.html; drawer wrappers with mock context. Docs: development/component-playground.md. |
| 2026-02-20 | Performance & resilience     | Static assets: StaticFilesWithCache in main.py sets Cache-Control: public, max-age=3600 for /static. Offline/failure: connection-error banner in base.html (role=alert, aria-live=polite) shown on htmx:responseError and htmx:sendError; dismissible. Offline indicator shown when navigator.onLine is false, hidden on online. window.showConnectionError() for optional fetch use. base.css .connection-error-banner, .offline-indicator. Docs: reference/performance-resilience.md. |
| 2026-02-20 | Truck animation → bug report | Removed all ZIP/member-cards loading behavior; HTMX direct swap. Bug report: truck + status during submit via fetch; success page (?submitted=1) is green box only, no truck. Fix: server responds in &lt;20ms so redirect was at 600ms—animation never visible. Now 3.5s truck animation with dashed horizontal path; truck drives left → stops at center ("Picking up bug") → drives to end ("Sent!"); 🐛 at center fades when truck stops (picked up). Status steps: "Sending…" → "Picking up bug" (1.1s) → "Sent!" (2.8s). base.css report-bug-truck-path (dashed), report-bug-bug, report-bug-truck-drive 3-phase keyframes; report_bug.html statusSteps, setTimeout timing. Cleanup: responsive track (width 100%, max 280px, min 200px); --report-bug-truck-duration; prefers-reduced-motion (no movement, bug fades only); small-screen smaller emoji; JS TRUCK_DURATION_MS, clearStatusTimers before redirect. |
| 2026-02-20 | Beta banner (minimal)        | Solid #f0f0f0, max-width 1120px to match .container, no gradient/animation, minimal text + link + dismiss; responsive padding to align. base.css. |
| 2026-02-20 | Micro-interactions (confetti-like) | Pop/thunk on key moments: "I sent it!" (pop+shimmer), End call (thunk), card Called!/Emailed! (pop), reminder Copy/Calendar (pop). **Start My Outreach** CTA: canvas-confetti on click (origin at button), amber/gold palette; confetti script in base.html head; prefers-reduced-motion skips confetti. Report-bug success has no animation. base.css keyframes + utilities; advocacy-email, advocacy-form, index.html. prefers-reduced-motion respected. |
| 2026-02-20 | Beta banner + in-app bug report | Site-wide dismissible bar; “Report a bug” → in-app form at /report-bug (no GitHub/service). Form: description, optional email, stored in bug_reports; optional email to BETA_BANNER_EMAIL if SMTP set. Email body: timestamp, email, issue, page, screenshot (inline + link or “No image sent”), IP/User-Agent. Override with FEEDBACK_URL or EMAIL for external/mailto. |
| 2026-02-20 | Bug nudge in advocacy drawer | No longer in drawer header. Call: "Something went wrong? Report a bug" at end of script (after End call) and in voicemail (_advocacy_drawer_call). Email: same copy under action bar (Open in your email app / I sent it) (_advocacy_drawer_email). No-answer drawer same copy. Gated on show_beta_banner and beta_banner_feedback_url. advocacy-drawer.css .drawer-bug-nudge, .gmail-bug-nudge. |
| 2026-02-20 | Form submission security     | CSRF (double-submit cookie XSRF-TOKEN + body), rate limiting (bug report, request-code, verify-code), page_url validation (http/https only). /report-bug exempt from API key. security.py; docs env vars + Security section. |
| 2026-02-20 | Turnstile CAPTCHA (optional) | Cloudflare Turnstile on bug report form when ILGA_TURNSTILE_SITE_KEY + SECRET_KEY set. Free 1M/mo; server-side siteverify. feedback router + report_bug template. |
| 2026-02-20 | Report bug description min length | "What went wrong" requires ≥20 chars. Backend: BUG_REPORT_DESCRIPTION_MIN_LENGTH; redirect ?error=description_short. Template: minlength, inline error #description-error, aria-invalid/aria-describedby. Client-side validation before fetch; clear error on input when valid. base.css: .report-bug-field--error, .report-bug-field-error (red text, border, focus ring). |
| 2026-02-20 | Report bug email validation | Optional reporter email validated server-side (one @, dot in domain, length limits); redirect ?error=email_invalid. Template: inline error #email-error, aria-invalid/aria-describedby; client-side isValidEmail on input/blur and submit with graceful messages (e.g. "Please enter a valid email address or leave the field empty"). |
| 2026-02-20 | Branch cleanup               | Removed commented intel nav, consolidated tooltip init to `initAdvocacyTooltips`. |
| 2026-02-20 | Defect pass                  | test/lint pass; smoke GET /advocacy, POST search, drawer 404; power broker, Tippy verified. |
| 2026-02-20 | Recommendation chip tooltips | Tippy.js: one tooltip at a time, appendTo body, theme recommendation-chip; Popper 2 + Tippy 6. |
| 2026-02-20 | Power Broker logic           | Chair for topic (default Transportation) or highest Moneyball outside district; exclude_senate_district/exclude_house_district. |
| 2026-02-20 | Potential Ally removed       | 3 cards only: Your Senator, Your Rep, Power Broker; seating redo later. |
| 2026-02-20 | The Land of Kei branding     | SITE_NAME, META_DESCRIPTION, footer, auth email, .env.example, docs. Advocacy page title now uses ILGA_SITE_NAME ("{{ site_name }} — Find Your Targets"). |
| 2026-02-20 | Dev ZIP autofill             | ILGA_DEV_MODE=1 autofills hero ZIP 60601 when no ?zip=. |
| 2026-02-20 | Auth strip outreach progress | Signed-in users see "Called X legislators and sent Y emails" under verified email. GET /outreach/my-stats; refresh on sign-in and after recording call/email. index.html #auth-strip-progress, advocacy-form.css .auth-strip-progress. |
| 2026-02-20 | ZIP in URL on search         | history.replaceState ?zip=XXXXX on form submit; shareable links. |
| 2026-02-20 | Member cards enter animation | card-fade-slide-up stagger; prefers-reduced-motion respected. |
| 2026-02-21 | Hero unique outreach + outreach cleanup | Hero ticker: single number = total unique outreach (distinct user+member pairs, call/email). Copy "Join X+ Illinois residents who've already taken action." get_outreach_aggregate in outreach.py. One-time script scripts/clean_outreach_funky_mama_only.py. Seed by profile: prod=moratyle@gmail.com, dev=funky_mama11@gmail.com. Docs: db-and-outreach.md. |
| 2026-02-20 | Hero copy refresh             | Subhead: statutory gap + pre-written script in under a minute; ticker "Join X+ Illinois residents who've already taken action this week"; CTA "Start My Outreach". advocacy.py hero_subhead (2 places), index.html ticker + find-btn. |
| 2026-02-20 | Hero animation sequence       | Sharpie fades in first on load (0s); after --hero-highlight-delay (0.5s) the underline/highlighter reveal runs. advocacy-form.css: .advocacy-hero --hero-highlight-delay, .hero-headline-mark::after animation-delay. |
| 2026-02-20 | Hero "Fix" sharpie circle     | Sharpie PNG overlay around word "Fix" in hero line 1. advocacy.py hero_headline_line1_* vars; index.html .hero-headline-sharpie-wrap + img sharpie.png; advocacy-form.css .hero-headline-sharpie, scale/position fixed to word. |
| 2026-02-20 | Hero headline responsive size | Desktop unchanged (clamp 1.75rem–2.75rem). Tablet ≤768px: clamp(2.25rem, 6.5vw, 3.5rem). Phone ≤480px: clamp(2rem, 11vw, 3.25rem). advocacy-form.css .hero-headline in media queries — max-width-as-possible at small breakpoints. |
| 2026-02-20 | Hardball advocacy landing    | Threat headline, Anton/Impact, two-column hero, CTA "Start My Outreach", ticker, trust badges, hero alignment. |
| 2026-02-20 | Refactor round               | Intelligence/explore routers, CSS split; main ~1700 lines. |
| 2026-02-20 | Advocacy router cleanup      | Removed debug logging; E501/F401 fixed; lint + test pass. |
| 2026-02-20 | Drawer open/close animation  | Snappy slide: --drawer-duration 0.28s, --drawer-ease cubic-bezier(0.33,1,0.68,1); overlay + panel in sync; no fade-in on panel. prefers-reduced-motion: 0.01s. advocacy-drawer.css. |
| 2026-02-20 | Drawer close-then-open       | Open another drawer → close first, then open after 300ms (DRAWER_CLOSE_MS). |
| 2026-02-20 | Call drawer overflow         | overflow-x hidden, min-width 0, word-break on mobile. |
| 2026-02-20 | Mobile drawer URL/scroll     | 95dvh, overscroll-behavior contain, body scroll lock + restore. |
| 2026-02-20 | Advocacy mobile responsiveness | Media queries moved from base.css into advocacy-drawer/cards/form. |
| 2026-02-20 | Privacy and Terms + footer   | Privacy policy at /privacy and Terms of use at /terms; legal router (routers/legal.py); templates privacy.html, terms.html with .legal-page styling; footer in base.html updated with Privacy and Terms links (always shown); docs app-overview Legal and trust. |
| 2026-02-20 | Security hardening           | CSP (Content-Security-Policy-Report-Only by default; ILGA_CSP_ENFORCE=1 to enforce; ILGA_CSP_REPORT_URI optional). HSTS opt-in (ILGA_HSTS_ENABLED=1 when site is HTTPS). Prod startup warning if ILGA_APP_BASE_URL is not https://. Docs: deployment Security headers, env vars (CSP/HSTS), status-report HTTPS+base URL. |
| 2026-02-20 | Post-prod                    | OG/Twitter cards, canonical, Umami, advocacy SEO globals. |
| 2026-02-20 | SEO sitemap and robots       | GET /sitemap.xml (key pages: /, /advocacy, /intelligence, /explore) and GET /robots.txt (allow all, Sitemap line); both use APP_BASE_URL. Exempt from API key. Docs: environment-variables, deployment, app-overview. |
| 2026-02-20 | Automated deploy Vultr       | Push main → CI → SSH deploy; DEPLOY_HOST/USER/SSH_KEY. |
| 2026-02-19 | Deployment prep              | Lint, Procfile, status-report, deployment.md, vultr-deployment-guide, startup banner URLs. |
| 2026-02-20 | Typography normalization     | Removed fixed/inconsistent text sizes and styles that caused bugs. variables.css: added --line-height-tight. All static/css: font-size and line-height use design tokens (--font-size-base, --font-size-body, --font-size-label, --font-size-sm, --font-size-xs, --font-size-h1, --font-size-h2, --line-height-body, --line-height-tight). No more px line-heights or arbitrary em/rem; hero clamp() kept for responsive headlines. base, advocacy-cards, advocacy-drawer, advocacy-form, advocacy-email, intelligence-dashboard, intelligence-tables. |
| 2026-02-20 | Mobile experience overhaul  | Typography: variables.css font-size tokens, :root 17px at 480px; base.css body/headings use tokens. Cards: rem-based mobile text, larger member photos (96/88px), 44px touch targets (card-details-toggle, beta-banner-dismiss). Drawer/email: larger drawer photo (80px), bumped smallest font sizes. Intelligence: responsive summary grid, .intel-table-scroll-wrap, font-size floor 0.8rem at 480px. Templates: explanation partial classes, predictions/bill inline font-sizes to rem. |
| 2026-02-20 | Mobile typography scale-up   | variables.css: at 480px base 18px, h1 2rem, h2 1.375rem, body/label tokens. Hero: .hero-inner full-width on phone; .hero-headline 1.875rem (phone) / 1.75rem (tablet), eyebrow/subhead larger. base.css: explicit body/h1/h2/footer at 480px. advocacy-drawer.css: mobile labels 0.9375rem, panel title 1.2rem. Branch: feature/mobile-typography. |
| 2026-02-20 | Moneyball help circle (mobile) | .moneyball-help no longer forced to 44px on mobile; stays content-sized (1.1em) so circle fits the "?" only. advocacy-cards.css. |
| 2026-02-20 | Badges keyboard-accessible   | tabindex=0, role=button, aria-label; Tippy focus trigger; focus ring. |
| 2026-02-19 | Accessibility pass           | Drawer role=dialog, focus trap, landmarks, role=alert/status. |
| 2026-02-19 | Humanized advocacy copy      | Conversational tone across drawer, errors, guide steps. |
| 2026-02-19 | ZIP search loading animation  | (Removed 2026-02-20: truck moved to bug report success; ZIP now direct HTMX swap.) |
| 2026-02-19 | Advocacy error states        | Drawer load 404/5xx, I sent auth/network, wrapup error, empty results, drawer-email-open sync, email mobile. |
| 2026-02-19 | Mobile bottom sheet drawer   | ≤768px bottom sheet, drag handle, safe-area-inset for notch. |
| 2026-02-18 | Lint                         | All E402/E501 fixed; per-file ignores; make lint. |
| 2026-02-19 | Pre-commit                   | E501 in config/advocacy; make pre-commit. |
| 2026-02-18 | Status report                | status-report.md: broken/buggy/missing, deployment checklist. |
| 2026-02-19 | DB setup docs                | db-and-outreach.md: data flow, fire pill DB-driven, checklist. |
| 2026-02-18/19 | Guided Email (Mad Libs)   | Traveling outline, mad lib blanks, constituent toggle, To pill, after-call smart email, drawer header constituent. |
| 2026-02-18 | Outreach scripts             | Call/voicemail/email templates; target_type; wrapup call_date. |
| 2026-02-18 | Deployment readiness         | / → /advocacy; /intelligence exempt from API key; deployment.md. |
| 2026-02-18 | Outreach heat pill            | 🔥 N from COUNT distinct user_id; two-DB; seed mock advocates. |
| 2026-02-18 | Unified bill scrape          | scrape_bill_complete(), fulltext externalized, incremental_bill_scrape, _log.py. |
| 2026-02-18 | Auth + outreach DB           | SQLite, email-code auth, OutreachEvent, heat/called/emailed state, interest poll, seed, smoke test. |
| 2026-02-18 | Cursor subagents             | backend.md, tests.md. |
| 2026-02-18 | Dev Bar                      | ?dev floating toolbar; Advocacy/Intelligence/Explore panels; deep-link; Ctrl+Shift+D. |
| 2026-02-18 | Separate dev/prod cache      | cache/ sacred; cache/dev/ for dev; dev-reset. |
| 2026-02-18 | Dev mock data                | committees.json roster/bill_numbers; bills action_history/vote_events/slips. |
| 2026-02-18 | Member card no-email UX      | Hint under Email button; Call/Email same height; primary vs secondary. |
| 2026-02-18 | Favicon                      | /favicon.ico truck SVG. |
| 2026-02-18 | Member cards badges under name | Grid: photo col1, name+badges col2; mobile contents. |
| 2026-02-18 | Call drawer email/name width | 75% right-aligned. |
| 2026-02-18 | Call drawer gray             | #ebebf0 for them/capture bubbles. |
| 2026-02-18 | Email drawer click-to-copy To | Copy member email; green Copied feedback. |
| 2026-02-18 | Advocacy drawer width        | Call 420px; email 50vw. |
| 2026-02-18 | Gmail-style email drawer     | htmx.process() after fetch; compose UI, I sent, copy row, banners. |
| 2026-02-17 | Startup summary + URLs        | Steps 10–12 (graph, ML, influence); Services block APP_BASE_URL. |
| 2026-02-17 | Advocacy overhaul            | Hero, auto-submit ZIP, Call/Email buttons, phone panel, streamlined chips, HTMX drawer, wrap-up, error states. |
| 2026-02-17 | Member photos                | photo_url; scrape-members, refresh-photos. |
| 2026-02-17 | Responsive + sidebar         | Breakpoints; results sidebar <details>; panels full width mobile. |
| 2026-02-14 | CI ML extras                 | pip install -e ".[dev,ml]" in CI. |
| 2026-02-14 | Lint (ML)                    | features.py ft_tfidf_id_to_idx; member_value.py F841. |
| 2026-02-14 | Member Value Model           | Ridge LOO-CV, value residual, recruitment rankings; Step 6; /intelligence/recruitment. |
| 2026-02-14 | Full-text leakage fix        | FULLTEXT_DROP_COLUMNS; advance model no fulltext. |
| 2026-02-14 | Bill-to-law reference        | ilga_rules.json bill_to_law_process; intelligence_bill.html. |
| 2026-02-14 | Bill page last action        | From action_history; fallback latest; full actions table. |
| 2026-02-14 | Pipeline stage rollback      | HB3356 Rule 19(b); current_stage vs highest_stage. |
| 2026-02-14 | SHAP explanations            | GET /api/bills/{id}/explanation; Why This Score. |
| (various)  | GraphQL search              | search.py; SearchResultType, search() resolver. |

---

## Done (this session)

| Topic                          | Summary |
|--------------------------------|--------|
| Code cleanup (main.py)         | Removed dead `_load_stale_cache_fallback` (lifespan uses `etl.load_stale_cache_fallback`). Removed unused `get_bill_status_urls` re-export and wrong comment (scripts don't import from main). Dropped unused imports: `ILGAScraper`, `load_bill_cache`. Query.bills() now uses `safe_parse_date` directly; kept `_parse_bill_date`/`_safe_parse_date` aliases for tests. |
| Branch reconciliation (prod-improvements) | API key: added /privacy and /terms to exempt set so legal pages are public when ILGA_API_KEY is set. Sitemap: added /privacy and /terms to _SITEMAP_PATHS. status-report.md: routes list now includes feedback + legal; notes custom error pages, catch-all 404, CSP/HSTS, static cache; Legal and SEO paths documented as API-key exempt. CTA kept as "Start Outreaching" per product choice. |
| Advocacy/router cleanup        | `member_lookup.is_constituent_for_zip_member(state, zip_code, member)`; `advocacy_helpers.party_abbr_for_member(member)`; shared `_hero_context()` in advocacy router. Replaced 4× is_constituent blocks, 3× party_abbr blocks, 2× hero dicts; explore uses party_abbr_for_member. |
| Pre-push cleanup               | Removed leftover hero-image debug instrumentation (fetch to localhost:7246/ingest) from index.html; tests 303 pass, lint clean. |
| Full-text feature caps         | FULLTEXT_MAX_FEATURES 400, FULLTEXT_MAX_TOKENS 2000; env overrides. |
| ML Step 3 tuning                | n_iter 20, trimmed grid, verbose=2, ILGA_ML_SKIP_TUNE=1. |
| Smart tiered index scanning     | <24h skip, <7d tail-only, >7d full; scrape_metadata timestamps. |
| Pipeline resilience             | Preserve vote_events/witness_slips on re-scrape; unified scrape.py; Makefile 8 targets. |
| Vote PDF tally mismatch        | Middle-initial E/P/N/A disambiguation; SMART_A lookahead. |
| Vote/slip scraper robustness   | Data-verified progress, dual-track, heuristic skip, batch saves, merge at startup. |
| Lint hardening                  | E501/W293; pre-commit Ruff + make test. |
| Incremental scrape-votes        | Progress file, resumable, parallel, ETA. |
| Discovery-based bill index     | _discover_doc_types(); all doc types; chamber J for EO/JSR/AM. |
| Separation of concerns         | Bill vote_events/witness_slips; single-page index; unified committees.json; GraphQL IDs. |

---

## Previously done

| Topic                           | Summary |
|---------------------------------|--------|
| Bill index scraping             | Clear progress logs; checkpoint after every page. |
| Chamber-specific associated     | associated_senator / associated_representatives in GraphQL + exporter. |
| Full bill index + scrape-200    | Pagination; make scrape-200, make scrape-full. |
| Scrape → dev/prod pipeline      | make scrape (prod), make scrape-dev; ILGA_LOAD_ONLY=1. |
| Evidence-based script hints     | _build_script_hint_*; "How we pick targets"; Moneyball definition. |
| Scorecard / Moneyball clarity   | metrics_definitions.py; empirical first; metricsGlossary. |
| Advocacy full-data clarity      | Banner actual member_count/zip_count; make dev-full. |

---

## Next (when you're ready)

| Area        | Item                    | Notes |
|-------------|-------------------------|-------|
| Power Map   | Weighted edges           | Shared bill count for pruning/edge thickness. |
| Power Map   | House committee mapping  | _CATEGORY_COMMITTEES (currently Senate only). |
| Power Map   | Free-text topic search   | e.g. "kei truck" → bills → committees → highlight members. |
| Power Map   | Mobile detail panel      | Bottom sheet on small screens. |
| Power Map   | Path visualization       | Dotted paths from user legislators to chairs when topic+ZIP set. |
| Power Map   | Time-based animation     | Co-sponsorship network over session. |
| Pipeline    | Run full pipeline        | make scrape (daily ~2 min or FULL=1); verify GraphQL + advocacy. |
| Prod        | Shift to prod            | ILGA_PROFILE=prod, CORS, optional API_KEY. |

---

## Production (checklist for deployment)

| # | Check |
|---|--------|
| 1 | Pre-populate `cache/` (full or incremental scrape) so startup is load-only. |
| 2 | Set `ILGA_PROFILE=prod` (turns off dev + seed). |
| 3 | Set `ILGA_CORS_ORIGINS` to your front-end origin(s). |
| 4 | Set `ILGA_API_KEY` to protect GraphQL (prod warns if empty). |
| 5 | Use `GET /health` for readiness (`ready` when members loaded). |

---

## ML Pipeline (`feature/ml-pipeline` branch)

**New `src/ilga_graph/ml/` package** — automated "Legislative Intelligence Engine" (v2). One command: `make ml-run`.

### v2 improvements (what changed)

| Problem in v1 | Fix in v2 |
|---------------|-----------|
| Test set 3 pos / 2900 neg | Evaluate on mature bills (120+ days); 412 advanced / 1650 stuck. |
| Single algorithm           | Compare 4 algorithms, 5-fold stratified CV; best auto-selected. |
| No hyperparameter tuning   | RandomizedSearchCV 40 iter; GBT n_estimators=300, max_depth=9, lr=0.05. |
| No calibration             | CalibratedClassifierCV isotonic. |
| 49% DBSCAN outliers        | Agglomerative clustering; 100% classified, k=10. |
| Anomaly = big = suspicious | Coordination features (name dup, unanimity, top org share); explains WHY. |
| No quality report          | model_quality.json with trust, comparison, strengths/issues. |

### Current results (v2)

| Metric | Value |
|--------|--------|
| Bill prediction CV ROC-AUC | 0.984 ± 0.005 |
| Bill prediction Test ROC-AUC | 0.910 |
| Test precision (advanced) | 96.3% |
| Accuracy on mature bills | 94.9% |
| Entity resolution | 100% |
| Coalition clustering | 10 blocs, 100% classified |
| Anomaly detection | 102 bills (8%) with reasons |
| Pipeline runtime | ~4 min |

### What it produces (in `processed/`)

| Output | Description |
|--------|--------------|
| bill_scores.parquet | Every bill scored; mature = labels, immature = forecasts. |
| model_quality.json | Trust assessment, metrics, top features. |
| coalitions.parquet | Legislators → voting bloc. |
| member_embeddings.parquet | 32-dim vectors. |
| slip_anomalies.parquet | Coordination signals + reasons. |
| fact_vote_casts.parquet | Vote casts with member_id. |
| dim_*.parquet, fact_*.parquet | Star schema. |

### How to run

```bash
make ml-setup    # One time
make ml-run      # Full pipeline ~4 min
```

### Pipeline steps (all automated)

1. **Data Pipeline** — Flatten cache to Parquet.
2. **Entity Resolution** — 385 names → member IDs; 100% resolved.
3. **Bill Outcome Prediction** — 4 algorithms, tune, calibrate; mature-bill eval.
4. **Coalition Discovery** — Agreement graph + Agglomerative; 10 blocs.
5. **Anomaly Detection** — Coordination features, Isolation Forest.

### Individual steps (optional)

```bash
make ml-pipeline
make ml-resolve    # AUTO=1 for non-interactive
make ml-predict
```

### v3: Intelligence Dashboard & Self-Correcting Feedback Loop

Backtester, chained scrape→ml-run, ml_loader.py, GraphQL 7 resolvers, /intelligence tabbed UI (Predictions, Coalitions, Anomalies, Model Quality & Accuracy).

### v4: Coalition Naming, Bill Pipeline Stage, Stuck-Bill Analysis

characterize_coalitions(), compute_bill_stage(), classify_stuck_status(); BillScore stage/stuck fields; GraphQL filters; predictions/coalitions dashboard updates.

### Next (ML backlog)

- Gold labels (bill_labels_gold.json): eval and/or override training.
- Low-confidence UX: uncertain rows, calibration curve.
- Individual vote prediction, poison-pill detector, committee prediction, accuracy trend, per-bill history, influence trend.

---

## Backlog / Future

| Item | Status |
|------|--------|
| Deploy from `production` branch (not main) | Open; plan in .cursor/plans. |
| Unified GraphQL `search` query | Done. |
| Search: fuzzy, index, vote/slip entities, autocomplete | Open. |
| Full bill index make scrape-full | Done. |
| make scrape-votes from cache | Done. |
| Advocacy: member photos, richer script, email links | Open. |
| Advocacy: CSS to static when it grows | Open. |
| Advocacy: HTMX drill-down on card | Open. |
| Advocacy: interactive map (ZIP/district) | Open. |
| Full Census crosswalk for prod | Open. |

---

## Done (summary)

| Theme | What was done |
|-------|----------------|
| Modularity 1–2 | config.py ILGA_PROFILE, .env; try/except ETL; checkpoint every 50 bills. |
| Cache & seed | committees.json; mocks/dev/ SEED_MODE; normalized members/bills; vote/slip cache. |
| ETL | load_or_scrape_data, compute_analytics, export_vault; ILGA_LOAD_ONLY. |
| Resilience | HTTPAdapter retries, startup timings CSV, ETL-phase summary table. |
| Witness slips & GraphQL | billSlipAnalytics, memberSlipAlignment, advancement; graphql/README. |
| Bills-first & incremental | Legislation pages; incremental_bill_scrape; sponsor_ids. |
| Server & CI | main.py single app; ruff; tests. |
| Seating chart | seat_block_id, seat_ring, seatmate_names, seatmate_affinity; seating.py; Senate-only. |
| SSR Advocacy | zip_crosswalk, GET /advocacy, POST /advocacy/search, 3 cards, policy filter. |
| Steps 3–5 | etl.py, analytics_cache, loaders; batch_load in resolvers. |
| Moneyball v2 | Shell bill filter, institutional weight, chair-first Power Broker. |
