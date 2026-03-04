# Design system

Single source of truth for UI primitives, tokens, and component taxonomy. Use this doc to decide canonical vs legacy class names and to consolidate similar components.

## Canonical sources

- **Tokens (colors, spacing, typography, radii, shadows):** [src/ilga_graph/static/css/variables.css](../../src/ilga_graph/static/css/variables.css)
- **Primitives (buttons, inputs, cards, badges, score bar):** [src/ilga_graph/static/css/components.css](../../src/ilga_graph/static/css/components.css)

All new UI must use only canonical classes from `components.css` and tokens from `variables.css`.

---

## Concepts and taxonomy

| Concept | Canonical class(es) | CSS file | Legacy / alternate class names |
|--------|---------------------|----------|--------------------------------|
| **Primary button** | `btn btn--primary` | components.css | `drawer-btn-primary` (advocacy-email.css), `hero-signin-btn` (advocacy-form.css), `admin-login-btn` (admin-login.css), `mock-preset-btn`, `mock-apply-btn` (admin-mocks.css), `intel-tab` (intelligence-dashboard.css), `brief-states-filter-btn`, `brief-states-view-btn` (base.css), `gmail-from-verify-btn`, `gmail-from-code-submit`, `gmail-send-btn` (advocacy-email.css) |
| **Secondary button** | `btn btn--secondary` | components.css | `drawer-btn-secondary` (advocacy-email.css), `hero-signin-cancel`, `admin-login-resend` (admin-login.css), `admin-login-back` |
| **Ghost button** | `btn btn--ghost` | components.css | — |
| **Story / accent CTA** | `btn btn--story` | components.css | Orange CTA for “Share your story”, community actions |
| **Success button** | `btn btn--success` | components.css | Green confirm (e.g. poll submit) |
| **Input** | `input` (with class `input`) | components.css | `hero-signin-input`, `hero-signin-code-input` (advocacy-form.css), `admin-login-input`, `admin-login-code-input` (admin-login.css), `gmail-field-input`, `gmail-from-input`, `gmail-subject-input` (advocacy-email.css), `mock-zip-input`, `mock-contact-input` (admin-mocks.css) |
| **Card** | `card`, `card--elevated`, `card--flush` | components.css | Feature wrappers (e.g. intro-card, kei-poll-card) compose `.card` where appropriate |
| **Badge** | `badge`, `badge--success`, `badge--warning`, `badge--info`, `badge--neutral` | components.css | — |
| **Score bar** | `score-bar`, `score-bar__fill`, `score-bar__fill-inner`, `score-bar__label` | components.css | Used by intelligence; modifiers `.score-bar--sm`, `--lg`; fill colors via `.score-high`, `.score-mid`, etc. |
| **Link as button** | `<a class="btn btn--primary">` or `btn btn--secondary` | components.css | Use same classes on `<a>` when the CTA navigates |

---

## Typography

| Use case | Token | When to use |
|----------|--------|--------------|
| **Body (default)** | `--font-serif` | Default body text (Georgia); used site-wide unless overridden. |
| **UI / home** | `--font-sans` | Home page and many UI surfaces; system stack. |
| **Hero / landing** | `--font-hero` | Hero headlines and landing blocks (Helvetica Neue, etc.). |

Font sizes and line heights are in `variables.css` (`--font-size-base`, `--font-size-h1`, `--line-height-body`, etc.). Mobile overrides live in the same file.

---

## Consolidation strategies

### Alias (fast)

For a legacy class that should look like a canonical one, add a rule so the legacy class inherits or duplicates the canonical styles. No template changes; visual consistency first.

- Example: `.drawer-btn-primary { /* same as .btn.btn--primary */ }` or re-use a shared set of properties.
- Use when you want quick visual parity before migrating markup.

### Migrate (clean)

Replace legacy class names in templates with canonical ones (e.g. `drawer-btn drawer-btn-primary` → `btn btn--primary`), then remove the legacy rules from feature CSS. Do one feature or one page at a time.

1. Run [scripts/audit_css_usage.py](../../scripts/audit_css_usage.py) and smoke-test after each batch.
2. Prefer migrating when you are already touching that template or feature.

### Rule for new UI

New UI must use only canonical classes from `components.css` and tokens from `variables.css`.

---

## Related

- [UI dead-code purge](ui-cleanup.md) — Auditing and removing dead CSS/JS/templates.
- [Component playground](component-playground.md) — Dev-only scene-based playground at `/dev/playground`.
