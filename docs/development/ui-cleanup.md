# UI dead-code purge

This doc describes how to safely audit and remove dead CSS, JS, and HTML (templates) without breaking the UI.

## Principles

1. **Audit before delete** — Always run the audit scripts and review the report before removing anything.
2. **Small batches** — Remove in small batches (e.g. one CSS file or a few rules); run tests and smoke-check after each batch.
3. **Safelist liberally** — Dynamic classes (Jinja, JS-injected, third-party) and entry-point functions should be in the scripts’ safelists so they are never flagged as dead.
4. **Document in TODOS** — Log what was purged in [TODOS.md](../../TODOS.md) (Refactor row).

## Scripts

All scripts live in `scripts/` and are run from the repo root.

### CSS: `audit_css_usage.py`

- **What it does:** Scans all templates for `class="..."` and `id="..."` (and `{% block body_class %}...{% endblock %}`, and JS `getElementById` / `querySelector` / `closest`). Parses all CSS files under `src/ilga_graph/static/css/` and flags rules whose leading class or id is not in the “used” set and not in the safelist.
- **Safelist:** Prefixes like `tippy-`, `htmx-`, `drawer-`, `power-badge-`, etc., and literals like `power-badge`, `open`, `active`. See `SAFELIST_PREFIXES` and `SAFELIST_LITERALS` in the script.
- **Run:** `python3 scripts/audit_css_usage.py` (prints candidate-dead rules). Add `--report` to write `audit_css_report.txt`.
- **Caveats:** Conditional classes (e.g. `class="foo {% if x %}bar{% endif %}"`) may not be fully extracted; avoid removing rules that might apply to such classes. Prefer removing only rules you’ve confirmed are unused (e.g. by grep).

### JS: `audit_js_usage.py`

- **What it does:** Collects function definitions (`function name(` and `var/let/const name = function`) from inline scripts in templates and from `static/js/*.js`. Counts references (calls, `window.name`, callback refs like `= name` or `, name)`). Reports names that are defined but have no references beyond the definition (candidate dead).
- **Safelist:** Entry points and known-used names (e.g. `closeAdvocacyDrawer`, `getCsrfToken`, `initAuthStrip`). See `SAFELIST` in the script.
- **Run:** `python3 scripts/audit_js_usage.py` (prints candidate-dead names). Add `--report` to write `audit_js_report.txt`.
- **Caveats:** IIFEs and callbacks passed as function references (e.g. `addEventListener('click', foo)`) are only detected if the ref pattern matches. Many “candidate dead” functions are actually used as callbacks; verify with grep before removing.

### Templates: `audit_template_reachability.py`

- **What it does:** Collects template names from Python (`TemplateResponse("...")`, `get_template("...")`, and `"template": "..."` in dicts) and from templates (`{% include "..." %}` and `{% extends "..." %}`). Reports template files that are never referenced.
- **Run:** `python3 scripts/audit_template_reachability.py` (prints unreachable templates). Add `--report` to write `audit_template_reachability_report.txt`.
- **Caveats:** Templates rendered via a variable (e.g. `TemplateResponse(tpl)` where `tpl` is set elsewhere) are not detected; the script only finds string literals.

## Workflow

1. Run all three audits:  
   `python3 scripts/audit_css_usage.py --report`  
   `python3 scripts/audit_js_usage.py --report`  
   `python3 scripts/audit_template_reachability.py --report`
2. Review the reports. For CSS/JS, confirm with grep that a selector or function is truly unused before removing.
3. Remove in small batches. After each batch:
   - Run `make test`.
   - Manually smoke-test key pages: `/`, `/advocacy` (ZIP + drawer), `/the-issue`, `/legislator-brief`, `/updates`, one admin page.
4. Update TODOS.md (Refactor row) with what was removed.

## Regression

- **Unit tests:** `make test` (pytest).
- **Visual:** Manual smoke of the routes above. Optionally add Playwright (or similar) tests that load key pages and assert no console errors and critical selectors exist.
