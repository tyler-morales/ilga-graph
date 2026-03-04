# Live accessibility scanner

Runs **axe-core** (WCAG 2.1) against the dev server and prints violations to the terminal.

## Setup

```bash
cd scripts/a11y
npm install
```

## Run

1. Start the dev server (e.g. from project root: `uvicorn src.ilga_graph.main:app --reload` or your usual command).
2. From `scripts/a11y`:

```bash
npm run a11y
```

- Default URL: `http://localhost:8000`
- Custom URL: `node scan.js https://localhost:8443` or `A11Y_BASE_URL=https://localhost:8443 npm run a11y`

## Output

The script prints each violation with impact, rule id, description, help URL, and affected DOM nodes (first 10). Exit code is 1 if any violations are found.

## Phase 2

Use the **accessibility-agent** subagent for static analysis: semantic HTML, ARIA, alt text, keyboard/focus, and contrast in templates and CSS.
