# GitHub Actions Example

Use this workflow to run on PR commit pushes (`synchronize`) and keep one PR comment updated.

```yaml
name: PR UI Screenshot Update

on:
  pull_request:
    types: [opened, reopened, synchronize]

permissions:
  contents: write
  pull-requests: write

jobs:
  ui-screenshot:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: |
          npm ci
          npx playwright install --with-deps chromium

      - name: Start app
        run: |
          ./scripts/start_web.sh &
          sleep 5

      - name: Update PR screenshot comment
        env:
          GH_TOKEN: ${{ github.token }}
          GITHUB_REPOSITORY: ${{ github.repository }}
        run: |
          python3 skills/pr-ui-screenshot-update/scripts/update_pr_ui_screenshot.py \
            --base-ref "${{ github.event.pull_request.base.sha }}" \
            --head-ref "${{ github.event.pull_request.head.sha }}" \
            --url "http://127.0.0.1:8000" \
            --pr-number "${{ github.event.pull_request.number }}"

      - name: Upload screenshot artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: pr-ui-screenshots
          path: tmp/pr-ui-screenshots/*.png
          if-no-files-found: ignore
```

Notes:
- Use `--commit-screenshot --push` only if you want images committed to the PR branch for inline markdown rendering.
- If your UI runs on another host/port, change `--url`.
- Ensure your app is reachable before screenshot capture.
