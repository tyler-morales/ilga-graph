#!/usr/bin/env node
/**
 * Live a11y scan: runs axe-core against the dev server and prints violations to the terminal.
 * Usage: npm run a11y   (default: http://localhost:8000)
 *        node scan.js [url]
 *
 * Start the dev server first (e.g. uvicorn from project root), then run from scripts/a11y.
 */

const puppeteer = require('puppeteer');
const axeCore = require('axe-core');
const baseUrl = process.argv[2] || process.env.A11Y_BASE_URL || 'http://localhost:8000';

function formatViolation(v) {
  const lines = [
    `  [${v.impact}] ${v.id}`,
    `    ${v.help}`,
    `    ${v.description}`,
    `    Help: ${v.helpUrl}`,
  ];
  if (v.nodes && v.nodes.length) {
    lines.push('    Nodes:');
    v.nodes.slice(0, 10).forEach((node) => {
      lines.push(`      - ${node.html.replace(/\s+/g, ' ').slice(0, 120)}`);
      if (node.failureSummary) lines.push(`        ${node.failureSummary}`);
    });
    if (v.nodes.length > 10) lines.push(`      ... and ${v.nodes.length - 10} more`);
  }
  return lines.join('\n');
}

async function run() {
  console.log(`\n  A11y scan: ${baseUrl}\n`);
  const browser = await puppeteer.launch({ headless: 'new' });
  try {
    const page = await browser.newPage();
    await page.setBypassCSP(true);
    await page.goto(baseUrl, { waitUntil: 'networkidle0', timeout: 30000 });
    await page.addScriptTag({ path: require.resolve('axe-core/axe.min.js') });

    const results = await page.evaluate(() => window.axe.run({ runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] } }));

    const violations = results.violations || [];
    if (violations.length === 0) {
      console.log('  No accessibility violations reported.\n');
      return;
    }
    console.log(`  Violations: ${violations.length}\n`);
    violations.forEach((v) => console.log(formatViolation(v) + '\n'));
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error('Scan failed:', err.message);
  process.exit(1);
});
