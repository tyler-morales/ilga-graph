#!/usr/bin/env node
/**
 * Minify all JS in src/ilga_graph/static/js/ to .min.js in the same dir.
 * Run from repo root. Used by: npm run minify:js, make minify
 */
const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const staticDir = path.join(__dirname, "..", "src", "ilga_graph", "static", "js");
if (!fs.existsSync(staticDir)) {
  console.log("No static/js dir, skip.");
  process.exit(0);
}
const files = fs.readdirSync(staticDir).filter((f) => f.endsWith(".js") && !f.endsWith(".min.js"));

let ok = true;
for (const f of files) {
  const input = path.join(staticDir, f);
  const output = path.join(staticDir, f.replace(/\.js$/, ".min.js"));
  try {
    execSync(`npx terser "${input}" -o "${output}"`, { stdio: "inherit" });
    console.log(`  ${f} -> ${path.basename(output)}`);
  } catch (e) {
    ok = false;
  }
}
process.exit(ok ? 0 : 1);
