#!/usr/bin/env node
/**
 * Minify all CSS in src/ilga_graph/static/css/ to .min.css in the same dir.
 * Run from repo root. Used by: npm run minify:css, make minify
 */
const fs = require("fs");
const path = require("path");
const CleanCSS = require("clean-css");

const staticDir = path.join(__dirname, "..", "src", "ilga_graph", "static", "css");
const files = fs.readdirSync(staticDir).filter((f) => f.endsWith(".css") && !f.endsWith(".min.css"));

const minifier = new CleanCSS({});
let ok = true;
for (const f of files) {
  const inputPath = path.join(staticDir, f);
  const outputPath = path.join(staticDir, f.replace(/\.css$/, ".min.css"));
  try {
    const result = minifier.minify(fs.readFileSync(inputPath, "utf8"));
    if (result.errors.length) {
      console.error(result.errors);
      ok = false;
    } else {
      fs.writeFileSync(outputPath, result.styles);
      console.log(`  ${f} -> ${path.basename(outputPath)}`);
    }
  } catch (e) {
    console.error(e);
    ok = false;
  }
}
process.exit(ok ? 0 : 1);
