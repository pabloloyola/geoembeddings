const fs = require("fs");
const { chromium } = require("playwright");

const fragmentPath = process.argv[2] || "kanto-simulator-explorer.html";

(async () => {
  const browser = await chromium.launch({headless: true});
  const errors = [];
  for (const width of [736, 320]) {
    const page = await browser.newPage({viewport: {width, height: 1000}});
    page.on("pageerror", (error) => errors.push(String(error)));
    const fragment = fs.readFileSync(fragmentPath, "utf8");
    const base = `<style>
      :root{--background:#fff;--foreground:#17202a;--card:#f5f7f8;--card-foreground:#17202a;--muted:#edf0f2;--muted-foreground:#66717c;--border:#cbd1d6;--green:#16845b;--red:#c43d3d;--viz-series-1:#2864dc;--viz-series-2:#38a57a;--viz-series-3:#e28b30;--viz-series-4:#8b62d7;--viz-series-5:#d34e70}
      *{box-sizing:border-box}body{margin:16px;font:14px/1.45 system-ui;color:var(--foreground);background:var(--background)}
      .viz-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.card{padding:12px;background:var(--card);border-radius:8px}.viz-stat-value{font-size:24px}.text-muted{color:var(--muted-foreground)}.text-small{font-size:12px}.viz-controls,.viz-row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.form-label{display:grid;gap:4px}.form-select{font:inherit;padding:6px}.viz-badge{padding:3px 8px;border-radius:10px;background:var(--muted)}.table{width:100%;border-collapse:collapse}.table td,.table th{padding:6px;border-bottom:1px solid var(--border);text-align:left}.text-center{text-align:center!important}
    </style>`;
    await page.setContent(base + fragment, {waitUntil: "load"});
    await page.waitForTimeout(200);
    const metrics = await page.evaluate(() => ({
      users: document.querySelectorAll("#kse-user option").length,
      days: document.querySelectorAll("#kse-day option").length,
      svgMarks: document.querySelectorAll("#kse-timeline circle,#kse-timeline rect").length,
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      checks: document.querySelectorAll("#kse-checks tr").length,
    }));
    if (metrics.users < 1 || metrics.days < 1 || metrics.svgMarks < 1 || metrics.checks < 1 || metrics.scrollWidth > metrics.clientWidth) {
      errors.push(width + "px validation: " + JSON.stringify(metrics));
    }
    await page.screenshot({path: `/tmp/kanto-explorer-${width}.png`, fullPage: true});
    await page.close();
  }
  await browser.close();
  if (errors.length) {
    console.error(JSON.stringify(errors, null, 2));
    process.exit(1);
  }
  console.log("Explorer rendered at 736px and 320px with no script or overflow errors.");
})();
