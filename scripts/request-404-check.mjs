// Prüft jede Sitemap- und Weiterleitungsseite auf 4xx/5xx-Antworten und Head-Basics
// (<!doctype html>, <meta charset>, <meta name="viewport">, <title>). Zusätzlich
// eine iPhone-14-Emulation von community.html (Breite = Gerätebreite, Burger-Menü aktiv).
//
// Start: node scripts/request-404-check.mjs     (braucht ein Node mit Playwright-Zugriff)
// Scheitert sichtbar bei jeder Seite mit fehlender Antwort oder fehlendem Head-Element.
import { chromium, devices } from 'playwright';
const BASE = 'https://gobetter.trade';
const SITEMAP = ['/', '/fair-value-pro.html', '/portfolio-tracker.html', '/investment-planner.html',
  '/daten-quelle.html', '/plaene.html', '/partner.html', '/community.html', '/akademie/',
  '/akademie/modul-01-grundlagen.html','/akademie/modul-02-fundamentalanalyse.html',
  '/akademie/modul-03-chartanalyse.html','/akademie/modul-04-einstieg.html',
  '/akademie/modul-05-risikomanagement.html','/akademie/modul-06-verkaufsstrategien.html',
  '/akademie/modul-07-psychologie.html','/akademie/modul-08-strategie.html',
  '/agb.html','/datenschutz.html','/impressum.html'];
const REDIRECTS = ['/pro-mode.html','/fair-value-pro/pro-mode.html','/fair-value-pro/daten-quelle.html',
  '/trading-desk.html','/fuer-autoren.html'];

const browser = await chromium.launch({ headless: true });

function checkHeadBasics(html) {
  const issues = [];
  if (!/<!doctype html/i.test(html)) issues.push('kein <!doctype html>');
  if (!/<meta\s+charset=/i.test(html)) issues.push('kein <meta charset>');
  if (!/<meta\s+name="viewport"/i.test(html)) issues.push('kein <meta name="viewport">');
  if (!/<title>[^<]+<\/title>/i.test(html)) issues.push('kein <title>');
  return issues;
}

let total = 0, bad = 0, headBad = 0;
for (const p of [...SITEMAP, ...REDIRECTS]) {
  const url = BASE + p;
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  const errs = [];
  page.on('response', r => {
    const u = r.url();
    if (u.startsWith(BASE) && r.status() >= 400) errs.push(`${r.status()} ${u}`);
  });
  // HTML separat per request fetch, damit Redirect-Stubs mit meta-refresh nicht stören.
  const htmlResp = await page.request.get(url);
  const html = await htmlResp.text();
  const headIssues = checkHeadBasics(html);
  await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
  // Durchscrollen, damit lazy-geladene Assets Antworten melden.
  const h = await page.evaluate(() => document.documentElement.scrollHeight);
  for (let i = 1; i <= 6; i++) { await page.evaluate(y => window.scrollTo(0, y), Math.round(h*i/6)); await page.waitForTimeout(200); }
  await page.waitForLoadState('networkidle');
  total++;
  const lines = [];
  if (errs.length) { bad++; lines.push(...errs.map(e => '    404/5xx: ' + e)); }
  if (headIssues.length) { headBad++; lines.push(...headIssues.map(e => '    HEAD: ' + e)); }
  if (lines.length) { console.log(`✗ ${url}`); lines.forEach(l => console.log(l)); }
  else console.log(`✓ ${url}`);
  await ctx.close();
}

console.log('\n=== iPhone 14 Emulation: community.html ===');
const iphone = devices['iPhone 14'];
const ctx = await browser.newContext({ ...iphone });
const page = await ctx.newPage();
await page.goto(BASE + '/community.html', { waitUntil: 'load' });
await page.waitForSelector('header.gbn', { timeout: 5000 });
const m = await page.evaluate(() => ({
  w: window.innerWidth, dpr: window.devicePixelRatio,
  bodyW: document.body.getBoundingClientRect().width,
  scrollW: document.documentElement.scrollWidth,
  burgerVisible: getComputedStyle(document.querySelector('.gbn__burger')).display !== 'none',
  linksVisible: getComputedStyle(document.querySelector('.gbn__links')).display !== 'none',
}));
console.log(`  innerWidth=${m.w}  dpr=${m.dpr}  bodyW=${m.bodyW}  scrollW=${m.scrollW}`);
console.log(`  Nav: burger=${m.burgerVisible}  desktop-links=${m.linksVisible}`);
if (m.scrollW > m.w + 1) console.log(`  ⚠ horizontales Scrollen: ${m.scrollW - m.w}px`);
if (!m.burgerVisible || m.linksVisible) console.log(`  ⚠ Nav-Modus falsch`);
await ctx.close();

await browser.close();
console.log(`\n${total} Seiten, ${bad} mit fehlerhaften Anfragen, ${headBad} mit Head-Mängeln.`);
