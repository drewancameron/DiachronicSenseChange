#!/usr/bin/env node
/**
 * Generate a printable PDF from docs/index.html using Puppeteer.
 *
 * Set viewport to the exact PDF content width (699px), apply styles,
 * run alignment, use emulateMediaType('screen') so page.pdf() doesn't
 * re-layout.
 */

import puppeteer from 'puppeteer';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const args = process.argv.slice(2);
const outputIdx = args.indexOf('--output');
const outputPath = outputIdx >= 0
  ? path.resolve(args[outputIdx + 1])
  : path.join(ROOT, 'output', 'blood_meridian.pdf');

const htmlPath = path.join(ROOT, 'docs', 'index.html');

async function main() {
  console.log(`Rendering ${htmlPath}`);

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Step 1: Load at wide viewport so the page renders with gloss panel visible
  await page.setViewport({ width: 1200, height: 800 });
  await page.emulateMediaType('screen');
  await page.goto(`file://${htmlPath}`, { waitUntil: 'networkidle0', timeout: 30000 });
  await page.evaluate(() => document.fonts.ready);

  // Step 2: Inject styles for PDF
  await page.evaluate(() => {
    document.querySelectorAll('.fb-tab, .fb-panel').forEach(el => el.style.display = 'none');

    const style = document.createElement('style');
    style.textContent = `
      body { max-width: none !important; padding: 0.5rem !important; margin: 0 !important; }
      .page-body { display: grid !important; grid-template-columns: 1fr 220px !important; gap: 0 0.6rem !important; }
      .gloss-panel { display: block !important; padding-left: 0.5rem !important; }
      .main-text { font-size: 0.95rem !important; line-height: 1.9 !important; hyphens: none !important; overflow-wrap: normal !important; word-break: normal !important; }
      .mg { font-size: 0.6rem !important; line-height: 1.2 !important; }
      .gw:hover { background: none !important; }
    `;
    document.head.appendChild(style);
  });

  // Step 3: Resize viewport to exact PDF content width
  await page.setViewport({ width: 700, height: 800 });

  // Step 4: Force the gloss panel visible (viewport is now < 800px breakpoint)
  await page.evaluate(() => {
    // Must override inline since the media query fired
    document.querySelectorAll('.page-body').forEach(el => {
      el.style.cssText = 'display: grid !important; grid-template-columns: 1fr 220px !important; gap: 0 0.6rem !important;';
    });
    document.querySelectorAll('.gloss-panel').forEach(el => {
      el.style.cssText = 'display: block !important; padding-left: 0.5rem !important; position: relative !important;';
    });
  });

  // Step 5: Wait for layout to fully settle at 699px
  await new Promise(r => setTimeout(r, 500));
  await page.evaluate(() => void document.body.offsetHeight);
  await new Promise(r => setTimeout(r, 500));

  // Step 6: Run alignment at 699px — this matches the PDF width exactly
  await page.evaluate(() => {
    const GAP = 3;
    document.querySelectorAll('.page-body').forEach(body => {
      const panel = body.querySelector('.gloss-panel');
      if (!panel) return;
      const bodyTop = body.getBoundingClientRect().top + window.scrollY;
      const panelWidth = panel.offsetWidth;
      const colWidth = Math.floor((panelWidth - 12) / 2);
      const mgs = panel.querySelectorAll('.mg');

      const heights = [];
      mgs.forEach(mg => {
        mg.style.width = colWidth + 'px';
        heights.push(mg.getBoundingClientRect().height);
      });

      let col1Bottom = 0, col2Bottom = 0;
      mgs.forEach((mg, i) => {
        const anchor = document.getElementById(mg.dataset.for);
        if (!anchor) { mg.style.display = 'none'; return; }
        const idealTop = anchor.getBoundingClientRect().top + window.scrollY - bodyTop;
        const top1 = Math.max(idealTop, col1Bottom + GAP);
        const top2 = Math.max(idealTop, col2Bottom + GAP);
        const useCol1 = Math.abs(top1 - idealTop) <= Math.abs(top2 - idealTop);
        if (useCol1) {
          mg.style.top = top1 + 'px';
          mg.style.left = '0px';
          col1Bottom = top1 + heights[i];
        } else {
          mg.style.top = top2 + 'px';
          mg.style.left = (colWidth + 12) + 'px';
          col2Bottom = top2 + heights[i];
        }
      });
    });
  });

  await new Promise(r => setTimeout(r, 200));

  // Verify gw497
  const check = await page.evaluate(() => {
    const mg = document.querySelector('.mg[data-for="gw497"]');
    const anchor = document.getElementById('gw497');
    const body = mg.closest('.page-body');
    const bodyTop = body.getBoundingClientRect().top + window.scrollY;
    const ideal = anchor.getBoundingClientRect().top + window.scrollY - bodyTop;
    return {
      mgTop: Math.round(parseFloat(mg.style.top)),
      idealTop: Math.round(ideal),
      diff: Math.round(parseFloat(mg.style.top) - ideal),
      bodyWidth: document.body.clientWidth,
    };
  });
  console.log(`gw497: mgTop=${check.mgTop} idealTop=${check.idealTop} diff=${check.diff} bodyWidth=${check.bodyWidth}`);

  // Step 7: Generate PDF
  await page.pdf({
    path: outputPath,
    format: 'A4',
    margin: { top: '15mm', bottom: '15mm', left: '15mm', right: '10mm' },
    printBackground: true,
    displayHeaderFooter: false,
  });

  const size = fs.statSync(outputPath).size;
  console.log(`Wrote ${outputPath} (${(size/1024).toFixed(0)}K)`);
  await browser.close();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
