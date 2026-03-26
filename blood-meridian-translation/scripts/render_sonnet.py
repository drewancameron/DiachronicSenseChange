#!/usr/bin/env python3
"""Quick renderer for the Sonnet comparison drafts — same layout, no glosses."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts_sonnet"
OUTPUT = ROOT / "output"

def italicise_loans(greek: str) -> str:
    return re.sub(r'\*(\S+)', r'<i>\1</i>', greek)

def main():
    files = sorted(DRAFTS.glob("*.txt"))
    paras = []
    for f in files:
        text = f.read_text("utf-8").strip()
        text = italicise_loans(text)
        # Split on double newlines for paragraph breaks within a passage
        for block in text.split("\n\n"):
            block = block.replace("\n", " ").strip()
            if block:
                paras.append(block)

    html = """<!DOCTYPE html>
<html lang="el"><head><meta charset="utf-8">
<title>Ὁ Αἱματόεις Μεσημβρινός — Sonnet</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=GFS+Didot&display=swap');
@page { size: A4; margin: 3cm; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'GFS Didot', serif; max-width: 620px; margin: 0 auto; padding: 3rem 2rem; line-height: 2.1; }
.title-page { text-align: center; margin: 4rem 0 5rem; page-break-after: always; }
.chapter-num { font-size: 1.6rem; text-align: center; margin: 0 0 1.8rem; color: #333; letter-spacing: 0.15em; }
.main-text { font-size: 1.15rem; text-align: justify; }
.para { text-indent: 1.5em; margin-bottom: 0.6rem; }
.para:first-child { text-indent: 0; }
.tag { position: fixed; top: 1rem; right: 1rem; background: #EF4444; color: #fff; padding: 0.3rem 0.8rem; font-size: 0.75rem; border-radius: 4px; font-family: sans-serif; }
</style></head><body>
<div class="tag">SONNET</div>
<div class="title-page">
  <div style="font-size:0.95rem;letter-spacing:0.2em;color:#555;margin-bottom:2rem">ΚΟΡΜΑΚ ΜΑΚΚΑΡΘΥ</div>
  <div style="font-size:2.4rem;line-height:1.2;margin-bottom:0.6rem">Ὁ Αἱματόεις<br>Μεσημβρινός</div>
  <div style="font-size:1rem;color:#555;margin-bottom:2rem">ἢ<br>Τὸ Ἑσπέριον Ἐρύθημα</div>
  <div style="font-size:0.85rem;color:#888;margin-top:3rem">εἰς τὴν Ἑλλάδα φωνὴν μεταφρασθέν</div>
</div>
<div class="chapter-num">Ι</div>
<div class="main-text">
"""
    for i, p in enumerate(paras):
        html += f'<p class="para">{p}</p>\n'

    html += "</div></body></html>"

    out = OUTPUT / "blood_meridian_sonnet.html"
    out.write_text(html, "utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
