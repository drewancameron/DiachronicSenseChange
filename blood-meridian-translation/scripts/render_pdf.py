#!/usr/bin/env python3
"""
Render blood_meridian to PDF using WeasyPrint.

Since WeasyPrint doesn't execute JS, we generate a PDF-specific HTML
where glosses appear as simple right-column entries alongside their
paragraph, using a two-column CSS grid that WeasyPrint handles natively.

Usage:
  python3 scripts/render_pdf.py
  python3 scripts/render_pdf.py -o output/custom.pdf
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"
OUTPUT = ROOT / "output"


def italicise_loans(greek: str) -> str:
    return re.sub(r'\*(\S+)', r'<i>\1</i>', greek)


def build_pdf_html(passage_ids: list[str]) -> str:
    """Build a PDF-optimised HTML with static two-column layout."""

    # Gather all paragraphs with their glosses
    paragraphs = []  # list of {"text_parts": [...], "glosses": [...]}

    for passage_id in passage_ids:
        primary_path = DRAFTS / passage_id / "primary.txt"
        glosses_path = APPARATUS / passage_id / "marginal_glosses.json"
        if not primary_path.exists():
            continue

        draft_text = primary_path.read_text("utf-8").strip()
        draft_paras = [p.strip() for p in draft_text.split("\n\n") if p.strip()]

        if glosses_path.exists():
            marginal = json.load(open(glosses_path))
        else:
            sents = [s.strip() for s in re.split(r'(?<=[.;·!])\s+', draft_text) if s.strip()]
            marginal = {"sentences": [{"index": i, "greek": s, "glosses": []} for i, s in enumerate(sents)]}

        # Map sentences to paragraphs
        para_starters = set()
        for para in draft_paras:
            first_sent = re.split(r'(?<=[.;·!])\s+', para)
            if first_sent:
                para_starters.add(first_sent[0].strip())

        current_para_sents = []
        current_para_glosses = []

        for mg_sent in marginal["sentences"]:
            grk = mg_sent["greek"]
            glosses = mg_sent.get("glosses", [])

            # New paragraph?
            if grk in para_starters and current_para_sents:
                paragraphs.append({
                    "text_parts": current_para_sents,
                    "glosses": current_para_glosses,
                })
                current_para_sents = []
                current_para_glosses = []

            current_para_sents.append(italicise_loans(grk))
            for g in glosses:
                if g.get("note"):
                    current_para_glosses.append(g)

        if current_para_sents:
            paragraphs.append({
                "text_parts": current_para_sents,
                "glosses": current_para_glosses,
            })

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=GFS+Didot&display=swap');

  @page {{
    size: A4;
    margin: 2.5cm 1.5cm 2.5cm 1.5cm;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'GFS Didot', 'Times New Roman', serif;
    color: #1a1a1a;
    font-size: 10.5pt;
    line-height: 1.9;
  }}

  /* Title page */
  .title-page {{
    text-align: center;
    padding-top: 8cm;
    page-break-after: always;
  }}
  .title-page .author {{ font-size: 9pt; letter-spacing: 0.2em; color: #333; margin-bottom: 1.5cm; }}
  .title-page .title {{ font-size: 22pt; line-height: 1.3; margin-bottom: 0.3cm; }}
  .title-page .sub {{ font-size: 10pt; color: #333; margin-bottom: 1cm; }}
  .title-page .note {{ font-size: 8pt; color: #666; }}

  .chapter-num {{
    font-size: 14pt; text-align: center;
    margin: 0 0 1cm; color: #333; letter-spacing: 0.15em;
  }}

  /* Each paragraph row: text left, glosses right */
  .para-row {{
    display: flex;
    gap: 1.2cm;
    margin-bottom: 0.15cm;
  }}
  .para-text {{
    flex: 1;
    text-align: justify;
    hyphens: auto;
    text-indent: 1.2em;
  }}
  .para-row:first-child .para-text {{
    text-indent: 0;
  }}
  .para-gloss {{
    width: 5.5cm;
    flex-shrink: 0;
    font-size: 7pt;
    line-height: 1.3;
    color: #555;
    border-left: 0.5pt solid #d0ccc0;
    padding-left: 0.4cm;
    column-count: 2;
    column-gap: 0.3cm;
  }}
  .para-gloss .ge {{
    margin-bottom: 0.1cm;
    break-inside: avoid;
  }}
  .para-gloss .w {{
    font-weight: 600;
    color: #333;
  }}
  .para-gloss .n {{
    color: #666;
  }}
</style>
</head>
<body>

<div class="title-page">
  <div class="author">ΚΟΡΜΑΚ ΜΑΚΚΑΡΘΥ</div>
  <div class="title">Ὁ Αἱματόεις<br>Μεσημβρινός</div>
  <div class="sub">ἢ<br>Τὸ Ἑσπέριον Ἐρύθημα</div>
  <div class="note">εἰς τὴν Ἑλλάδα φωνὴν μεταφρασθέν</div>
</div>

<div class="chapter-num">Ι</div>
"""

    # Display-time density cap for PDF: limit glosses per paragraph
    # to avoid the gloss column overwhelming the text column.
    PDF_MAX_GLOSSES_PER_PARA = 6

    for i, para in enumerate(paragraphs):
        text = " ".join(para["text_parts"])
        gloss_html = ""
        displayed = 0
        for g in para["glosses"]:
            if displayed >= PDF_MAX_GLOSSES_PER_PARA:
                break
            anchor = g["anchor"]
            note = g["note"]
            displayed += 1
            gloss_html += (
                f'<div class="ge">'
                f'<span class="w">{anchor}</span> '
                f'<span class="n">{note}</span>'
                f'</div>'
            )

        html += f"""<div class="para-row">
  <div class="para-text">{text}</div>
  <div class="para-gloss">{gloss_html}</div>
</div>
"""

    html += "</body></html>"
    return html


def render_pdf(pdf_path: Path, passage_ids: list[str]):
    from weasyprint import HTML

    print("  Building PDF-optimised HTML...")
    html_str = build_pdf_html(passage_ids)

    # Save intermediate HTML for debugging
    debug_path = OUTPUT / "blood_meridian_pdf.html"
    debug_path.write_text(html_str, encoding="utf-8")

    print("  Rendering PDF with WeasyPrint...")
    HTML(string=html_str, base_url=str(ROOT)).write_pdf(str(pdf_path))

    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Wrote {pdf_path.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path,
                        default=OUTPUT / "blood_meridian.pdf")
    args = parser.parse_args()

    passage_ids = sorted(
        d.name for d in DRAFTS.iterdir()
        if d.is_dir() and (d / "primary.txt").exists()
    )

    render_pdf(args.output, passage_ids)


if __name__ == "__main__":
    main()
