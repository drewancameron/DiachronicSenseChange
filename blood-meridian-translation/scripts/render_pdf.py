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


def _ge_class(g: dict) -> str:
    t = g.get("_type", "")
    if t == "echo":
        return "echo"
    elif t == "attestation":
        return "attest"
    return ""


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

        # Load echoes and attestations
        echoes_path = APPARATUS / passage_id / "echoes.json"
        attestations_path = APPARATUS / passage_id / "thematic_attestations.json"
        echoes = json.load(open(echoes_path)) if echoes_path.exists() else []
        attestations = json.load(open(attestations_path)) if attestations_path.exists() else []

        for mg_sent in marginal["sentences"]:
            grk = mg_sent["greek"]
            glosses = list(mg_sent.get("glosses", []))

            # Merge echoes — keep full data for footnote rendering
            for echo in echoes:
                phrase = echo.get("greek", "")
                if phrase and phrase[:15] in grk:
                    glosses.append({
                        "anchor": phrase[:25],
                        "source": echo.get("source", ""),
                        "source_quote": echo.get("source_quote", ""),
                        "note": echo.get("note", ""),
                        "_type": "echo",
                    })

            # Merge thematic attestations
            for att in attestations:
                word = att.get("word", "")
                if word and word in grk:
                    glosses.append({
                        "anchor": word,
                        "source": f'{att.get("author", "")}, {att.get("work", "")}',
                        "source_quote": "",
                        "note": "",
                        "_type": "attestation",
                    })

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
    gap: 0.4cm;
    margin-bottom: 0.1cm;
    align-items: stretch;  /* gloss column stretches to paragraph height */
    break-inside: auto;
  }}
  .para-text {{
    flex: 1;
    text-align: justify;
    hyphens: auto;
    text-indent: 1.2em;
    break-inside: auto;
  }}
  .para-row:first-child .para-text {{
    text-indent: 0;
  }}
  .para-gloss {{
    width: 6.5cm;
    flex-shrink: 0;
    font-size: 6.5pt;
    line-height: 1.2;
    color: #555;
    padding-left: 0.3cm;
    display: flex;
    flex-direction: column;
    justify-content: space-around;  /* distribute glosses across paragraph height */
  }}
  .ge {{
    margin-bottom: 0.06cm;
    break-inside: avoid;
  }}
  .ge .w {{
    font-weight: 600;
    color: #333;
    font-size: 6pt;
  }}
  .ge .n {{
    color: #666;
  }}
  /* Per-page footnotes using CSS float:footnote */
  .fn-inline {{
    float: footnote;
    font-size: 6pt;
    line-height: 1.3;
    color: #555;
  }}
  .fn-inline .fn-source {{
    font-style: italic;
  }}
  .fn-inline .fn-quote {{
    font-family: 'GFS Didot', serif;
  }}
  .fn-inline::footnote-call {{
    font-size: 5pt;
    vertical-align: super;
    color: #888;
  }}
  .fn-inline::footnote-marker {{
    font-weight: 600;
    color: #666;
  }}
  @page {{
    @footnote {{
      border-top: 0.5pt solid #ccc;
      padding-top: 0.2cm;
    }}
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

    all_footnotes = []
    fn_counter = 1

    for i, para in enumerate(paragraphs):
        text = " ".join(para["text_parts"])

        # Separate vocab glosses from footnotes
        gloss_html = ""
        for g in para["glosses"]:
            if g.get("_type") in ("echo", "attestation"):
                phrase = g.get("anchor", "")[:15]
                source = g.get("source", "")
                source_quote = g.get("source_quote", "")
                note = g.get("note", "")
                fn_content = f'<span class="fn-source">{source}</span>'
                if source_quote:
                    fn_content += f' — <span class="fn-quote">{source_quote}</span>'
                if note:
                    fn_content += f' ({note})'
                fn_span = f'<span class="fn-inline">{fn_content}</span>'
                if phrase and phrase in text:
                    pos = text.find(phrase)
                    end = text.find(" ", pos + len(phrase))
                    if end < 0:
                        end = len(text)
                    text = text[:end] + fn_span + text[end:]
                g["_number"] = fn_counter
                all_footnotes.append(g)
                fn_counter += 1
            else:
                if g.get("rank", 1) > 2:
                    continue
                gloss_html += (
                    f'<div class="ge">'
                    f'<span class="w">{g["anchor"]}</span> '
                    f'<span class="n">{g["note"]}</span>'
                    f'</div>'
                )

        indent = "" if i == 0 else ' style="text-indent: 1.2em;"'

        html += f"""<div class="para-row">
  <div class="para-text"{indent}>{text}</div>
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
