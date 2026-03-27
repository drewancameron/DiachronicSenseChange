#!/usr/bin/env python3
"""
Render Blood Meridian translation to PDF via Typst.

Typst natively handles:
  - Per-page footnotes with automatic numbering
  - Margin notes (glosses) aligned to text position
  - Proper paragraph flow across page breaks
  - Polytonic Greek fonts

Usage:
  python3 scripts/render_typst.py
  python3 scripts/render_typst.py -o output/custom.pdf
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"
PASSAGES = ROOT / "passages"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


def escape_typst(text: str) -> str:
    """Escape special Typst characters."""
    # Typst uses # for code, * for bold, _ for italic, etc.
    text = text.replace("\\", "\\\\")
    text = text.replace("#", "\\#")
    text = text.replace("*", "\\*")
    text = text.replace("_", "\\_")
    text = text.replace("@", "\\@")
    text = text.replace("$", "\\$")
    text = text.replace("<", "\\<")
    text = text.replace(">", "\\>")
    return text


def build_typst(passage_ids: list[str]) -> str:
    """Build a Typst document from passages."""

    doc = []

    # Page setup — use grid layout for text + glosses
    doc.append(r"""
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2cm, right: 1.5cm),
  numbering: "1",
)
#set text(font: "GFS Didot", size: 10.5pt, lang: "el")
#set par(justify: true, leading: 0.9em)
""")

    # Title page
    doc.append("""
#align(center + horizon)[
  #text(size: 9pt, tracking: 0.2em, fill: rgb("#333333"))[ΚΟΡΜΑΚ ΜΑΚΚΑΡΘΥ]
  #v(1.5cm)
  #text(size: 22pt)[Ὁ Αἱματόεις\\ Μεσημβρινός]
  #v(0.3cm)
  #text(size: 10pt, fill: rgb("#333333"))[ἢ\\ Τὸ Ἑσπέριον Ἐρύθημα]
  #v(1cm)
  #text(size: 8pt, fill: rgb("#666666"))[εἰς τὴν Ἑλλάδα φωνὴν μεταφρασθέν]
]
#pagebreak()
""")

    # Chapter number
    doc.append("""
#v(2cm)
#align(center)[#text(size: 14pt, tracking: 0.15em, fill: rgb("#333333"))[Ι]]
#v(1cm)

// Begin two-column grid: text (left, flexible) + glosses (right, fixed)
#grid(
  columns: (1fr, 4.5cm),
  column-gutter: 0.8cm,
""")

    # Process each passage
    for passage_id in passage_ids:
        primary_path = DRAFTS / passage_id / "primary.txt"
        glosses_path = APPARATUS / passage_id / "marginal_glosses.json"
        echoes_path = APPARATUS / passage_id / "echoes.json"

        if not primary_path.exists():
            continue

        draft_text = primary_path.read_text("utf-8").strip()
        paragraphs = [p.strip() for p in draft_text.split("\n\n") if p.strip()]

        # Load glosses
        if glosses_path.exists():
            marginal = json.load(open(glosses_path))
        else:
            marginal = {"sentences": []}

        # Load echoes
        echoes = []
        if echoes_path.exists():
            echoes = json.load(open(echoes_path))

        # Build a map: sentence text → glosses
        sent_glosses = {}
        for mg_sent in marginal.get("sentences", []):
            grk = mg_sent.get("greek", "")
            gls = mg_sent.get("glosses", [])
            if grk and gls:
                sent_glosses[grk[:30]] = gls

        # Build a map: phrase prefix → echo
        echo_map = {}
        for echo in echoes:
            phrase = echo.get("greek", "")
            if phrase:
                echo_map[phrase[:15]] = echo

        # Render each paragraph
        for para_idx, para in enumerate(paragraphs):
            sents = re.split(r'(?<=[.·;!])\s+', para)
            sents = [s.strip() for s in sents if s.strip()]

            # Collect all glosses for this paragraph
            para_gloss_entries = []

            para_typst = []
            for sent in sents:
                sent_escaped = escape_typst(sent)

                # Handle *-marked loanwords → italic
                sent_escaped = re.sub(
                    r'\\\*(\S+)',
                    r'#text(style: "italic")[\1]',
                    sent_escaped
                )

                # Find glosses for this sentence
                matched_glosses = []
                for key, gls in sent_glosses.items():
                    if sent.startswith(key[:25]) or key.startswith(sent[:25]):
                        matched_glosses = gls
                        break

                # Collect glosses (don't insert inline)
                for g in matched_glosses:
                    rank = g.get("rank", 1)
                    if rank > 2:
                        continue
                    anchor = escape_typst(g.get("anchor", ""))
                    gnote = escape_typst(g.get("note", ""))
                    if anchor:
                        para_gloss_entries.append((anchor, gnote))

                # Insert footnotes for echoes (these are inline)
                for prefix, echo in echo_map.items():
                    if prefix in sent:
                        source = escape_typst(echo.get("source", ""))
                        source_quote = escape_typst(echo.get("source_quote", ""))
                        note = escape_typst(echo.get("note", ""))

                        fn_body = f'#emph[{source}]'
                        if source_quote:
                            fn_body += f' — {source_quote}'
                        if note:
                            fn_body += f' ({note})'

                        esc_prefix = escape_typst(prefix)
                        if esc_prefix in sent_escaped:
                            pos = sent_escaped.find(esc_prefix) + len(esc_prefix)
                            end = sent_escaped.find(" ", pos)
                            if end < 0:
                                end = len(sent_escaped)
                            sent_escaped = (
                                sent_escaped[:end] +
                                f'#footnote[{fn_body}]' +
                                sent_escaped[end:]
                            )

                para_typst.append(sent_escaped)

            # Join sentences
            full_para = " ".join(para_typst)

            # Grid cell 1: text
            if para_idx == 0:
                doc.append(f'  [#par(first-line-indent: 0pt)[{full_para}]],')
            else:
                doc.append(f'  [{full_para}],')

            # Grid cell 2: glosses
            if para_gloss_entries:
                gloss_lines = []
                for anchor, gnote in para_gloss_entries:
                    gloss_lines.append(
                        f'    #block(spacing: 0.25em)[#text(weight: "bold", fill: rgb("#333"), size: 6.5pt)[{anchor}] #text(fill: rgb("#666"), size: 6.5pt)[{gnote}]]'
                    )
                doc.append('  [#set text(size: 7pt)\n  #set par(leading: 0.35em, justify: false)\n' + '\n'.join(gloss_lines) + '\n  ],')
            else:
                doc.append('  [],')  # empty gloss cell

    # Close the grid
    doc.append(")")
    return "\n".join(doc)


def render_pdf(pdf_path: Path, passage_ids: list[str]):
    """Build Typst source and compile to PDF."""
    print("  Building Typst source...")
    typst_src = build_typst(passage_ids)

    # Write intermediate .typ file
    typ_path = OUTPUT / "blood_meridian.typ"
    typ_path.write_text(typst_src, encoding="utf-8")

    print("  Compiling with Typst...")
    result = subprocess.run(
        ["typst", "compile", str(typ_path), str(pdf_path)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  Typst errors:\n{result.stderr}")
        # Try to compile anyway — warnings are ok
        if not pdf_path.exists():
            print("  ✗ PDF not generated")
            return

    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Wrote {pdf_path.name} ({size_mb:.1f} MB)")


def main():
    import argparse
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
