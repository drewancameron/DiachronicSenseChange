#!/usr/bin/env python3
"""
Render translated passages as an Ørberg-style PDF via Typst.

Produces a .typ file with:
  - Greek text in the main column
  - Marginal glosses using Typst margin notes (auto-positioned per page)
  - Footnote apparatus for echoes/attestations
  - Proper pagination with glosses following their anchor words

Usage:
  python3 scripts/render_pdf.py          # generate .typ and compile to PDF
  python3 scripts/render_pdf.py --typ    # generate .typ only (no compile)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

MAX_APPARATUS_PER_PARA = 2


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def escape_typst(s: str) -> str:
    """Escape special Typst characters in text."""
    s = s.replace("\\", "\\\\")
    s = s.replace("#", "\\#")
    s = s.replace("$", "\\$")
    s = s.replace("@", "\\@")
    s = s.replace("<", "\\<")
    s = s.replace(">", "\\>")
    # '=' at start of content creates a heading in Typst
    if s.startswith("="):
        s = "\\=" + s[1:]
    return s


def load_passages(passage_ids: list[str]) -> list[dict]:
    """Load all paragraphs with their glosses and footnotes."""
    all_sentences = []

    for passage_id in passage_ids:
        primary_path = DRAFTS / passage_id / "primary.txt"
        glosses_path = APPARATUS / passage_id / "marginal_glosses.json"
        if not primary_path.exists():
            continue

        if glosses_path.exists():
            marginal = load_json(glosses_path)
        else:
            text = primary_path.read_text("utf-8").strip()
            sents = [s.strip() for s in re.split(r'(?<=[.;·!])\s+', text) if s.strip()]
            marginal = {"sentences": [{"index": i, "greek": s, "glosses": []} for i, s in enumerate(sents)]}

        draft_text = primary_path.read_text("utf-8").strip()
        paragraphs_raw = [p.strip() for p in draft_text.split("\n\n") if p.strip()]

        para_starters = set()
        for para in paragraphs_raw:
            first_sent = re.split(r'(?<=[.;·!])\s+', para)
            if first_sent:
                para_starters.add(first_sent[0].strip())

        echoes_path = APPARATUS / passage_id / "echoes.json"
        attestations_path = APPARATUS / passage_id / "thematic_attestations.json"
        echoes = []
        attestations = []
        if echoes_path.exists():
            echoes = json.load(open(echoes_path))
        if attestations_path.exists():
            attestations = json.load(open(attestations_path))

        for mg_sent in marginal["sentences"]:
            grk = mg_sent["greek"]
            glosses = [g for g in mg_sent["glosses"] if g.get("rank", 1) <= 2]

            sent_footnotes = []
            for echo in echoes:
                phrase = echo.get("greek", "")
                if phrase and phrase[:15] in grk:
                    sent_footnotes.append(echo)
            for att in attestations:
                word = att.get("word", "")
                if word and word in grk:
                    sent_footnotes.append({
                        "greek": word,
                        "source": f'{att.get("author", "")}, {att.get("work", "")}',
                        "source_quote": "",
                        "note": "",
                        "_type": "attestation",
                    })

            if grk in para_starters and all_sentences and not all_sentences[-1].get("para_break"):
                if any(not item.get("para_break") for item in all_sentences):
                    all_sentences.append({"para_break": True})

            all_sentences.append({
                "greek": grk,
                "glosses": glosses,
                "footnotes": sent_footnotes,
            })
        all_sentences.append({"para_break": True})

    # Group into paragraphs
    paragraphs = []
    cur = {"sentences": [], "glosses": [], "footnotes": []}
    for item in all_sentences:
        if item.get("para_break"):
            if cur["sentences"]:
                paragraphs.append(cur)
            cur = {"sentences": [], "glosses": [], "footnotes": []}
            continue
        cur["sentences"].append(item["greek"])
        cur["glosses"].extend(item["glosses"])
        for fn in item.get("footnotes", []):
            if fn.get("source_quote") or (fn.get("note") and len(fn.get("note", "")) > 10):
                cur["footnotes"].append(fn)
    if cur["sentences"]:
        paragraphs.append(cur)

    return paragraphs


def render_typst(passage_ids: list[str]) -> str:
    """Generate a Typst document string."""
    paragraphs = load_passages(passage_ids)

    typ = []

    # Preamble — per-sentence grid with gloss column
    typ.append(r"""#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2cm, right: 1.5cm),
  numbering: "1",
)
#set text(font: "Didot", size: 11pt, lang: "el")
#set par(justify: true, leading: 0.85em)

// Smaller footnotes for apparatus
#show footnote.entry: set text(size: 6.5pt)

// Gloss entry in the margin column
#let gloss(anchor, note) = {
  block(above: 1.5pt, below: 0pt,
    text(size: 7pt)[#strong[#anchor] #text(fill: rgb("#555"))[#note]]
  )
}
""")

    # Title page
    typ.append(r"""
#align(center + horizon)[
  #text(size: 9pt, tracking: 0.2em, fill: rgb("#333"))[ΚΟΡΜΑΚ ΜΑΚΚΑΡΘΥ]
  #v(1.5cm)
  #text(size: 22pt)[Ὁ Αἱματόεις\ Μεσημβρινός]
  #v(0.3cm)
  #text(size: 10pt, fill: rgb("#333"))[ἢ\ Τὸ Ἑσπέριον Ἐρύθημα]
  #v(1cm)
  #text(size: 8pt, fill: rgb("#666"))[εἰς τὴν Ἑλλάδα φωνὴν μεταφρασθέν]
]
#pagebreak()

#v(2cm)
#align(center)[#text(size: 14pt, tracking: 0.15em, fill: rgb("#333"))[Ι]]
#v(1cm)
""")

    # Render paragraphs as grid rows: text (left) + glosses (right)
    first_para = True

    for para in paragraphs:
        greek_text = " ".join(para["sentences"])

        # Handle loans: *word → italic
        greek_text = re.sub(r'\*(\S+)', r'_\1_', greek_text)

        # Escape Typst specials but preserve _word_ italics
        greek_text = greek_text.replace("_", "\x00ITAL\x00")
        greek_text = escape_typst(greek_text)
        greek_text = greek_text.replace("\x00ITAL\x00", "_")

        # Add footnotes for echoes/attestations
        footnotes = para["footnotes"]
        if footnotes:
            seen_phrases = {}
            for fn in footnotes:
                phrase = fn.get("greek", "")[:20]
                existing = seen_phrases.get(phrase)
                if not existing or len(fn.get("source_quote", "")) > len(existing.get("source_quote", "")):
                    seen_phrases[phrase] = fn
            deduped = list(seen_phrases.values())
            ranked = sorted(deduped, key=lambda f: (
                f.get("rank", 2),
                -len(f.get("source_quote", "")),
                -len(f.get("note", "")),
            ))
            selected = ranked[:MAX_APPARATUS_PER_PARA]

            for fn in selected:
                phrase = fn.get("greek", "").strip()
                source = escape_typst(fn.get("source", ""))
                source_quote = escape_typst(fn.get("source_quote", ""))
                note = escape_typst(fn.get("note", ""))

                fn_body = f'_{source}_'
                if source_quote:
                    fn_body += f': {source_quote}'
                if note:
                    fn_body += f' — {note}'

                phrase_esc = escape_typst(phrase[:15])
                pos = greek_text.find(phrase_esc)
                if pos >= 0:
                    end = pos + len(phrase_esc)
                    while end < len(greek_text) and greek_text[end] not in ' .,;·!?\n':
                        end += 1
                    greek_text = greek_text[:end] + f'#footnote[{fn_body}]' + greek_text[end:]

        # Build gloss column entries
        gloss_lines = []
        seen_anchors = set()
        for g in para["glosses"]:
            anchor = g["anchor"]
            if anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            note = g["note"]
            note = note.replace("_", "\x00ITAL\x00")
            note = escape_typst(note)
            note = note.replace("\x00ITAL\x00", "_")
            anchor_esc = escape_typst(anchor)
            gloss_lines.append(f'    #gloss[{anchor_esc}][{note}]')

        gloss_col = "\n".join(gloss_lines) if gloss_lines else ""

        # Paragraph indent
        indent = "0pt" if first_para else "1.5em"
        first_para = False

        # Emit as grid row: text | glosses
        typ.append(f"""#block(spacing: 0.4em, grid(
  columns: (1fr, 4.5cm),
  column-gutter: 0.5cm,
  [#par(first-line-indent: {indent})[{greek_text}]],
  [#set par(leading: 0.25em, justify: false)
{gloss_col}
  ],
))""")

    return "\n".join(typ)


def main():
    typ_only = "--typ" in sys.argv

    passage_ids = sorted(
        d.name for d in DRAFTS.iterdir()
        if d.is_dir() and (d / "primary.txt").exists()
    )

    typst_content = render_typst(passage_ids)
    typ_path = OUTPUT / "blood_meridian.typ"
    typ_path.write_text(typst_content, encoding="utf-8")
    print(f"Wrote {typ_path}")

    if not typ_only:
        pdf_path = OUTPUT / "blood_meridian.pdf"
        result = subprocess.run(
            ["typst", "compile", str(typ_path), str(pdf_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"Wrote {pdf_path}")
        else:
            print(f"Typst compile error:\n{result.stderr[:2000]}")
            sys.exit(1)


if __name__ == "__main__":
    main()
