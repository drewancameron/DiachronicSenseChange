#!/usr/bin/env python3
"""Build a Typst file from primary.txt + marginal_glosses.json per chapter.

Typst handles margin notes with proper page-boundary awareness.

Usage:
  python3 scripts/build_typst.py                    # generate .typ
  python3 scripts/build_typst.py --compile           # generate + compile to PDF
"""

import json
import re
import subprocess
import unicodedata
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"
PASSAGES = ROOT / "passages"
OUT_DIR = ROOT / "output"
OUT_TYP = OUT_DIR / "blood_meridian.typ"


def _is_greek(path: Path) -> bool:
    text = path.read_text("utf-8").strip()[:10]
    return bool(text) and "GREEK" in unicodedata.name(text[0], "")


def get_chapter(d: str) -> str:
    p = PASSAGES / f"{d}.json"
    return json.load(open(p)).get("chapter", "?") if p.exists() else "?"


def get_dirs() -> list[str]:
    return sorted(
        d.name for d in APPARATUS.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (DRAFTS / d.name / "primary.txt").exists()
        and _is_greek(DRAFTS / d.name / "primary.txt")
    )


def _get_idf(word):
    """Get IDF score for a Greek word."""
    stripped = "".join(c for c in unicodedata.normalize("NFD", word)
                      if unicodedata.category(c) != "Mn").lower()
    if stripped in _idf_lemma:
        return _idf_lemma[stripped].get("idf", 5)
    if stripped in _idf_form:
        return _idf_form[stripped].get("idf", 5)
    return 12  # unknown = probably rare


_idf_lemma = {}
_idf_form = {}


def _load_idf():
    global _idf_lemma, _idf_form
    if _idf_lemma:
        return
    try:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from auto_gloss import load_idf
        _idf_lemma, _idf_form = load_idf()
    except Exception:
        pass


# Layout constants
CHARS_PER_LINE = 65     # chars per text line (measured from PDF output)
TEXT_LINE_HT = 11.0     # pt per text line (measured from PDF output)
GLOSS_ENTRY_HT = 9.0   # pt per gloss entry (6.5pt font + spacing + bold overhead)


def build_chunk(chapter_dir: str, recently_glossed: set) -> tuple[str, set]:
    """Build Typst content for one chunk. Returns (typst_content, glossed_words)."""
    _load_idf()

    glosses_path = APPARATUS / chapter_dir / "marginal_glosses.json"
    primary_path = DRAFTS / chapter_dir / "primary.txt"

    with open(glosses_path) as f:
        data = json.load(f)
    with open(primary_path) as f:
        paragraphs = [line.strip() for line in f if line.strip()]

    # Build gloss map
    gloss_map = {}
    for s in data["sentences"]:
        for g in s.get("glosses", []):
            anchor = g["anchor"]
            if anchor not in gloss_map:
                gloss_map[anchor] = g["note"]

    parts = []
    glossed_this_chunk = set()

    for para in paragraphs:
        # Find glossable words (not recently glossed)
        gloss_candidates = []
        anchors = sorted(gloss_map.keys(), key=len, reverse=True)
        used_positions = set()

        for anchor in anchors:
            if anchor in recently_glossed or anchor in glossed_this_chunk:
                continue
            pos = para.find(anchor)
            if pos < 0:
                continue
            overlap = False
            for up_start, up_end in used_positions:
                if pos < up_end and pos + len(anchor) > up_start:
                    overlap = True
                    break
            if overlap:
                continue

            note = gloss_map[anchor]
            idf = _get_idf(anchor)
            gloss_candidates.append((idf, pos, anchor, note))
            used_positions.add((pos, pos + len(anchor)))

        # Sort by IDF descending (rarest first)
        gloss_candidates.sort(key=lambda x: -x[0])

        # No Python-side cap — Typst's clip: true handles overflow.
        # IDF sort ensures rarest words appear at top of each block.

        # Re-sort by position for display order
        gloss_entries = [(pos, anchor, note) for _, pos, anchor, note in gloss_candidates]
        gloss_entries.sort(key=lambda x: x[0])

        for _, anchor, _ in gloss_entries:
            glossed_this_chunk.add(anchor)

        # Build Typst paragraph with margin glosses
        if gloss_entries:
            note_lines = []
            for _, anchor, note in gloss_entries:
                # Escape special Typst characters
                anchor_esc = anchor.replace('[', '\\[').replace(']', '\\]').replace('#', '\\#')
                note_esc = note.replace('[', '\\[').replace(']', '\\]').replace('#', '\\#')
                # Truncate long definitions to avoid wrapping
                combined = f"{anchor_esc} {note_esc}"
                if len(combined) > 35:
                    note_esc = note_esc[:35 - len(anchor_esc) - 3] + "…"
                note_lines.append(f"*{anchor_esc}* {note_esc}")

            notes_block = " \\\n  ".join(note_lines)
            para_esc = para.replace('[', '\\[').replace(']', '\\]').replace('#', '\\#')

            parts.append(
                f'#glossed-para(\n  [{para_esc}],\n  [{notes_block}]\n)\n'
            )
        else:
            para_esc = para.replace('[', '\\[').replace(']', '\\]').replace('#', '\\#')
            parts.append(f'#plain-para([{para_esc}])\n')

    return "\n".join(parts), glossed_this_chunk


def build_document():
    """Build the complete Typst document."""
    dirs = get_dirs()
    print(f"Building from {len(dirs)} chunks")

    # Group by chapter
    chapters = OrderedDict()
    for d in dirs:
        ch = get_chapter(d)
        if ch not in chapters:
            chapters[ch] = []
        chapters[ch].append(d)

    # Preamble
    doc = '''// Blood Meridian in Ancient Greek — Typst typeset
#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 2.5cm, right: 2cm),
)

#set text(
  font: "Times New Roman",
  size: 11pt,
  lang: "el",
)

// Glossed paragraph: grid layout, text height constrains gloss column.
// Gloss column is clipped to text height to prevent gaps.
// Python pre-computes how many glosses fit (adaptive constraint).
#let glossed-para(body, notes) = {
  layout(size => {
    let text-block = block(width: size.width - 5cm, [
      #set par(first-line-indent: 1.5em, leading: 0.9em)
      #body
    ])
    let text-height = measure(text-block).height
    let notes-block = text(size: 6.5pt, fill: luma(80), notes)
    let notes-height = measure(block(width: 4.5cm, notes-block)).height
    // Use the taller of the two (but prefer text height)
    let row-height = if notes-height <= text-height {
      text-height
    } else {
      // Glosses taller than text: clip glosses to text height
      text-height
    }
    grid(
      columns: (1fr, 4.5cm),
      column-gutter: 0.5cm,
      rows: (row-height,),
      block(height: row-height, clip: false, text-block),
      block(height: row-height, clip: true, notes-block),
    )
  })
  v(0.4em)
}

// Plain paragraph (no glosses)
#let plain-para(body) = {
  block[#set par(first-line-indent: 1.5em, leading: 0.9em)
    #body]
  v(0.4em)
}

// Title
#align(center)[
  #text(size: 24pt)[Blood Meridian]
  #v(0.3em)
  #text(size: 12pt, style: "italic")[translated into Ancient Greek]
  #v(0.3em)
  #text(size: 10pt)[CORMAC McCARTHY]
]

#pagebreak()

'''

    # Build chapters
    recently_glossed = set()
    RECENCY_WINDOW = 6
    recent_sets = []

    for roman, chunk_dirs in chapters.items():
        print(f"  Chapter {roman}: {len(chunk_dirs)} chunks")

        doc += f'\n#align(center)[#text(size: 18pt)[{roman}]]\n#v(1em)\n\n'

        for d in chunk_dirs:
            content, glossed = build_chunk(d, recently_glossed)
            doc += content + "\n"
            recent_sets.append(glossed)

            # Update recency
            recently_glossed = set()
            for s in recent_sets[-RECENCY_WINDOW:]:
                recently_glossed.update(s)

        doc += "\n#pagebreak()\n\n"

    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    typ_path = Path(args.output) if args.output else OUT_TYP
    typ_path.parent.mkdir(parents=True, exist_ok=True)

    doc = build_document()
    typ_path.write_text(doc)
    print(f"Wrote {typ_path} ({len(doc):,} bytes)")

    if args.compile:
        pdf_path = typ_path.with_suffix(".pdf")
        print("  Compiling with typst...")
        result = subprocess.run(
            ["typst", "compile", str(typ_path), str(pdf_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print(f"Wrote {pdf_path}")
        else:
            print(f"typst errors: {result.stderr[:500]}")


def _check_gaps(pdf_path: str) -> list:
    """Find paragraphs where gloss column is taller than text column."""
    import fitz
    doc = fitz.open(pdf_path)
    gaps = []
    for pn in range(len(doc)):
        page = doc[pn]
        pw = page.rect.width
        mx = pw * 0.6
        blocks = page.get_text("dict")["blocks"]
        tbs = sorted([b for b in blocks if "lines" in b and b["bbox"][0] <= mx],
                     key=lambda b: b["bbox"][1])
        mbs = sorted([b for b in blocks if "lines" in b and b["bbox"][0] > mx],
                     key=lambda b: b["bbox"][1])
        for tb in tbs:
            t_ht = tb["bbox"][3] - tb["bbox"][1]
            for mb in mbs:
                if abs(mb["bbox"][1] - tb["bbox"][1]) < 15:
                    m_ht = mb["bbox"][3] - mb["bbox"][1]
                    if m_ht > t_ht + 5:
                        text = "".join(s["text"] for l in tb["lines"] for s in l["spans"])
                        gaps.append(text[:50])
                    break
    doc.close()
    return gaps


if __name__ == "__main__":
    main()
