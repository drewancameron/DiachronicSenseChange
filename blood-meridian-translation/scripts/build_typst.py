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


def build_chunk(chapter_dir: str, recently_glossed: set) -> tuple[str, set]:
    """Build Typst content for one chunk. Returns (typst_content, glossed_words)."""
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
        gloss_entries = []
        anchors = sorted(gloss_map.keys(), key=len, reverse=True)
        used_positions = set()

        for anchor in anchors:
            if anchor in recently_glossed or anchor in glossed_this_chunk:
                continue
            pos = para.find(anchor)
            if pos < 0:
                continue
            # Check for overlap with already-used positions
            overlap = False
            for up_start, up_end in used_positions:
                if pos < up_end and pos + len(anchor) > up_start:
                    overlap = True
                    break
            if overlap:
                continue

            note = gloss_map[anchor]
            gloss_entries.append((pos, anchor, note))
            used_positions.add((pos, pos + len(anchor)))
            glossed_this_chunk.add(anchor)

        # Sort by position
        gloss_entries.sort(key=lambda x: x[0])

        # Build Typst paragraph with margin glosses
        if gloss_entries:
            note_lines = []
            for _, anchor, note in gloss_entries:
                # Escape special Typst characters
                anchor_esc = anchor.replace('[', '\\[').replace(']', '\\]').replace('#', '\\#')
                note_esc = note.replace('[', '\\[').replace(']', '\\]').replace('#', '\\#')
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

// Paragraph with margin glosses — grid-based layout
#let glossed-para(body, notes) = {
  grid(
    columns: (1fr, 4.5cm),
    column-gutter: 0.5cm,
    [#set par(first-line-indent: 1.5em, leading: 0.9em)
     #body],
    text(size: 6.5pt, fill: luma(80), notes),
  )
  v(0.4em)
}

// Plain paragraph (no glosses)
#let plain-para(body) = {
  grid(
    columns: (1fr, 4.5cm),
    column-gutter: 0.5cm,
    [#set par(first-line-indent: 1.5em, leading: 0.9em)
     #body],
    [],
  )
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
    RECENCY_WINDOW = 10
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
        print("Compiling with typst...")
        pdf_path = typ_path.with_suffix(".pdf")
        result = subprocess.run(
            ["typst", "compile", str(typ_path), str(pdf_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print(f"Wrote {pdf_path}")
        else:
            print(f"typst errors:")
            print(result.stderr[:500])


if __name__ == "__main__":
    main()
