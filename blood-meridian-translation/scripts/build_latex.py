#!/usr/bin/env python3
"""Build a LaTeX file from primary.txt + marginal_glosses.json per chapter.

Uses XeLaTeX for Unicode Greek with custom glossmargin.sty for margin glosses.
Glosses are batched per paragraph and emitted as single margin blocks,
avoiding the overlap problems of individual \\marginpar/\\marginnote calls.

Usage:
  python3 scripts/build_latex.py                    # generate .tex
  python3 scripts/build_latex.py --compile          # generate .tex and run xelatex
  python3 scripts/build_latex.py --output out.tex   # custom output path
"""

import json
import os
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
OUT_TEX = OUT_DIR / "blood_meridian.tex"


def _is_greek(path: Path) -> bool:
    text = path.read_text("utf-8").strip()[:10]
    return bool(text) and "GREEK" in unicodedata.name(text[0], "")


def get_chapter(chapter_dir: str) -> str:
    p_path = PASSAGES / f"{chapter_dir}.json"
    if p_path.exists():
        return json.load(open(p_path)).get("chapter", "?")
    return "?"


def get_chapter_dirs() -> list[str]:
    return sorted(
        d.name for d in APPARATUS.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (DRAFTS / d.name / "primary.txt").exists()
        and _is_greek(DRAFTS / d.name / "primary.txt")
    )


def tex_escape(s: str) -> str:
    s = s.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
    s = s.replace("#", "\\#").replace("_", "\\_")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def build_chunk(chapter_dir: str) -> str:
    """Build LaTeX for one chunk with \\gloss commands."""
    glosses_path = APPARATUS / chapter_dir / "marginal_glosses.json"
    primary_path = DRAFTS / chapter_dir / "primary.txt"

    with open(glosses_path) as f:
        data = json.load(f)
    with open(primary_path) as f:
        paragraphs = [line.strip() for line in f if line.strip()]

    sentences = data["sentences"]

    # Build a map of anchor → note for this chunk
    gloss_map = {}  # anchor_text → note
    for s in sentences:
        for g in s.get("glosses", []):
            anchor = g["anchor"]
            note = g["note"]
            if anchor not in gloss_map:
                gloss_map[anchor] = note

    # Process each paragraph: collect glosses, estimate heights, compute offsets
    #
    # Strategy: track a "margin watermark" — how far down the margin is used.
    # Each paragraph's text occupies some height. Each gloss block occupies
    # some height. If the gloss block would extend below the text, push it
    # down (negative offset from paragraph start) or accept overflow.
    #
    # Estimated heights (at our font sizes):
    #   Text line: ~17pt (11pt font × 1.6 stretch)
    #   Gloss entry: ~20pt (7pt font × 2 lines + 2pt gap)
    #   Chars per text line: ~65

    TEXT_LINE_HT = 17  # pt per text line (11pt × 1.6 leading)
    
    CHARS_PER_TEXT_LINE = 55  # chars per text line (with wide margin)
    

    # Track recently glossed words — don't repeat within RECENCY_WINDOW paragraphs
    RECENCY_WINDOW = 8
    recently_glossed = set()
    recent_glosses = []  # list of sets per paragraph

    parts = []

    for para in paragraphs:
        para_tex = tex_escape(para)

        # Update recency window
        recently_glossed = set()
        for s in recent_glosses[-RECENCY_WINDOW:]:
            recently_glossed.update(s)

        glossed_this_para = set()

        # Collect glossable words with their positions in the paragraph
        anchors = sorted(gloss_map.keys(), key=len, reverse=True)
        gloss_positions = []  # (char_position, anchor, note)

        for anchor in anchors:
            if anchor in recently_glossed or anchor in glossed_this_para:
                continue
            anchor_esc = tex_escape(anchor)
            pos = para_tex.find(anchor_esc)
            if pos < 0:
                continue
            note = gloss_map[anchor]
            note_esc = tex_escape(note)
            gloss_positions.append((pos, anchor, anchor_esc, note_esc))
            glossed_this_para.add(anchor)

        # Sort by position in text
        gloss_positions.sort(key=lambda x: x[0])

        # All glosses in a paragraph go into ONE marginpar block
        # placed at the first glossed word. This ensures:
        # 1. One \marginpar per paragraph (stays within float limits)
        # 2. Block aligned with the paragraph start
        # 3. \marginpar collision avoidance handles inter-paragraph stacking

        if len(gloss_positions) >= 1:
            # Build note block (single or merged)
            entries = []
            for _, anchor, anchor_esc, note_esc in gloss_positions:
                entries.append(f"\\textbf{{{anchor_esc}}} {note_esc}")
            merged_note = "\\\\[2pt]".join(entries)

            # Place \marginpar at the FIRST glossed word in the text
            # LaTeX aligns \marginpar with the line where it's called
            first_anchor_esc = gloss_positions[0][2]
            replacement = f"\\glossmerged{{{first_anchor_esc}}}{{{merged_note}}}"
            para_tex = para_tex.replace(first_anchor_esc, replacement, 1)

        recent_glosses.append(glossed_this_para)
        parts.append(para_tex + "\n")

    return "\n".join(parts)


def build_page():
    """Build the complete LaTeX document."""
    chapter_dirs = get_chapter_dirs()
    print(f"Building from {len(chapter_dirs)} chunks")

    # Group by chapter
    chapters = OrderedDict()
    for d in chapter_dirs:
        ch = get_chapter(d)
        if ch not in chapters:
            chapters[ch] = []
        chapters[ch].append(d)

    # Build chapter content
    body = ""
    for roman, dirs in chapters.items():
        print(f"  Chapter {roman}: {len(dirs)} chunks")
        chunks = []
        for d in dirs:
            chunks.append(build_chunk(d))

        body += f"""
\\begin{{center}}
\\Large {roman}
\\end{{center}}
\\vspace{{1em}}

{"".join(chunks)}

\\newpage
"""

    doc = f"""\\documentclass[11pt,a4paper]{{article}}

% XeLaTeX for Unicode
\\usepackage{{fontspec}}
\\usepackage{{polyglossia}}
\\setdefaultlanguage{{greek}}
\\setotherlanguage{{english}}

% Greek font
\\setmainfont{{Times New Roman}}[Script=Greek, Ligatures=TeX]
\\newfontfamily\\greekfont{{Times New Roman}}[Script=Greek]

% Page layout: wide right margin for glosses
\\usepackage[
  a4paper,
  top=2cm,
  bottom=2.5cm,
  left=2.5cm,
  right=5.5cm,
  marginparwidth=4.5cm,
  marginparsep=0.5cm,
]{{geometry}}

% Custom gloss system
\\usepackage{{glossmargin}}

% Line spacing
\\usepackage{{setspace}}
\\setstretch{{1.6}}

% Paragraph formatting
\\setlength{{\\parindent}}{{1.5em}}
\\setlength{{\\parskip}}{{0.4em}}

% Margin note spacing — prevents overflow at page bottom
\\setlength{{\\marginparpush}}{{8pt}}

\\pagestyle{{plain}}

\\title{{Blood Meridian}}
\\author{{Cormac McCarthy}}
\\date{{Translated into Ancient Greek}}

\\begin{{document}}

\\maketitle
\\thispagestyle{{empty}}
\\newpage

{body}

\\end{{document}}
"""

    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    tex_path = Path(args.output) if args.output else OUT_TEX
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    doc = build_page()
    tex_path.write_text(doc)
    print(f"Wrote {tex_path} ({len(doc):,} bytes)")

    if args.compile:
        # Copy .sty to output dir
        sty_src = ROOT / "output" / "glossmargin.sty"
        sty_dst = tex_path.parent / "glossmargin.sty"
        if sty_src.exists() and sty_src != sty_dst:
            import shutil
            shutil.copy2(sty_src, sty_dst)

        print("Compiling with xelatex...")
        xelatex = "/Library/TeX/texbin/xelatex"
        if not os.path.exists(xelatex):
            xelatex = "xelatex"

        for pass_num in range(2):
            result = subprocess.run(
                [xelatex, "-interaction=nonstopmode",
                 f"-output-directory={tex_path.parent}", str(tex_path)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0 and pass_num == 1:
                print("xelatex errors:")
                for line in result.stdout.split("\n")[-20:]:
                    if line.strip():
                        print(f"  {line}")

        pdf_path = tex_path.with_suffix(".pdf")
        if pdf_path.exists():
            print(f"Wrote {pdf_path}")
        else:
            print("PDF not generated — check log")


if __name__ == "__main__":
    main()
