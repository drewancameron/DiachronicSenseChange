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
    GLOSS_LINE_HT = 10  # pt per gloss line (7pt font + 3pt gap)
    CHARS_PER_TEXT_LINE = 55  # chars per text line (with wide margin)
    CHARS_PER_GLOSS_LINE = 30  # chars per gloss margin line (4cm wide, small font)

    parts = []
    margin_used = 0  # pt of margin space consumed so far (relative to text)

    for para in paragraphs:
        para_tex = tex_escape(para)

        # Estimate text height of this paragraph
        text_lines = max(1, len(para) // CHARS_PER_TEXT_LINE + 1)
        text_ht = text_lines * TEXT_LINE_HT + 7  # +7pt for parskip

        # Find which glosses appear in this paragraph
        para_glosses = []
        anchors = sorted(gloss_map.keys(), key=len, reverse=True)
        for anchor in anchors:
            anchor_esc = tex_escape(anchor)
            if anchor_esc in para_tex:
                note = gloss_map[anchor]
                note_esc = tex_escape(note)
                para_glosses.append(f"\\textbf{{{anchor_esc}}} {note_esc}")

        if para_glosses:
            # Cap glosses: gloss block should not be taller than the text
            # At ~3 gloss lines per entry in two columns, each entry pair = ~30pt
            # Cap to fit within text_ht
            max_pairs = max(2, int(text_ht / 30))
            max_glosses = max_pairs * 2
            if len(para_glosses) > max_glosses:
                para_glosses = para_glosses[:max_glosses]

            # Estimate gloss block height
            # Each column is ~16 chars wide at 6.5pt in 4.5cm/2
            col1_lines = 0
            col2_lines = 0
            for i, g in enumerate(para_glosses):
                entry_lines = max(2, (len(g) + 8) // 14)  # conservative: 14 chars/line
                if i % 2 == 0:
                    col1_lines += entry_lines
                else:
                    col2_lines += entry_lines
            gloss_lines = max(col1_lines, col2_lines)
            gloss_ht = gloss_lines * GLOSS_LINE_HT + 8  # generous gap

            # Offset to avoid overlapping with previous paragraph's glosses
            # Add a minimum 8pt gap between blocks
            offset = max(0, margin_used + 8) if margin_used > 0 else 0

            # Split glosses into two columns (alternate)
            col1 = []
            col2 = []
            for i, g in enumerate(para_glosses):
                if i % 2 == 0:
                    col1.append(g)
                else:
                    col2.append(g)

            if col2:
                # Two-column layout with computed offset
                c1_tex = "\\\\[3pt]".join(col1)
                c2_tex = "\\\\[3pt]".join(col2)
                parts.append(
                    f"\\glossblock{{{c1_tex}}}{{{c2_tex}}}{{{offset}pt}}%\n{para_tex}\n"
                )
            else:
                # Single column with computed offset
                block = "\\\\[3pt]".join(para_glosses)
                parts.append(
                    f"\\glossblocksingle{{{block}}}{{{offset}pt}}%\n{para_tex}\n"
                )

            margin_used = offset + gloss_ht - text_ht
        else:
            parts.append(para_tex + "\n")
            # No glosses: text passes, reducing margin debt
            margin_used = max(0, margin_used - text_ht)

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
  bottom=2cm,
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

\\pagestyle{{plain}}

\\title{{\\Huge Ὁ Αἱματόεις Μεσημβρινός\\\\[0.3em]
\\large ἢ Τὸ Ἑσπέριον Ἐρύθημα\\\\[0.5em]
\\normalsize εἰς τὴν Ἑλλάδα φωνὴν μεταφρασθέν}}
\\author{{ΚΟΡΜΑΚ ΜΑΚΚΑΡΘΥ}}
\\date{{}}

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
