#!/usr/bin/env python3
"""Build docs/index.html from primary.txt + marginal_glosses.json per chapter.

Produces a clean reader edition: Greek text with margin glosses, no apparatus.
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"
OUT = ROOT / "docs" / "index.html"

def _is_greek(path: Path) -> bool:
    """Check if a file starts with Greek text (not an English refusal)."""
    import unicodedata
    text = path.read_text("utf-8").strip()[:10]
    return bool(text) and "GREEK" in unicodedata.name(text[0], "")

# Chapter dirs in order — must have Greek primary.txt and marginal_glosses.json
CHAPTER_DIRS = sorted(
    d.name for d in APPARATUS.iterdir()
    if d.is_dir() and not d.name.startswith("_")
    and (DRAFTS / d.name / "primary.txt").exists()
    and _is_greek(DRAFTS / d.name / "primary.txt")
)

# Roman numerals for chapter headings
ROMAN = [
    "", "Ι", "ΙΙ", "ΙΙΙ", "ΙV", "V", "VΙ", "VΙΙ", "VΙΙΙ", "ΙΧ", "Χ",
    "ΧΙ", "ΧΙΙ", "ΧΙΙΙ", "ΧΙV", "ΧV", "ΧVΙ", "ΧVΙΙ", "ΧVΙΙΙ", "ΧΙΧ", "ΧΧ",
]


def wrap_anchors(sentence_greek: str, glosses: list, gw_offset: int) -> tuple[str, int, list]:
    """Wrap glossed words in <span class="gw"> tags.

    Returns (html_string, number_of_glosses_wrapped, glosses_in_text_order).
    """
    if not glosses:
        return html_escape(sentence_greek), 0, []

    # Sort glosses by position in sentence (first occurrence), longest first for overlaps
    positioned = []
    for g in glosses:
        anchor = g["anchor"]
        idx = sentence_greek.find(anchor)
        if idx == -1:
            # Try case-insensitive / normalized match
            lower = sentence_greek.lower()
            idx = lower.find(anchor.lower())
        if idx >= 0:
            positioned.append((idx, len(anchor), g))

    # Sort by position; for same position, longest first
    positioned.sort(key=lambda x: (x[0], -x[1]))

    # Remove overlapping spans (keep first/longest)
    kept = []
    end = 0
    for pos, length, g in positioned:
        if pos >= end:
            kept.append((pos, length, g))
            end = pos + length

    # Build HTML by slicing the sentence
    parts = []
    cursor = 0
    gw_count = 0
    for pos, length, g in kept:
        # Text before this anchor
        if pos > cursor:
            parts.append(html_escape(sentence_greek[cursor:pos]))
        # The anchor span
        gw_id = gw_offset + gw_count
        anchor_text = sentence_greek[pos:pos + length]
        title = html_escape_attr(g["note"])
        parts.append(
            f'<span class="gw" id="gw{gw_id}" title="{title}">'
            f'{html_escape(anchor_text)}</span>'
        )
        gw_count += 1
        cursor = pos + length

    # Remaining text
    if cursor < len(sentence_greek):
        parts.append(html_escape(sentence_greek[cursor:]))

    return "".join(parts), gw_count, [g for _, _, g in kept]


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def html_escape_attr(s: str) -> str:
    return html_escape(s).replace('"', "&quot;")


def build_chapter(chapter_dir: str, chapter_num: int, gw_offset: int) -> tuple[str, str, int]:
    """Build main-text HTML and gloss-panel HTML for one chapter.

    Returns (main_html, gloss_html, new_gw_offset).
    """
    glosses_path = APPARATUS / chapter_dir / "marginal_glosses.json"
    with open(glosses_path) as f:
        data = json.load(f)

    main_parts = []
    gloss_parts = []
    gw_id = gw_offset

    # Group sentences into paragraphs.
    # primary.txt is one paragraph per line (most chapters are single-paragraph).
    primary_path = DRAFTS / chapter_dir / "primary.txt"
    with open(primary_path) as f:
        paragraphs = [line.strip() for line in f if line.strip()]

    # Walk through sentences, assigning to paragraphs by matching text
    sentences = data["sentences"]
    sent_idx = 0

    for para_text in paragraphs:
        para_html_parts = []
        # Consume sentences that belong to this paragraph
        remaining = para_text
        while sent_idx < len(sentences):
            s = sentences[sent_idx]
            sg = s["greek"]
            # Check if this sentence starts the remaining paragraph text
            if remaining.startswith(sg):
                wrapped, count, ordered_glosses = wrap_anchors(sg, s.get("glosses", []), gw_id)
                # Emit gloss panel entries in text order
                for i, g in enumerate(ordered_glosses):
                    gloss_parts.append(
                        f'<div class="mg" data-for="gw{gw_id + i}">'
                        f'<span class="w">{html_escape(g["anchor"])}</span> '
                        f'<span class="n">{html_escape(g["note"])}</span></div>'
                    )
                gw_id += count
                para_html_parts.append(wrapped)
                remaining = remaining[len(sg):].lstrip()
                sent_idx += 1
            else:
                # Sentence doesn't match — might be a paragraph boundary issue.
                # Try to find it anywhere in remaining
                pos = remaining.find(sg)
                if pos >= 0:
                    # There's unmatched text before it
                    if pos > 0:
                        para_html_parts.append(html_escape(remaining[:pos]))
                    wrapped, count, ordered_glosses = wrap_anchors(sg, s.get("glosses", []), gw_id)
                    for i, g in enumerate(ordered_glosses):
                        gloss_parts.append(
                            f'<div class="mg" data-for="gw{gw_id + i}">'
                            f'<span class="w">{html_escape(g["anchor"])}</span> '
                            f'<span class="n">{html_escape(g["note"])}</span></div>'
                        )
                    gw_id += count
                    para_html_parts.append(wrapped)
                    remaining = remaining[pos + len(sg):].lstrip()
                    sent_idx += 1
                else:
                    break

        # Any remaining unmatched text in the paragraph
        if remaining.strip():
            para_html_parts.append(html_escape(remaining))

        main_parts.append(
            f'<p class="para">{" ".join(para_html_parts)}</p>'
        )

    main_html = "\n".join(main_parts)
    gloss_html = "\n".join(gloss_parts)
    return main_html, gloss_html, gw_id


def get_chapter(chapter_dir: str) -> str:
    """Get the chapter number from the passage JSON."""
    p_path = ROOT / "passages" / f"{chapter_dir}.json"
    if p_path.exists():
        return json.load(open(p_path)).get("chapter", "?")
    return "?"


def build_page():
    """Build the complete HTML page."""
    # Group chunks by chapter
    from collections import OrderedDict
    chapters = OrderedDict()
    gw_offset = 0

    for chapter_dir in CHAPTER_DIRS:
        ch = get_chapter(chapter_dir)
        if ch not in chapters:
            chapters[ch] = {"mains": [], "glosses": []}
        main_html, gloss_html, gw_offset = build_chapter(chapter_dir, 0, gw_offset)
        chapters[ch]["mains"].append(main_html)
        chapters[ch]["glosses"].append(gloss_html)

    # Build page body: one page-body div per chapter
    body_parts = []
    for roman, ch_data in chapters.items():
        all_main = "\n".join(ch_data["mains"])
        all_glosses = "\n".join(ch_data["glosses"])
        body_parts.append(f"""\
<div style="height:2rem"></div>
<div class="chapter-num">{roman}</div>
<div class="page-body">
<div class="main-text">
{all_main}
</div>
<div class="gloss-panel">
{all_glosses}
</div>
</div>""")

    chapters_html = "\n".join(body_parts)

    page = f"""\
<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="utf-8">
<title>Ὁ Αἱματόεις Μεσημβρινός</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=GFS+Didot&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'GFS Didot', 'Times New Roman', serif;
    background: #fff;
    color: #1a1a1a;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
  }}

  /* Title page */
  .title-page {{
    text-align: center; margin: 0; padding: 4rem 2rem;
    min-height: 60vh;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
  }}
  .title-page .author {{ font-size: 0.95rem; letter-spacing: 0.2em; color: #333; margin-top: 2.25rem; margin-bottom: 1.25rem; }}
  .title-page .title {{ font-size: 2.6rem; line-height: 1.2; margin-bottom: 0; color: #1a1a1a; }}
  .title-page .sub {{ font-size: 1.05rem; color: #333; margin-bottom: 1.25rem; }}
  .title-page .note {{ font-size: 0.85rem; color: #666; margin-top: -0.1rem; }}

  .chapter-num {{
    font-size: 1.6rem; text-align: center;
    margin: 0 0 1.5rem; color: #333; letter-spacing: 0.15em;
  }}

  /* Two-panel layout: Greek (left) + glosses (right) */
  .page-body {{
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 0 1.2rem;
    position: relative;
    margin-bottom: 2rem;
  }}

  /* Main Greek text column */
  .main-text {{
    font-size: 1.1rem;
    line-height: 2;
    text-align: justify;
    hyphens: auto;
  }}
  .main-text .para {{
    text-indent: 1.5em;
    margin-bottom: 0.5rem;
  }}
  .main-text .para:first-child {{
    text-indent: 0;
  }}

  /* Gloss panel */
  .gloss-panel {{
    position: relative;
    padding-left: 1rem;
  }}

  .mg {{
    position: absolute;
    width: 100%;
    font-size: 0.75rem;
    line-height: 1.35;
    color: #555;
  }}
  .mg .w {{
    font-weight: 600;
    color: #333;
  }}
  .mg .n {{
    color: #666;
  }}

  .gw {{ cursor: help; }}
  .gw:hover {{ background: #F5F0E0; }}

  @media (max-width: 800px) {{
    .page-body {{ grid-template-columns: 1fr; }}
    .gloss-panel {{ display: none; }}
    /* On mobile, show glosses as tooltips only */
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

{chapters_html}

<div class="fb-tab" onclick="document.querySelector('.fb-panel').classList.toggle('open')">Feedback</div>

<div class="fb-panel">
  <button class="fb-close" onclick="this.parentElement.classList.remove('open')">&times;</button>
  <h3>Feedback</h3>
  <form id="fb-form" action="https://formspree.io/f/xwvwvaop" method="POST">
    <label>What passage does this concern?</label>
    <input type="text" name="passage" placeholder="e.g. paragraph 1, or general">
    <label>Type of feedback</label>
    <select name="type">
      <option>Translation quality</option>
      <option>Grammar error</option>
      <option>Wrong word / sense</option>
      <option>Gloss suggestion</option>
      <option>Display / formatting</option>
      <option>Other</option>
    </select>
    <label>Your feedback</label>
    <textarea name="feedback" required placeholder="Your comments..."></textarea>
    <label>Your name (optional)</label>
    <input type="text" name="name" placeholder="Optional">
    <input type="hidden" name="_subject" value="Blood Meridian Translation Feedback">
    <button type="submit">Send</button>
    <div class="fb-thanks">Thank you for your feedback!</div>
  </form>
</div>

<style>
  .fb-tab {{
    position: fixed; top: 1rem; right: 0;
    background: #444; color: #fff; padding: 0.4rem 0.8rem;
    font-size: 0.75rem; cursor: pointer; border-radius: 4px 0 0 4px;
    writing-mode: horizontal-tb; z-index: 100;
  }}
  .fb-tab:hover {{ background: #666; }}
  .fb-panel {{
    display: none; position: fixed; top: 0; right: 0;
    width: 320px; height: 100vh; background: #fafaf8;
    box-shadow: -2px 0 8px rgba(0,0,0,0.15); z-index: 101;
    padding: 1.5rem; overflow-y: auto;
    font-family: system-ui, sans-serif;
  }}
  .fb-panel.open {{ display: block; }}
  .fb-panel h3 {{ margin: 0 0 0.8rem; font-size: 1rem; color: #333; }}
  .fb-panel label {{ display: block; font-size: 0.8rem; color: #555; margin: 0.6rem 0 0.2rem; }}
  .fb-panel textarea, .fb-panel input, .fb-panel select {{
    width: 100%; padding: 0.4rem; font-size: 0.85rem;
    border: 1px solid #ccc; border-radius: 3px;
  }}
  .fb-panel textarea {{ height: 120px; resize: vertical; }}
  .fb-panel button {{
    margin-top: 0.8rem; padding: 0.5rem 1.2rem;
    background: #444; color: #fff; border: none;
    border-radius: 3px; cursor: pointer; font-size: 0.85rem;
  }}
  .fb-panel button:hover {{ background: #666; }}
  .fb-panel .fb-close {{
    position: absolute; top: 0.5rem; right: 0.8rem;
    background: none; border: none; font-size: 1.2rem;
    color: #999; cursor: pointer; padding: 0;
  }}
  .fb-thanks {{ display: none; color: #4a4; font-size: 0.9rem; margin-top: 1rem; }}
</style>

<script>
// Align margin glosses vertically next to their anchor words
function alignGlosses() {{
  document.querySelectorAll('.page-body').forEach(body => {{
    const panel = body.querySelector('.gloss-panel');
    if (!panel) return;
    const bodyTop = body.getBoundingClientRect().top + window.scrollY;
    const panelWidth = panel.offsetWidth;
    const colWidth = Math.floor((panelWidth - 12) / 2);

    const mgs = panel.querySelectorAll('.mg');
    let col1Bottom = 0;
    let col2Bottom = 0;

    mgs.forEach(mg => {{
      const forId = mg.dataset.for;
      const anchor = document.getElementById(forId);
      if (!anchor) {{ mg.style.display = 'none'; return; }}

      mg.style.width = colWidth + 'px';

      const anchorRect = anchor.getBoundingClientRect();
      let idealTop = anchorRect.top + window.scrollY - bodyTop;

      let top1 = Math.max(idealTop, col1Bottom + 2);
      let top2 = Math.max(idealTop, col2Bottom + 2);

      let useCol1 = true;
      if (Math.abs(top2 - idealTop) < Math.abs(top1 - idealTop)) {{
        useCol1 = false;
      }}

      if (useCol1) {{
        mg.style.top = top1 + 'px';
        mg.style.left = '0px';
        col1Bottom = top1 + mg.offsetHeight;
      }} else {{
        mg.style.top = top2 + 'px';
        mg.style.left = (colWidth + 12) + 'px';
        col2Bottom = top2 + mg.offsetHeight;
      }}
    }});
  }});
}}

window.addEventListener('load', alignGlosses);
window.addEventListener('resize', alignGlosses);

// Feedback form
document.addEventListener('DOMContentLoaded', function() {{
  const form = document.getElementById('fb-form');
  if (form) {{
    form.addEventListener('submit', function(e) {{
      e.preventDefault();
      fetch(this.action, {{method:'POST', body: new FormData(this), headers:{{'Accept':'application/json'}}}})
        .then(r => {{
          if (r.ok) {{
            this.reset();
            this.querySelector('.fb-thanks').style.display = 'block';
            setTimeout(() => this.querySelector('.fb-thanks').style.display = 'none', 3000);
          }}
        }});
    }});
  }}
}});
</script>
</body>
</html>"""

    return page


if __name__ == "__main__":
    print(f"Building from {len(CHAPTER_DIRS)} chapters: {', '.join(CHAPTER_DIRS)}")
    html = build_page()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"Wrote {OUT} ({len(html):,} bytes)")
