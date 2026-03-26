#!/usr/bin/env python3
"""
Render translated passages as an Ørberg-style HTML reading page.
Two-column layout: Greek text (large, left), glosses panel (right).
Greek preserves McCarthy's paragraphing. Glosses aligned by JS to their anchor words.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

gloss_counter = 0


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def italicise_loans(greek: str) -> str:
    return re.sub(r'\*(\S+)', r'<i>\1</i>', greek)


def highlight_anchors(greek: str, glosses: list) -> str:
    global gloss_counter
    # Italicise loans FIRST, before inserting HTML spans
    result = italicise_loans(greek)
    for g in glosses:
        anchor = g["anchor"]
        gid = f"gw{gloss_counter}"
        g["_id"] = gid
        gloss_counter += 1
        italicised_anchor = f"<i>{anchor}</i>"
        note_escaped = g["note"].replace('"', '&quot;')
        if italicised_anchor in result:
            replacement = (
                f'<span class="gw" id="{gid}" '
                f'title="{note_escaped}"><i>{anchor}</i></span>'
            )
            result = result.replace(italicised_anchor, replacement, 1)
        else:
            pattern = re.escape(anchor)
            replacement = (
                f'<span class="gw" id="{gid}" '
                f'title="{note_escaped}">{anchor}</span>'
            )
            result = re.sub(pattern, replacement, result, count=1)
    return result


def _gloss_css_class(g: dict) -> str:
    """Return CSS class for a gloss entry based on its type."""
    t = g.get("_type", "")
    if t == "echo":
        return "echo"
    elif t == "attestation":
        return "attest"
    return ""


def render_passage(passage_ids: list[str]) -> str:
    global gloss_counter
    gloss_counter = 0

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

        # Detect paragraph boundaries from the draft text (blank lines = speaker changes)
        draft_text = primary_path.read_text("utf-8").strip()
        paragraphs = [p.strip() for p in draft_text.split("\n\n") if p.strip()]

        para_starters = set()
        for para in paragraphs:
            first_sent = re.split(r'(?<=[.;·!])\s+', para)
            if first_sent:
                para_starters.add(first_sent[0].strip())

        # Load echoes and thematic attestations for this passage
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
            glosses = list(mg_sent["glosses"])  # copy so we can append

            # Echoes become footnotes, not margin glosses
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

            # Filter glosses by rank if present (keep rank 1 and 2, drop 3 if tight)
            # For now keep all — the renderer can thin later based on density
            glosses = [g for g in glosses if g.get("rank", 1) <= 2]

            if grk in para_starters and all_sentences and not all_sentences[-1].get("para_break"):
                if any(not item.get("para_break") for item in all_sentences):
                    all_sentences.append({"para_break": True})
            highlighted = highlight_anchors(grk, glosses)
            all_sentences.append({
                "html": highlighted,
                "glosses": glosses,
                "footnotes": sent_footnotes,
            })
        all_sentences.append({"para_break": True})

    # Collect all glosses with IDs for the margin
    all_glosses = []
    seen = set()
    for item in all_sentences:
        if item.get("para_break"):
            continue
        for g in item["glosses"]:
            key = g.get("_id", g["anchor"])
            if key not in seen:
                seen.add(key)
                all_glosses.append(g)

    html = []
    html.append("""<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="utf-8">
<title>Ὁ Αἱματόεις Μεσημβρινός</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=GFS+Didot&display=swap');

  @page { size: A4; margin: 2cm 1.5cm; }
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'GFS Didot', 'Times New Roman', serif;
    background: #fff;
    color: #1a1a1a;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
  }

  /* Title page */
  .title-page {
    text-align: center; margin: 0; padding: 4rem 2rem;
    page-break-after: always;
    min-height: 60vh;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
  }
  .title-page .author { font-size: 0.95rem; letter-spacing: 0.2em; color: #333; margin-top: 2.25rem; margin-bottom: 1.25rem; }
  .title-page .title { font-size: 2.6rem; line-height: 1.2; margin-bottom: 0; color: #1a1a1a; }
  .title-page .sub { font-size: 1.05rem; color: #333; margin-bottom: 1.25rem; }
  .title-page .note { font-size: 0.85rem; color: #666; margin-top: -0.1rem; }

  .chapter-num {
    font-size: 1.6rem; text-align: center;
    margin: 0 0 1.5rem; color: #333; letter-spacing: 0.15em;
  }

  /* Two-panel layout: Greek (left) + glosses (right) */
  .page-body {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 0 1.2rem;
    position: relative;
  }

  /* Main Greek text column */
  .main-text {
    font-size: 1.1rem;
    line-height: 2;
    text-align: justify;
    hyphens: auto;
  }
  .main-text .para {
    text-indent: 1.5em;
    margin-bottom: 0.5rem;
  }
  .main-text .para:first-child {
    text-indent: 0;
  }

  /* Gloss panel: two-column CSS flow inside */
  .gloss-panel {
    position: relative;
    border-left: none;
    padding-left: 1rem;
  }

  .mg {
    position: absolute;
    width: 100%;
    font-size: 0.75rem;
    line-height: 1.35;
    color: #555;
  }
  .mg .w {
    font-weight: 600;
    color: #333;
  }
  .mg .n {
    color: #666;
  }
  /* Footnote markers in the text */
  .fn-marker {
    font-size: 0.65rem;
    vertical-align: super;
    color: #888;
    margin-left: 1px;
    cursor: help;
  }

  /* Footnote section at bottom */
  .footnotes {
    margin-top: 2rem;
    padding-top: 0.8rem;
    border-top: 1px solid #ccc;
    font-size: 0.75rem;
    line-height: 1.4;
    color: #555;
  }
  .fn-entry {
    margin-bottom: 0.4rem;
  }
  .fn-num {
    font-weight: 600;
    color: #666;
    margin-right: 0.3rem;
  }
  .fn-source {
    font-style: italic;
  }
  .fn-quote {
    font-family: 'GFS Didot', serif;
  }

  .gw { cursor: help; }
  .gw:hover { background: #F5F0E0; }

  @media print {
    body { padding: 0; max-width: none; }
    .gw:hover { background: none; }
  }
  @media (max-width: 800px) {
    .page-body { grid-template-columns: 1fr; }
    .gloss-panel { display: none; }
  }
</style>
</head>
<body>
""")

    # Title page
    html.append("""
<div class="title-page">
  <div class="author">ΚΟΡΜΑΚ ΜΑΚΚΑΡΘΥ</div>
  <div class="title">Ὁ Αἱματόεις<br>Μεσημβρινός</div>
  <div class="sub">ἢ<br>Τὸ Ἑσπέριον Ἐρύθημα</div>
  <div class="note">εἰς τὴν Ἑλλάδα φωνὴν μεταφρασθέν</div>
</div>
""")

    html.append('<div style="height:3rem"></div>')
    html.append('<div class="chapter-num">Ι</div>')
    html.append('<div class="page-body">')

    # Collect footnotes with running numbers
    all_footnotes = []
    fn_counter = 1

    # Column 1: flowing Greek text with paragraphs + footnote markers
    html.append('<div class="main-text">')
    para_parts = []
    for item in all_sentences:
        if item.get("para_break"):
            if para_parts:
                html.append(f'<p class="para">{" ".join(para_parts)}</p>')
                para_parts = []
            continue

        sent_html = item["html"]

        # Insert footnote markers for echoes/attestations
        for fn in item.get("footnotes", []):
            phrase = fn.get("greek", "")
            if phrase and phrase[:12] in sent_html:
                marker = f'<sup class="fn-marker" title="see footnote {fn_counter}">{fn_counter}</sup>'
                # Insert marker after the phrase
                insert_pos = sent_html.find(phrase[:12])
                if insert_pos >= 0:
                    # Find end of the phrase in the HTML
                    end_pos = insert_pos + len(phrase[:20])
                    # Place marker after the nearest word boundary
                    space_pos = sent_html.find(" ", end_pos)
                    if space_pos < 0:
                        space_pos = len(sent_html)
                    sent_html = sent_html[:space_pos] + marker + sent_html[space_pos:]

                fn["_number"] = fn_counter
                all_footnotes.append(fn)
                fn_counter += 1

        para_parts.append(sent_html)

    if para_parts:
        html.append(f'<p class="para">{" ".join(para_parts)}</p>')
    html.append('</div>')

    # Gloss panel: vocabulary glosses only (no echoes/attestations)
    html.append('<div class="gloss-panel">')
    for g in all_glosses:
        if g.get("_type") in ("echo", "attestation"):
            continue  # these are now footnotes
        gid = g.get("_id", "")
        html.append(
            f'<div class="mg" data-for="{gid}">'
            f'<span class="w">{g["anchor"]}</span> '
            f'<span class="n">{g["note"]}</span>'
            f'</div>'
        )
    html.append('</div>')

    html.append('</div>')  # page-body

    # Footnote section
    if all_footnotes:
        html.append('<div class="footnotes">')
        for fn in all_footnotes:
            num = fn.get("_number", "")
            source = fn.get("source", "")
            source_quote = fn.get("source_quote", "")
            note = fn.get("note", "")
            entry = f'<div class="fn-entry"><span class="fn-num">{num}.</span>'
            entry += f'<span class="fn-source">{source}</span>'
            if source_quote:
                entry += f' — <span class="fn-quote">{source_quote}</span>'
            if note:
                entry += f' ({note})'
            entry += '</div>'
            html.append(entry)
        html.append('</div>')

    # JS to position glosses in two columns within the panel
    html.append("""
<script>
function alignGlosses() {
  const panel = document.querySelector('.gloss-panel');
  if (!panel) return;
  const body = document.querySelector('.page-body');
  const bodyTop = body.getBoundingClientRect().top + window.scrollY;
  const panelWidth = panel.offsetWidth;
  const colWidth = Math.floor((panelWidth - 12) / 2);

  const mgs = panel.querySelectorAll('.mg');

  let col1Bottom = 0;
  let col2Bottom = 0;

  mgs.forEach(mg => {
    const forId = mg.dataset.for;
    const anchor = document.getElementById(forId);
    if (!anchor) { mg.style.display = 'none'; return; }

    mg.style.width = colWidth + 'px';

    const anchorRect = anchor.getBoundingClientRect();
    let idealTop = anchorRect.top + window.scrollY - bodyTop;

    let top1 = Math.max(idealTop, col1Bottom + 2);
    let top2 = Math.max(idealTop, col2Bottom + 2);

    let useCol1 = true;
    if (Math.abs(top2 - idealTop) < Math.abs(top1 - idealTop)) {
      useCol1 = false;
    }

    if (useCol1) {
      mg.style.top = top1 + 'px';
      mg.style.left = '0px';
      col1Bottom = top1 + mg.offsetHeight;
    } else {
      mg.style.top = top2 + 'px';
      mg.style.left = (colWidth + 12) + 'px';
      col2Bottom = top2 + mg.offsetHeight;
    }
  });
}

window.addEventListener('load', alignGlosses);
window.addEventListener('resize', alignGlosses);
</script>
""")

    html.append('</body></html>')
    return "\n".join(html)


def main():
    if len(sys.argv) < 2:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )
    else:
        passage_ids = sys.argv[1:]

    html = render_passage(passage_ids)
    out_path = OUTPUT / "blood_meridian.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
