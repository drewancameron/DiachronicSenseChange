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
MAX_APPARATUS_PER_PARA = 2


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
  /* Footnote markers */
  .fn-marker { font-size: 0.6rem; color: #999; }

  /* Inline apparatus: small block between paragraphs */
  .apparatus {
    font-size: 0.7rem;
    line-height: 1.3;
    color: #888;
    margin: 0.1rem 0 0.6rem 1.5em;
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

    # Column 1: flowing Greek text with paragraphs + inline apparatus after each
    html.append('<div class="main-text">')
    para_parts = []
    para_footnotes = []
    fn_counter = 1

    for item in all_sentences:
        if item.get("para_break"):
            if para_parts:
                if para_footnotes:
                    # Deduplicate by Greek phrase — keep the one with longest source_quote
                    seen_phrases = {}
                    for fn in para_footnotes:
                        phrase = fn.get("greek", "")[:20]
                        existing = seen_phrases.get(phrase)
                        if not existing or len(fn.get("source_quote", "")) > len(existing.get("source_quote", "")):
                            seen_phrases[phrase] = fn
                    deduped = list(seen_phrases.values())

                    # Select best entries
                    ranked = sorted(deduped, key=lambda f: (
                        f.get("rank", 2),  # Sonnet's significance rank (1=best)
                        -len(f.get("source_quote", "")),  # longer quote = better
                        -len(f.get("note", "")),
                    ))
                    selected = ranked[:MAX_APPARATUS_PER_PARA]

                    # Find position of each echo phrase in the plain text
                    # to assign numbers in text order
                    joined = " ".join(para_parts)
                    # Strip HTML tags for position finding
                    plain = re.sub(r'<[^>]+>', '', joined)
                    for fn in selected:
                        phrase = fn.get("greek", "")
                        # Find the END position of the phrase in plain text
                        pos = plain.find(phrase[:15])
                        if pos >= 0:
                            fn["_text_pos"] = pos + len(phrase)
                        else:
                            fn["_text_pos"] = len(plain)

                    # Sort by position in text, then assign numbers
                    selected.sort(key=lambda f: f.get("_text_pos", 0))
                    for fn in selected:
                        fn["_num"] = fn_counter
                        fn_counter += 1

                    # Insert markers into the HTML at the right positions
                    # Work backwards so insertions don't shift later positions
                    # Build plain-text version for position finding
                    plain = re.sub(r'<[^>]+>', '', joined)

                    for fn in reversed(selected):
                        phrase = fn.get("greek", "").strip()
                        if not phrase:
                            continue

                        # Find the phrase in the PLAIN text to get the end position
                        # Try full phrase first, then progressively shorter prefixes
                        plain_pos = -1
                        match_len = 0
                        for try_len in range(len(phrase), 4, -1):
                            p = plain.find(phrase[:try_len])
                            if p >= 0:
                                plain_pos = p
                                match_len = try_len
                                break

                        if plain_pos < 0:
                            continue

                        # The end position in plain text
                        plain_end = plain_pos + match_len

                        # Now find where plain_end maps to in the HTML
                        # Walk through joined, tracking plain-text position
                        html_insert = 0
                        plain_idx = 0
                        in_tag = False
                        for i, ch in enumerate(joined):
                            if ch == '<':
                                in_tag = True
                            elif ch == '>':
                                in_tag = False
                                continue
                            if not in_tag:
                                if plain_idx == plain_end:
                                    html_insert = i
                                    break
                                plain_idx += 1
                        else:
                            html_insert = len(joined)

                        # Skip past any closing tags at this position
                        while html_insert < len(joined) and joined[html_insert:html_insert+2] == '</':
                            close = joined.find('>', html_insert)
                            if close >= 0:
                                html_insert = close + 1
                            else:
                                break

                        marker = f'<sup class="fn-marker">{fn["_num"]}</sup>'
                        joined = joined[:html_insert] + marker + joined[html_insert:]
                        # Update plain too so subsequent searches account for shift
                        plain = re.sub(r'<[^>]+>', '', joined)

                    html.append(f'<p class="para">{joined}</p>')

                    # Emit apparatus in number order
                    selected.sort(key=lambda f: f["_num"])
                    refs = []
                    for fn in selected:
                        source = fn.get("source", "")
                        source_quote = fn.get("source_quote", "")
                        note = fn.get("note", "")
                        ref = f'<sup>{fn["_num"]}</sup> <em>{source}</em>'
                        if source_quote:
                            ref += f': {source_quote}'
                        if note:
                            ref += f' — {note}'
                        refs.append(ref)
                    html.append(
                        f'<div class="apparatus">{";&nbsp; ".join(refs)}</div>'
                    )
                else:
                    html.append(f'<p class="para">{" ".join(para_parts)}</p>')
                para_parts = []
                para_footnotes = []
            continue

        para_parts.append(item["html"])
        for fn in item.get("footnotes", []):
            if fn.get("source_quote") or (fn.get("note") and len(fn.get("note", "")) > 10):
                para_footnotes.append(fn)

    if para_parts:
        html.append(f'<p class="para">{" ".join(para_parts)}</p>')
        if para_footnotes:
            ranked = sorted(para_footnotes, key=lambda f: (
                -len(f.get("source_quote", "")),
                -len(f.get("note", "")),
            ))
            refs = []
            for fn in ranked[:MAX_APPARATUS_PER_PARA]:
                source = fn.get("source", "")
                source_quote = fn.get("source_quote", "")
                note = fn.get("note", "")
                ref = f'<em>{source}</em>'
                if source_quote:
                    ref += f': {source_quote}'
                if note:
                    ref += f' — {note}'
                refs.append(ref)
            html.append(
                f'<div class="apparatus">{";&nbsp; ".join(refs)}</div>'
            )
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

    # Feedback tab
    html.append("""
<style>
  .fb-tab {
    position: fixed; top: 1rem; right: 0;
    background: #444; color: #fff; padding: 0.4rem 0.8rem;
    font-size: 0.75rem; cursor: pointer; border-radius: 4px 0 0 4px;
    writing-mode: horizontal-tb; z-index: 100;
  }
  .fb-tab:hover { background: #666; }
  .fb-panel {
    display: none; position: fixed; top: 0; right: 0;
    width: 320px; height: 100vh; background: #fafaf8;
    box-shadow: -2px 0 8px rgba(0,0,0,0.15); z-index: 101;
    padding: 1.5rem; overflow-y: auto;
    font-family: system-ui, sans-serif;
  }
  .fb-panel.open { display: block; }
  .fb-panel h3 { margin: 0 0 0.8rem; font-size: 1rem; color: #333; }
  .fb-panel label { display: block; font-size: 0.8rem; color: #555; margin: 0.6rem 0 0.2rem; }
  .fb-panel textarea, .fb-panel input, .fb-panel select {
    width: 100%; padding: 0.4rem; font-size: 0.85rem;
    border: 1px solid #ccc; border-radius: 3px;
  }
  .fb-panel textarea { height: 120px; resize: vertical; }
  .fb-panel button {
    margin-top: 0.8rem; padding: 0.5rem 1.2rem;
    background: #444; color: #fff; border: none;
    border-radius: 3px; cursor: pointer; font-size: 0.85rem;
  }
  .fb-panel button:hover { background: #666; }
  .fb-panel .fb-close {
    position: absolute; top: 0.5rem; right: 0.8rem;
    background: none; border: none; font-size: 1.2rem;
    color: #999; cursor: pointer; padding: 0;
  }
  .fb-thanks { display: none; color: #4a4; font-size: 0.9rem; margin-top: 1rem; }
</style>

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
      <option>Apparatus / echo</option>
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

<script>
document.getElementById('fb-form').addEventListener('submit', function(e) {
  e.preventDefault();
  fetch(this.action, {method:'POST', body: new FormData(this), headers:{'Accept':'application/json'}})
    .then(r => {
      if (r.ok) {
        this.reset();
        this.querySelector('.fb-thanks').style.display = 'block';
        setTimeout(() => this.querySelector('.fb-thanks').style.display = 'none', 3000);
      }
    });
});
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
