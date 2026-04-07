#!/usr/bin/env python3
"""
V7 Pipeline: Mechanical draft → Opus polish → glosses from provenance.

The mechanical-first pipeline:
1. Stanza parse English
2. Woodhouse/LSJ vocabulary lookup (with Haiku synonym disambiguation)
3. Construction mapping from parallel corpus
4. Haiku verification pass
5. Opus polish (default: Sophoclean style)
6. Glosses from translation provenance (no LLM needed)

Usage:
  python3 scripts/pipeline_v7.py 001_see_the_child_he
  python3 scripts/pipeline_v7.py --all
  python3 scripts/pipeline_v7.py 001_see_the_child_he --style septuagint
  python3 scripts/pipeline_v7.py 001_see_the_child_he --dry-run
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PASSAGES = ROOT / "passages"
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"

sys.path.insert(0, str(SCRIPTS))

STYLES = {
    "sophocles": (
        "Sophocles — austere grandeur, compressed syntax, bare and loaded words, "
        "riddling irony, fate-laden silences. Strip language to bone. "
        "Every word must earn its place."
    ),
    "septuagint": (
        "the Septuagint translators — paratactic καί-chains, Hebraic simplicity, "
        "sparse particles, present-tense narrative, echoing Genesis and Joshua"
    ),
    "thucydides": (
        "Thucydides — dense periodic sentences, abstract nouns, compressed "
        "participles, gnomic aorists, clinical detachment from violence"
    ),
    "euripides": (
        "Euripides — iambic trimeter rhythm where possible, stichomythic "
        "compression for dialogue, pathetic irony, the mundane made tragic"
    ),
    "koine": (
        "a Koine narrative prose writer — fluid, clear, readable. "
        "Attic vocabulary with Koine simplicity. Natural and unaffected."
    ),
}


def translate_passage(passage_id: str, style: str = "sophocles",
                      dry_run: bool = False, force: bool = False) -> dict | None:
    """Run the full V7 pipeline on one passage."""
    p_path = PASSAGES / f"{passage_id}.json"
    if not p_path.exists():
        print(f"  {passage_id}: not found")
        return None

    en_text = json.load(open(p_path)).get("text", "")
    if not en_text:
        return None

    draft_dir = DRAFTS / passage_id
    draft_path = draft_dir / "primary.txt"

    if draft_path.exists() and not force and not dry_run:
        existing = draft_path.read_text("utf-8").strip()
        if existing:
            print(f"  {passage_id}: draft exists. Use --force to overwrite.")
            return None

    draft_dir.mkdir(parents=True, exist_ok=True)

    # === STAGE 1-4: Mechanical draft ===
    print(f"\n  [{passage_id}] MECHANICAL DRAFT")
    t0 = time.time()

    from mechanical_assembler import assemble_passage
    from construction_dispatcher import _load_vocab, _save_haiku_cache

    _load_vocab()
    pairs = assemble_passage(passage_id)
    _save_haiku_cache()

    if not pairs:
        print(f"    No sentences produced")
        return None

    # Build draft block
    draft_lines = []
    for en, grc in pairs:
        draft_lines.append(f"EN: {en}")
        draft_lines.append(f"GR: {grc}")
        draft_lines.append("")
    draft_block = "\n".join(draft_lines)

    mech_time = time.time() - t0
    print(f"    {len(pairs)} sentences in {mech_time:.0f}s")

    if dry_run:
        for en, grc in pairs:
            print(f"    EN: {en}")
            print(f"    GR: {grc}")
        return None

    # Save mechanical draft
    (draft_dir / "stage0_mechanical.txt").write_text(draft_block)

    # === STAGE 5: Opus polish ===
    print(f"\n  [{passage_id}] OPUS POLISH (style: {style})")

    style_desc = STYLES.get(style, STYLES["sophocles"])

    prompt = f"""You are rewriting a mechanical Ancient Greek translation in the distinctive style of {style_desc}.

Transform this draft into prose that reads as natural, powerful Ancient Greek. You may freely reorder, adjust morphology, add particles and articles, choose between synonyms where the draft's choice feels flat.

Preserve:
- The core meaning of each sentence
- McCarthy's sentence boundaries and paragraph structure
- Fragments as fragments — do not add verbs to bare noun phrases
- The register: Attic vocabulary, not Homeric epic

English source:
{en_text}

Mechanical draft:
{draft_block}

Output ONLY the Greek text."""

    import anthropic
    client = anthropic.Anthropic()

    t0 = time.time()
    r = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    polished = r.content[0].text.strip()
    polish_time = time.time() - t0
    cost = r.usage.input_tokens / 1e6 * 15 + r.usage.output_tokens / 1e6 * 75
    print(f"    {polish_time:.0f}s, ${cost:.4f}")
    print(f"    {polished[:100]}...")

    # Save polished
    (draft_dir / "stage1_opus_polished.txt").write_text(polished)
    draft_path.write_text(polished + "\n")

    # === STAGE 6: Glosses from provenance ===
    print(f"\n  [{passage_id}] GLOSSES FROM PROVENANCE")

    gloss_data = build_provenance_glosses(passage_id, pairs, polished)

    if gloss_data:
        out_dir = APPARATUS / passage_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "marginal_glosses.json"
        with open(out_path, "w") as f:
            json.dump(gloss_data, f, ensure_ascii=False, indent=2)
        total_glosses = sum(len(s["glosses"]) for s in gloss_data["sentences"])
        print(f"    {total_glosses} glosses written")

    return {
        "passage_id": passage_id,
        "greek": polished,
        "cost": cost,
        "mechanical_time": mech_time,
        "polish_time": polish_time,
    }


def build_provenance_glosses(passage_id: str,
                              mechanical_pairs: list[tuple[str, str]],
                              polished_greek: str) -> dict | None:
    """Build glosses from the translation provenance chain.

    We know: English word → Greek lemma (from Woodhouse/LSJ) → definition.
    No LLM needed — the gloss IS the dictionary definition we already looked up.

    Skips common words that an intermediate Greek reader already knows.
    """
    import unicodedata
    from mechanical_assembler import get_provenance
    from mechanical_glosser import lsj_lookup, lsj_antonym, _load_lsj

    _load_lsj()

    # Common lemmas an intermediate reader knows — don't gloss
    COMMON = {
        "εἶναι", "ἔχειν", "λέγειν", "ποιεῖν", "γίγνεσθαι", "ἔρχεσθαι",
        "ἰέναι", "ὁρᾶν", "εἰδέναι", "δοκεῖν", "βούλεσθαι", "δεῖν",
        "δύνασθαι", "φέρειν", "λαμβάνειν", "διδόναι", "τιθέναι",
        "ἀνήρ", "γυνή", "ἄνθρωπος", "θεός", "παῖς", "πατήρ", "μήτηρ",
        "πόλις", "γῆ", "ὕδωρ", "πῦρ", "οἶκος", "ὁδός", "χείρ",
        "λόγος", "ἔργον", "ὄνομα", "μέγας", "πολύς", "καλός", "κακός",
        "ἀγαθός", "πᾶς", "ἄλλος", "αὐτός", "ἐγώ", "σύ", "οὗτος",
        "ἐκεῖνος", "ὅς", "τίς", "εἷς", "δύο", "τρεῖς",
    }

    provenance = get_provenance()
    if not provenance:
        return None

    # Build gloss entries from provenance
    gloss_entries = {}  # lemma → note string

    for prov in provenance:
        lemma = prov["lemma"]
        english = prov["english"]

        if lemma in COMMON or lemma in gloss_entries or len(lemma) <= 2:
            continue

        # The English word IS the gloss — we know the sense because we chose it
        en_word = english.lower().strip(".,;:!?")
        defn = en_word

        # Clean
        defn = re.sub(r"^(the|a|an|to)\s+", "", defn.strip(), flags=re.I)
        defn = re.sub(r"\s+(of|with|for|from|in|on|at|to)(\s+(a|an|the))?\s*$",
                       "", defn)
        defn = defn.rstrip(". ")
        if not defn or len(defn) < 2:
            continue

        # Antonym
        antonym = lsj_antonym(lemma)
        note = f"= {defn}"
        if antonym:
            ant = antonym.strip()
            if (len(ant) > 1 and len(ant) < 30 and
                any("GREEK" in unicodedata.name(c, "") for c in ant[:3]
                    if c.isalpha())):
                note += f" ↔ {ant}"

        gloss_entries[lemma] = note

    # Split polished Greek into sentences
    greek_sentences = [s.strip() for s in
                       re.split(r'(?<=[.;·!])\s+', polished_greek) if s.strip()]

    # Match glosses to polished text by finding inflected forms
    mg_sentences = []
    used = set()

    for i, sent in enumerate(greek_sentences):
        sent_glosses = []
        for lemma, note in gloss_entries.items():
            if lemma in used:
                continue
            # Match on first 3-4 chars of lemma (prefix matching for inflection)
            stem = lemma[:min(4, len(lemma))]
            if stem not in sent:
                continue
            # Find the actual word in the sentence
            anchor = None
            for word in re.findall(
                    r'[\u0370-\u03FF\u1F00-\u1FFF\u0300-\u036F]+', sent):
                if word.startswith(stem) or word.lower().startswith(stem.lower()):
                    anchor = word
                    break
            if anchor:
                sent_glosses.append({
                    "anchor": anchor,
                    "note": note,
                    "rank": 1,
                })
                used.add(lemma)

        mg_sentences.append({
            "index": i,
            "greek": sent,
            "glosses": sent_glosses,
        })

    return {"sentences": mg_sentences}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V7: Mechanical → Opus → Glosses")
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--style", type=str, default="sophocles",
                        choices=list(STYLES.keys()),
                        help="Opus polish style (default: sophocles)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.all:
        passage_ids = sorted(p.stem for p in PASSAGES.glob("*.json")
                            if not p.stem.startswith("exp_"))
    elif args.passages:
        passage_ids = args.passages
    else:
        parser.print_help()
        return

    print(f"╔{'═'*60}╗")
    print(f"║  V7: Mechanical → Opus polish ({args.style})          ║")
    print(f"╚{'═'*60}╝")

    start = time.time()
    total_cost = 0
    for pid in passage_ids:
        result = translate_passage(pid, style=args.style,
                                   dry_run=args.dry_run, force=args.force)
        if result:
            total_cost += result["cost"]

    from construction_dispatcher import _save_haiku_cache
    from morpheus_check import _save_cache
    _save_haiku_cache()
    _save_cache()

    print(f"\n{'═'*60}")
    print(f"  Done in {time.time()-start:.0f}s — cost: ${total_cost:.4f}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
