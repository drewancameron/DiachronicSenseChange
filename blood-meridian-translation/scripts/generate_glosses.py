#!/usr/bin/env python3
"""
Generate Ørberg-style marginal glosses using IDF-based word selection
and a single LLM call for definitions.

Pipeline:
1. IDF auto-glosser identifies rare words (threshold-based)
2. Single Sonnet call generates Ørberg-style definitions for all flagged words
3. Output written to apparatus/<passage_id>/marginal_glosses.json

Usage:
  python3 scripts/generate_glosses.py 001_see_the_child
  python3 scripts/generate_glosses.py --all
  python3 scripts/generate_glosses.py --all --threshold 9
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
PASSAGES = ROOT / "passages"
APPARATUS = ROOT / "apparatus"
GLOSSARY = ROOT / "glossary"

sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_THRESHOLD = 9.0


def load_idf():
    from auto_gloss import load_idf as _load
    return _load()


def get_words_to_gloss(passage_id: str, lemma_idf: dict, form_idf: dict,
                       threshold: float) -> list[dict]:
    from auto_gloss import propose_glosses
    return propose_glosses(passage_id, lemma_idf, form_idf, threshold, threshold + 2)


def generate_glosses_for_passage(passage_id: str, lemma_idf: dict, form_idf: dict,
                                  threshold: float, dry_run: bool = False) -> bool:
    primary_path = DRAFTS / passage_id / "primary.txt"
    if not primary_path.exists():
        print(f"  {passage_id}: no primary.txt")
        return False

    greek = primary_path.read_text("utf-8").strip()

    # Get words to gloss
    words = get_words_to_gloss(passage_id, lemma_idf, form_idf, threshold)
    if not words:
        print(f"  {passage_id}: no words need glossing")
        return False

    print(f"  {passage_id}: {len(words)} words to gloss")

    # Split Greek into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.;·!])\s+', greek) if s.strip()]

    if dry_run:
        for w in words:
            print(f"    {w['word']:>25}  IDF={w['idf']:.1f}")
        return False

    # Single Sonnet call to generate all definitions
    word_block = "\n".join(f"- {w['word']}" for w in words)

    prompt = f"""Generate Ørberg-style Ancient Greek glosses for these words. Each word appears in the translation below.

Ørberg notation rules:
- "= synonym" for simple definitions (e.g., ὠχρός = λευκὸς ὡς νοσῶν ↔ ἐρυθρός)
- "< root" for derivations (e.g., λίνεον < λίνον = ἐκ λίνου πεποιημένον)
- "↔ antonym" for contrast (e.g., ἰσχνός = λεπτός ↔ παχύς)
- "·" to decompose compounds (e.g., ξυλο·κόπος = ξύλα κόπτων)
- "ἐνταῦθα =" for contextual meaning when a word has a special sense here
- Keep definitions SHORT — ideally under 8 Greek words
- Use ONLY Ancient Greek in the definitions, never English
- For verbs, give the meaning; for rare noun forms, give nominative + meaning
- For participles, show the root verb

Words to gloss:
{word_block}

Greek text (for context):
{greek}

Output as JSON array:
[{{"word": "ὠχρός", "note": "= λευκὸς ὡς νοσῶν ↔ ἐρυθρός"}}, ...]

Output ONLY the JSON array, no other text."""

    import anthropic
    client = anthropic.Anthropic()

    t0 = time.time()
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = r.content[0].text.strip()
    cost = r.usage.input_tokens / 1e6 * 3 + r.usage.output_tokens / 1e6 * 15
    print(f"    Sonnet: {time.time()-t0:.0f}s, {r.usage.input_tokens}+{r.usage.output_tokens} tok, ${cost:.4f}")

    # Parse JSON response
    try:
        if "```" in response_text:
            response_text = re.search(r'```(?:json)?\s*\n?(.*?)```',
                                       response_text, re.DOTALL).group(1)
        gloss_defs = json.loads(response_text)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"    Failed to parse response: {e}")
        print(f"    Response: {response_text[:200]}")
        return False

    # Build gloss lookup
    gloss_lookup = {}
    for g in gloss_defs:
        word = g.get("word", "")
        note = g.get("note", "")
        if word and note:
            gloss_lookup[word] = note

    print(f"    Got {len(gloss_lookup)} definitions")

    # Build marginal_glosses.json structure
    mg_sentences = []
    for i, sent in enumerate(sentences):
        sent_glosses = []
        for w in words:
            word = w["word"]
            if word in sent and word in gloss_lookup:
                sent_glosses.append({
                    "anchor": word,
                    "note": gloss_lookup[word],
                    "rank": 1,
                })
        mg_sentences.append({
            "index": i,
            "greek": sent,
            "glosses": sent_glosses,
        })

    # Write output
    out_dir = APPARATUS / passage_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "marginal_glosses.json"
    with open(out_path, "w") as f:
        json.dump({"sentences": mg_sentences}, f, ensure_ascii=False, indent=2)
    print(f"    Wrote {out_path}")

    total_glosses = sum(len(s["glosses"]) for s in mg_sentences)
    print(f"    {total_glosses} glosses across {len(sentences)} sentences")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )
    elif args.passages:
        passage_ids = args.passages
    else:
        parser.print_help()
        return

    lemma_idf, form_idf = load_idf()
    print(f"IDF loaded: {len(lemma_idf):,} lemmas, threshold={args.threshold}")
    print()

    t0 = time.time()
    for pid in passage_ids:
        generate_glosses_for_passage(pid, lemma_idf, form_idf,
                                     args.threshold, args.dry_run)
        print()

    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
