#!/usr/bin/env python3
"""
V2 gloss generator: let Sonnet choose what to gloss.

Instead of mechanical frequency-based candidate selection (which misses
common words due to broken lemmatisation), we give Sonnet the full Greek
text and let it decide which words an intermediate reader needs help with.

Usage:
  python3 scripts/build_glosses_v2.py 001_see_the_child
  python3 scripts/build_glosses_v2.py --all
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DRAFTS = ROOT / "drafts"
PASSAGES = ROOT / "passages"
APPARATUS = ROOT / "apparatus"
GLOSSARY = ROOT / "glossary"

sys.path.insert(0, str(SCRIPTS))

GLOSS_PROMPT = """You are generating Ørberg-style marginal glosses for an Ancient Greek text (a translation of McCarthy's Blood Meridian). These appear in the margin of a reader-edition.

## The Ørberg Principle
Explain rare Ancient Greek words using only common Ancient Greek words that the reader already knows. The reader knows basic Attic/Koine vocabulary (~top 500 lemmas: articles, εἰμί, ἔχω, ποιέω, λέγω, ἀνήρ, γυνή, πόλις, μέγας, etc.). Your gloss must use ONLY such common words.

## Notation
- `=` synonym: `= λεπτός, ἀσθενής`
- `<` derivation: `< λίνον· = ἐκ λίνου πεποιημένον`
- `↔` antonym: `↔ παχύς`
- `·` compound: `ξυλο·κόπος`
- `+` case: `σκαλεύω + αἰτ.`
- `()` grammar/source: `(παρακ. μτχ.)`, `(Ὅμ.)`
- `ἐνταῦθα` contextual meaning

Abbreviations: αἰτ. γεν. δοτ. = cases; ἑν. πλ. = number; ἐν. μέσ. παθ. = voice; παρακ. ἀόρ. μέλλ. = tense; μτχ. ἀπρ. ὑποτ. εὐκτ. = mood

## Exemplars
- ἰσχνός → `= λεπτός, ἀσθενὴς τὸ σῶμα ↔ παχύς, εὔρωστος`
- σκαλεύει → `σκαλεύω + αἰτ. = κινεῖ τοὺς ἄνθρακας σιδήρῳ ↔ σβέννυμι`
- ξυλοκόπων καὶ ὑδροφόρων → `ξυλο·κόπος · ὑδρο·φόρος = δοῦλοι ταπεινοί (ΙΗΣ. ΝΑΥ. θ´ 21)`
- πτώσσει → `↔ ἵσταται ὀρθός· = ὀκλάζει ὥσπερ θηρίον δεδοικός (Ὅμ.)`

## Rules
1. ENTIRELY in Greek — no English, no Latin script.
2. Gloss ONLY words that an intermediate Greek reader would not know. Skip: articles, common verbs (εἰμί, ἔχω, ποιέω, λέγω, φέρω, βαίνω, γιγνώσκω, ὁράω, etc.), common nouns (ἀνήρ, γυνή, παῖς, θεός, πόλις, γῆ, ὕδωρ, πῦρ, etc.), common adjectives (μέγας, μικρός, καλός, κακός, etc.), all particles, prepositions, pronouns, demonstratives, numerals.
3. DO gloss: rare vocabulary, neologisms (marked with *), compounds, words used in unusual senses, hapax legomena, technical terms, biblical/Homeric resonances.
4. Under 80 characters per gloss. Dense and terse.
5. Distribute glosses evenly through the passage — don't front-load.
6. Multi-word anchors allowed for phrases that form a unit.
7. Target: roughly one gloss per 6-8 words of Greek text.

## English source (for context)
{english}

## Greek text to gloss
{greek}

Return a JSON array. The "anchor" must exactly match text in the Greek.
Group by sentence — include the sentence text and its glosses.
[
  {{"sentence": "first sentence of Greek...", "glosses": [
    {{"anchor": "word", "note": "Ørberg gloss"}},
    ...
  ]}},
  ...
]
Output ONLY the JSON."""


def call_sonnet(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.·;!])\s+', text)
    result = []
    for chunk in raw:
        for sub in chunk.split("\n\n"):
            sub = sub.strip()
            if sub:
                result.append(sub)
    return result


def build_passage_glosses(passage_id: str) -> bool:
    draft_path = DRAFTS / passage_id / "primary.txt"
    passage_path = PASSAGES / f"{passage_id}.json"
    if not draft_path.exists():
        return False

    greek = draft_path.read_text("utf-8").strip()
    english = ""
    if passage_path.exists():
        english = json.load(open(passage_path)).get("text", "")

    words = len(greek.split())
    print(f"  {passage_id}: {words} words")

    # For long passages, batch by splitting into chunks
    BATCH_WORDS = 200
    if words > BATCH_WORDS:
        sents = split_sentences(greek)
        batches = []
        current_batch = []
        current_words = 0
        for s in sents:
            current_batch.append(s)
            current_words += len(s.split())
            if current_words >= BATCH_WORDS:
                batches.append(current_batch)
                current_batch = []
                current_words = 0
        if current_batch:
            batches.append(current_batch)
    else:
        batches = [split_sentences(greek)]

    all_results = []
    for i, batch_sents in enumerate(batches):
        batch_greek = " ".join(batch_sents)
        n_batches = len(batches)
        if n_batches > 1:
            print(f"    Batch {i+1}/{n_batches} ({len(batch_greek.split())} words)...")

        prompt = GLOSS_PROMPT.format(english=english, greek=batch_greek)
        raw = call_sonnet(prompt)

        try:
            clean = re.sub(r'^```json\s*', '', raw)
            clean = re.sub(r'\s*```$', '', clean)
            batch_result = json.loads(clean)
            all_results.extend(batch_result)
        except json.JSONDecodeError:
            print(f"    WARNING: could not parse response for batch {i+1}")

    # Build output in standard format
    sents = split_sentences(greek)
    output = {"passage_id": passage_id, "style": "Ørberg", "sentences": []}

    # Map Sonnet's sentence-grouped results back to our sentence list
    sonnet_map = {}
    for r in all_results:
        s_text = r.get("sentence", "")
        glosses = r.get("glosses", [])
        # Match by prefix (Sonnet may truncate)
        for i, sent in enumerate(sents):
            if sent.startswith(s_text[:30]) or s_text.startswith(sent[:30]):
                sonnet_map[i] = glosses
                break

    total = 0
    for i, sent in enumerate(sents):
        glosses = sonnet_map.get(i, [])
        output["sentences"].append({
            "index": i,
            "greek": sent,
            "glosses": glosses,
        })
        total += len(glosses)

    # Write
    out_dir = APPARATUS / passage_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "marginal_glosses.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"    ✓ {total} glosses")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
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

    for pid in passage_ids:
        build_passage_glosses(pid)


if __name__ == "__main__":
    main()
