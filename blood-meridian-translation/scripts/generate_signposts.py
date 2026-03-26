#!/usr/bin/env python3
"""
Step 3: Generate construction signpost cards for Blood Meridian passages.

For each English construction in a BM passage, produces a distribution card
showing what real translators did with similar constructions, with nearest
examples from the parallel corpus.

Uses kernel smoothing over tree edit distance when exact matches are sparse.

Usage:
  python3 scripts/generate_signposts.py                         # all passages
  python3 scripts/generate_signposts.py 005_saint_louis         # one passage
  python3 scripts/generate_signposts.py --for-prompt 013        # output prompt-ready text
"""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASSAGES = ROOT / "passages"
MODEL_DIR = ROOT / "models" / "construction_model"
MODEL_PATH = MODEL_DIR / "cond_distributions.json"
PAIRS_PATH = MODEL_DIR / "construction_pairs.jsonl"
SIGNPOST_DIR = ROOT / "signposts"
DB_PATH = ROOT.parent / "db" / "diachronic.db"

_en_nlp = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        import stanza
        _en_nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                                   verbose=False)
    return _en_nlp


def load_model() -> dict:
    if MODEL_PATH.exists():
        return json.load(open(MODEL_PATH))
    return {}


def load_english(passage_id: str) -> str:
    p = PASSAGES / f"{passage_id}.json"
    if p.exists():
        return json.load(open(p)).get("text", "")
    return ""


def extract_en_constructions(text: str) -> list[dict]:
    """Extract English constructions from a BM passage."""
    nlp = _get_en_nlp()
    doc = nlp(text)

    # Reuse the extraction logic from the parallel pipeline
    sys.path.insert(0, str(ROOT / "scripts"))
    from extract_parallel_constructions import extract_constructions
    return extract_constructions(doc, "en")


def find_nearest_examples(en_type: str, en_text: str, n: int = 3) -> list[dict]:
    """Find nearest parallel corpus examples for a given construction type.

    Samples from diverse sources (not just the first N from Thucydides).
    Future: use tree edit distance for kernel-smoothed matching.
    """
    if not PAIRS_PATH.exists():
        return []

    # Collect by source for diversity
    by_source = {}
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            source = rec.get("source", "unknown")
            for pair in rec.get("pairs", []):
                if pair["en_type"] == en_type and pair["grc_type"] != "none":
                    if source not in by_source:
                        by_source[source] = []
                    if len(by_source[source]) < 3:  # cap per source
                        by_source[source].append({
                            "en_text": pair.get("en_text", "")[:80],
                            "grc_type": pair["grc_type"],
                            "source": source,
                        })
            if len(by_source) >= 10:  # enough sources
                break

    # Pick one from each source for diversity, round-robin
    examples = []
    seen_texts = set()
    for source in sorted(by_source.keys()):
        for ex in by_source[source]:
            if ex["en_text"] not in seen_texts:
                seen_texts.add(ex["en_text"])
                examples.append(ex)
                break
        if len(examples) >= n:
            break

    return examples


def find_related_word_clusters(en_text: str, n_clusters: int = 2) -> list[dict]:
    """Find related word clusters from the parallel corpus for key content words.

    Uses the collocate index if available, otherwise does a simple database lookup
    for Greek words that co-occur with the translation of English content words.
    """
    nlp = _get_en_nlp()
    doc = nlp(en_text[:500])

    # Extract content words (nouns, verbs, adjectives)
    content_words = []
    for sent in doc.sentences:
        for w in sent.words:
            if w.upos in ("NOUN", "VERB", "ADJ") and len(w.text) > 3:
                content_words.append(w.lemma)

    if not content_words or not DB_PATH.exists():
        return []

    # Look up Greek co-occurring vocabulary from aligned translations
    clusters = []
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    for en_word in content_words[:5]:  # top 5 content words
        # Find SHORT Greek passages whose English alignment contains this word
        # (short passages = tighter word-level correspondence)
        cur.execute("""
            SELECT p.greek_text
            FROM alignments a
            JOIN passages p ON a.passage_id = p.passage_id
            WHERE a.aligned_text LIKE ?
              AND a.aligned_text NOT LIKE '[awaiting%'
              AND LENGTH(a.aligned_text) < 200
              AND LENGTH(p.greek_text) < 300
            LIMIT 30
        """, (f"% {en_word} %",))

        rows = cur.fetchall()
        if not rows:
            continue

        # Extract Greek content words from these passages (filter function words)
        import re
        GREEK_STOPWORDS = {
            "καί", "καὶ", "τοῦ", "τῶν", "τὴν", "τὸν", "τῆς", "τοῖς", "τὰς",
            "τὰ", "τοὺς", "τῷ", "τό", "τὸ", "δὲ", "δέ", "μὲν", "μέν",
            "γὰρ", "γάρ", "οὐ", "οὐκ", "οὐχ", "μή", "μὴ", "ἐν", "εἰς",
            "ἐκ", "ἐξ", "πρός", "πρὸς", "ἀπό", "ἀπὸ", "ὑπό", "ὑπὸ",
            "περί", "περὶ", "κατά", "κατὰ", "μετά", "μετὰ", "διά", "διὰ",
            "ὡς", "ὅτι", "ἐπί", "ἐπὶ", "παρά", "παρὰ", "αὐτοῦ", "αὐτῶν",
            "αὐτὸν", "αὐτήν", "αὐτῷ", "αὐτοῖς", "αὐτὸς", "οὖν", "ἀλλά",
            "ἀλλὰ", "εἶναι", "ἐστιν", "ἐστὶ", "ἐστί", "ἦν", "τις",
            "τινα", "τινὲς", "αὐτόν", "αὐτήν", "ταῦτα", "τοῦτο",
            "οὗτος", "αὕτη", "ἐστι", "νῦν", "ἔτι",
        }
        greek_words = Counter()
        for (grc_text,) in rows:
            for tok in re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]+', grc_text):
                if len(tok) > 2 and tok not in GREEK_STOPWORDS:
                    greek_words[tok] += 1

        # Top co-occurring Greek content words
        top_greek = [w for w, c in greek_words.most_common(8) if c >= 2]
        if top_greek:
            clusters.append({
                "english_word": en_word,
                "greek_cluster": top_greek,
                "n_passages": len(rows),
            })

        if len(clusters) >= n_clusters:
            break

    conn.close()
    return clusters


def build_signpost_card(construction: dict, model: dict) -> dict:
    """Build a signpost card for one English construction."""
    en_type = construction["type"]

    # Get distribution from model
    dist_info = model.get("overall", {}).get(en_type, {})
    distribution = dist_info.get("distribution", {})
    total = dist_info.get("total", 0)

    # Get nearest examples
    examples = find_nearest_examples(en_type, construction.get("text", ""))

    # Recommendation (exclude 'none' which means alignment miss, not a real choice)
    real_dist = {k: v for k, v in distribution.items() if k != "none"}
    if real_dist:
        # Renormalise excluding 'none'
        real_total = sum(v["probability"] for v in real_dist.values())
        if real_total > 0:
            real_dist = {k: {"probability": v["probability"] / real_total, "count": v["count"]}
                         for k, v in real_dist.items()}
        top_type = max(real_dist, key=lambda k: real_dist[k]["probability"])
        top_prob = real_dist[top_type]["probability"]
        matched = sum(v["count"] for v in real_dist.values())
        recommendation = f"{top_type} ({top_prob:.0%} of {matched} matched examples)"

        # Nuance: for relative clauses, the global distribution favours genitive
        # absolute, but short defining relatives ("the man who X", "the thing
        # which Y") should almost always stay as relative clauses in Greek too.
        # Override recommendation when the English has a clear defining relative.
        if en_type == "relative_clause":
            head = construction.get("head_word", "")
            text = construction.get("text", "")
            # Short defining relative: head noun + who/which/that + verb
            # These are the cases where Greek naturally uses ὅς/ἥ/ὅ too
            if head and len(text) < 80:
                recommendation = (
                    f"relative_clause — short defining relative "
                    f"('{head}...{construction.get('verb', '')}') → "
                    f"keep as ὅς/ἥ/ὅ + finite verb. "
                    f"Global dist: rel.cl. {real_dist.get('relative_clause', {}).get('probability', 0):.0%}, "
                    f"gen.abs. {real_dist.get('genitive_absolute', {}).get('probability', 0):.0%} "
                    f"— but gen.abs. is for circumstantial, not defining"
                )
            else:
                recommendation += (
                    f". Note: if this is a defining relative ('the X who Y'), "
                    f"prefer ὅς + finite verb; gen.abs. is for circumstantial clauses"
                )

        distribution = real_dist  # use renormalised for output
    elif distribution:
        recommendation = f"Constructions found but none aligned to Greek — check manually"
    else:
        recommendation = f"No data for '{en_type}' in parallel corpus"

    # Related word clusters
    clusters = find_related_word_clusters(construction.get("text", ""))

    return {
        "en_type": en_type,
        "en_scale": construction.get("scale", ""),
        "en_text": construction.get("text", "")[:100],
        "en_head": construction.get("head_word", ""),
        "en_verb": construction.get("verb", ""),
        "distribution": {k: v["probability"] for k, v in distribution.items()},
        "total_examples": total,
        "recommendation": recommendation,
        "nearest_examples": examples,
        "word_clusters": clusters,
    }


def format_signposts_for_prompt(cards: list[dict]) -> str:
    """Format signpost cards as text for injection into a translation prompt."""
    if not cards:
        return ""

    sections = ["## Construction Signposts (from 142K parallel corpus)\n"]

    for card in cards:
        if not card["distribution"]:
            continue

        sections.append(f"**{card['en_type']}**: \"{card['en_text']}\"")

        # Distribution
        dist_lines = []
        for grc_type, prob in sorted(card["distribution"].items(), key=lambda x: -x[1]):
            if prob >= 0.03:  # only show ≥3%
                dist_lines.append(f"  - {grc_type}: {prob:.0%}")
        if dist_lines:
            sections.append("  Distribution in real translations:")
            sections.extend(dist_lines)

        # Recommendation
        sections.append(f"  → Recommendation: {card['recommendation']}")

        # Examples
        if card["nearest_examples"]:
            sections.append("  Examples:")
            for ex in card["nearest_examples"][:2]:
                sections.append(f"    [{ex['source']}] EN: {ex['en_text']}")
                sections.append(f"      → GRC used: {ex['grc_type']}")

        # Word clusters
        if card.get("word_clusters"):
            sections.append("  Related Greek vocabulary:")
            for cl in card["word_clusters"]:
                sections.append(f"    '{cl['english_word']}' → {', '.join(cl['greek_cluster'][:5])}")

        sections.append("")

    return "\n".join(sections)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--for-prompt", action="store_true",
                        help="Output prompt-ready text")
    args = parser.parse_args()

    model = load_model()
    if not model:
        print("No model found. Run build_construction_model.py first.")
        return

    if args.passages:
        passage_ids = args.passages
    else:
        passage_ids = sorted(
            p.stem for p in PASSAGES.glob("*.json")
        )

    SIGNPOST_DIR.mkdir(parents=True, exist_ok=True)

    for pid in passage_ids:
        print(f"\n  Generating signposts for {pid}...")
        en_text = load_english(pid)
        if not en_text:
            continue

        constructions = extract_en_constructions(en_text)
        clause_level = [c for c in constructions if c["scale"] in ("clause", "sentence")]

        if not clause_level:
            print(f"    No clause-level constructions found")
            continue

        # Deduplicate: keep the most specific construction per sentence
        # (relative_clause > coordination_chain > fragment)
        PRIORITY = {"relative_clause": 3, "conditional": 3, "temporal": 3,
                     "coordination_chain": 2, "fragment": 1}
        seen_texts = {}
        deduped = []
        for c in clause_level:
            text_key = c.get("text", "")[:60]
            prio = PRIORITY.get(c["type"], 0)
            if text_key not in seen_texts or prio > seen_texts[text_key]:
                seen_texts[text_key] = prio
                # Remove previous lower-priority entry for same text
                deduped = [d for d in deduped if d.get("text", "")[:60] != text_key]
                deduped.append(c)
        clause_level = deduped

        cards = [build_signpost_card(c, model) for c in clause_level]

        # Save JSON
        with open(SIGNPOST_DIR / f"{pid}.json", "w") as f:
            json.dump(cards, f, ensure_ascii=False, indent=2)

        # Print prompt-ready text
        if args.for_prompt:
            prompt_text = format_signposts_for_prompt(cards)
            print(prompt_text)
        else:
            n_with_data = sum(1 for c in cards if c["distribution"])
            print(f"    {len(cards)} constructions, {n_with_data} with distribution data")

        # Save prompt text
        prompt_text = format_signposts_for_prompt(cards)
        with open(SIGNPOST_DIR / f"{pid}_prompt.txt", "w") as f:
            f.write(prompt_text)


if __name__ == "__main__":
    main()
