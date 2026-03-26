#!/usr/bin/env python3
"""
Contextual vocabulary lookup from the parallel corpus.

For each content word in a McCarthy sentence, finds how Greek translators
handled that word (or a synonym) in a similar grammatical context.

Returns attestation-backed vocabulary suggestions, not generic word lists.

Usage:
  python3 scripts/vocab_lookup.py "He is pale and thin, he wears a thin and ragged linen shirt."
  python3 scripts/vocab_lookup.py --passage 001_see_the_child
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT.parent / "db" / "diachronic.db"
PASSAGES = ROOT / "passages"

_en_nlp = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        import stanza
        _en_nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                                   verbose=False)
    return _en_nlp


# Common English words that won't yield useful Greek matches
STOPWORDS = {
    "be", "have", "do", "say", "go", "get", "make", "know", "think", "take",
    "come", "see", "want", "give", "use", "find", "tell", "ask", "work",
    "seem", "feel", "try", "leave", "call", "man", "woman", "thing", "time",
    "way", "day", "year", "people", "part", "place", "case", "point",
    "hand", "eye", "head", "face", "side", "one", "two", "first", "last",
    "new", "old", "good", "great", "high", "long", "large", "small",
    "other", "own", "right", "same", "much", "more", "most", "also",
}


def extract_content_words(text: str) -> list[dict]:
    """Extract content words with their grammatical roles using stanza."""
    nlp = _get_en_nlp()
    doc = nlp(text)

    words = []
    for sent in doc.sentences:
        for w in sent.words:
            if w.upos in ("NOUN", "VERB", "ADJ", "ADV") and w.lemma.lower() not in STOPWORDS:
                if len(w.lemma) < 3:
                    continue

                # Determine grammatical role
                role = w.deprel
                head = next((h for h in sent.words if h.id == w.head), None)

                context = ""
                if w.upos == "ADJ" and head:
                    context = f"modifying '{head.lemma}'"
                elif w.upos == "NOUN" and role in ("nsubj", "nsubj:pass"):
                    context = "as subject"
                elif w.upos == "NOUN" and role == "obj":
                    context = "as object"
                elif w.upos == "NOUN" and role == "obl":
                    prep = next((c for c in sent.words if c.head == w.id and c.deprel == "case"), None)
                    if prep:
                        context = f"after '{prep.text}'"
                elif w.upos == "VERB" and role == "root":
                    context = "main verb"

                words.append({
                    "text": w.text,
                    "lemma": w.lemma,
                    "upos": w.upos,
                    "role": role,
                    "context": context,
                })

    # Deduplicate by lemma
    seen = set()
    unique = []
    for w in words:
        if w["lemma"] not in seen:
            seen.add(w["lemma"])
            unique.append(w)

    return unique


def lookup_word_in_corpus(lemma: str, upos: str, context: str = "",
                           max_results: int = 3) -> list[dict]:
    """Find how Greek translators rendered this English word in context."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Build search patterns — the lemma and common inflections/synonyms
    patterns = [f"% {lemma} %", f"% {lemma}s %", f"% {lemma}ed %",
                f"% {lemma}ing %", f"% {lemma},%", f"% {lemma}.%"]

    # Also add synonyms for common words
    synonyms = _get_synonyms(lemma, upos)
    for syn in synonyms:
        patterns.extend([f"% {syn} %", f"% {syn}s %", f"% {syn}ed %"])

    results = []
    seen_greek = set()

    for pat in patterns:
        cur.execute("""
            SELECT a.aligned_text, p.greek_text, d.title
            FROM alignments a
            JOIN passages p ON a.passage_id = p.passage_id
            JOIN documents d ON p.document_id = d.document_id
            WHERE a.alignment_method = 'reference_match'
              AND LOWER(a.aligned_text) LIKE LOWER(?)
              AND LENGTH(a.aligned_text) BETWEEN 15 AND 200
              AND LENGTH(p.greek_text) BETWEEN 15 AND 300
            LIMIT 5
        """, (pat,))

        for en, grc, src in cur.fetchall():
            grc_clean = grc.replace("\n", " ").strip()
            # Skip if we've seen very similar Greek
            grc_key = grc_clean[:50]
            if grc_key in seen_greek:
                continue
            seen_greek.add(grc_key)

            results.append({
                "english": en.strip()[:120],
                "greek": grc_clean[:120],
                "source": src,
            })

            if len(results) >= max_results:
                break

        if len(results) >= max_results:
            break

    conn.close()
    return results


def _get_synonyms(lemma: str, upos: str) -> list[str]:
    """Return a few synonyms for common McCarthy vocabulary."""
    syn_map = {
        # Adjectives
        "pale": ["wan", "pallid", "white", "ashen"],
        "thin": ["lean", "gaunt", "slender", "meagre", "emaciated"],
        "ragged": ["tattered", "torn", "worn"],
        "dark": ["gloomy", "murky", "sombre", "black"],
        # Nouns
        "fire": ["hearth", "flame", "blaze"],
        "wolf": ["wolves"],
        "wolves": ["wolf"],
        "woods": ["forest", "grove", "wood"],
        "field": ["plain", "meadow"],
        "snow": ["frost", "ice"],
        "drink": ["wine", "drunk", "drunken", "intoxicated"],
        "shirt": ["garment", "tunic", "cloth"],
        "child": ["boy", "lad", "youth"],
        "poet": ["bard", "singer"],
        "schoolmaster": ["teacher", "tutor", "instructor"],
        # Verbs
        "crouch": ["squat", "cower", "stoop"],
        "stoke": ["kindle", "tend", "feed"],
        "harbor": ["shelter", "conceal", "hide"],
        "lie": ["recline", "rest"],
        "quote": ["recite", "repeat", "cite"],
        "watch": ["observe", "gaze", "behold"],
    }
    return syn_map.get(lemma.lower(), [])


def format_vocab_for_prompt(word: dict, corpus_hits: list[dict]) -> str:
    """Format vocabulary lookup as prompt text."""
    if not corpus_hits:
        return ""

    role_desc = f" ({word['context']})" if word['context'] else ""
    lines = [f"'{word['text']}' [{word['upos'].lower()}{role_desc}]:"]

    for hit in corpus_hits:
        lines.append(f"  [{hit['source'][:25]}] \"{hit['english'][:60]}\"")
        lines.append(f"    → {hit['greek'][:60]}")

    return "\n".join(lines)


def lookup_passage(passage_id: str) -> str:
    """Generate vocabulary section for a BM passage."""
    p_path = PASSAGES / f"{passage_id}.json"
    if not p_path.exists():
        return ""

    en_text = json.load(open(p_path)).get("text", "")
    content_words = extract_content_words(en_text)

    sections = ["## Vocabulary Guidance (from parallel corpus)\n"]

    for w in content_words:
        hits = lookup_word_in_corpus(w["lemma"], w["upos"], w["context"])
        formatted = format_vocab_for_prompt(w, hits)
        if formatted:
            sections.append(formatted)

    return "\n\n".join(sections)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--passage", type=str)
    args = parser.parse_args()

    if args.passage:
        print(lookup_passage(args.passage))
    elif args.text:
        content_words = extract_content_words(args.text)
        for w in content_words:
            hits = lookup_word_in_corpus(w["lemma"], w["upos"], w["context"])
            formatted = format_vocab_for_prompt(w, hits)
            if formatted:
                print(formatted)
                print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
