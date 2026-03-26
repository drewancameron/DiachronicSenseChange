#!/usr/bin/env python3
"""
Find occurrences of pilot lemmata in the ingested Greek passages.

Searches for surface forms of target lemmata using Unicode-aware
matching. Creates occurrence records with local Greek context windows.

Note: This performs surface-form matching. For better results,
use a morphological analyzer (e.g., CLTK) to identify lemmatized forms.
This script provides a baseline that can be improved with morphological data.
"""

import re
import sqlite3
import unicodedata
from pathlib import Path

import yaml

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
LEMMATA_PATH = Path(__file__).parent.parent / "config" / "pilot_lemmata.yaml"

# Context window size (words on each side of target)
CONTEXT_WINDOW = 15


def normalize_greek(text: str) -> str:
    """Normalize Greek text for matching: strip accents, lowercase."""
    # Decompose unicode, strip combining marks (accents, breathings)
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower()


def build_surface_forms(lemma_greek: str) -> list[str]:
    """
    Generate common surface forms for a Greek lemma.

    This is a simplified approach — a proper implementation would use
    morphological tables. For now we match the normalized stem.
    """
    base = normalize_greek(lemma_greek)
    # Strip common endings to get approximate stem
    stems = [base]

    # For nouns: try removing common endings
    for ending in ["ος", "ον", "η", "ης", "α", "ις", "ων", "ους"]:
        norm_ending = normalize_greek(ending)
        if base.endswith(norm_ending) and len(base) > len(norm_ending) + 1:
            stem = base[:-len(norm_ending)]
            stems.append(stem)

    return stems


def find_in_passage(text: str, lemma_greek: str) -> list[dict]:
    """Find occurrences of a lemma in a Greek passage."""
    norm_text = normalize_greek(text)
    stems = build_surface_forms(lemma_greek)

    # Also try matching the full normalized lemma form
    norm_lemma = normalize_greek(lemma_greek)

    matches = []
    words = text.split()
    norm_words = [normalize_greek(w) for w in words]

    for i, (word, norm_word) in enumerate(zip(words, norm_words)):
        # Check if this word matches any stem
        is_match = False
        for stem in stems:
            if len(stem) >= 3 and norm_word.startswith(stem):
                is_match = True
                break

        if not is_match:
            continue

        # Extract context window
        start = max(0, i - CONTEXT_WINDOW)
        end = min(len(words), i + CONTEXT_WINDOW + 1)
        context = " ".join(words[start:end])

        matches.append({
            "surface_form": word,
            "token_offset": i,
            "context": context,
        })

    return matches


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find pilot lemma occurrences")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--lemmata", type=Path, default=LEMMATA_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max passages to search (for testing)")
    args = parser.parse_args()

    # Load pilot lemmata
    with open(args.lemmata) as f:
        config = yaml.safe_load(f)

    lemmata = config["pilot_lemmata"]
    print(f"Searching for {len(lemmata)} pilot lemmata")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    # Get all Greek passages
    query = """
        SELECT p.passage_id, p.document_id, p.reference, p.greek_text,
               d.period, d.genre
        FROM passages p
        JOIN documents d ON p.document_id = d.document_id
    """
    if args.limit:
        query += f" LIMIT {args.limit}"

    passages = conn.execute(query).fetchall()
    print(f"Searching {len(passages)} Greek passages")

    total_occurrences = 0

    for lemma_info in lemmata:
        lemma_greek = lemma_info["lemma_greek"]
        lemma_translit = lemma_info["lemma_transliteration"]
        count = 0

        for passage_id, doc_id, reference, greek_text, period, genre in passages:
            if not greek_text:
                continue

            matches = find_in_passage(greek_text, lemma_greek)
            for m in matches:
                # Create token record
                cur = conn.execute(
                    """INSERT INTO tokens (passage_id, surface_form, lemma,
                                          token_offset, is_target)
                       VALUES (?, ?, ?, ?, 1)""",
                    (passage_id, m["surface_form"], lemma_greek,
                     m["token_offset"]),
                )
                token_id = cur.lastrowid

                # Create occurrence record
                conn.execute(
                    """INSERT INTO occurrences
                       (token_id, lemma, passage_id, document_id,
                        greek_context, period, genre)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (token_id, lemma_greek, passage_id, doc_id,
                     m["context"], period, genre),
                )
                count += 1

        conn.commit()
        total_occurrences += count
        print(f"  {lemma_translit} ({lemma_greek}): {count} occurrences")

    conn.close()
    print(f"\nTotal occurrences found: {total_occurrences}")


if __name__ == "__main__":
    main()
