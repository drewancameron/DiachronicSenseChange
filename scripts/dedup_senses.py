#!/usr/bin/env python3
"""
Deduplicate and clean the sense inventory.

Three operations:
1. Remove Wiktionary entries that are derived/compound words, not senses
2. Merge near-duplicate LLM-extracted senses (fuzzy string matching)
3. Retain Wiktionary's genuine sense definitions as reference senses

The goal is a clean, manageable inventory where each lemma has
5-15 well-defined senses, not 400 compound words.
"""

import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"


def normalize_for_comparison(text: str) -> str:
    """Normalize a sense label for fuzzy comparison."""
    text = text.lower().strip()
    # Remove parenthetical clarifications
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove common filler words
    for word in ['of', 'the', 'a', 'an', 'or', 'and', 'as']:
        text = re.sub(rf'\b{word}\b', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove trailing punctuation
    text = text.strip('.,;:/')
    return text


def jaccard_words(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    words_a = set(normalize_for_comparison(a).split())
    words_b = set(normalize_for_comparison(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def is_compound_or_derived(label: str) -> bool:
    """Check if a 'sense' is actually a derived/compound word, not a definition."""
    # Greek words (with transliteration in parens)
    if re.match(r'^[α-ωἀ-ᾷ]', label):
        return True
    # Transliterated Greek compounds
    if re.match(r'^[a-z]+olog', label):
        return True
    # Meta-commentary about tables/appendices
    if any(x in label.lower() for x in ['table gives', 'appendix:', 'declension',
                                         'inflectional', 'dialectal']):
        return True
    # Very short labels that are just references
    if len(label) < 3:
        return True
    return False


def merge_senses(conn, lemma: str, dry_run: bool = False) -> dict:
    """Merge near-duplicate senses for one lemma."""

    senses = conn.execute("""
        SELECT sense_id, sense_label, notes,
               (SELECT COUNT(*) FROM candidate_labels WHERE sense_id = si.sense_id) as usage_count
        FROM sense_inventory si
        WHERE lemma = ?
        ORDER BY sense_id
    """, (lemma,)).fetchall()

    stats = {"removed_compounds": 0, "merged": 0, "kept": 0}

    # Phase 1: Remove compound/derived words and meta-text
    to_delete = []
    real_senses = []

    for sid, label, notes, usage in senses:
        if is_compound_or_derived(label):
            to_delete.append(sid)
            stats["removed_compounds"] += 1
        else:
            real_senses.append((sid, label, notes, usage))

    # Phase 2: Merge near-duplicates among remaining senses
    # Group senses by similarity
    merged_groups = []  # list of (canonical_id, [member_ids])
    used = set()

    # Sort by usage count (most-used first = canonical)
    real_senses.sort(key=lambda x: -x[3])

    for i, (sid_a, label_a, notes_a, usage_a) in enumerate(real_senses):
        if sid_a in used:
            continue

        group = [sid_a]
        used.add(sid_a)

        for j, (sid_b, label_b, notes_b, usage_b) in enumerate(real_senses):
            if sid_b in used or sid_b == sid_a:
                continue

            similarity = jaccard_words(label_a, label_b)
            # Also check if one label is a substring of the other
            norm_a = normalize_for_comparison(label_a)
            norm_b = normalize_for_comparison(label_b)
            is_substring = norm_a in norm_b or norm_b in norm_a
            if similarity >= 0.4 or is_substring:
                group.append(sid_b)
                used.add(sid_b)
                stats["merged"] += 1

        merged_groups.append((sid_a, group))

    stats["kept"] = len(merged_groups)

    if dry_run:
        return stats

    # Apply deletions
    if to_delete:
        # Reassign any candidate_labels pointing to deleted senses
        # (shouldn't happen since compounds have 0 usage, but be safe)
        for sid in to_delete:
            conn.execute("DELETE FROM candidate_labels WHERE sense_id = ?", (sid,))
            conn.execute("DELETE FROM sense_inventory WHERE sense_id = ?", (sid,))

    # Apply merges
    for canonical_id, group in merged_groups:
        if len(group) > 1:
            # Point all candidate_labels to the canonical sense
            for member_id in group[1:]:
                conn.execute(
                    "UPDATE candidate_labels SET sense_id = ? WHERE sense_id = ?",
                    (canonical_id, member_id),
                )
                conn.execute(
                    "DELETE FROM sense_inventory WHERE sense_id = ?",
                    (member_id,),
                )

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deduplicate sense inventory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    # Before stats
    total_before = conn.execute("SELECT COUNT(*) FROM sense_inventory").fetchone()[0]
    print(f"Sense inventory before: {total_before}")
    print()

    pilot_lemmata = [
        "κόσμος", "λόγος", "ψυχή", "ἀρετή", "δίκη", "τέχνη",
        "νόμος", "φύσις", "δαίμων", "σῶμα", "θεός", "χάρις",
    ]

    translit = {
        "κόσμος": "kosmos", "λόγος": "logos", "ψυχή": "psyche",
        "ἀρετή": "arete", "δίκη": "dike", "τέχνη": "techne",
        "νόμος": "nomos", "φύσις": "physis", "δαίμων": "daimon",
        "σῶμα": "soma", "θεός": "theos", "χάρις": "charis",
    }

    for lemma in pilot_lemmata:
        before = conn.execute(
            "SELECT COUNT(*) FROM sense_inventory WHERE lemma = ?", (lemma,)
        ).fetchone()[0]

        stats = merge_senses(conn, lemma, dry_run=args.dry_run)

        print(f"{translit[lemma]:>10s} ({lemma}): {before:>3d} → "
              f"{stats['kept']:>2d} kept  "
              f"({stats['removed_compounds']} compounds, {stats['merged']} merged)",
              flush=True)

    if not args.dry_run:
        conn.commit()

        # Show final state
        print(f"\n{'='*60}")
        total_after = conn.execute("SELECT COUNT(*) FROM sense_inventory").fetchone()[0]
        print(f"Sense inventory after: {total_after} (was {total_before})")
        print()

        # Show cleaned senses for each lemma
        for lemma in pilot_lemmata:
            senses = conn.execute("""
                SELECT si.sense_label,
                       (SELECT COUNT(*) FROM candidate_labels cl WHERE cl.sense_id = si.sense_id) as n
                FROM sense_inventory si
                WHERE si.lemma = ?
                ORDER BY n DESC
            """, (lemma,)).fetchall()

            print(f"\n{translit[lemma]} ({lemma}): {len(senses)} senses")
            for label, n in senses:
                marker = "●" if n > 0 else "○"
                print(f"  {marker} {label[:70]}" + (f" [{n} occ]" if n > 0 else ""))

    conn.close()


if __name__ == "__main__":
    main()
