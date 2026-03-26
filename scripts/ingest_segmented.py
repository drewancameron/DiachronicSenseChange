#!/usr/bin/env python3
"""
Ingest segmented PDF content into the main database tables.

Takes the segmented_content table (from PDF segmentation) and:
1. Creates author/document records for new sources
2. Inserts Greek text segments as passages
3. Creates translation records for English segments
4. Links commentary/lexical notes to the notes system
5. Runs occurrence finding on new Greek passages

This bridges the PDF segmentation output into the same schema
used by the Perseus TEI pipeline.
"""

import re
import sqlite3
import unicodedata
from pathlib import Path

import yaml

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
CONFIG_DIR = Path(__file__).parent.parent / "config"
LEMMATA_PATH = CONFIG_DIR / "pilot_lemmata.yaml"

# Map source filenames to author/period metadata
# This supplements the YAML source registries
SOURCE_METADATA = {
    "Delphi_Complete_Works_of_Aristotle": ("Aristotle", "Ἀριστοτέλης", "classical", -384, -322),
    "Delphi_Complete_Works_of_Thucydides": ("Thucydides", "Θουκυδίδης", "classical", -460, -400),
    "Delphi_Complete_Works_of_Euripides": ("Euripides", "Εὐριπίδης", "classical", -480, -406),
    "Delphi_Complete_Works_of_Aristophanes": ("Aristophanes", "Ἀριστοφάνης", "classical", -446, -386),
    "Delphi_Complete_Works_of_Polybius": ("Polybius", "Πολύβιος", "hellenistic", -200, -118),
    "Delphi_Complete_Works_of_Apollonius": ("Apollonius of Rhodes", "Ἀπολλώνιος", "hellenistic", -295, -215),
    "Delphi_Complete_Works_of_Dio_Chrysostom": ("Dio Chrysostom", "Δίων Χρυσόστομος", "imperial", 40, 115),
    "Delphi_Complete_Works_of_Appian": ("Appian", "Ἀππιανός", "imperial", 95, 165),
    "Complete_Works_of_Diodorus_Siculus": ("Diodorus Siculus", "Διόδωρος", "hellenistic", -90, -30),
    "Works_of_Nonnus": ("Nonnus", "Νόννος", "imperial", 400, 470),
    "works_of_Hesiod": ("Hesiod", "Ἡσίοδος", "archaic", -700, -650),
    "Septuagint": ("Septuagint", None, "koine", -300, -100),
    "Dodds-Euripides-Bacchae": ("Euripides", "Εὐριπίδης", "classical", -480, -406),
    "Aeschylus-Agamemnon": ("Aeschylus", "Αἰσχύλος", "classical", -525, -456),
    "Himmelhoch": ("Aeschylus", "Αἰσχύλος", "classical", -525, -456),
    "Raeburn": ("Aeschylus", "Αἰσχύλος", "classical", -525, -456),
    "Basel_Commentary": ("Homer", "Ὅμηρος", "homeric", -750, -700),
    "Sansone": ("Plato", "Πλάτων", "classical", -428, -348),
    "plato_the-republic": ("Plato", "Πλάτων", "classical", -428, -348),
    "Leaf": ("Homer", "Ὅμηρος", "homeric", -750, -700),
    "Plutarchs_Lives": ("Plutarch", "Πλούταρχος", "imperial", 46, 120),
    "Plutarch_-_J_L_Moles": ("Plutarch", "Πλούταρχος", "imperial", 46, 120),
    "Lucian_Volume_VI": ("Lucian", "Λουκιανός", "imperial", 125, 180),
    "Odyssey_Books_13-24": ("Homer", "Ὅμηρος", "homeric", -750, -700),
    "If_Not_Winter": ("Sappho", "Σαπφώ", "archaic", -630, -570),
    "Sophocles_0561": ("Sophocles", "Σοφοκλῆς", "classical", -496, -406),
    "bacchae.*dual": ("Euripides", "Εὐριπίδης", "classical", -480, -406),
}

CONTEXT_WINDOW = 15


def normalize_greek(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower()


def has_greek_chars(text: str) -> bool:
    """Check if text contains Greek Unicode characters."""
    for ch in text:
        if "\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF":
            return True
    return False


def match_source_metadata(source_file: str) -> tuple | None:
    """Match a source filename to author metadata."""
    for pattern, meta in SOURCE_METADATA.items():
        if pattern.lower() in source_file.lower():
            return meta
    return None


def ensure_author(conn, name, name_greek, period, fl_start, fl_end):
    """Get or create an author record."""
    row = conn.execute(
        "SELECT author_id FROM authors WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        """INSERT INTO authors (name, name_greek, period, floruit_start, floruit_end)
           VALUES (?, ?, ?, ?, ?)""",
        (name, name_greek, period, fl_start, fl_end),
    )
    return cur.lastrowid


def ensure_document(conn, author_id, title, period, source_file):
    """Get or create a document record."""
    row = conn.execute(
        "SELECT document_id FROM documents WHERE title = ? AND author_id = ?",
        (title, author_id),
    ).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        """INSERT INTO documents (author_id, title, period, notes)
           VALUES (?, ?, ?, ?)""",
        (author_id, title, period, f"From segmented PDF: {source_file}"),
    )
    return cur.lastrowid


def find_lemma_occurrences(text: str, lemma_greek: str) -> list[dict]:
    """Find occurrences of a lemma in Greek text."""
    norm_lemma = normalize_greek(lemma_greek)
    stems = [norm_lemma]
    for ending in ["ος", "ον", "η", "ης", "α", "ις", "ων", "ους"]:
        ne = normalize_greek(ending)
        if norm_lemma.endswith(ne) and len(norm_lemma) > len(ne) + 1:
            stems.append(norm_lemma[:-len(ne)])

    words = text.split()
    norm_words = [normalize_greek(w) for w in words]
    matches = []

    for i, (word, nw) in enumerate(zip(words, norm_words)):
        for stem in stems:
            if len(stem) >= 3 and nw.startswith(stem):
                start = max(0, i - CONTEXT_WINDOW)
                end = min(len(words), i + CONTEXT_WINDOW + 1)
                context = " ".join(words[start:end])
                matches.append({
                    "surface_form": word,
                    "token_offset": i,
                    "context": context,
                })
                break

    return matches


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest segmented PDF content")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--find-occurrences", action="store_true", default=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    # Get all segmented content
    segments = conn.execute("""
        SELECT segment_id, source_file, source_page_start, source_page_end,
               segment_type, text_content, reference, rights_status,
               may_redistribute
        FROM segmented_content
        WHERE segment_type IN ('greek_text', 'translation_text',
                               'commentary_note', 'lexical_note')
        ORDER BY source_file, source_page_start
    """).fetchall()

    print(f"Processing {len(segments)} segments from segmented_content")

    # Load pilot lemmata
    lemmata = []
    if LEMMATA_PATH.exists():
        with open(LEMMATA_PATH) as f:
            config = yaml.safe_load(f)
        lemmata = config.get("pilot_lemmata", [])
        print(f"Loaded {len(lemmata)} pilot lemmata")

    stats = {
        "greek_passages": 0,
        "translation_segments": 0,
        "notes_added": 0,
        "occurrences_found": 0,
    }

    current_source = None
    author_id = None
    doc_id = None
    edition_id = None

    for seg_id, source_file, pg_start, pg_end, seg_type, text, ref, rights, may_redist in segments:
        if not text or len(text.strip()) < 10:
            continue

        # Set up author/document for new source files
        if source_file != current_source:
            current_source = source_file
            meta = match_source_metadata(source_file)
            if meta:
                name, name_greek, period, fl_start, fl_end = meta
                author_id = ensure_author(conn, name, name_greek, period, fl_start, fl_end)
                doc_title = f"{name} (from {source_file[:50]})"
                doc_id = ensure_document(conn, author_id, doc_title, period, source_file)

                # Create edition record
                cur = conn.execute(
                    """INSERT OR IGNORE INTO editions
                       (document_id, source_url, rights_status, format)
                       VALUES (?, ?, ?, 'pdf')""",
                    (doc_id, source_file, rights or "unknown"),
                )
                edition_id = cur.lastrowid or conn.execute(
                    "SELECT edition_id FROM editions WHERE source_url = ?",
                    (source_file,),
                ).fetchone()[0]

                print(f"\n  Source: {name} ({period})")
            else:
                print(f"\n  Source: {source_file} (no metadata match)")
                author_id = doc_id = edition_id = None

        if not doc_id:
            continue

        if seg_type == "greek_text" and has_greek_chars(text):
            # Create segment and passage records
            cur = conn.execute(
                """INSERT INTO segments (edition_id, segment_type, reference, raw_text)
                   VALUES (?, 'book', ?, ?)""",
                (edition_id, ref or f"pages_{pg_start}-{pg_end}", source_file),
            )
            seg_db_id = cur.lastrowid

            cur = conn.execute(
                """INSERT INTO passages
                   (segment_id, document_id, reference, greek_text, word_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (seg_db_id, doc_id, ref or f"p{pg_start}-{pg_end}",
                 text, len(text.split())),
            )
            passage_id = cur.lastrowid
            stats["greek_passages"] += 1

            # Find pilot lemma occurrences
            if args.find_occurrences and lemmata:
                period = meta[2] if meta else None
                for lemma_info in lemmata:
                    matches = find_lemma_occurrences(text, lemma_info["lemma_greek"])
                    for m in matches:
                        cur2 = conn.execute(
                            """INSERT INTO tokens
                               (passage_id, surface_form, lemma, token_offset, is_target)
                               VALUES (?, ?, ?, ?, 1)""",
                            (passage_id, m["surface_form"],
                             lemma_info["lemma_greek"], m["token_offset"]),
                        )
                        conn.execute(
                            """INSERT INTO occurrences
                               (token_id, lemma, passage_id, document_id,
                                greek_context, period, genre)
                               VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                            (cur2.lastrowid, lemma_info["lemma_greek"],
                             passage_id, doc_id, m["context"], period),
                        )
                        stats["occurrences_found"] += 1

        elif seg_type == "translation_text":
            stats["translation_segments"] += 1
            # Store as a translation record linked to the document
            conn.execute(
                """INSERT OR IGNORE INTO translations
                   (document_id, translator, source_url, rights_status)
                   VALUES (?, ?, ?, ?)""",
                (doc_id, "from_pdf_segmentation", source_file, rights or "unknown"),
            )

        elif seg_type in ("commentary_note", "lexical_note"):
            stats["notes_added"] += 1
            conn.execute(
                """INSERT INTO extracted_notes
                   (source_file, language, reference, note_text, note_type,
                    classification_method)
                   VALUES (?, 'eng', ?, ?, ?, 'pdf_segmentation')""",
                (source_file, ref, text, seg_type),
            )

        if stats["greek_passages"] % 200 == 0 and stats["greek_passages"] > 0:
            conn.commit()
            print(f"    {stats['greek_passages']} Greek passages, "
                  f"{stats['occurrences_found']} occurrences...", flush=True)

    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"Ingestion complete:")
    print(f"  Greek passages added: {stats['greek_passages']:,}")
    print(f"  Translation segments: {stats['translation_segments']:,}")
    print(f"  Notes added: {stats['notes_added']:,}")
    print(f"  Pilot lemma occurrences: {stats['occurrences_found']:,}")


if __name__ == "__main__":
    main()
