#!/usr/bin/env python3
"""
WP2 Task 2.3: Align extracted notes to passages and occurrences.

Links the 48K extracted notes (from TEI and PDF segmentation) back to
specific passages and target token occurrences by:
1. Reference matching (e.g., note ref "1.5.2" → passage ref "1.5.2")
2. Lemma mention matching (note text mentions a pilot lemma → link to occurrences)
3. Source-file matching (notes from commentary on X → passages from X)
"""

import re
import sqlite3
import unicodedata
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"


def normalize_greek(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower()


def setup_tables(conn):
    """Create the note-alignment linking table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_alignments (
            alignment_id INTEGER PRIMARY KEY,
            note_id INTEGER NOT NULL,
            passage_id INTEGER,
            occurrence_id INTEGER,
            alignment_method TEXT,  -- 'reference_match', 'lemma_mention', 'source_match'
            confidence REAL DEFAULT 0.5,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (note_id) REFERENCES extracted_notes(note_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_note_align_note
            ON note_alignments(note_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_note_align_passage
            ON note_alignments(passage_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_note_align_occurrence
            ON note_alignments(occurrence_id)
    """)


def align_by_reference(conn) -> int:
    """
    Match notes to passages by canonical reference.
    E.g., a note with reference "1.5.2" links to passages with matching refs.
    """
    # Get notes with references
    notes = conn.execute("""
        SELECT note_id, reference, source_file
        FROM extracted_notes
        WHERE reference IS NOT NULL AND reference != '?' AND reference != 'header'
    """).fetchall()

    # Build a passage reference index by source file
    # Notes from Perseus English files → match to Greek passages of same work
    aligned = 0
    batch = []

    for note_id, ref, source_file in notes:
        # Extract TLG work ID from source file path
        tlg_match = re.findall(r'(tlg\d{4})[./]', source_file)
        if not tlg_match:
            continue

        # Find passages with matching reference and same author
        # Use LIKE to handle reference prefix matching
        passages = conn.execute("""
            SELECT p.passage_id FROM passages p
            JOIN segments s ON p.segment_id = s.segment_id
            JOIN editions e ON s.edition_id = e.edition_id
            WHERE e.source_url LIKE ? AND p.reference LIKE ?
            LIMIT 5
        """, (f"%{tlg_match[0]}%", f"%{ref}%")).fetchall()

        for (pid,) in passages:
            batch.append((note_id, pid, 'reference_match', 0.7))
            aligned += 1

        if len(batch) >= 1000:
            conn.executemany(
                """INSERT OR IGNORE INTO note_alignments
                   (note_id, passage_id, alignment_method, confidence)
                   VALUES (?, ?, ?, ?)""",
                batch,
            )
            conn.commit()
            batch = []

    if batch:
        conn.executemany(
            """INSERT OR IGNORE INTO note_alignments
               (note_id, passage_id, alignment_method, confidence)
               VALUES (?, ?, ?, ?)""",
            batch,
        )
        conn.commit()

    return aligned


def align_by_lemma_mention(conn, pilot_lemmata: list[str]) -> int:
    """
    Find notes that mention a pilot lemma and link them to
    occurrences of that lemma.
    """
    # Get lexical and commentary notes
    notes = conn.execute("""
        SELECT note_id, note_text, reference, source_file
        FROM extracted_notes
        WHERE note_type IN ('lexical_note', 'commentary_note')
        AND length(note_text) > 30
    """).fetchall()

    aligned = 0
    batch = []

    for note_id, note_text, ref, source_file in notes:
        note_lower = note_text.lower()
        note_norm = normalize_greek(note_text)

        for lemma in pilot_lemmata:
            lemma_norm = normalize_greek(lemma)
            lemma_trans = {
                "κόσμος": "kosmos", "λόγος": "logos", "ψυχή": "psyche",
                "ἀρετή": "arete", "δίκη": "dike", "τέχνη": "techne",
                "νόμος": "nomos", "φύσις": "physis", "δαίμων": "daimon",
                "σῶμα": "soma", "θεός": "theos", "χάρις": "charis",
            }.get(lemma, "")

            # Check if note mentions this lemma (Greek or transliterated)
            if lemma_norm in note_norm or lemma_trans in note_lower:
                # Find occurrences of this lemma near the note's reference
                if ref and ref != '?':
                    occs = conn.execute("""
                        SELECT o.occurrence_id FROM occurrences o
                        JOIN passages p ON o.passage_id = p.passage_id
                        WHERE o.lemma = ? AND p.reference LIKE ?
                        LIMIT 3
                    """, (lemma, f"%{ref}%")).fetchall()
                else:
                    # No reference — just link to any occurrence of this lemma
                    # from the same author/source
                    occs = conn.execute("""
                        SELECT o.occurrence_id FROM occurrences o
                        WHERE o.lemma = ?
                        LIMIT 1
                    """, (lemma,)).fetchall()

                for (occ_id,) in occs:
                    batch.append((note_id, occ_id, 'lemma_mention', 0.6))
                    aligned += 1

        if len(batch) >= 1000:
            conn.executemany(
                """INSERT OR IGNORE INTO note_alignments
                   (note_id, occurrence_id, alignment_method, confidence)
                   VALUES (?, ?, ?, ?)""",
                batch,
            )
            conn.commit()
            batch = []

    if batch:
        conn.executemany(
            """INSERT OR IGNORE INTO note_alignments
               (note_id, occurrence_id, alignment_method, confidence)
               VALUES (?, ?, ?, ?)""",
            batch,
        )
        conn.commit()

    return aligned


def main():
    conn = sqlite3.connect(DB_PATH)
    setup_tables(conn)

    # Clear previous alignments
    conn.execute("DELETE FROM note_alignments")
    conn.commit()

    pilot_lemmata = [
        "κόσμος", "λόγος", "ψυχή", "ἀρετή", "δίκη", "τέχνη",
        "νόμος", "φύσις", "δαίμων", "σῶμα", "θεός", "χάρις",
    ]

    print("Phase 1: Align notes by reference...")
    n1 = align_by_reference(conn)
    print(f"  {n1:,} note-passage alignments")

    print("\nPhase 2: Align notes by lemma mention...")
    n2 = align_by_lemma_mention(conn, pilot_lemmata)
    print(f"  {n2:,} note-occurrence alignments")

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM note_alignments").fetchone()[0]
    by_method = conn.execute(
        "SELECT alignment_method, COUNT(*) FROM note_alignments GROUP BY alignment_method"
    ).fetchall()

    print(f"\nTotal note alignments: {total:,}")
    for method, count in by_method:
        print(f"  {method}: {count:,}")

    conn.close()


if __name__ == "__main__":
    main()
