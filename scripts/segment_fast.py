#!/usr/bin/env python3
"""
Fast rule-based segmentation for predictable PDF content.

Uses Unicode character detection to classify chunks as Greek or English
without any API calls. Works well for:
- Delphi editions (clear Greek/English/dual sections)
- Loeb volumes (facing-page Greek/English)
- Pure English translations
- Septuagint (mixed Greek/English facing pages)

For commentaries with genuinely mixed content (Dodds, Denniston-Page,
Sansone, etc.), use segment_with_openai.py instead.
"""

import json
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
CLEANED_DIR = Path(__file__).parent.parent / "corpus" / "cleaned" / "manual_text"


def greek_char_ratio(text: str) -> float:
    """What fraction of alphabetic characters are Greek?"""
    greek = 0
    latin = 0
    for ch in text:
        if "\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF":
            greek += 1
        elif ch.isalpha():
            latin += 1
    total = greek + latin
    return greek / total if total > 0 else 0.0


def classify_chunk(text: str) -> str:
    """Classify a chunk by character content."""
    if len(text.strip()) < 20:
        return "noise"

    ratio = greek_char_ratio(text)

    if ratio > 0.7:
        return "greek_text"
    elif ratio > 0.3:
        return "mixed"  # significant Greek + English
    elif ratio > 0.05:
        return "commentary_note"  # English with some Greek quoted
    else:
        # Pure English — check if it's paratext
        lower = text[:500].lower()
        paratext_markers = [
            "contents", "table of contents", "copyright", "isbn",
            "published by", "delphi classics", "introduction",
            "bibliography", "index", "preface", "note on the text",
            "oceanofpdf", "list of", "acknowledgment",
        ]
        if any(m in lower for m in paratext_markers):
            return "paratext"
        return "translation_text"


def segment_file(json_path: Path, conn: sqlite3.Connection) -> dict:
    """Segment a single extracted PDF using rules."""
    with open(json_path) as f:
        doc = json.load(f)

    source_file = doc["source_file"]
    rights = doc.get("rights_status", "unknown")
    may_redist = 1 if doc.get("may_redistribute", False) else 0
    chunks = doc["chunks"]

    # Check if already done
    existing = conn.execute(
        "SELECT COUNT(*) FROM segmented_content WHERE source_file = ?",
        (source_file,),
    ).fetchone()[0]
    if existing >= int(len(chunks) * 0.9):
        return {"status": "skipped", "existing": existing}

    # Clear partial
    if existing > 0:
        conn.execute(
            "DELETE FROM segmented_content WHERE source_file = ?",
            (source_file,),
        )

    stats = {"total": len(chunks), "by_type": {}}

    for chunk in chunks:
        text = chunk["combined_text"]

        # For mixed chunks, try to split into Greek and English sub-segments
        seg_type = classify_chunk(text)
        stats["by_type"][seg_type] = stats["by_type"].get(seg_type, 0) + 1

        if seg_type == "mixed":
            # Split by page breaks and classify each page separately
            pages = chunk.get("pages", [])
            if pages:
                for page_data in pages:
                    page_text = page_data["text"]
                    page_type = classify_chunk(page_text)
                    if page_type in ("noise", "mixed"):
                        # mixed at page level — call it commentary
                        if page_type == "mixed":
                            page_type = "commentary_note"
                        else:
                            continue
                    conn.execute(
                        """INSERT INTO segmented_content
                           (source_file, source_page_start, source_page_end,
                            segment_type, text_content, rights_status,
                            may_redistribute, classification_method)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 'rule_based')""",
                        (source_file, page_data["page"], page_data["page"],
                         page_type, page_text, rights, may_redist),
                    )
            else:
                # No page-level data, store whole chunk
                conn.execute(
                    """INSERT INTO segmented_content
                       (source_file, source_page_start, source_page_end,
                        segment_type, text_content, rights_status,
                        may_redistribute, classification_method)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'rule_based')""",
                    (source_file, chunk["start_page"], chunk["end_page"],
                     seg_type, text, rights, may_redist),
                )
        elif seg_type != "noise":
            conn.execute(
                """INSERT INTO segmented_content
                   (source_file, source_page_start, source_page_end,
                    segment_type, text_content, rights_status,
                    may_redistribute, classification_method)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'rule_based')""",
                (source_file, chunk["start_page"], chunk["end_page"],
                 seg_type, text, rights, may_redist),
            )

    conn.commit()
    return {"status": "done", "total": len(chunks), "by_type": stats["by_type"]}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast rule-based segmentation")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--input-dir", type=Path, default=CLEANED_DIR)
    parser.add_argument("--skip-pattern", type=str, default="Liddell-Scott",
                        help="Skip files matching this pattern")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS segmented_content (
            segment_id INTEGER PRIMARY KEY,
            source_file TEXT NOT NULL,
            source_page_start INTEGER,
            source_page_end INTEGER,
            segment_type TEXT,
            text_content TEXT NOT NULL,
            reference TEXT,
            rights_status TEXT,
            may_redistribute INTEGER DEFAULT 0,
            classification_method TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    json_files = sorted(args.input_dir.glob("*.json"))
    print(f"Processing {len(json_files)} files...")

    total_done = 0
    total_skipped = 0

    for jf in json_files:
        name = jf.stem
        if args.skip_pattern and args.skip_pattern in name:
            continue

        result = segment_file(jf, conn)

        if result["status"] == "skipped":
            total_skipped += 1
        else:
            total_done += 1
            print(f"  {name[:60]}: {result['by_type']}")

    conn.close()

    # Final stats
    conn = sqlite3.connect(args.db)
    total = conn.execute("SELECT COUNT(*) FROM segmented_content").fetchone()[0]
    files = conn.execute("SELECT COUNT(DISTINCT source_file) FROM segmented_content").fetchone()[0]
    print(f"\nDone: {total_done} files segmented, {total_skipped} skipped")
    print(f"Total: {files} files, {total:,} segments in DB")

    print("\nBy type:")
    for row in conn.execute(
        "SELECT segment_type, COUNT(*) FROM segmented_content GROUP BY segment_type ORDER BY COUNT(*) DESC"
    ).fetchall():
        print(f"  {row[0]}: {row[1]:,}")
    conn.close()


if __name__ == "__main__":
    main()
