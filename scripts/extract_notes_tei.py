#!/usr/bin/env python3
"""
Extract notes, footnotes, and paratext from Perseus TEI XML files.

This is the rule-based (free) layer of the segmentation pipeline.
Perseus TEI files contain <note> elements inline within the text.
This script extracts them as separate typed segments and links them
back to their parent passage.

No API calls needed — this is purely structural extraction.
"""

import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
RAW_DIR = Path(__file__).parent.parent / "corpus" / "raw" / "perseus"


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def extract_text_only(el: ET.Element) -> str:
    """Extract text, skipping child elements."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(extract_text_only(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def classify_note(note_el: ET.Element, context: str = "") -> str:
    """
    Classify a <note> element by type based on attributes and content.
    Returns one of: footnote, commentary_note, lexical_note, apparatus, paratext
    """
    resp = note_el.get("resp", "").lower()
    note_type = note_el.get("type", "").lower()
    place = note_el.get("place", "").lower()
    text = extract_text_only(note_el).lower()

    # Check explicit type attributes
    if note_type in ("textual", "critical"):
        return "apparatus"
    if note_type in ("editorial", "commentary"):
        return "commentary_note"
    if note_type == "marginal":
        return "paratext"

    # Check resp attribute
    if resp in ("loeb", "editor"):
        return "footnote"
    if resp in ("translator",):
        return "footnote"

    # Content-based heuristics
    if len(text) < 5:
        return "footnote"  # Short notes are typically numbered footnotes

    # Look for lexical discussion markers
    lexical_markers = ["cf.", "lit.", "literally", "i.e.", "sc.", "scil.",
                       "the word", "meaning", "sense of", "translat"]
    if any(m in text for m in lexical_markers):
        return "lexical_note"

    # Look for textual apparatus markers
    apparatus_markers = ["ms.", "mss.", "codex", "codd.", "reading",
                         "variant", "omit", "insert", "corrupt"]
    if any(m in text for m in apparatus_markers):
        return "apparatus"

    # Default to footnote
    return "footnote"


def extract_notes_from_file(filepath: Path) -> list[dict]:
    """
    Extract all <note> elements from a TEI file with their context.
    """
    try:
        tree = ET.parse(filepath)
    except ET.ParseError:
        return []

    root = tree.getroot()
    for el in root.iter():
        el.tag = strip_ns(el.tag)

    notes = []

    # Track current position in the document hierarchy
    def walk(element, ref_parts=None):
        if ref_parts is None:
            ref_parts = []

        n = element.get("n", "")
        if element.tag in ("div", "div1", "div2", "div3") and n:
            if not n.startswith("urn:"):
                current_ref = ref_parts + [n]
            else:
                current_ref = ref_parts
        else:
            current_ref = ref_parts

        for child in element:
            if child.tag == "note":
                note_text = extract_text_only(child)
                if note_text and len(note_text.strip()) > 0:
                    note_type = classify_note(child)
                    ref = ".".join(current_ref) if current_ref else "?"

                    # Get surrounding context
                    parent_text = ""
                    if element.text:
                        parent_text = element.text[:100]

                    notes.append({
                        "reference": ref,
                        "note_text": note_text.strip(),
                        "note_type": note_type,
                        "resp": child.get("resp", ""),
                        "anchored": child.get("anchored", ""),
                        "parent_context": parent_text,
                        "source_file": str(filepath),
                    })
            else:
                walk(child, current_ref)

    body = root.find(".//body")
    if body is not None:
        walk(body)

    # Also extract from teiHeader (prefaces, etc.)
    header = root.find(".//teiHeader")
    if header is not None:
        for note in header.iter("note"):
            text = extract_text_only(note)
            if text and len(text) > 10:
                notes.append({
                    "reference": "header",
                    "note_text": text.strip(),
                    "note_type": "paratext",
                    "resp": note.get("resp", ""),
                    "anchored": "",
                    "parent_context": "",
                    "source_file": str(filepath),
                })

    return notes


def detect_language(filepath: Path) -> str:
    name = filepath.name.lower()
    if "grc" in name:
        return "grc"
    elif "eng" in name:
        return "eng"
    return "unknown"


def parse_tlg_ids(filepath: Path) -> tuple[str, str]:
    parts = filepath.parts
    tlg_author = tlg_work = ""
    for i, p in enumerate(parts):
        if p.startswith("tlg") and len(p) == 7:
            tlg_author = p
            if i + 1 < len(parts) and parts[i + 1].startswith("tlg"):
                tlg_work = parts[i + 1]
    return tlg_author, tlg_work


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract notes from Perseus TEI")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print notes without storing")
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    xml_files = sorted(args.raw_dir.rglob("*.xml"))
    if args.limit_files:
        xml_files = xml_files[:args.limit_files]

    print(f"Processing {len(xml_files)} TEI files...")

    stats = {
        "files_processed": 0,
        "files_with_notes": 0,
        "total_notes": 0,
        "by_type": {},
        "by_language": {"grc": 0, "eng": 0, "unknown": 0},
    }

    all_notes = []

    for filepath in xml_files:
        notes = extract_notes_from_file(filepath)
        lang = detect_language(filepath)
        stats["files_processed"] += 1

        if notes:
            stats["files_with_notes"] += 1
            stats["total_notes"] += len(notes)
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + len(notes)

            for n in notes:
                n["language"] = lang
                ntype = n["note_type"]
                stats["by_type"][ntype] = stats["by_type"].get(ntype, 0) + 1

            all_notes.extend(notes)

        if stats["files_processed"] % 100 == 0:
            print(f"  Processed {stats['files_processed']}/{len(xml_files)} files, "
                  f"{stats['total_notes']} notes so far...")

    print(f"\n{'='*50}")
    print(f"Extraction complete:")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files with notes: {stats['files_with_notes']}")
    print(f"  Total notes extracted: {stats['total_notes']}")
    print(f"\n  By type:")
    for ntype, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        print(f"    {ntype}: {count}")
    print(f"\n  By source language:")
    for lang, count in sorted(stats["by_language"].items(), key=lambda x: -x[1]):
        print(f"    {lang}: {count}")

    if args.dry_run:
        print(f"\n--- Sample notes ---")
        for n in all_notes[:10]:
            print(f"\n  [{n['note_type']}] ref={n['reference']} lang={n['language']}")
            print(f"  {n['note_text'][:150]}")
        return

    # Store in database
    conn = sqlite3.connect(args.db)

    # Create a notes table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extracted_notes (
            note_id INTEGER PRIMARY KEY,
            source_file TEXT NOT NULL,
            language TEXT,
            reference TEXT,
            note_text TEXT NOT NULL,
            note_type TEXT CHECK(note_type IN (
                'footnote', 'commentary_note', 'lexical_note',
                'apparatus', 'paratext', 'noise'
            )),
            resp TEXT,
            parent_context TEXT,
            classification_method TEXT DEFAULT 'rule_based',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_notes_type ON extracted_notes(note_type)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_notes_ref ON extracted_notes(reference)
    """)

    # Clear previous extractions
    conn.execute("DELETE FROM extracted_notes WHERE classification_method = 'rule_based'")

    for n in all_notes:
        conn.execute(
            """INSERT INTO extracted_notes
               (source_file, language, reference, note_text, note_type,
                resp, parent_context, classification_method)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'rule_based')""",
            (n["source_file"], n["language"], n["reference"],
             n["note_text"], n["note_type"], n["resp"], n["parent_context"]),
        )

    conn.commit()
    final_count = conn.execute("SELECT COUNT(*) FROM extracted_notes").fetchone()[0]
    conn.close()

    print(f"\nStored {final_count} notes in database")


if __name__ == "__main__":
    main()
