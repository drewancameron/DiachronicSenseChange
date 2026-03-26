#!/usr/bin/env python3
"""
Align Greek passages to English translations using canonical references.

Perseus TEI files share canonical reference schemes but with different
structures:
- Greek: <l n="1"> line elements within <div subtype="Book" n="1">
- English: <p> blocks with <milestone n="1" unit="line"/> markers
           within <div subtype="book/card" n="...">

This script extracts English text at section/card level and aligns
to Greek passages by matching book numbers (and line ranges where possible).
"""

import re
import sqlite3
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
RAW_DIR = Path(__file__).parent.parent / "corpus" / "raw" / "perseus"


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def extract_text_clean(el: ET.Element) -> str:
    """Extract text, skipping note elements."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        tag = strip_ns(child.tag)
        if tag == "note":
            # Skip note content but keep tail
            if child.tail:
                parts.append(child.tail)
        else:
            parts.append(extract_text_clean(child))
            if child.tail:
                parts.append(child.tail)
    return "".join(parts)


def parse_english_tei(filepath: Path) -> dict:
    """
    Parse an English TEI file and extract text organized by section reference.

    Returns: {
        "translator": str,
        "sections": {
            "1": "full text of book/section 1...",
            "1.1": "text of card/subsection 1.1...",
            ...
        }
    }
    """
    try:
        tree = ET.parse(filepath)
    except ET.ParseError:
        return None

    root = tree.getroot()
    for el in root.iter():
        el.tag = strip_ns(el.tag)

    result = {"translator": "", "sections": {}}

    # Get translator
    editor_el = root.find(".//titleStmt/editor[@role='translator']")
    if editor_el is not None:
        result["translator"] = "".join(editor_el.itertext()).strip()
    else:
        editor_el = root.find(".//titleStmt/editor")
        if editor_el is not None:
            result["translator"] = "".join(editor_el.itertext()).strip()

    body = root.find(".//body")
    if body is None:
        return result

    def walk_divs(element, ref_parts=None):
        if ref_parts is None:
            ref_parts = []

        for child in element:
            tag = child.tag
            n = child.get("n", "")
            subtype = child.get("subtype", "")

            if tag == "div" and n:
                # Skip URN-style top-level divs
                if n.startswith("urn:"):
                    walk_divs(child, ref_parts)
                    continue

                current_ref = ref_parts + [n]
                ref_key = ".".join(current_ref)

                # Extract all text from this div level
                text = extract_text_clean(child)
                if text and len(text.strip()) > 10:
                    result["sections"][ref_key] = text.strip()

                # Also recurse for finer-grained sections
                walk_divs(child, current_ref)

            elif tag in ("div1", "div2", "div3") and n:
                current_ref = ref_parts + [n]
                ref_key = ".".join(current_ref)
                text = extract_text_clean(child)
                if text and len(text.strip()) > 10:
                    result["sections"][ref_key] = text.strip()
                walk_divs(child, current_ref)

    walk_divs(body)
    return result


def extract_ref_book(ref: str) -> str | None:
    """
    Extract the book/section number from a full reference.
    "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2.1.16" -> "1"
    "1.16" -> "1"
    """
    # Strip CTS URN prefix
    if "perseus-grc" in ref or "perseus-eng" in ref:
        parts = ref.split(".")
        # Find the part after the edition identifier
        for i, p in enumerate(parts):
            if "perseus-" in p:
                if i + 1 < len(parts):
                    return parts[i + 1]
    # Simple dotted reference
    parts = ref.split(".")
    if parts:
        return parts[0]
    return None


def extract_ref_section(ref: str) -> str | None:
    """
    Extract section reference (book.line or book.section).
    "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2.1.16" -> "1.16"
    """
    if "perseus-grc" in ref or "perseus-eng" in ref:
        parts = ref.split(".")
        for i, p in enumerate(parts):
            if "perseus-" in p:
                remaining = parts[i + 1:]
                if remaining:
                    return ".".join(remaining)
    parts = ref.split(".")
    return ref


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Align translations to Greek passages")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    raw_dir = args.raw_dir
    eng_files = sorted(raw_dir.rglob("*eng*.xml"))
    print(f"Found {len(eng_files)} English translation files")

    total_aligned = 0
    files_with_alignments = 0

    for eng_file in eng_files:
        parsed = parse_english_tei(eng_file)
        if not parsed or not parsed["sections"]:
            continue

        # Extract TLG IDs from path
        parts = eng_file.parts
        tlg_author = tlg_work = ""
        for i, p in enumerate(parts):
            if p.startswith("tlg") and len(p) == 7:
                tlg_author = p
                if i + 1 < len(parts) and parts[i + 1].startswith("tlg"):
                    tlg_work = parts[i + 1]

        if not tlg_author:
            continue

        work_key = f"{tlg_author}.{tlg_work}"

        # Find document
        doc_row = conn.execute(
            "SELECT document_id FROM documents WHERE tlg_work_id = ?",
            (work_key,),
        ).fetchone()
        if not doc_row:
            continue
        doc_id = doc_row[0]

        # Get or create translation record
        trans_row = conn.execute(
            "SELECT translation_id FROM translations WHERE source_url = ?",
            (str(eng_file),),
        ).fetchone()

        if not trans_row:
            translator = parsed["translator"] or "unknown"
            cur = conn.execute(
                """INSERT INTO translations
                   (document_id, translator, source_url, rights_status, has_notes)
                   VALUES (?, ?, ?, 'creative_commons', 1)""",
                (doc_id, translator, str(eng_file)),
            )
            trans_id = cur.lastrowid
        else:
            trans_id = trans_row[0]

        # Get Greek passages for this document
        greek_passages = conn.execute(
            "SELECT passage_id, reference FROM passages WHERE document_id = ?",
            (doc_id,),
        ).fetchall()

        if not greek_passages:
            continue

        # Build index of Greek passages by book number
        greek_by_book = defaultdict(list)
        for pid, ref in greek_passages:
            book = extract_ref_book(ref)
            section = extract_ref_section(ref)
            if book:
                greek_by_book[book].append((pid, section))

        # Try to align English sections to Greek passages
        file_aligned = 0
        eng_sections = parsed["sections"]

        for pid, ref in greek_passages:
            section = extract_ref_section(ref)
            if not section:
                continue

            # Try exact section match first
            eng_text = None
            confidence = 0.0

            if section in eng_sections:
                eng_text = eng_sections[section]
                confidence = 0.85
            else:
                # Try book-level match
                book = extract_ref_book(ref)
                if book and book in eng_sections:
                    # Book-level alignment — lower confidence, truncate text
                    full_text = eng_sections[book]
                    eng_text = full_text[:500] + "..." if len(full_text) > 500 else full_text
                    confidence = 0.4

            if eng_text:
                conn.execute(
                    """INSERT OR IGNORE INTO alignments
                       (passage_id, translation_id, aligned_text,
                        alignment_type, alignment_confidence, alignment_method)
                       VALUES (?, ?, ?, 'translation', ?, 'reference_match')""",
                    (pid, trans_id, eng_text, confidence),
                )
                file_aligned += 1

        if file_aligned > 0:
            files_with_alignments += 1
            total_aligned += file_aligned
            conn.commit()

        if files_with_alignments % 20 == 0 and files_with_alignments > 0:
            print(f"  Processed {files_with_alignments} files with alignments, "
                  f"{total_aligned} total alignments...")

    conn.commit()

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM alignments").fetchone()[0]
    high_conf = conn.execute(
        "SELECT COUNT(*) FROM alignments WHERE alignment_confidence >= 0.8"
    ).fetchone()[0]
    trans_count = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]

    print(f"\nAlignment complete:")
    print(f"  Translation files processed: {files_with_alignments}")
    print(f"  Translation records: {trans_count}")
    print(f"  Total alignments: {total}")
    print(f"  High-confidence (≥0.8): {high_conf}")

    conn.close()


if __name__ == "__main__":
    main()
