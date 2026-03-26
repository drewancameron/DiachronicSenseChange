#!/usr/bin/env python3
"""
Ingest parsed TEI XML files into the SQLite database.

Walks the raw/perseus directory, parses each TEI file,
and populates authors, documents, editions/translations, segments,
and passages tables. Handles both Greek texts and English translations.
"""

import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
RAW_DIR = Path(__file__).parent.parent / "corpus" / "raw" / "perseus"

# TLG author mapping (extending from acquire script)
TLG_AUTHORS = {
    "tlg0012": ("Homer", "Ὅμηρος", "homeric", -750, -700),
    "tlg0020": ("Hesiod", "Ἡσίοδος", "archaic", -700, -650),
    "tlg0019": ("Pindar", "Πίνδαρος", "archaic", -518, -438),
    "tlg0085": ("Aeschylus", "Αἰσχύλος", "classical", -525, -456),
    "tlg0011": ("Sophocles", "Σοφοκλῆς", "classical", -496, -406),
    "tlg0006": ("Euripides", "Εὐριπίδης", "classical", -480, -406),
    "tlg0002": ("Aristophanes", "Ἀριστοφάνης", "classical", -446, -386),
    "tlg0016": ("Herodotus", "Ἡρόδοτος", "classical", -484, -425),
    "tlg0003": ("Thucydides", "Θουκυδίδης", "classical", -460, -400),
    "tlg0032": ("Xenophon", "Ξενοφῶν", "classical", -430, -354),
    "tlg0059": ("Plato", "Πλάτων", "classical", -428, -348),
    "tlg0086": ("Aristotle", "Ἀριστοτέλης", "classical", -384, -322),
    "tlg0014": ("Demosthenes", "Δημοσθένης", "classical", -384, -322),
    "tlg0010": ("Isocrates", "Ἰσοκράτης", "classical", -436, -338),
    "tlg0540": ("Lysias", "Λυσίας", "classical", -445, -380),
    "tlg0557": ("Epictetus", "Ἐπίκτητος", "imperial", 50, 135),
    "tlg0541": ("Polybius", "Πολύβιος", "hellenistic", -200, -118),
    "tlg0007": ("Plutarch", "Πλούταρχος", "imperial", 46, 120),
    "tlg0062": ("Lucian", "Λουκιανός", "imperial", 125, 180),
    "tlg2003": ("Strabo", "Στράβων", "imperial", -64, 24),
    "tlg7000": ("New Testament", None, "koine", 50, 120),
    "tlg0527": ("Septuagint", None, "koine", -300, -100),
}

# Known TLG work IDs for major works
TLG_WORKS = {
    "tlg0012.tlg001": ("Iliad", "Ἰλιάς", "epic", -750),
    "tlg0012.tlg002": ("Odyssey", "Ὀδύσσεια", "epic", -725),
    "tlg0020.tlg001": ("Theogony", "Θεογονία", "epic", -700),
    "tlg0020.tlg002": ("Works and Days", "Ἔργα καὶ Ἡμέραι", "didactic", -700),
    "tlg0020.tlg003": ("Shield of Heracles", "Ἀσπὶς Ἡρακλέους", "epic", -580),
    "tlg0003.tlg001": ("History of the Peloponnesian War", None, "historiography", -411),
    "tlg0016.tlg001": ("Histories", "Ἱστορίαι", "historiography", -440),
}


def strip_ns(tag: str) -> str:
    """Strip XML namespace from tag."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def extract_text(el: ET.Element) -> str:
    """Recursively extract text from element."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def detect_language(filepath: Path) -> str:
    """Detect language from filename."""
    name = filepath.name.lower()
    if "grc" in name:
        return "grc"
    elif "eng" in name:
        return "eng"
    elif "lat" in name:
        return "lat"
    return "unknown"


def parse_tlg_path(filepath: Path) -> tuple[str, str]:
    """Extract TLG author and work IDs from filepath."""
    parts = filepath.parts
    for i, p in enumerate(parts):
        if p.startswith("tlg") and len(p) == 7:
            author_id = p
            if i + 1 < len(parts) and parts[i + 1].startswith("tlg"):
                work_id = parts[i + 1]
            else:
                work_id = ""
            return author_id, work_id
    return "", ""


def parse_tei_metadata(filepath: Path) -> dict:
    """Parse TEI header for metadata."""
    try:
        tree = ET.parse(filepath)
    except ET.ParseError as e:
        return {"error": str(e)}

    root = tree.getroot()

    # Strip namespaces for easier querying
    for el in root.iter():
        el.tag = strip_ns(el.tag)

    meta = {"title": "", "author": "", "editor": "", "translator": ""}

    title_el = root.find(".//titleStmt/title")
    if title_el is not None:
        meta["title"] = extract_text(title_el)

    author_el = root.find(".//titleStmt/author")
    if author_el is not None:
        meta["author"] = extract_text(author_el)

    editor_el = root.find(".//titleStmt/editor")
    if editor_el is not None:
        role = editor_el.get("role", "editor")
        if role == "translator":
            meta["translator"] = extract_text(editor_el)
        else:
            meta["editor"] = extract_text(editor_el)

    return meta


def extract_passages(filepath: Path) -> list[dict]:
    """Extract text passages with references from TEI body."""
    try:
        tree = ET.parse(filepath)
    except ET.ParseError:
        return []

    root = tree.getroot()
    for el in root.iter():
        el.tag = strip_ns(el.tag)

    passages = []

    # Strategy: find the deepest div structure and extract text units
    body = root.find(".//body")
    if body is None:
        return []

    # Collect reference path as we recurse
    def walk_divs(element, ref_parts=None):
        if ref_parts is None:
            ref_parts = []

        tag = element.tag
        n = element.get("n", "")

        if tag == "div" and n:
            current_ref = ref_parts + [n]
        elif tag in ("div1", "div2", "div3") and n:
            current_ref = ref_parts + [n]
        else:
            current_ref = ref_parts

        # Extract lines
        for line in element.findall("l"):
            ln = line.get("n", "")
            text = extract_text(line)
            if text:
                ref = ".".join(current_ref + [ln]) if ln else ".".join(current_ref)
                passages.append({"reference": ref, "text": text, "unit": "line"})

        # Extract paragraphs (for prose)
        for para in element.findall("p"):
            text = extract_text(para)
            if text and len(text) > 5:
                ref = ".".join(current_ref) if current_ref else "?"
                passages.append({"reference": ref, "text": text, "unit": "paragraph"})

        # Recurse into child divs
        for child in element:
            if child.tag in ("div", "div1", "div2", "div3"):
                walk_divs(child, current_ref)

    walk_divs(body)

    # If nothing found via structured walk, try flat extraction
    if not passages:
        for el in body.iter():
            if el.tag in ("l", "p"):
                text = extract_text(el)
                n = el.get("n", "")
                if text and len(text) > 3:
                    passages.append({"reference": n or "?", "text": text, "unit": el.tag})

    return passages


def ensure_author(conn: sqlite3.Connection, tlg_id: str) -> int:
    """Get or create author record."""
    cur = conn.execute("SELECT author_id FROM authors WHERE tlg_id = ?", (tlg_id,))
    row = cur.fetchone()
    if row:
        return row[0]

    info = TLG_AUTHORS.get(tlg_id, (tlg_id, None, None, None, None))
    name, name_greek, period, fl_start, fl_end = info
    cur = conn.execute(
        """INSERT INTO authors (name, name_greek, tlg_id, floruit_start, floruit_end, period)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (name, name_greek, tlg_id, fl_start, fl_end, period),
    )
    return cur.lastrowid


def ensure_document(conn: sqlite3.Connection, author_id: int, tlg_author: str,
                    tlg_work: str, meta: dict, period: str) -> int:
    """Get or create document record."""
    work_key = f"{tlg_author}.{tlg_work}"
    cur = conn.execute("SELECT document_id FROM documents WHERE tlg_work_id = ?", (work_key,))
    row = cur.fetchone()
    if row:
        return row[0]

    work_info = TLG_WORKS.get(work_key)
    if work_info:
        title, title_greek, genre, approx_date = work_info
    else:
        title = meta.get("title", tlg_work)
        title_greek = None
        genre = None
        approx_date = None

    cur = conn.execute(
        """INSERT INTO documents (author_id, title, title_greek, tlg_work_id, genre,
                                  approximate_date, period)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (author_id, title, title_greek, work_key, genre, approx_date, period),
    )
    return cur.lastrowid


def ingest_file(conn: sqlite3.Connection, filepath: Path) -> dict:
    """Ingest a single TEI XML file into the database."""
    lang = detect_language(filepath)
    tlg_author, tlg_work = parse_tlg_path(filepath)

    if not tlg_author:
        return {"status": "skipped", "reason": "no TLG ID"}

    meta = parse_tei_metadata(filepath)
    if "error" in meta:
        return {"status": "error", "reason": meta["error"]}

    author_info = TLG_AUTHORS.get(tlg_author, (tlg_author, None, None, None, None))
    period = author_info[2]

    author_id = ensure_author(conn, tlg_author)
    doc_id = ensure_document(conn, author_id, tlg_author, tlg_work, meta, period)

    # Create edition or translation record
    if lang == "grc":
        cur = conn.execute(
            """INSERT OR IGNORE INTO editions
               (document_id, editor, source_url, rights_status, format)
               VALUES (?, ?, ?, 'creative_commons', 'xml')""",
            (doc_id, meta.get("editor", ""), str(filepath)),
        )
        source_id = cur.lastrowid or conn.execute(
            "SELECT edition_id FROM editions WHERE source_url = ?", (str(filepath),)
        ).fetchone()[0]
    elif lang == "eng":
        translator = meta.get("translator") or meta.get("editor", "unknown")
        cur = conn.execute(
            """INSERT OR IGNORE INTO translations
               (document_id, translator, source_url, rights_status)
               VALUES (?, ?, ?, 'creative_commons')""",
            (doc_id, translator, str(filepath)),
        )
        source_id = cur.lastrowid
    else:
        return {"status": "skipped", "reason": f"unsupported language: {lang}"}

    # Extract and store passages
    raw_passages = extract_passages(filepath)

    if lang == "grc" and raw_passages:
        # Create a segment record for this file
        cur = conn.execute(
            """INSERT INTO segments (edition_id, segment_type, reference, raw_text)
               VALUES (?, 'book', ?, ?)""",
            (source_id, f"{tlg_author}.{tlg_work}", filepath.name),
        )
        segment_id = cur.lastrowid

        for p in raw_passages:
            conn.execute(
                """INSERT INTO passages (segment_id, document_id, reference, greek_text, word_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (segment_id, doc_id, p["reference"], p["text"], len(p["text"].split())),
            )

    return {
        "status": "ok",
        "lang": lang,
        "author": author_info[0],
        "passages": len(raw_passages),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Perseus TEI files into DB")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"Raw directory not found: {args.raw_dir}")
        print("Run acquire_pilot_corpus.py first.")
        return

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    xml_files = sorted(args.raw_dir.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files to ingest")

    stats = {"ok": 0, "skipped": 0, "error": 0, "total_passages": 0}
    for i, f in enumerate(xml_files):
        result = ingest_file(conn, f)
        status = result.get("status", "error")
        stats[status] = stats.get(status, 0) + 1
        if status == "ok":
            stats["total_passages"] += result.get("passages", 0)

        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  Processed {i+1}/{len(xml_files)} files...")

    conn.commit()
    conn.close()

    print(f"\nIngestion complete:")
    print(f"  OK: {stats['ok']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Total Greek passages: {stats['total_passages']}")


if __name__ == "__main__":
    main()
