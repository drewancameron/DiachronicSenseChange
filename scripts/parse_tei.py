#!/usr/bin/env python3
"""
Parse TEI XML files from Perseus into structured segments.

Extracts Greek text and English translations, preserving canonical
references (CTS URNs, book/line numbers) for alignment.
"""

import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"

# TEI namespace
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def extract_text_content(element: ET.Element) -> str:
    """Recursively extract text from an element, stripping tags."""
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(extract_text_content(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def parse_tei_header(root: ET.Element) -> dict:
    """Extract metadata from the TEI header."""
    header = root.find(".//tei:teiHeader", NS)
    if header is None:
        header = root.find(".//teiHeader")

    metadata = {
        "title": "",
        "author": "",
        "editor": "",
        "language": "",
        "source_desc": "",
    }

    if header is None:
        return metadata

    # Try with and without namespace
    for ns_prefix in [NS, {}]:
        prefix = "tei:" if ns_prefix else ""

        title_el = header.find(f".//{prefix}titleStmt/{prefix}title", ns_prefix or None)
        if title_el is not None and title_el.text:
            metadata["title"] = title_el.text.strip()

        author_el = header.find(f".//{prefix}titleStmt/{prefix}author", ns_prefix or None)
        if author_el is not None:
            metadata["author"] = extract_text_content(author_el)

        lang_el = header.find(f".//{prefix}language", ns_prefix or None)
        if lang_el is not None:
            metadata["language"] = lang_el.get("ident", "")

        if metadata["title"]:
            break

    return metadata


def parse_tei_body(root: ET.Element) -> list[dict]:
    """Extract text divisions with references from TEI body."""
    segments = []

    # Find all div elements (with or without namespace)
    divs = root.findall(".//tei:div", NS) or root.findall(".//div")

    if not divs:
        # Try finding text content directly
        body = root.find(".//tei:body", NS) or root.find(".//body")
        if body is not None:
            text = extract_text_content(body)
            if text:
                segments.append({
                    "reference": "1",
                    "text": text,
                    "div_type": "body",
                })
        return segments

    for div in divs:
        div_type = div.get("type", "section")
        div_n = div.get("n", "")
        div_subtype = div.get("subtype", "")

        # Look for line groups, paragraphs, or direct text
        lines = div.findall(".//tei:l", NS) or div.findall(".//l")
        paras = div.findall(".//tei:p", NS) or div.findall(".//p")

        if lines:
            for line in lines:
                n = line.get("n", "")
                text = extract_text_content(line)
                if text:
                    ref = f"{div_n}.{n}" if div_n and n else (n or div_n or "?")
                    segments.append({
                        "reference": ref,
                        "text": text,
                        "div_type": div_type,
                    })
        elif paras:
            for i, para in enumerate(paras, 1):
                text = extract_text_content(para)
                if text:
                    ref = f"{div_n}.{i}" if div_n else str(i)
                    segments.append({
                        "reference": ref,
                        "text": text,
                        "div_type": div_type,
                    })
        else:
            text = extract_text_content(div)
            if text:
                segments.append({
                    "reference": div_n or "?",
                    "text": text,
                    "div_type": div_type,
                })

    return segments


def parse_tei_file(filepath: Path) -> dict:
    """Parse a TEI XML file and return structured data."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Strip namespace prefix for easier querying
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    metadata = parse_tei_header(root)
    segments = parse_tei_body(root)

    # Detect language from filename or metadata
    filename = filepath.name
    if "grc" in filename:
        metadata["language"] = "grc"
    elif "eng" in filename:
        metadata["language"] = "eng"

    return {
        "filepath": str(filepath),
        "metadata": metadata,
        "segments": segments,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse TEI XML files")
    parser.add_argument("files", nargs="+", type=Path, help="TEI XML files to parse")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    args = parser.parse_args()

    for filepath in args.files:
        result = parse_tei_file(filepath)
        meta = result["metadata"]
        segs = result["segments"]

        print(f"\n{'='*60}")
        print(f"File: {filepath.name}")
        print(f"Title: {meta['title']}")
        print(f"Author: {meta['author']}")
        print(f"Language: {meta['language']}")
        print(f"Segments: {len(segs)}")

        if not args.summary and segs:
            print(f"\nFirst 3 segments:")
            for seg in segs[:3]:
                text_preview = seg["text"][:100] + "..." if len(seg["text"]) > 100 else seg["text"]
                print(f"  [{seg['reference']}] {text_preview}")


if __name__ == "__main__":
    main()
