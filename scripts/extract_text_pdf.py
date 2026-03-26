#!/usr/bin/env python3
"""
Extract text from manually-acquired PDF files.

Uses PyMuPDF (fitz) for text extraction, producing page-by-page
plain text that can then be sent to the OpenAI API for segmentation.
"""

import json
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed. Run: pip install PyMuPDF", file=sys.stderr)
    sys.exit(1)

import yaml

MANUAL_DIR = Path(__file__).parent.parent / "corpus" / "raw" / "manual"
OUTPUT_DIR = Path(__file__).parent.parent / "corpus" / "cleaned" / "manual_text"
SOURCES_YAML = Path(__file__).parent.parent / "config" / "manual_sources.yaml"


def extract_pdf_text(filepath: Path, pages_per_chunk: int = 5) -> list[dict]:
    """
    Extract text from a PDF, returning chunks of pages.

    Each chunk contains multiple pages to give the segmentation model
    enough context to classify content type.
    """
    doc = fitz.open(filepath)
    total_pages = len(doc)
    chunks = []

    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk, total_pages)
        text_parts = []

        for page_num in range(start, end):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                text_parts.append({
                    "page": page_num + 1,
                    "text": text.strip(),
                })

        if text_parts:
            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(
                f"[Page {p['page']}]\n{p['text']}" for p in text_parts
            )
            chunks.append({
                "start_page": start + 1,
                "end_page": end,
                "pages": text_parts,
                "combined_text": combined_text,
                "char_count": len(combined_text),
            })

    doc.close()
    return chunks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from manual PDFs")
    parser.add_argument("--input-dir", type=Path, default=MANUAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pages-per-chunk", type=int, default=5)
    parser.add_argument("--file", type=str, help="Process single file")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load source registry for metadata
    sources_meta = {}
    if SOURCES_YAML.exists():
        with open(SOURCES_YAML) as f:
            config = yaml.safe_load(f)
        for s in config.get("sources", []):
            sources_meta[s["filename"]] = s

    if args.file:
        pdf_files = [args.input_dir / args.file]
    else:
        pdf_files = sorted(args.input_dir.glob("*.pdf"))

    print(f"Processing {len(pdf_files)} PDF files...")

    for filepath in pdf_files:
        if not filepath.exists():
            print(f"  [skip] {filepath.name} — not found")
            continue

        meta = sources_meta.get(filepath.name, {})
        rights = meta.get("rights_status", "unknown")
        title = meta.get("title", filepath.stem)

        print(f"\n{'='*50}")
        print(f"File: {filepath.name}")
        print(f"Title: {title}")
        print(f"Rights: {rights}")
        print(f"Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

        chunks = extract_pdf_text(filepath, args.pages_per_chunk)
        total_chars = sum(c["char_count"] for c in chunks)
        total_pages = max(c["end_page"] for c in chunks) if chunks else 0

        print(f"Pages: {total_pages}")
        print(f"Chunks: {len(chunks)}")
        print(f"Total chars: {total_chars:,}")
        print(f"Est. tokens: ~{total_chars // 4:,}")

        # Save extracted text
        stem = filepath.stem.replace(" ", "_")
        output_path = args.output_dir / f"{stem}.json"

        output = {
            "source_file": filepath.name,
            "title": title,
            "rights_status": rights,
            "may_redistribute": meta.get("may_redistribute", False),
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "est_tokens": total_chars // 4,
            "chunks": chunks,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Saved to: {output_path.name}")

        # Show a sample
        if chunks:
            sample = chunks[min(2, len(chunks) - 1)]
            print(f"\n  Sample (pages {sample['start_page']}-{sample['end_page']}):")
            print(f"  {sample['combined_text'][:200]}...")

    print(f"\n{'='*50}")
    print(f"All PDF text extracted to {args.output_dir}")


if __name__ == "__main__":
    main()
