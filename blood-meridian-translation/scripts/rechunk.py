#!/usr/bin/env python3
"""
Re-chunk source text into translation-sized passages (~500-1200 chars).

Groups small paragraphs (dialogue) and splits long ones, while recording
paragraph boundaries so translated chunks can be reassembled with
McCarthy's original line breaks.

Each chunk's JSON stores:
  - "text": the English source (paragraphs joined by \\n\\n)
  - "para_breaks": list of character positions where \\n\\n occurs
  - "chapter": chapter number
  - "source_chunk": which chunk of the original chapter this is

Usage:
  python3 scripts/rechunk.py                    # preview chunks from source text
  python3 scripts/rechunk.py --write            # write passage JSON files
  python3 scripts/rechunk.py --target 800       # target chunk size
  python3 scripts/rechunk.py --chapter II       # specific chapter
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "passages" / "McCarthy_Blood_Meridian.txt"
PASSAGES = ROOT / "passages"
DRAFTS = ROOT / "drafts"

# Target chunk size
TARGET_MIN = 400
TARGET_MAX = 1200
TARGET_IDEAL = 700

# Chapter boundaries (line numbers from the source file)
CHAPTER_LINES = {
    "I": (0, 245),
    "II": (246, 581),
    "III": (582, 979),
    "IV": (980, 1150),
}


def load_chapter(chapter: str) -> list[str]:
    """Load a chapter as a list of paragraphs (split on blank lines)."""
    start, end = CHAPTER_LINES[chapter]
    lines = open(SOURCE).readlines()
    chapter_text = "".join(lines[start:end]).strip()
    # Split on blank lines (McCarthy's paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', chapter_text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_paragraphs(paragraphs: list[str], target_min: int = TARGET_MIN,
                     target_max: int = TARGET_MAX) -> list[dict]:
    """Group paragraphs into chunks of roughly target size.

    Returns list of chunks, each with:
      - "paragraphs": list of paragraph texts
      - "text": joined text with \\n\\n between paragraphs
      - "char_count": total character count
    """
    chunks = []
    current_paras = []
    current_chars = 0

    for para in paragraphs:
        para_len = len(para)

        # If this single paragraph is too long, split it at sentence boundaries
        if para_len > target_max:
            # Flush current accumulator first
            if current_paras:
                chunks.append(_make_chunk(current_paras))
                current_paras = []
                current_chars = 0

            # Split long paragraph into sentence groups
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_group = []
            sent_chars = 0
            for sent in sentences:
                if sent_chars + len(sent) > target_max and sent_group:
                    # Emit current sentence group as a single-paragraph chunk
                    chunks.append(_make_chunk([" ".join(sent_group)],
                                              is_split=True))
                    sent_group = []
                    sent_chars = 0
                sent_group.append(sent)
                sent_chars += len(sent) + 1
            if sent_group:
                chunks.append(_make_chunk([" ".join(sent_group)],
                                          is_split=True))
            continue

        # Would adding this paragraph make the chunk too big?
        new_size = current_chars + para_len + (2 if current_paras else 0)
        if new_size > target_max and current_paras:
            # Emit current chunk, start new one with this paragraph
            chunks.append(_make_chunk(current_paras))
            current_paras = [para]
            current_chars = para_len
        else:
            current_paras.append(para)
            current_chars = new_size

            # If we've reached a good size, emit
            if current_chars >= target_min and para_len > 100:
                # Only break here if the next paragraph would push us over
                # or we're at a natural scene break (long paragraph followed by short)
                chunks.append(_make_chunk(current_paras))
                current_paras = []
                current_chars = 0

    # Flush remaining
    if current_paras:
        # If tiny, merge with previous chunk
        if chunks and current_chars < target_min // 2:
            prev = chunks[-1]
            prev["paragraphs"].extend(current_paras)
            prev["text"] = "\n\n".join(prev["paragraphs"])
            prev["char_count"] = len(prev["text"])
        else:
            chunks.append(_make_chunk(current_paras))

    return chunks


def _make_chunk(paragraphs: list[str], is_split: bool = False) -> dict:
    text = "\n\n".join(paragraphs)
    return {
        "paragraphs": paragraphs,
        "text": text,
        "char_count": len(text),
        "para_count": len(paragraphs),
        "is_split": is_split,  # True if this chunk is part of a split paragraph
    }


def make_slug(text: str) -> str:
    """Generate a short descriptive slug from the first few words."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()[:4]
    return "_".join(words)


def segment_sentences(text: str) -> list[dict]:
    """Split text into sentences with discourse hints."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = []
    for i, s in enumerate(raw):
        s = s.strip()
        if not s:
            continue
        hint = "narrative"
        words = s.split()
        if len(words) <= 5 and not any(w in s.lower() for w in ['said', 'called', 'cried']):
            hint = "fragment"
        elif any(marker in s.lower() for marker in [', said ', ' said ', ', called ']):
            hint = "dialogue"
        sentences.append({"index": i, "text": s, "discourse_hint": hint})
    for i, sent in enumerate(sentences):
        sent["index"] = i
    return sentences


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write passage JSON files")
    parser.add_argument("--target", type=int, default=TARGET_IDEAL,
                        help=f"Target chunk size in chars (default: {TARGET_IDEAL})")
    parser.add_argument("--chapter", type=str, default=None,
                        help="Process specific chapter (I, II, III, IV)")
    parser.add_argument("--start-num", type=int, default=1,
                        help="Starting passage number")
    args = parser.parse_args()

    global TARGET_MIN, TARGET_MAX
    TARGET_MIN = int(args.target * 0.6)
    TARGET_MAX = int(args.target * 1.7)

    chapters = [args.chapter] if args.chapter else list(CHAPTER_LINES.keys())
    passage_num = args.start_num

    for chapter in chapters:
        print(f"\n{'='*60}")
        print(f"  Chapter {chapter}")
        print(f"{'='*60}")

        paragraphs = load_chapter(chapter)
        print(f"  {len(paragraphs)} paragraphs, {sum(len(p) for p in paragraphs):,} chars")

        chunks = chunk_paragraphs(paragraphs, TARGET_MIN, TARGET_MAX)

        print(f"  → {len(chunks)} chunks")
        print()

        sizes = [c["char_count"] for c in chunks]
        print(f"  Size range: {min(sizes)}-{max(sizes)} chars")
        print(f"  Mean: {sum(sizes)/len(sizes):.0f} chars")
        print()

        for i, chunk in enumerate(chunks):
            slug = make_slug(chunk["text"])
            pid = f"{passage_num:03d}_{slug}"
            split_mark = " [SPLIT]" if chunk.get("is_split") else ""
            print(f"  {pid}: {chunk['char_count']:>5} chars, "
                  f"{chunk['para_count']} paras, "
                  f"{len(segment_sentences(chunk['text']))} sent{split_mark}")
            print(f"    {chunk['text'][:80]}...")

            if args.write:
                sentences = segment_sentences(chunk["text"])
                passage = {
                    "id": pid,
                    "chapter": chapter,
                    "title": slug.replace("_", " "),
                    "source_lines": [0, 0],
                    "discourse_types": ["narrative"],
                    "priority": 2,
                    "notes": f"Auto-chunked from Chapter {chapter}",
                    "text": chunk["text"],
                    "sentences": sentences,
                    "sentence_count": len(sentences),
                    "glossary_terms_needed": [],
                    "translation_status": "pending",
                    "para_breaks": [i for i, c in enumerate(chunk["text"])
                                    if chunk["text"][i:i+2] == "\n\n"],
                }
                out_path = PASSAGES / f"{pid}.json"
                with open(out_path, "w") as f:
                    json.dump(passage, f, indent=2, ensure_ascii=False)
                (DRAFTS / pid).mkdir(parents=True, exist_ok=True)
                print(f"    → wrote {out_path}")

            passage_num += 1
            print()


if __name__ == "__main__":
    main()
