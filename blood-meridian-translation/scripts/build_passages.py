#!/usr/bin/env python3
"""
Extract passages from McCarthy_Blood_Meridian.txt based on the passage manifest
and write individual JSON files into passages/.
"""

import json
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "config" / "passage_manifest.yaml"
SOURCE = ROOT / "passages" / "McCarthy_Blood_Meridian.txt"
OUT_DIR = ROOT / "passages"


def load_source_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def load_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["passages"]


def extract_passage(lines: list[str], start: int, end: int) -> str:
    """Extract lines [start, end] (1-indexed, inclusive) and return as text."""
    selected = lines[start - 1 : end]
    return "".join(selected).strip()


def segment_sentences(text: str) -> list[str]:
    """
    Rough sentence segmentation for McCarthy's prose.
    McCarthy often uses sentence fragments and omits punctuation,
    so we split on '. ' and '? ' and '! ' but preserve fragments.
    """
    import re

    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def classify_sentence(sent: str) -> str:
    """Basic heuristic discourse-type tag for a sentence."""
    # Dialogue markers
    if any(
        marker in sent
        for marker in [", said ", ", cried ", " said:", " called ", " says "]
    ):
        return "dialogue"
    if sent.startswith(("Said ", "Says ", "Cried ")):
        return "dialogue"

    # Check for direct speech patterns (starts with capital after quote-like context)
    if any(sent.startswith(w) for w in ("Let", "Oh ", "Well ", "I ", "You ", "We ")):
        return "dialogue"

    # Short fragments are often description or catalogue
    words = sent.split()
    if len(words) <= 6 and not any(
        marker in sent for marker in [" said", " cried", " called"]
    ):
        return "fragment"

    return "narrative"


def build_passage_json(entry: dict, lines: list[str]) -> dict:
    start, end = entry["lines"]
    text = extract_passage(lines, start, end)
    sentences = segment_sentences(text)

    return {
        "id": entry["id"],
        "chapter": entry["chapter"],
        "title": entry["title"],
        "source_lines": entry["lines"],
        "discourse_types": entry["discourse"],
        "priority": entry["priority"],
        "notes": entry.get("notes", "").strip(),
        "text": text,
        "sentences": [
            {
                "index": i,
                "text": s,
                "discourse_hint": classify_sentence(s),
            }
            for i, s in enumerate(sentences)
        ],
        "sentence_count": len(sentences),
        "glossary_terms_needed": [],  # to be filled during translation
        "translation_status": "pending",
    }


def main():
    lines = load_source_lines(SOURCE)
    manifest = load_manifest(MANIFEST)

    for entry in manifest:
        passage = build_passage_json(entry, lines)
        out_path = OUT_DIR / f"{entry['id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(passage, f, ensure_ascii=False, indent=2)
        print(f"  wrote {out_path.name}  ({passage['sentence_count']} sentences)")

    print(f"\n{len(manifest)} passages extracted.")


if __name__ == "__main__":
    main()
