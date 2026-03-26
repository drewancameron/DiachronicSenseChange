#!/usr/bin/env python3
"""
Fix ellipses in marginal_glosses.json files by replacing truncated
greek fields with full sentences from the actual draft primary.txt.

The glosses files have sentence-indexed entries. This script reads
the draft, splits into sentences, and replaces each gloss entry's
greek field with the corresponding full sentence.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"


def split_sentences(text: str) -> list[str]:
    """Split Greek text into sentences. Handles . · ; as terminators."""
    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r'(?<=[.·;!])\s+', text)
    # Also split on double-newline (paragraph breaks)
    result = []
    for chunk in raw:
        for sub in chunk.split("\n\n"):
            sub = sub.strip()
            if sub:
                result.append(sub)
    return result


def fix_passage(passage_id: str) -> int:
    """Fix ellipses in one passage's glosses. Returns number of fixes."""
    draft_path = DRAFTS / passage_id / "primary.txt"
    gloss_path = APPARATUS / passage_id / "marginal_glosses.json"

    if not draft_path.exists() or not gloss_path.exists():
        return 0

    draft_text = draft_path.read_text("utf-8").strip()
    sentences = split_sentences(draft_text)

    with open(gloss_path) as f:
        glosses = json.load(f)

    fixes = 0
    for entry in glosses.get("sentences", []):
        idx = entry.get("index", 0)
        old_greek = entry.get("greek", "")

        if "..." in old_greek or "…" in old_greek:
            if idx < len(sentences):
                entry["greek"] = sentences[idx]
                fixes += 1

    if fixes > 0:
        with open(gloss_path, "w") as f:
            json.dump(glosses, f, ensure_ascii=False, indent=2)

    return fixes


def main():
    total = 0
    for d in sorted(APPARATUS.iterdir()):
        if d.is_dir() and (d / "marginal_glosses.json").exists():
            n = fix_passage(d.name)
            if n > 0:
                print(f"  {d.name}: fixed {n} ellipses")
                total += n

    if total == 0:
        print("  No ellipses found.")
    else:
        print(f"\n  Total: {total} fixes")


if __name__ == "__main__":
    main()
