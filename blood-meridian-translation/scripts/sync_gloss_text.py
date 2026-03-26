#!/usr/bin/env python3
"""
Sync the 'greek' field in marginal_glosses.json with the actual
sentences in the corresponding primary.txt draft.

The renderer uses the gloss file's greek field, so if the draft
was edited, the gloss file must be updated to match.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"


def split_sentences(text: str) -> list[str]:
    """Split Greek text into sentences."""
    raw = re.split(r'(?<=[.·;!])\s+', text)
    result = []
    for chunk in raw:
        for sub in chunk.split("\n\n"):
            sub = sub.strip()
            if sub:
                result.append(sub)
    return result


def sync_passage(passage_id: str) -> int:
    draft_path = DRAFTS / passage_id / "primary.txt"
    gloss_path = APPARATUS / passage_id / "marginal_glosses.json"

    if not draft_path.exists() or not gloss_path.exists():
        return 0

    draft_text = draft_path.read_text("utf-8").strip()
    draft_sents = split_sentences(draft_text)

    with open(gloss_path) as f:
        glosses = json.load(f)

    fixes = 0
    for entry in glosses.get("sentences", []):
        idx = entry.get("index", 0)
        old_greek = entry.get("greek", "")

        if idx < len(draft_sents):
            new_greek = draft_sents[idx]
            if old_greek != new_greek:
                entry["greek"] = new_greek
                fixes += 1

    # Also sync sentence count — if draft has more sentences, add empty entries
    while len(glosses["sentences"]) < len(draft_sents):
        glosses["sentences"].append({
            "index": len(glosses["sentences"]),
            "greek": draft_sents[len(glosses["sentences"])],
            "glosses": [],
        })
        fixes += 1

    if fixes > 0:
        with open(gloss_path, "w") as f:
            json.dump(glosses, f, ensure_ascii=False, indent=2)

    return fixes


def main():
    total = 0
    for d in sorted(APPARATUS.iterdir()):
        if d.is_dir() and (d / "marginal_glosses.json").exists():
            n = sync_passage(d.name)
            if n > 0:
                print(f"  {d.name}: synced {n} sentences")
                total += n

    if total == 0:
        print("  All gloss files already in sync.")
    else:
        print(f"\n  Total: {total} fixes")


if __name__ == "__main__":
    main()
