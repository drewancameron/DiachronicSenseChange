#!/usr/bin/env python3
"""
Re-index glosses to the correct sentence by finding which sentence
actually contains each anchor word. Fixes the mismatch caused by
sync_gloss_text.py renumbering sentences.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPARATUS = ROOT / "apparatus"


def reindex_passage(passage_id: str) -> int:
    gloss_path = APPARATUS / passage_id / "marginal_glosses.json"
    if not gloss_path.exists():
        return 0

    with open(gloss_path) as f:
        data = json.load(f)

    sentences = data.get("sentences", [])
    if not sentences:
        return 0

    # Collect all glosses that are misplaced
    orphans = []
    for sent in sentences:
        new_glosses = []
        for g in sent.get("glosses", []):
            anchor = g["anchor"]
            if anchor in sent["greek"]:
                new_glosses.append(g)
            else:
                orphans.append(g)
        sent["glosses"] = new_glosses

    # Re-place orphans into the correct sentence
    fixes = 0
    for g in orphans:
        anchor = g["anchor"]
        placed = False
        for sent in sentences:
            if anchor in sent["greek"]:
                sent["glosses"].append(g)
                placed = True
                fixes += 1
                break
        if not placed:
            # Try partial match (first word of multi-word anchor)
            first_word = anchor.split()[0] if " " in anchor else None
            if first_word:
                for sent in sentences:
                    if first_word in sent["greek"]:
                        sent["glosses"].append(g)
                        placed = True
                        fixes += 1
                        break
            if not placed:
                print(f"    ORPHANED in {passage_id}: {anchor}")

    if fixes > 0:
        with open(gloss_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return fixes


def main():
    total = 0
    for d in sorted(APPARATUS.iterdir()):
        if d.is_dir() and (d / "marginal_glosses.json").exists():
            n = reindex_passage(d.name)
            if n > 0:
                print(f"  {d.name}: re-indexed {n} glosses")
                total += n

    if total == 0:
        print("  All glosses correctly indexed.")
    else:
        print(f"\n  Total: {total} re-indexed")


if __name__ == "__main__":
    main()
