#!/usr/bin/env python3
"""
Post-processing: auto-mark loanwords, transliterations, and neologisms with *.

For each word in each draft:
  1. If it matches a *-marked IDF glossary entry → ensure * prefix
  2. If the 4-tier attestation pipeline (Morpheus → corpus → diachronic DB → whitelist)
     cannot find it → add * prefix
  3. Words already *-prefixed are left alone

This ensures the renderer italicises all non-classical vocabulary consistently.

Usage:
  python3 scripts/mark_loans.py              # process all passages
  python3 scripts/mark_loans.py --dry-run    # show what would change
  python3 scripts/mark_loans.py 001_see_the_child  # one passage
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
GLOSSARY = ROOT / "glossary" / "idf_glossary.json"
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))


def load_idf_starred() -> set[str]:
    """Load all *-marked Greek forms from the IDF glossary."""
    if not GLOSSARY.exists():
        return set()
    d = json.load(open(GLOSSARY))
    starred = set()
    for category, entries in d.items():
        if category == '_meta' or not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            ag = entry.get('ancient_greek', '')
            for match in re.findall(r'\*(\S+)', ag):
                starred.add(match)
    return starred


def is_greek_word(text: str) -> bool:
    """Check if token contains Greek characters."""
    clean = text.strip(".,·;:—–«»()[]!\"' *")
    return bool(re.search(r'[\u0370-\u03FF\u1F00-\u1FFF]', clean))


def process_passage(passage_id: str, dry_run: bool = False) -> dict:
    """Process one passage using morpheus_check's 4-tier attestation."""
    from morpheus_check import (
        check_passage, _load_cache, _save_cache,
    )

    primary = DRAFTS / passage_id / "primary.txt"
    if not primary.exists():
        return {"skipped": True}

    text = primary.read_text("utf-8").strip()
    idf_starred = load_idf_starred()

    # Run morpheus_check to get unattested words (uses 4-tier fallback)
    issues = check_passage(passage_id)
    _save_cache()

    unattested_words = set()
    for issue in issues:
        if issue["type"] == "unattested_word":
            unattested_words.add(issue["word"])

    # Also check against IDF starred forms
    tokens = re.findall(r'\S+', text)
    changes = []

    for tok in tokens:
        if tok.startswith("*"):
            continue
        if not is_greek_word(tok):
            continue

        clean = tok.rstrip(".,·;:—–«»()[]!\"' ")

        # Check IDF starred
        if clean in idf_starred:
            changes.append({"word": clean, "reason": "IDF glossary", "token": tok})
            continue

        # Check if unattested by 4-tier pipeline
        if clean in unattested_words:
            changes.append({"word": clean, "reason": "unattested (4-tier)", "token": tok})

    if changes and not dry_run:
        new_text = text
        # Apply in reverse to preserve positions
        for ch in sorted(changes, key=lambda c: text.find(c["token"]), reverse=True):
            idx = new_text.find(ch["token"])
            if idx >= 0:
                new_text = new_text[:idx] + "*" + new_text[idx:]
        primary.write_text(new_text + "\n", encoding="utf-8")

    return {"changes": changes}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.passages:
        passage_ids = args.passages
    else:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )

    total_changes = 0
    for pid in passage_ids:
        result = process_passage(pid, dry_run=args.dry_run)
        if result.get("skipped"):
            continue
        changes = result.get("changes", [])
        if changes:
            print(f"\n  {pid}: {len(changes)} words to mark")
            for ch in changes:
                print(f"    *{ch['word']}  ({ch['reason']})")
            total_changes += len(changes)
        else:
            print(f"  {pid}: ✓ all words attested")

    label = "would mark" if args.dry_run else "marked"
    print(f"\n  Total: {label} {total_changes} words across {len(passage_ids)} passages")


if __name__ == "__main__":
    main()
