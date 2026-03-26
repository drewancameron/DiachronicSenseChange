#!/usr/bin/env python3
"""
Check paragraph boundaries and sentence continuity across all passages.

Verifies:
  1. Every Greek passage ends with proper terminal punctuation (. · ; !)
  2. Every Greek passage begins with a capital or sensible continuation
  3. No content gaps between passages (English sentence count alignment)
  4. Paragraph breaks in Greek match paragraph breaks in English
  5. No orphaned fragments at passage boundaries
  6. Sentence length sanity (flags overly long sentences)

Usage:
  python3 scripts/check_boundaries.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
PASSAGES = ROOT / "passages"
SOURCE = ROOT / "passages" / "McCarthy_Blood_Meridian.txt"

MAX_SENTENCE_WORDS = 45  # flag sentences longer than this


def load_passage_manifest():
    """Load passage IDs and their line ranges from the manifest."""
    import yaml
    manifest_path = ROOT / "config" / "passage_manifest.yaml"
    with open(manifest_path) as f:
        data = yaml.safe_load(f)
    return data["passages"]


def load_english_lines():
    """Load the English source as lines."""
    with open(SOURCE, encoding="utf-8") as f:
        return f.readlines()


def get_english_text(passage: dict, lines: list[str]) -> str:
    """Extract English text for a passage from source lines."""
    start, end = passage["lines"]
    return "".join(lines[start - 1:end]).strip()


def split_greek_sentences(text: str) -> list[str]:
    """Split Greek text into sentences."""
    sents = re.split(r'(?<=[.·;!])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def split_english_sentences(text: str) -> list[str]:
    """Split English text into sentences."""
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sents if s.strip()]


def check_terminal_punctuation(text: str, passage_id: str) -> list[dict]:
    """Check that text ends with proper punctuation."""
    issues = []
    text = text.rstrip()
    if text and text[-1] not in '.·;!':
        last_30 = text[-60:] if len(text) > 60 else text
        issues.append({
            "type": "missing_terminal_punct",
            "passage": passage_id,
            "severity": "warning",
            "message": f"passage does not end with terminal punctuation",
            "context": f"...{last_30}",
        })
    return issues


def check_opening(text: str, passage_id: str, is_first: bool) -> list[dict]:
    """Check that text begins sensibly."""
    issues = []
    text = text.lstrip()
    if not text:
        issues.append({
            "type": "empty_passage",
            "passage": passage_id,
            "severity": "error",
            "message": "passage is empty",
        })
        return issues

    first_char = text[0]
    # Greek text should start with a capital Greek letter, or a lowercase
    # if it's a continuation (which would be a problem)
    if not is_first and first_char.islower():
        issues.append({
            "type": "lowercase_opening",
            "passage": passage_id,
            "severity": "info",
            "message": f"passage starts with lowercase '{first_char}' — continuation from previous?",
            "context": text[:60],
        })
    return issues


def check_sentence_lengths(text: str, passage_id: str) -> list[dict]:
    """Flag overly long sentences that may need splitting."""
    issues = []
    sents = split_greek_sentences(text)
    for i, sent in enumerate(sents):
        words = sent.split()
        if len(words) > MAX_SENTENCE_WORDS:
            issues.append({
                "type": "long_sentence",
                "passage": passage_id,
                "severity": "info",
                "message": f"sentence {i} has {len(words)} words (>{MAX_SENTENCE_WORDS})",
                "context": sent[:80] + "...",
            })
    return issues


def check_sentence_alignment(eng_text: str, grk_text: str, passage_id: str) -> list[dict]:
    """Check that Greek and English have roughly similar sentence counts."""
    issues = []
    eng_sents = split_english_sentences(eng_text)
    grk_sents = split_greek_sentences(grk_text)

    eng_n = len(eng_sents)
    grk_n = len(grk_sents)

    if eng_n == 0 or grk_n == 0:
        return issues

    ratio = grk_n / eng_n
    if ratio < 0.5:
        issues.append({
            "type": "sentence_gap",
            "passage": passage_id,
            "severity": "warning",
            "message": f"Greek has far fewer sentences than English ({grk_n} vs {eng_n}, ratio {ratio:.2f}) — content may be missing",
        })
    elif ratio > 2.0:
        issues.append({
            "type": "sentence_excess",
            "passage": passage_id,
            "severity": "info",
            "message": f"Greek has many more sentences than English ({grk_n} vs {eng_n}, ratio {ratio:.2f}) — over-splitting?",
        })

    return issues


def check_paragraph_alignment(passages: list[dict], eng_lines: list[str]) -> list[dict]:
    """Check that paragraph breaks in Greek match the English source."""
    issues = []

    for i in range(len(passages) - 1):
        curr = passages[i]
        next_p = passages[i + 1]

        curr_end = curr["lines"][1]
        next_start = next_p["lines"][0]

        # Check if there's a gap in line numbers (missing English content)
        if next_start > curr_end + 1:
            # Lines between passages
            gap_lines = eng_lines[curr_end:next_start - 1]
            gap_text = "".join(gap_lines).strip()
            if gap_text and not all(c in ' \n\t' for c in gap_text):
                issues.append({
                    "type": "content_gap",
                    "passage": f"{curr['id']} → {next_p['id']}",
                    "severity": "warning",
                    "message": f"English lines {curr_end+1}–{next_start-1} are not covered by any passage",
                    "context": gap_text[:80],
                })

    return issues


def check_seam_continuity(passage_ids: list[str]) -> list[dict]:
    """Check that the join between consecutive passages reads smoothly."""
    issues = []

    prev_text = None
    prev_id = None

    for pid in passage_ids:
        draft_path = DRAFTS / pid / "primary.txt"
        if not draft_path.exists():
            continue

        text = draft_path.read_text("utf-8").strip()

        if prev_text is not None:
            # Check the seam: last sentence of prev + first sentence of current
            prev_sents = split_greek_sentences(prev_text)
            curr_sents = split_greek_sentences(text)

            if prev_sents and curr_sents:
                last = prev_sents[-1]
                first = curr_sents[0]

                # Check last sentence ends properly
                if last and last[-1] not in '.·;!':
                    issues.append({
                        "type": "seam_no_terminal",
                        "passage": f"{prev_id} → {pid}",
                        "severity": "warning",
                        "message": f"last sentence of {prev_id} lacks terminal punctuation",
                        "context": f"END: ...{last[-50:]}\nSTART: {first[:50]}...",
                    })

                # Check for duplicate content at boundary
                last_words = last.split()[-5:]
                first_words = first.split()[:5]
                if last_words == first_words:
                    issues.append({
                        "type": "seam_duplicate",
                        "passage": f"{prev_id} → {pid}",
                        "severity": "warning",
                        "message": "duplicate content at passage boundary",
                        "context": f"END: {' '.join(last_words)}\nSTART: {' '.join(first_words)}",
                    })

        prev_text = text
        prev_id = pid

    return issues


def main():
    manifest = load_passage_manifest()
    eng_lines = load_english_lines()

    passage_ids = sorted(
        d.name for d in DRAFTS.iterdir()
        if d.is_dir() and (d / "primary.txt").exists()
    )

    all_issues = []

    print("Checking individual passages...")
    for i, pid in enumerate(passage_ids):
        draft_path = DRAFTS / pid / "primary.txt"
        grk_text = draft_path.read_text("utf-8").strip()

        # Find matching manifest entry for English
        manifest_entry = next((p for p in manifest if p["id"] == pid), None)
        eng_text = ""
        if manifest_entry:
            eng_text = get_english_text(manifest_entry, eng_lines)

        issues = []
        issues.extend(check_terminal_punctuation(grk_text, pid))
        issues.extend(check_opening(grk_text, pid, is_first=(i == 0)))
        issues.extend(check_sentence_lengths(grk_text, pid))
        if eng_text:
            issues.extend(check_sentence_alignment(eng_text, grk_text, pid))

        all_issues.extend(issues)

    print("Checking passage boundaries...")
    all_issues.extend(check_paragraph_alignment(manifest, eng_lines))

    print("Checking seam continuity...")
    all_issues.extend(check_seam_continuity(passage_ids))

    # Report
    print(f"\n{'='*60}")
    print(f"Boundary Check: {len(passage_ids)} passages")
    print(f"{'='*60}")

    warnings = [i for i in all_issues if i["severity"] == "warning"]
    errors = [i for i in all_issues if i["severity"] == "error"]
    infos = [i for i in all_issues if i["severity"] == "info"]

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for i in errors:
            print(f"  ✗ [{i['passage']}] {i['message']}")
            if "context" in i:
                print(f"    {i['context'][:100]}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for i in warnings:
            print(f"  ⚠ [{i['passage']}] {i['message']}")
            if "context" in i:
                for line in i["context"].split("\n"):
                    print(f"    {line[:100]}")

    if infos:
        print(f"\nINFO ({len(infos)}):")
        for i in infos:
            print(f"  ℹ [{i['passage']}] {i['message']}")

    if not all_issues:
        print("\n  ✓ All boundaries clean.")

    print(f"\nTotal: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info")


if __name__ == "__main__":
    main()
