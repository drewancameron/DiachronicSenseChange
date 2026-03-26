#!/usr/bin/env python3
"""
Structural matching for construction signposts.

Instead of giving every relative clause the same global distribution,
this module extracts structural features from a specific English construction
and finds the nearest matches in our 54K extracted pairs by structural
similarity. The distribution comes from THOSE neighbors, not the global pool.

Structural features for a relative clause:
  - sentence_length: short (<10 words), medium (10-25), long (>25)
  - clause_length: number of words in the relative clause itself
  - clause_position: final, medial, initial in the sentence
  - head_noun_type: common noun, proper noun, pronoun, demonstrative
  - is_defining: restrictive (no comma) vs non-restrictive
  - rel_pronoun: who, which, that, whose, where, etc.
  - clause_complexity: simple (1 verb), complex (2+ verbs), has_coordination

These features create a structural fingerprint. We match against fingerprints
extracted from our 54K parallel pairs and compute the local distribution
from the k nearest neighbors.

Usage:
  from structural_match import find_structural_matches
  matches = find_structural_matches(en_construction, pairs_path, k=50)
"""

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS_PATH = ROOT / "models" / "construction_model" / "construction_pairs.jsonl"


def extract_structural_fingerprint(construction: dict, words: list = None,
                                     sent_text: str = "") -> dict:
    """Extract structural features from an English construction.

    Can work from either:
    - A construction dict (from extract_parallel_constructions.py)
    - A stanza sentence's word list (for live analysis of BM text)
    """
    fp = {
        "type": construction.get("type", "unknown"),
    }

    text = construction.get("text", sent_text)
    word_count = len(text.split()) if text else 0

    # Sentence length bucket
    if word_count <= 10:
        fp["sent_length"] = "short"
    elif word_count <= 25:
        fp["sent_length"] = "medium"
    else:
        fp["sent_length"] = "long"

    fp["word_count"] = word_count

    # Type-specific features
    ctype = construction.get("type", "")

    if ctype == "relative_clause":
        head = construction.get("head_word", "")
        verb = construction.get("verb", "")
        subtree = construction.get("subtree_deprels", [])

        # Clause complexity: count verbs in subtree
        n_deps = len(subtree) if subtree else 0
        fp["clause_deps"] = n_deps
        fp["clause_complexity"] = "simple" if n_deps <= 4 else "complex"

        # Has coordination within the relative clause?
        fp["has_coordination"] = "conj" in (subtree or [])

        # Head word info
        fp["head_word"] = head
        fp["verb"] = verb

    elif ctype == "coordination_chain":
        fp["coord_count"] = construction.get("count", 0)

    elif ctype == "conditional":
        fp["verb"] = construction.get("verb", "")

    elif ctype == "fragment":
        fp["word_count"] = word_count

    return fp


def fingerprint_distance(fp1: dict, fp2: dict) -> float:
    """Compute distance between two structural fingerprints.

    Lower = more similar. Uses a weighted feature comparison.
    """
    if fp1.get("type") != fp2.get("type"):
        return 100.0  # different construction types are maximally distant

    d = 0.0

    # Sentence length: 0 if same bucket, 1 if adjacent, 2 if far
    len_map = {"short": 0, "medium": 1, "long": 2}
    l1 = len_map.get(fp1.get("sent_length", "medium"), 1)
    l2 = len_map.get(fp2.get("sent_length", "medium"), 1)
    d += abs(l1 - l2) * 2.0  # weight: sentence length matters a lot

    # Word count difference (normalised)
    wc1 = fp1.get("word_count", 15)
    wc2 = fp2.get("word_count", 15)
    d += min(abs(wc1 - wc2) / 10.0, 3.0)  # cap at 3

    # Clause complexity
    if fp1.get("clause_complexity") != fp2.get("clause_complexity"):
        d += 1.5

    # Clause dependency count
    cd1 = fp1.get("clause_deps", 3)
    cd2 = fp2.get("clause_deps", 3)
    d += min(abs(cd1 - cd2) / 3.0, 2.0)

    # Coordination within clause
    if fp1.get("has_coordination") != fp2.get("has_coordination"):
        d += 1.0

    # Coordination count (for coordination chains)
    cc1 = fp1.get("coord_count", 0)
    cc2 = fp2.get("coord_count", 0)
    if cc1 > 0 or cc2 > 0:
        d += min(abs(cc1 - cc2) / 2.0, 3.0)

    return d


def find_structural_matches(query_construction: dict, k: int = 50,
                             max_scan: int = 54000) -> dict:
    """Find the k nearest structural matches in our extracted pairs.

    Returns:
    {
        "query_fingerprint": {...},
        "n_scanned": int,
        "n_matched": int,
        "distribution": {"relative_clause": 0.65, "genitive_absolute": 0.20, ...},
        "examples": [{"en_text": ..., "grc_type": ..., "source": ..., "distance": ...}, ...]
    }
    """
    if not PAIRS_PATH.exists():
        return {"distribution": {}, "examples": [], "n_scanned": 0, "n_matched": 0}

    query_fp = extract_structural_fingerprint(query_construction)
    query_type = query_fp["type"]

    # Scan pairs and compute distances
    candidates = []  # (distance, grc_type, en_text, source)

    with open(PAIRS_PATH, encoding="utf-8") as f:
        n_scanned = 0
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            source = rec.get("source", "")

            # Look at EN constructions of same type
            for en_c in rec.get("en_constructions", []):
                if en_c.get("type") != query_type:
                    continue

                # Extract fingerprint
                cand_fp = extract_structural_fingerprint(en_c)
                dist = fingerprint_distance(query_fp, cand_fp)

                # Find what GRC construction was paired with this EN one
                # Look in the aligned pairs
                grc_type = "none"
                for pair in rec.get("pairs", []):
                    if pair.get("en_type") == query_type and pair.get("en_text", "")[:50] == en_c.get("text", "")[:50]:
                        grc_type = pair.get("grc_type", "none")
                        break

                candidates.append((dist, grc_type, en_c.get("text", "")[:80], source))

            n_scanned += 1
            if n_scanned >= max_scan:
                break

    # Sort by distance, take top k
    candidates.sort(key=lambda x: x[0])
    top_k = candidates[:k]

    # Compute distribution from neighbors (excluding 'none')
    type_counts = Counter()
    for dist, grc_type, text, source in top_k:
        if grc_type != "none":
            type_counts[grc_type] += 1

    total = sum(type_counts.values())
    distribution = {}
    if total > 0:
        for grc_type, count in type_counts.most_common():
            distribution[grc_type] = round(count / total, 3)

    # Pick diverse examples (one per source)
    examples = []
    seen_sources = set()
    for dist, grc_type, text, source in top_k:
        if grc_type != "none" and source not in seen_sources:
            seen_sources.add(source)
            examples.append({
                "en_text": text,
                "grc_type": grc_type,
                "source": source,
                "distance": round(dist, 2),
            })
            if len(examples) >= 3:
                break

    return {
        "query_fingerprint": query_fp,
        "n_scanned": n_scanned,
        "n_matched": total,
        "distribution": distribution,
        "examples": examples,
    }


# ====================================================================
# Quick test
# ====================================================================

if __name__ == "__main__":
    # Test with a short defining relative clause
    short_relcl = {
        "type": "relative_clause",
        "head_word": "man",
        "verb": "spoke",
        "text": "He turned to the man who spoke.",
        "subtree_deprels": ["nsubj"],
    }

    print("Query: 'He turned to the man who spoke.'")
    print("(short defining relative clause)\n")
    result = find_structural_matches(short_relcl, k=50)
    print(f"Scanned: {result['n_scanned']} records")
    print(f"Matched: {result['n_matched']} neighbors (excl. 'none')")
    print(f"Fingerprint: {result['query_fingerprint']}")
    print(f"\nLocal distribution:")
    for grc_type, prob in sorted(result["distribution"].items(), key=lambda x: -x[1]):
        print(f"  {grc_type}: {prob:.0%}")
    print(f"\nNearest examples:")
    for ex in result["examples"]:
        print(f"  [{ex['source']}] d={ex['distance']:.1f}: {ex['en_text']}")
        print(f"    → {ex['grc_type']}")

    print("\n" + "="*60)

    # Test with a long complex relative clause
    long_relcl = {
        "type": "relative_clause",
        "head_word": "gentleman",
        "verb": "posing",
        "text": "In truth, the gentleman standing here before you posing as a minister of the Lord is not only totally illiterate but is also wanted by the law",
        "subtree_deprels": ["nsubj", "advmod", "obl", "conj", "cc", "advcl", "obl"],
    }

    print("\nQuery: 'the gentleman standing here before you posing...'")
    print("(long complex participial/relative)\n")
    result2 = find_structural_matches(long_relcl, k=50)
    print(f"Matched: {result2['n_matched']} neighbors")
    print(f"Fingerprint: {result2['query_fingerprint']}")
    print(f"\nLocal distribution:")
    for grc_type, prob in sorted(result2["distribution"].items(), key=lambda x: -x[1]):
        print(f"  {grc_type}: {prob:.0%}")
    print(f"\nNearest examples:")
    for ex in result2["examples"]:
        print(f"  [{ex['source']}] d={ex['distance']:.1f}: {ex['en_text']}")
        print(f"    → {ex['grc_type']}")
