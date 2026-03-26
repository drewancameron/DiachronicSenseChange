#!/usr/bin/env python3
"""
Demand-driven sentence pair library.

Given a McCarthy English sentence, finds structurally similar sentences
in our corpus and returns their Greek counterparts with tree decompositions.

Strategy:
  1. Decompose the McCarthy sentence into a tree
  2. Search the 142K paired EN sentences for structural matches
  3. If good matches exist → return them with their Greek + decompositions
  4. If not → find matching EN sentences among the 125K unpaired,
     translate their nearby Greek via Haiku, embed to verify, cache the new pair

The library of verified pairs grows incrementally.

Usage:
  python3 scripts/pair_library.py "See the child."
  python3 scripts/pair_library.py "He turned to the man who spoke."
  python3 scripts/pair_library.py --passage 003_the_mother_dead
"""

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT.parent / "db" / "diachronic.db"
LIBRARY_PATH = ROOT / "models" / "pair_library.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))

_en_nlp = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        import stanza
        _en_nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                                   verbose=False)
    return _en_nlp


# ====================================================================
# 1. Structural fingerprinting (lightweight, no full tree needed)
# ====================================================================

def fingerprint_sentence(sent) -> dict:
    """Extract structural fingerprint from a stanza sentence."""
    words = sent.words
    fp = {
        "word_count": len([w for w in words if w.upos != "PUNCT"]),
        "has_root_verb": any(w.deprel == "root" and w.upos == "VERB" for w in words),
        "clause_types": [],
        "n_coordination": sum(1 for w in words if w.deprel == "conj"),
        "has_relative": any(w.deprel == "acl:relcl" for w in words),
        "has_conditional": False,
        "has_temporal": False,
        "has_passive": any("Pass" in (w.feats or "") for w in words),
        "root_deprels": sorted(set(
            w.deprel for w in words
            if w.head == next((r.id for r in words if r.deprel == "root"), 0)
            and w.upos != "PUNCT"
        )),
    }

    # Classify clauses
    for w in words:
        if w.deprel == "acl:relcl":
            fp["clause_types"].append("relative")
        elif w.deprel == "advcl":
            for c in words:
                if c.head == w.id and c.deprel == "mark":
                    if c.text.lower() in ("if", "unless"):
                        fp["has_conditional"] = True
                        fp["clause_types"].append("conditional")
                    elif c.text.lower() in ("when", "while", "after", "before", "until"):
                        fp["has_temporal"] = True
                        fp["clause_types"].append("temporal")
                    else:
                        fp["clause_types"].append("adverbial")
                    break
        elif w.deprel in ("ccomp", "xcomp"):
            fp["clause_types"].append("complement")

    if not fp["has_root_verb"]:
        fp["clause_types"].append("fragment")

    fp["sentence_type"] = _classify_type(fp)
    return fp


def _classify_type(fp: dict) -> str:
    if not fp["has_root_verb"]:
        return "fragment"
    if not fp["clause_types"]:
        if fp["n_coordination"] >= 2:
            return "compound"
        return "simple"
    if fp["n_coordination"] >= 2 and fp["clause_types"]:
        return "compound_complex"
    return "complex"


def fingerprint_distance(fp1: dict, fp2: dict) -> float:
    """Distance between two sentence fingerprints."""
    d = 0.0

    # Word count
    wc_diff = abs(fp1["word_count"] - fp2["word_count"])
    d += min(wc_diff / 5.0, 4.0)

    # Sentence type
    if fp1["sentence_type"] != fp2["sentence_type"]:
        d += 3.0

    # Clause structure
    ct1 = sorted(fp1["clause_types"])
    ct2 = sorted(fp2["clause_types"])
    if ct1 != ct2:
        # Count shared vs different
        shared = len(set(ct1) & set(ct2))
        total = max(len(set(ct1) | set(ct2)), 1)
        d += (1 - shared / total) * 4.0

    # Coordination count
    d += min(abs(fp1["n_coordination"] - fp2["n_coordination"]) / 2.0, 2.0)

    # Feature mismatches
    for feat in ("has_relative", "has_conditional", "has_temporal", "has_passive"):
        if fp1.get(feat) != fp2.get(feat):
            d += 1.0

    # Root argument structure
    rd1 = set(fp1.get("root_deprels", []))
    rd2 = set(fp2.get("root_deprels", []))
    shared_rd = len(rd1 & rd2)
    total_rd = max(len(rd1 | rd2), 1)
    d += (1 - shared_rd / total_rd) * 2.0

    return d


# ====================================================================
# 2. Search the paired library
# ====================================================================

def search_paired(query_fp: dict, k: int = 5) -> list[dict]:
    """Search the 142K paired EN sentences for structural matches."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Get clean paired sentences
    cur.execute("""
        SELECT a.aligned_text, p.greek_text, d.title, d.period, p.passage_id
        FROM alignments a
        JOIN passages p ON a.passage_id = p.passage_id
        JOIN documents d ON p.document_id = d.document_id
        WHERE a.alignment_method = 'reference_match'
          AND a.aligned_text NOT LIKE '[awaiting%]' AND a.aligned_text != ''
          AND LENGTH(a.aligned_text) BETWEEN 10 AND 300
          AND LENGTH(p.greek_text) BETWEEN 10 AND 400
          AND a.aligned_text NOT LIKE '%[%'
    """)

    nlp = _get_en_nlp()
    candidates = []

    # Process in chunks to avoid loading all into memory
    while True:
        rows = cur.fetchmany(1000)
        if not rows:
            break

        for en_text, grc_text, title, period, pid in rows:
            # Quick pre-filter by word count (skip if way too different)
            en_wc = len(en_text.split())
            if abs(en_wc - query_fp["word_count"]) > 15:
                continue

            # Parse and fingerprint
            try:
                doc = nlp(en_text)
                for sent in doc.sentences:
                    fp = fingerprint_sentence(sent)
                    dist = fingerprint_distance(query_fp, fp)
                    candidates.append({
                        "distance": dist,
                        "english": en_text[:200],
                        "greek": grc_text.replace("\n", " ").strip()[:200],
                        "source": title,
                        "period": period or "",
                        "passage_id": pid,
                        "fingerprint": fp,
                    })
            except Exception:
                continue

        # Keep only top candidates so far (memory efficiency)
        candidates.sort(key=lambda x: x["distance"])
        candidates = candidates[:k * 5]

    conn.close()

    # Final sort and return top k
    candidates.sort(key=lambda x: x["distance"])
    return candidates[:k]


def search_paired_fast(query_fp: dict, k: int = 5, max_scan: int = 2000) -> list[dict]:
    """Fast search using pre-computed fingerprints from the library cache,
    falling back to database scan if cache is empty."""

    # Check library cache first
    if LIBRARY_PATH.exists():
        candidates = []
        with open(LIBRARY_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                fp = rec.get("en_fingerprint", {})
                if fp:
                    dist = fingerprint_distance(query_fp, fp)
                    rec["distance"] = dist
                    candidates.append(rec)

        if candidates:
            candidates.sort(key=lambda x: x["distance"])
            return candidates[:k]

    # Fall back to database scan (slower but works without cache)
    return search_paired_scan(query_fp, k=k, max_scan=max_scan)


def search_paired_scan(query_fp: dict, k: int = 5, max_scan: int = 2000) -> list[dict]:
    """Scan database for structural matches, parsing on the fly.
    Limits parsing to max_scan sentences for speed."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Pre-filter: get sentences with similar word count
    target_wc = query_fp["word_count"]
    min_chars = max(10, target_wc * 3)  # rough char estimate
    max_chars = target_wc * 15

    cur.execute("""
        SELECT a.aligned_text, p.greek_text, d.title, d.period, p.passage_id
        FROM alignments a
        JOIN passages p ON a.passage_id = p.passage_id
        JOIN documents d ON p.document_id = d.document_id
        WHERE a.alignment_method = 'reference_match'
          AND a.aligned_text NOT LIKE '[awaiting%]' AND a.aligned_text != ''
          AND LENGTH(a.aligned_text) BETWEEN ? AND ?
          AND LENGTH(p.greek_text) BETWEEN 10 AND 400
          AND a.aligned_text NOT LIKE '%[%'
        ORDER BY RANDOM()
        LIMIT ?
    """, (min_chars, max_chars, max_scan))

    rows = cur.fetchall()
    conn.close()

    nlp = _get_en_nlp()
    candidates = []

    for en_text, grc_text, title, period, pid in rows:
        try:
            doc = nlp(en_text)
            for sent in doc.sentences:
                fp = fingerprint_sentence(sent)
                dist = fingerprint_distance(query_fp, fp)
                candidates.append({
                    "distance": round(dist, 2),
                    "english": en_text[:200],
                    "greek": grc_text.replace("\n", " ").strip()[:200],
                    "source": title,
                    "period": period or "",
                    "passage_id": pid,
                    "en_fingerprint": fp,
                })
        except Exception:
            continue

    candidates.sort(key=lambda x: x["distance"])
    return candidates[:k]


# ====================================================================
# 3. On-demand translation for unpaired matches
# ====================================================================

def translate_and_verify(greek_text: str, expected_english: str = "") -> dict:
    """Translate Greek via Haiku, verify with embedding similarity."""
    import anthropic
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Translate
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="Translate this Ancient Greek to English literally. Output ONLY the translation.",
        messages=[{"role": "user", "content": greek_text}],
    )
    en_auto = msg.content[0].text.strip()

    result = {"greek": greek_text, "english_auto": en_auto}

    # Verify if we have expected English
    if expected_english:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embs = model.encode([en_auto, expected_english], normalize_embeddings=True)
        sim = float(np.dot(embs[0], embs[1]))
        result["similarity"] = round(sim, 3)
        result["verified"] = sim > 0.3

    return result


# ====================================================================
# 4. Cache to library
# ====================================================================

def cache_pair(pair: dict):
    """Add a verified pair to the library cache."""
    LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIBRARY_PATH, "a") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


# ====================================================================
# 5. Main: find matches for a McCarthy sentence
# ====================================================================

def find_matches(mccarthy_text: str, k: int = 5, verbose: bool = True) -> list[dict]:
    """Find structurally matched EN↔GRC pairs for a McCarthy sentence."""
    nlp = _get_en_nlp()
    doc = nlp(mccarthy_text)

    all_matches = []

    for sent in doc.sentences:
        fp = fingerprint_sentence(sent)

        if verbose:
            print(f'\n  "{sent.text}"')
            print(f"  Type: {fp['sentence_type']}, {fp['word_count']} words")
            print(f"  Clauses: {fp['clause_types'] or ['(none)']}")
            print(f"  Coord: {fp['n_coordination']}, Rel: {fp['has_relative']}, "
                  f"Cond: {fp['has_conditional']}, Temp: {fp['has_temporal']}")

        matches = search_paired_scan(fp, k=k, max_scan=3000)

        if verbose:
            print(f"  → {len(matches)} matches found")
            for m in matches[:3]:
                print(f"    d={m['distance']:.1f} [{m['source']}]")
                print(f"      EN: {m['english'][:70]}")
                print(f"      GR: {m['greek'][:70]}")

        all_matches.append({
            "mccarthy_text": sent.text,
            "fingerprint": fp,
            "matches": matches,
        })

    return all_matches


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--passage", type=str)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.passage:
        # Load English source for a BM passage
        passage_path = ROOT / "passages" / f"{args.passage}.json"
        if not passage_path.exists():
            print(f"Passage not found: {args.passage}")
            return
        en_text = json.load(open(passage_path)).get("text", "")
        find_matches(en_text, k=args.k)

    elif args.text:
        find_matches(args.text, k=args.k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
