#!/usr/bin/env python3
"""
Build lemma and form IDF tables from the retrieval corpus.

Processes ~57K documents (~29M tokens) to compute:
1. Lemma IDF: how rare is this word's dictionary entry?
2. Form IDF: how rare is this specific inflected form?

Uses Morpheus API for lemmatisation (with 2300+ entry cache).
Output: glossary/lemma_idf.json, glossary/form_idf.json

Usage:
  python3 scripts/build_idf.py              # full build
  python3 scripts/build_idf.py --sample 5000 # quick test on 5K docs
"""

import json
import math
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "retrieval" / "data" / "corpus.jsonl"
GLOSSARY = ROOT / "glossary"
GLOSSARY.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))


def strip_accents(s: str) -> str:
    """Remove Greek accents/breathings for form normalisation."""
    # Decompose, remove combining marks, recompose
    nfkd = unicodedata.normalize('NFD', s)
    stripped = ''.join(c for c in nfkd
                       if unicodedata.category(c) != 'Mn')
    return stripped.lower()


def tokenise_greek(text: str) -> list[str]:
    """Extract Greek word tokens from text."""
    # Match sequences of Greek characters (basic + extended + polytonic)
    return re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF\u0300-\u036F]+',
                      text.lower())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only N documents (0 = all)")
    args = parser.parse_args()

    # Skip Morpheus for bulk IDF — too slow for 29M tokens.
    # Use accent-stripped forms as proxy for lemmas. Good enough for IDF
    # since most inflected forms of a rare word are also rare.
    has_morpheus = False
    parse_word = None
    print("Using accent-stripped forms (no Morpheus — too slow for bulk)")

    # Count documents
    print(f"Reading corpus from {CORPUS}...")
    t0 = time.time()

    # Per-lemma document frequency
    lemma_df = defaultdict(int)      # lemma → number of docs containing it
    lemma_examples = {}               # lemma → example surface form

    # Per-form document frequency
    form_df = defaultdict(int)        # normalised_form → number of docs
    form_lemma = {}                   # normalised_form → most common lemma

    # Lemma cache to avoid repeated Morpheus calls
    lemma_cache = {}  # surface_form → lemma

    n_docs = 0
    n_tokens = 0
    checkpoint_interval = 5000

    with open(CORPUS) as f:
        for line_num, line in enumerate(f):
            if args.sample and line_num >= args.sample:
                break

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("greek", doc.get("text", ""))
            if not text:
                continue

            tokens = tokenise_greek(text)
            if not tokens:
                continue

            n_docs += 1
            n_tokens += len(tokens)

            # Track unique lemmas and forms in this document
            doc_lemmas = set()
            doc_forms = set()

            for tok in tokens:
                if len(tok) <= 2:
                    continue

                # Normalised form (accent-stripped)
                norm = strip_accents(tok)
                doc_forms.add(norm)

                # Lemmatise
                if tok in lemma_cache:
                    lemma = lemma_cache[tok]
                elif has_morpheus:
                    analyses = parse_word(tok)
                    if analyses:
                        lemma = analyses[0].get("lemma", norm)
                        lemma = strip_accents(lemma)
                    else:
                        lemma = norm
                    lemma_cache[tok] = lemma
                else:
                    lemma = norm

                doc_lemmas.add(lemma)

                # Track example forms
                if lemma not in lemma_examples:
                    lemma_examples[lemma] = tok
                if norm not in form_lemma:
                    form_lemma[norm] = lemma

            # Update document frequencies
            for lemma in doc_lemmas:
                lemma_df[lemma] += 1
            for form in doc_forms:
                form_df[form] += 1

            # Progress
            if n_docs % checkpoint_interval == 0:
                elapsed = time.time() - t0
                rate = n_docs / elapsed
                print(f"  {n_docs:,} docs, {n_tokens:,} tokens, "
                      f"{len(lemma_df):,} lemmas, {len(form_df):,} forms "
                      f"({rate:.0f} docs/s)")

                pass  # checkpoint

    elapsed = time.time() - t0
    print(f"\nProcessed {n_docs:,} docs, {n_tokens:,} tokens in {elapsed:.0f}s")
    print(f"  {len(lemma_df):,} unique lemmas")
    print(f"  {len(form_df):,} unique forms")

    # Compute IDF = log(N / df)
    print("\nComputing IDF scores...")

    lemma_idf = {}
    for lemma, df in lemma_df.items():
        idf = math.log(n_docs / df)
        lemma_idf[lemma] = {
            "df": df,
            "idf": round(idf, 3),
            "example": lemma_examples.get(lemma, ""),
        }

    form_idf = {}
    for form, df in form_df.items():
        idf = math.log(n_docs / df)
        form_idf[form] = {
            "df": df,
            "idf": round(idf, 3),
            "lemma": form_lemma.get(form, ""),
        }

    # Save
    lemma_path = GLOSSARY / "lemma_idf.json"
    form_path = GLOSSARY / "form_idf.json"

    # Sort by IDF descending for readability
    lemma_sorted = dict(sorted(lemma_idf.items(),
                                key=lambda x: -x[1]["idf"]))
    form_sorted = dict(sorted(form_idf.items(),
                               key=lambda x: -x[1]["idf"]))

    with open(lemma_path, "w") as f:
        json.dump({
            "_meta": {
                "n_docs": n_docs,
                "n_tokens": n_tokens,
                "n_lemmas": len(lemma_idf),
                "built": time.strftime("%Y-%m-%d %H:%M"),
            },
            "lemmas": lemma_sorted,
        }, f, ensure_ascii=False, indent=None)
    print(f"Wrote {lemma_path} ({len(lemma_idf):,} lemmas)")

    with open(form_path, "w") as f:
        json.dump({
            "_meta": {
                "n_docs": n_docs,
                "n_tokens": n_tokens,
                "n_forms": len(form_idf),
                "built": time.strftime("%Y-%m-%d %H:%M"),
            },
            "forms": form_sorted,
        }, f, ensure_ascii=False, indent=None)
    print(f"Wrote {form_path} ({len(form_idf):,} forms)")

    pass  # no Morpheus cache to save

    # Show distribution
    print("\nLemma IDF distribution:")
    thresholds = [2, 3, 4, 5, 6, 8, 10]
    for t in thresholds:
        count = sum(1 for v in lemma_idf.values() if v["idf"] >= t)
        pct = count / len(lemma_idf) * 100
        print(f"  IDF ≥ {t}: {count:>6,} lemmas ({pct:.1f}%)")

    print("\nSample rare lemmas (IDF > 8):")
    rare = [(k, v) for k, v in lemma_sorted.items() if v["idf"] > 8]
    for lemma, info in rare[:10]:
        print(f"  {info['example']:>20}  (lemma: {lemma}, df={info['df']}, idf={info['idf']:.1f})")

    print("\nSample common lemmas (IDF < 1):")
    common = [(k, v) for k, v in lemma_idf.items() if v["idf"] < 1]
    for lemma, info in sorted(common, key=lambda x: x[1]["idf"])[:10]:
        print(f"  {info['example']:>20}  (lemma: {lemma}, df={info['df']}, idf={info['idf']:.1f})")


if __name__ == "__main__":
    main()
