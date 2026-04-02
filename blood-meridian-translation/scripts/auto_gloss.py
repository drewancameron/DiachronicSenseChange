#!/usr/bin/env python3
"""
Auto-glosser: identify words in translations that need glossing
based on corpus IDF scores.

Uses lemma_idf.json and form_idf.json built by build_idf.py.

Two layers:
1. Rare lemma (IDF > threshold): word's dictionary entry is uncommon → gloss meaning
2. Rare form (IDF > threshold): specific inflection is uncommon → gloss morphology

Usage:
  python3 scripts/auto_gloss.py 001_see_the_child     # show proposed glosses
  python3 scripts/auto_gloss.py --all                  # all passages
  python3 scripts/auto_gloss.py --threshold 7          # adjust rarity cutoff
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
GLOSSARY = ROOT / "glossary"
PASSAGES = ROOT / "passages"

# Default thresholds
LEMMA_THRESHOLD = 6.0   # IDF above this → gloss the meaning
FORM_THRESHOLD = 8.0    # IDF above this → gloss the morphology (even if lemma is common)

# Never gloss these (function words, particles, articles, prepositions)
STOPWORDS = {
    'και', 'δε', 'τε', 'γαρ', 'ουν', 'μεν', 'αλλα', 'ουτε', 'μητε',
    'εις', 'εν', 'επι', 'προς', 'απο', 'εκ', 'εξ', 'δια', 'κατα',
    'μετα', 'παρα', 'περι', 'υπο', 'υπερ', 'συν', 'αντι', 'προ',
    'ο', 'η', 'το', 'του', 'της', 'τω', 'τη', 'τον', 'την',
    'των', 'τοις', 'ταις', 'τους', 'τας', 'τα', 'οι', 'αι',
    'αυτου', 'αυτης', 'αυτω', 'αυτον', 'αυτην', 'αυτοις', 'αυτους',
    'ος', 'ον', 'ην', 'ως', 'οτι', 'ει', 'αν', 'μη', 'ου', 'ουκ', 'ουχ',
    'εστι', 'εστιν', 'ειναι', 'ην',
    'ουτος', 'αυτος', 'εκεινος', 'τις', 'τι', 'τινα',
    'πας', 'πασα', 'παν', 'παντα', 'παντες', 'πασαι',
    'ουδεις', 'ουδεν', 'μηδεις',
}


def strip_accents(s: str) -> str:
    """Remove Greek accents/breathings for normalisation."""
    nfkd = unicodedata.normalize('NFD', s)
    stripped = ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')
    return stripped.lower()


def tokenise_greek(text: str) -> list[str]:
    """Extract Greek word tokens preserving original forms."""
    return re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]+', text)


def load_idf():
    """Load the IDF tables."""
    lemma_path = GLOSSARY / "lemma_idf.json"
    form_path = GLOSSARY / "form_idf.json"

    if not lemma_path.exists():
        print(f"IDF tables not found. Run build_idf.py first.")
        sys.exit(1)

    lemma_data = json.load(open(lemma_path))
    form_data = json.load(open(form_path))

    return lemma_data.get("lemmas", {}), form_data.get("forms", {})


def propose_glosses(passage_id: str, lemma_idf: dict, form_idf: dict,
                    lemma_threshold: float, form_threshold: float) -> list[dict]:
    """Propose glosses for words in a passage based on IDF scores."""
    primary_path = DRAFTS / passage_id / "primary.txt"
    if not primary_path.exists():
        return []

    greek = primary_path.read_text("utf-8").strip()
    tokens = tokenise_greek(greek)

    proposed = []
    seen = set()  # avoid duplicates

    for tok in tokens:
        norm = strip_accents(tok)

        # Skip short words and stopwords
        if len(norm) <= 2 or norm in STOPWORDS:
            continue

        # Skip already seen
        if norm in seen:
            continue
        seen.add(norm)

        # Look up IDF
        form_info = form_idf.get(norm, None)
        form_score = form_info["idf"] if form_info else 11.0  # unknown = very rare

        # Check if this form needs glossing
        needs_gloss = False
        reason = ""

        if form_info is None:
            # Word not in corpus at all — definitely gloss
            needs_gloss = True
            reason = "not in corpus"
        elif form_score >= lemma_threshold:
            needs_gloss = True
            reason = f"rare (IDF={form_score:.1f}, df={form_info['df']})"

        if needs_gloss:
            proposed.append({
                "word": tok,
                "normalised": norm,
                "idf": form_score,
                "df": form_info["df"] if form_info else 0,
                "reason": reason,
            })

    # Sort by IDF descending (rarest first)
    proposed.sort(key=lambda x: -x["idf"])
    return proposed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--threshold", type=float, default=LEMMA_THRESHOLD,
                        help=f"IDF threshold for glossing (default: {LEMMA_THRESHOLD})")
    parser.add_argument("--compact", action="store_true",
                        help="One-line output per word")
    args = parser.parse_args()

    if args.all:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )
    elif args.passages:
        passage_ids = args.passages
    else:
        parser.print_help()
        return

    lemma_idf, form_idf = load_idf()
    print(f"Loaded {len(lemma_idf):,} lemmas, {len(form_idf):,} forms")
    print(f"Threshold: IDF ≥ {args.threshold}")
    print()

    for pid in passage_ids:
        proposed = propose_glosses(pid, lemma_idf, form_idf,
                                   args.threshold, args.threshold + 2)
        if not proposed:
            print(f"  {pid}: no glosses needed")
            continue

        # Load source text for context
        p_path = PASSAGES / f"{pid}.json"
        en_text = ""
        if p_path.exists():
            en_text = json.load(open(p_path)).get("text", "")

        greek = (DRAFTS / pid / "primary.txt").read_text("utf-8").strip()

        print(f"  {pid}: {len(proposed)} words to gloss")
        if not args.compact:
            # Show the Greek text with proposed words highlighted
            total_tokens = len(tokenise_greek(greek))
            print(f"    ({len(proposed)}/{total_tokens} tokens = {len(proposed)/total_tokens*100:.0f}% coverage)")
        print()

        for g in proposed:
            if args.compact:
                print(f"    {g['word']:>25}  IDF={g['idf']:>5.1f}  df={g['df']:>5}  {g['reason']}")
            else:
                print(f"    {g['word']}")
                print(f"      IDF={g['idf']:.1f}, df={g['df']}, {g['reason']}")
        print()


if __name__ == "__main__":
    main()
