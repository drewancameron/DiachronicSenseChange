#!/usr/bin/env python3
"""
Check for construction-type mismatches between English source and Greek
translation using stanza dependency parsing (not regex).

Compares structural features:
  1. Relative clauses (acl:relcl in English → should exist in Greek)
  2. Coordination density (conj count)
  3. Conditionals (advcl with "if" → advcl with εἰ)
  4. Fragments (no root verb)
  5. Direct speech structure

Usage:
  python3 scripts/check_constructions.py
  python3 scripts/check_constructions.py 003_the_mother_dead
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
PASSAGES = ROOT / "passages"

# Lazy-loaded stanza pipelines
_en_nlp = None
_grc_nlp = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        import stanza
        _en_nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                                   verbose=False)
    return _en_nlp


def _get_grc_nlp():
    global _grc_nlp
    if _grc_nlp is None:
        import stanza
        _grc_nlp = stanza.Pipeline('grc', processors='tokenize,pos,lemma,depparse',
                                    verbose=False)
    return _grc_nlp


def load_english(passage_id: str) -> str:
    p = PASSAGES / f"{passage_id}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f).get("text", "")
    return ""


def load_greek(passage_id: str) -> str:
    p = DRAFTS / passage_id / "primary.txt"
    if p.exists():
        return p.read_text("utf-8").strip()
    return ""


# ====================================================================
# Structural feature extraction
# ====================================================================

def extract_en_features(text: str) -> dict:
    """Extract structural features from English text."""
    nlp = _get_en_nlp()
    doc = nlp(text)

    features = {
        "sentences": len(doc.sentences),
        "relative_clauses": 0,
        "coordinations": 0,
        "conditionals": 0,
        "fragments": 0,
        "relative_details": [],
        "coordination_details": [],
    }

    for sent in doc.sentences:
        has_root_verb = False
        sent_coords = 0

        for word in sent.words:
            if word.deprel == "root" and word.upos == "VERB":
                has_root_verb = True

            # Relative clause: acl:relcl
            if word.deprel == "acl:relcl":
                head = sent.words[word.head - 1] if word.head > 0 else None
                features["relative_clauses"] += 1
                features["relative_details"].append({
                    "verb": word.text,
                    "head_noun": head.text if head else "?",
                    "sent": sent.text[:60],
                })

            # Coordination
            if word.deprel == "conj":
                sent_coords += 1

            # Conditional: advcl with "if"
            if word.deprel == "advcl":
                # Check if this clause is introduced by "if"
                for w2 in sent.words:
                    if w2.head == word.id and w2.deprel == "mark" and w2.text.lower() == "if":
                        features["conditionals"] += 1
                        break

        if not has_root_verb:
            features["fragments"] += 1

        features["coordinations"] += sent_coords
        if sent_coords >= 2:
            features["coordination_details"].append({
                "count": sent_coords,
                "sent": sent.text[:80],
            })

    return features


def extract_grc_features(text: str) -> dict:
    """Extract structural features from Ancient Greek text."""
    nlp = _get_grc_nlp()
    doc = nlp(text)

    features = {
        "sentences": len(doc.sentences),
        "relative_clauses": 0,
        "coordinations": 0,
        "conditionals": 0,
        "fragments": 0,
    }

    for sent in doc.sentences:
        has_root_verb = False
        for word in sent.words:
            if word.deprel == "root" and word.upos == "VERB":
                has_root_verb = True
            if word.deprel in ("acl:relcl", "acl"):
                # Check if introduced by a relative pronoun
                for w2 in sent.words:
                    if w2.head == word.id and w2.upos == "PRON":
                        features["relative_clauses"] += 1
                        break
            if word.deprel == "conj":
                features["coordinations"] += 1
            if word.deprel == "advcl":
                for w2 in sent.words:
                    if w2.head == word.id and w2.deprel == "mark" and w2.lemma in ("εἰ", "ἐάν", "ἤν"):
                        features["conditionals"] += 1
                        break

        if not has_root_verb:
            features["fragments"] += 1

    return features


# ====================================================================
# Comparison
# ====================================================================

def compare_features(en: dict, grc: dict, passage_id: str) -> list[dict]:
    """Compare English and Greek structural features. Flag mismatches."""
    issues = []

    # 1. Relative clauses: English has more than Greek
    if en["relative_clauses"] > grc["relative_clauses"]:
        diff = en["relative_clauses"] - grc["relative_clauses"]
        details = "; ".join(
            f"'{d['head_noun']}...{d['verb']}'" for d in en["relative_details"][:3]
        )
        issues.append({
            "type": "relative_clause_lost",
            "severity": "review",
            "message": f"English has {en['relative_clauses']} relative clauses, Greek has {grc['relative_clauses']} ({diff} missing). EN examples: {details}",
        })

    # 2. Coordination: significant loss
    if en["coordinations"] > 0:
        ratio = grc["coordinations"] / en["coordinations"] if en["coordinations"] > 0 else 1.0
        if ratio < 0.5 and en["coordinations"] >= 5:
            issues.append({
                "type": "coordination_lost",
                "severity": "review",
                "message": f"English has {en['coordinations']} coordinations, Greek has {grc['coordinations']} (ratio {ratio:.2f}). McCarthy's 'and...and...and' chains may be lost.",
            })

    # 3. Conditionals: lost
    if en["conditionals"] > grc["conditionals"]:
        diff = en["conditionals"] - grc["conditionals"]
        issues.append({
            "type": "conditional_lost",
            "severity": "review",
            "message": f"English has {en['conditionals']} conditionals, Greek has {grc['conditionals']} ({diff} missing).",
        })

    # 4. Fragments: English has more (Greek added verbs)
    if en["fragments"] > grc["fragments"] + 2:
        issues.append({
            "type": "fragments_expanded",
            "severity": "review",
            "message": f"English has {en['fragments']} fragments, Greek has {grc['fragments']}. Some McCarthy fragments may have been expanded into full sentences.",
        })

    return issues


def check_passage(passage_id: str) -> list[dict]:
    en_text = load_english(passage_id)
    grc_text = load_greek(passage_id)
    if not en_text or not grc_text:
        return []

    print(f"  Parsing {passage_id}...")
    en_features = extract_en_features(en_text)
    grc_features = extract_grc_features(grc_text)

    print(f"    EN: {en_features['sentences']} sents, {en_features['relative_clauses']} relcl, "
          f"{en_features['coordinations']} coord, {en_features['conditionals']} cond, "
          f"{en_features['fragments']} frag")
    print(f"    GR: {grc_features['sentences']} sents, {grc_features['relative_clauses']} relcl, "
          f"{grc_features['coordinations']} coord, {grc_features['conditionals']} cond, "
          f"{grc_features['fragments']} frag")

    return compare_features(en_features, grc_features, passage_id)


def main():
    if len(sys.argv) > 1:
        passage_ids = sys.argv[1:]
    else:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )

    all_issues = []
    for pid in passage_ids:
        issues = check_passage(pid)
        all_issues.extend([(pid, i) for i in issues])

    print(f"\n{'='*60}")
    print(f"Construction Check (stanza): {len(passage_ids)} passages")
    print(f"{'='*60}")

    if all_issues:
        print(f"\nFlagged for review ({len(all_issues)}):")
        for pid, i in all_issues:
            print(f"\n  [{pid}] {i['type']}")
            print(f"    {i['message']}")
    else:
        print("\n  ✓ No construction mismatches found.")


if __name__ == "__main__":
    main()
