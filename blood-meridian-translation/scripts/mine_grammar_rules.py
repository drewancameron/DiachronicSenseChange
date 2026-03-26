#!/usr/bin/env python3
"""
Mine grammar rules empirically from UD Ancient Greek treebanks.

Reads the gold-standard CoNLL-U files (Perseus + PROIEL, ~417K tokens)
and extracts patterns that hold with high confidence:

  1. Preposition governance: for each preposition lemma, which case(s)
     does its governed noun take?  (ἐν → Dat 100%)
  2. Agreement patterns: det-noun, amod-noun — what % agree on
     Case/Gender/Number?
  3. Subject-verb agreement: what % of nsubj relations agree in Number?
     Special handling for neuter plural.
  4. Construction templates: genitive absolute, articular infinitive,
     accusative+infinitive — extract structural signatures.
  5. Verb government: which verbs take genitive / dative objects?

Output: a mined_rules.json that the grammar_engine can load and check.

Usage:
  python3 scripts/mine_grammar_rules.py
  python3 scripts/mine_grammar_rules.py --min-count 5 --min-confidence 0.95
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from conllu import parse as conllu_parse

ROOT = Path(__file__).resolve().parent.parent
TREEBANK_DIR = ROOT / "data" / "treebanks"
OUTPUT_PATH = ROOT / "config" / "mined_rules.json"


def load_all_treebanks() -> list:
    """Load all CoNLL-U files and return list of sentences."""
    all_sentences = []
    for path in sorted(TREEBANK_DIR.glob("*.conllu")):
        print(f"  Loading {path.name}...")
        text = path.read_text("utf-8")
        sentences = conllu_parse(text)
        all_sentences.extend(sentences)
    print(f"  Total: {len(all_sentences)} sentences")
    return all_sentences


def get_feat(token: dict, feat_name: str) -> str | None:
    """Get a feature value from a token's feats dict."""
    feats = token.get("feats") or {}
    return feats.get(feat_name)


# ====================================================================
# 1. Preposition governance
# ====================================================================

def mine_preposition_governance(sentences: list, min_count: int = 10) -> dict:
    """For each preposition, count what case its governed noun takes."""
    prep_cases = defaultdict(Counter)  # lemma → Counter of cases

    for sent in sentences:
        tokens = {t["id"]: t for t in sent if isinstance(t["id"], int)}

        for tok in tokens.values():
            if tok["upos"] != "ADP":
                continue
            if tok["deprel"] != "case":
                continue

            # In UD, ADP is a child of the noun (deprel=case), head is the noun
            head_id = tok["head"]
            if head_id not in tokens:
                continue
            head = tokens[head_id]
            case = get_feat(head, "Case")
            if case:
                prep_cases[tok["lemma"]][case] += 1

    # Build rules
    rules = {}
    for lemma, case_counts in sorted(prep_cases.items()):
        total = sum(case_counts.values())
        if total < min_count:
            continue

        # Find which cases are attested with > 1% frequency
        accepted = {}
        for case, count in case_counts.most_common():
            pct = count / total
            if pct >= 0.01:  # at least 1% to count as valid
                accepted[case] = round(pct, 4)

        rules[lemma] = {
            "total_attestations": total,
            "accepted_cases": accepted,
            "most_common": case_counts.most_common(1)[0][0],
        }

    return rules


# ====================================================================
# 2. Agreement patterns
# ====================================================================

def mine_agreement_patterns(sentences: list) -> dict:
    """Check how reliably det-noun and amod-noun agree."""
    checks = {
        "det_noun": {"agree": Counter(), "disagree": Counter(), "deprel": "det"},
        "amod_noun": {"agree": Counter(), "disagree": Counter(), "deprel": "amod"},
    }

    for sent in sentences:
        tokens = {t["id"]: t for t in sent if isinstance(t["id"], int)}

        for tok in tokens.values():
            for check_name, check_info in checks.items():
                if tok["deprel"] != check_info["deprel"]:
                    continue
                head_id = tok["head"]
                if head_id not in tokens:
                    continue
                head = tokens[head_id]

                for feat in ("Case", "Gender", "Number"):
                    tok_val = get_feat(tok, feat)
                    head_val = get_feat(head, feat)
                    if tok_val and head_val:
                        if tok_val == head_val:
                            check_info["agree"][feat] += 1
                        else:
                            check_info["disagree"][feat] += 1

    results = {}
    for check_name, check_info in checks.items():
        feat_rates = {}
        for feat in ("Case", "Gender", "Number"):
            total = check_info["agree"][feat] + check_info["disagree"][feat]
            if total > 0:
                rate = check_info["agree"][feat] / total
                feat_rates[feat] = {
                    "agreement_rate": round(rate, 4),
                    "total": total,
                    "agree": check_info["agree"][feat],
                    "disagree": check_info["disagree"][feat],
                }
        results[check_name] = feat_rates

    return results


# ====================================================================
# 3. Subject-verb agreement + neuter plural
# ====================================================================

def mine_subject_verb_agreement(sentences: list) -> dict:
    """Check subject-verb Number agreement, including neuter plural rule."""
    stats = {
        "total": 0,
        "number_agree": 0,
        "number_disagree": 0,
        "neuter_plural_singular_verb": 0,
        "neuter_plural_plural_verb": 0,
        "neuter_plural_total": 0,
    }

    for sent in sentences:
        tokens = {t["id"]: t for t in sent if isinstance(t["id"], int)}

        for tok in tokens.values():
            if tok["deprel"] not in ("nsubj", "nsubj:pass"):
                continue
            head_id = tok["head"]
            if head_id not in tokens:
                continue
            head = tokens[head_id]

            subj_num = get_feat(tok, "Number")
            verb_num = get_feat(head, "Number")
            subj_gender = get_feat(tok, "Gender")

            if not subj_num or not verb_num:
                continue

            stats["total"] += 1

            if subj_num == verb_num:
                stats["number_agree"] += 1
            else:
                stats["number_disagree"] += 1

            # Neuter plural specifically
            if subj_gender == "Neut" and subj_num == "Plur":
                stats["neuter_plural_total"] += 1
                if verb_num == "Sing":
                    stats["neuter_plural_singular_verb"] += 1
                elif verb_num == "Plur":
                    stats["neuter_plural_plural_verb"] += 1

    return stats


# ====================================================================
# 4. Verb government (genitive/dative objects)
# ====================================================================

def mine_verb_government(sentences: list, min_count: int = 5) -> dict:
    """Find verbs that regularly take genitive or dative objects."""
    verb_obj_cases = defaultdict(Counter)

    for sent in sentences:
        tokens = {t["id"]: t for t in sent if isinstance(t["id"], int)}

        for tok in tokens.values():
            if tok["deprel"] not in ("obj", "iobj", "obl"):
                continue
            head_id = tok["head"]
            if head_id not in tokens:
                continue
            head = tokens[head_id]

            if head["upos"] != "VERB":
                continue

            case = get_feat(tok, "Case")
            if not case:
                continue

            verb_obj_cases[head["lemma"]][case] += 1

    # Find verbs with non-accusative objects
    interesting = {}
    for lemma, case_counts in sorted(verb_obj_cases.items()):
        total = sum(case_counts.values())
        if total < min_count:
            continue

        # Only keep verbs where genitive or dative is the primary case
        top_case = case_counts.most_common(1)[0]
        if top_case[0] in ("Gen", "Dat") and top_case[1] / total >= 0.3:
            interesting[lemma] = {
                "total": total,
                "cases": dict(case_counts.most_common()),
                "primary_case": top_case[0],
                "confidence": round(top_case[1] / total, 4),
            }

    return interesting


# ====================================================================
# 5. Construction templates
# ====================================================================

def mine_constructions(sentences: list) -> dict:
    """Count construction frequencies to establish baselines."""
    stats = {
        "genitive_absolute": 0,
        "articular_infinitive": 0,
        "acc_inf": 0,
        "relative_clause": 0,
        "conditional": 0,
        "purpose_clause": 0,
        "result_clause": 0,
        "temporal_clause": 0,
        "total_sentences": len(sentences),
    }

    temporal_markers = {"ὅτε", "ἐπεί", "ἐπειδή", "πρίν", "ἕως", "μέχρι", "ἄχρι", "ἡνίκα"}

    for sent in sentences:
        tokens = {t["id"]: t for t in sent if isinstance(t["id"], int)}

        for tok in tokens.values():
            # Genitive absolute: participle in Gen
            if (get_feat(tok, "VerbForm") == "Part" and get_feat(tok, "Case") == "Gen"):
                # Check for genitive noun child
                for t2 in tokens.values():
                    if t2["head"] == tok["id"] and t2["upos"] in ("NOUN", "PROPN", "PRON"):
                        if get_feat(t2, "Case") == "Gen":
                            stats["genitive_absolute"] += 1
                            break

            # Articular infinitive
            if get_feat(tok, "VerbForm") == "Inf":
                for t2 in tokens.values():
                    if t2["head"] == tok["id"] and t2["upos"] == "DET" and t2["deprel"] == "det":
                        stats["articular_infinitive"] += 1
                        break

            # Relative clause
            if tok["deprel"] in ("acl:relcl", "acl"):
                for t2 in tokens.values():
                    if t2["head"] == tok["id"] and t2["upos"] == "PRON":
                        if t2.get("lemma") in ("ὅς", "ὅστις", "οἷος", "ὅσος"):
                            stats["relative_clause"] += 1
                            break

            # Conditional
            if tok["deprel"] == "advcl":
                for t2 in tokens.values():
                    if (t2["head"] == tok["id"] and t2["deprel"] == "mark"
                            and t2.get("lemma") in ("εἰ", "ἐάν", "ἤν", "ἄν")):
                        stats["conditional"] += 1
                        break

            # Purpose clause
            if tok["deprel"] == "advcl":
                for t2 in tokens.values():
                    if (t2["head"] == tok["id"] and t2["deprel"] == "mark"
                            and t2.get("lemma") in ("ἵνα", "ὅπως")):
                        if get_feat(tok, "Mood") in ("Sub", "Opt"):
                            stats["purpose_clause"] += 1
                        break

            # Result clause
            if tok["deprel"] in ("advcl", "xcomp", "ccomp"):
                for t2 in tokens.values():
                    if (t2["head"] == tok["id"] and t2["deprel"] == "mark"
                            and t2.get("lemma") == "ὥστε"):
                        stats["result_clause"] += 1
                        break

            # Temporal clause
            if tok["deprel"] == "advcl":
                for t2 in tokens.values():
                    if (t2["head"] == tok["id"] and t2["deprel"] == "mark"
                            and t2.get("lemma") in temporal_markers):
                        stats["temporal_clause"] += 1
                        break

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mine grammar rules from UD treebanks")
    parser.add_argument("--min-count", type=int, default=10,
                        help="Minimum attestation count for rules")
    parser.add_argument("--min-confidence", type=float, default=0.95,
                        help="Minimum confidence for strict rules")
    args = parser.parse_args()

    print("Loading UD Ancient Greek treebanks...")
    sentences = load_all_treebanks()

    print("\n1. Mining preposition governance...")
    prep_gov = mine_preposition_governance(sentences, min_count=args.min_count)
    print(f"   Found {len(prep_gov)} preposition lemmas")
    for lemma, info in list(prep_gov.items())[:5]:
        print(f"   {lemma}: {info['accepted_cases']} (n={info['total_attestations']})")

    print("\n2. Mining agreement patterns...")
    agreement = mine_agreement_patterns(sentences)
    for check, feats in agreement.items():
        for feat, info in feats.items():
            print(f"   {check}/{feat}: {info['agreement_rate']:.1%} "
                  f"({info['agree']}/{info['total']})")

    print("\n3. Mining subject-verb agreement...")
    subj_verb = mine_subject_verb_agreement(sentences)
    print(f"   Total nsubj pairs: {subj_verb['total']}")
    if subj_verb['total'] > 0:
        print(f"   Number agreement: {subj_verb['number_agree']/subj_verb['total']:.1%}")
    if subj_verb['neuter_plural_total'] > 0:
        sg_rate = subj_verb['neuter_plural_singular_verb'] / subj_verb['neuter_plural_total']
        print(f"   Neuter plural → singular verb: {sg_rate:.1%} "
              f"({subj_verb['neuter_plural_singular_verb']}/{subj_verb['neuter_plural_total']})")

    print("\n4. Mining verb government (gen/dat objects)...")
    verb_gov = mine_verb_government(sentences, min_count=args.min_count)
    print(f"   Found {len(verb_gov)} verbs with non-accusative primary government")
    for lemma, info in list(sorted(verb_gov.items(), key=lambda x: -x[1]['total']))[:10]:
        print(f"   {lemma}: {info['primary_case']} "
              f"({info['confidence']:.0%}, n={info['total']})")

    print("\n5. Mining construction frequencies...")
    constructions = mine_constructions(sentences)
    for k, v in constructions.items():
        if k != "total_sentences":
            print(f"   {k}: {v}")

    # Save
    output = {
        "source": "UD Ancient Greek Perseus + PROIEL",
        "total_sentences": len(sentences),
        "preposition_governance": prep_gov,
        "agreement_rates": agreement,
        "subject_verb_agreement": subj_verb,
        "verb_government": verb_gov,
        "construction_frequencies": constructions,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
