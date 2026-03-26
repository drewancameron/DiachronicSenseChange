#!/usr/bin/env python3
"""
Grammar rule engine for Ancient Greek — rules mined from UD treebanks.

Checks translations against empirical patterns extracted from 417K tokens
of gold-annotated Ancient Greek (Perseus + PROIEL treebanks).

Rules are data-driven (mined_rules.json) + declarative (grammar_rules.yaml).
The mined rules provide: preposition governance, verb government (gen/dat),
agreement baselines.  The YAML adds: construction detection, aspect, etc.

Pipeline:  stanza (grc) → UD parse → rule matching → violation list

Usage:
  python3 scripts/grammar_engine.py                          # all passages
  python3 scripts/grammar_engine.py 004_at_fourteen          # one passage
  python3 scripts/grammar_engine.py --conllu                 # also dump CoNLL-U
  python3 scripts/grammar_engine.py --report report.json     # JSON output
  python3 scripts/grammar_engine.py --verbose                # show full parses
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
RULES_PATH = ROOT / "config" / "grammar_rules.yaml"
MINED_RULES_PATH = ROOT / "config" / "mined_rules.json"
PASSAGES = ROOT / "passages"

# Lazy stanza pipeline
_grc_nlp = None


def _get_grc_nlp():
    global _grc_nlp
    if _grc_nlp is None:
        import stanza
        _grc_nlp = stanza.Pipeline(
            'grc', processors='tokenize,pos,lemma,depparse', verbose=False
        )
    return _grc_nlp


def load_rules() -> dict:
    """Load the grammar rule book (YAML + mined)."""
    with open(RULES_PATH) as f:
        rules = yaml.safe_load(f)
    # Also load mined rules if available
    if MINED_RULES_PATH.exists():
        with open(MINED_RULES_PATH) as f:
            rules["_mined"] = json.load(f)
    return rules


def parse_feats(feats_str: str | None) -> dict:
    """Parse UD feature string 'Case=Nom|Gender=Masc' into dict."""
    if not feats_str:
        return {}
    result = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


class ParsedSentence:
    """Wrapper around a stanza sentence for easy feature access."""

    def __init__(self, sent):
        self.sent = sent
        self.words = sent.words
        self._feats_cache = {}

    def feats(self, word) -> dict:
        if word.id not in self._feats_cache:
            self._feats_cache[word.id] = parse_feats(word.feats)
        return self._feats_cache[word.id]

    def head_word(self, word):
        if word.head == 0:
            return None
        return self.words[word.head - 1]

    def children(self, word):
        return [w for w in self.words if w.head == word.id]

    def children_by_deprel(self, word, deprels: list[str]):
        return [w for w in self.words if w.head == word.id and w.deprel in deprels]

    def text_window(self, word, radius=3):
        """Get text around a word for context."""
        idx = word.id - 1  # 0-based
        start = max(0, idx - radius)
        end = min(len(self.words), idx + radius + 1)
        tokens = [self.words[i].text for i in range(start, end)]
        return " ".join(tokens)


# ====================================================================
# Rule checkers
# ====================================================================

def check_agreement_rules(psent: ParsedSentence, rules: list[dict]) -> list[dict]:
    """Check agreement rules: head and dependent must share features."""
    issues = []

    for rule in rules:
        # Special rules handled separately
        if rule.get("special"):
            issues.extend(_check_special(psent, rule))
            continue

        head_upos_set = set(rule.get("head_upos", []))
        dep_upos_set = set(rule.get("dep_upos", []))
        deprel_set = set(rule.get("deprel", []))
        must_agree = rule.get("must_agree", [])
        head_feat_constraints = rule.get("head_feats", {})
        require_dep_feat = rule.get("require_dep_feat")
        exceptions = rule.get("exceptions", [])

        for word in psent.words:
            # word is the dependent
            if word.upos not in dep_upos_set:
                continue
            if word.deprel not in deprel_set:
                continue

            head = psent.head_word(word)
            if head is None:
                continue
            if head.upos not in head_upos_set:
                continue

            # Check head feature constraints
            head_feats = psent.feats(head)
            skip = False
            for feat_name, allowed_vals in head_feat_constraints.items():
                if head_feats.get(feat_name) not in allowed_vals:
                    skip = True
                    break
            if skip:
                continue

            dep_feats = psent.feats(word)

            # Require dep to have a specific feature?
            if require_dep_feat and require_dep_feat not in dep_feats:
                continue

            # Check agreement
            for feat in must_agree:
                head_val = head_feats.get(feat)
                dep_val = dep_feats.get(feat)

                if not head_val or not dep_val:
                    continue  # can't check if feature absent

                if head_val != dep_val:
                    # Check exceptions
                    is_exception = False
                    for exc in exceptions:
                        cond = exc.get("condition", {})
                        dep_cond = cond.get("dep_feats", {})
                        head_cond = cond.get("head_feats", {})
                        match = True
                        for k, v in dep_cond.items():
                            if dep_feats.get(k) != v:
                                match = False
                                break
                        for k, v in head_cond.items():
                            if head_feats.get(k) != v:
                                match = False
                                break
                        if match:
                            is_exception = True
                            break

                    if is_exception:
                        continue

                    issues.append({
                        "rule": rule["name"],
                        "type": "agreement",
                        "severity": rule.get("severity", "warning"),
                        "description": rule["description"],
                        "message": (
                            f"'{word.text}' ({dep_val}) ≠ '{head.text}' ({head_val}) "
                            f"— {feat} mismatch"
                        ),
                        "context": psent.text_window(word),
                        "head_word": head.text,
                        "dep_word": word.text,
                        "feature": feat,
                        "expected": head_val,
                        "found": dep_val,
                    })

    return issues


def _check_special(psent: ParsedSentence, rule: dict) -> list[dict]:
    """Handle specially-coded rules."""
    if rule["special"] == "relative_pronoun_agreement":
        return _check_relative_pronoun(psent, rule)
    return []


def _check_relative_pronoun(psent: ParsedSentence, rule: dict) -> list[dict]:
    """Check relative pronoun agrees with antecedent in gender and number."""
    issues = []

    for word in psent.words:
        # Find relative clause verbs
        if word.deprel not in ("acl:relcl", "acl"):
            continue

        antecedent = psent.head_word(word)
        if antecedent is None:
            continue

        ante_feats = psent.feats(antecedent)

        # Find the relative pronoun (child of the relcl verb, PRON)
        for child in psent.children(word):
            if child.upos != "PRON":
                continue
            # Check if it's a relative pronoun (lemma ὅς, ὅστις, etc.)
            if child.lemma not in ("ὅς", "ὅστις", "οἷος", "ὅσος", "ὅσπερ"):
                continue

            pron_feats = psent.feats(child)

            for feat in ("Gender", "Number"):
                ante_val = ante_feats.get(feat)
                pron_val = pron_feats.get(feat)
                if ante_val and pron_val and ante_val != pron_val:
                    issues.append({
                        "rule": rule["name"],
                        "type": "agreement",
                        "severity": rule.get("severity", "review"),
                        "description": rule["description"],
                        "message": (
                            f"Relative '{child.text}' ({pron_val}) ≠ "
                            f"antecedent '{antecedent.text}' ({ante_val}) — {feat}"
                        ),
                        "context": psent.text_window(child, radius=5),
                        "head_word": antecedent.text,
                        "dep_word": child.text,
                        "feature": feat,
                    })

    return issues


def check_government_rules(psent: ParsedSentence, rules: list[dict]) -> list[dict]:
    """Check preposition governance: prep requires specific case on its object."""
    issues = []

    for word in psent.words:
        if word.upos != "ADP":
            continue

        # Find matching rule by lemma
        matching_rule = None
        for rule in rules:
            if word.lemma in rule.get("lemma", []):
                matching_rule = rule
                break

        if not matching_rule:
            # Also try the surface form (for elided forms like ἀπ', δι', etc.)
            for rule in rules:
                if word.text in rule.get("lemma", []):
                    matching_rule = rule
                    break

        if not matching_rule:
            continue

        required_cases = matching_rule["required_case"]

        # In UD, the preposition is a child of the noun (case relation).
        # The noun is the head.
        governed = psent.head_word(word)
        if governed is None:
            continue

        gov_feats = psent.feats(governed)
        gov_case = gov_feats.get("Case")

        if not gov_case:
            continue

        if gov_case not in required_cases:
            issues.append({
                "rule": matching_rule["name"],
                "type": "government",
                "severity": matching_rule.get("severity", "warning"),
                "description": matching_rule["description"],
                "message": (
                    f"'{word.text}' ({word.lemma}) requires "
                    f"{'/'.join(required_cases)} but '{governed.text}' "
                    f"is {gov_case}"
                ),
                "context": psent.text_window(word),
                "preposition": word.text,
                "governed_word": governed.text,
                "expected_case": required_cases,
                "found_case": gov_case,
            })

    return issues


# ====================================================================
# Mined-rule checkers (use empirical patterns from treebanks)
# ====================================================================

def check_mined_prep_governance(psent: ParsedSentence, mined: dict) -> list[dict]:
    """Check preposition governance using empirically mined rules from treebanks.
    This replaces/supplements the YAML-based preposition rules with data from
    417K tokens of real Greek."""
    prep_rules = mined.get("preposition_governance", {})
    issues = []

    for word in psent.words:
        if word.upos != "ADP" or word.deprel != "case":
            continue

        rule = prep_rules.get(word.lemma)
        if not rule:
            continue

        # In UD, ADP is child of noun (case deprel)
        governed = psent.head_word(word)
        if governed is None:
            continue

        gov_case = psent.feats(governed).get("Case")
        if not gov_case:
            continue

        accepted = rule["accepted_cases"]
        if gov_case not in accepted:
            # How confident are we this is wrong?
            total = rule["total_attestations"]
            most_common = rule["most_common"]
            issues.append({
                "rule": f"mined_prep_{word.lemma}",
                "type": "government",
                "severity": "warning",
                "description": (
                    f"'{word.lemma}' governs {'/'.join(accepted.keys())} "
                    f"in {total} treebank attestations"
                ),
                "message": (
                    f"'{word.text}' ({word.lemma}) + '{governed.text}' "
                    f"in {gov_case} — expected {'/'.join(accepted.keys())} "
                    f"(most common: {most_common})"
                ),
                "context": psent.text_window(word),
                "treebank_evidence": accepted,
            })

    return issues


def check_mined_verb_government(psent: ParsedSentence, mined: dict) -> list[dict]:
    """Check verb government: flag if a verb that usually takes gen/dat
    has its object in accusative, or vice versa.  Based on treebank stats."""
    verb_rules = mined.get("verb_government", {})
    issues = []

    for word in psent.words:
        if word.deprel not in ("obj", "iobj", "obl"):
            continue

        head = psent.head_word(word)
        if head is None or head.upos != "VERB":
            continue

        rule = verb_rules.get(head.lemma)
        if not rule:
            continue

        obj_case = psent.feats(word).get("Case")
        if not obj_case:
            continue

        expected = rule["primary_case"]
        confidence = rule["confidence"]

        # Only flag if confidence is high and case doesn't match
        if confidence >= 0.5 and obj_case != expected:
            # But don't flag accusative objects of verbs that *sometimes* take acc
            verb_cases = rule.get("cases", {})
            if obj_case in verb_cases:
                continue  # this case is attested for this verb

            issues.append({
                "rule": f"mined_verb_gov_{head.lemma}",
                "type": "verb_government",
                "severity": "review",
                "description": (
                    f"'{head.lemma}' typically takes {expected} object "
                    f"({confidence:.0%} of {rule['total']} attestations)"
                ),
                "message": (
                    f"'{head.text}' ({head.lemma}) + '{word.text}' in "
                    f"{obj_case} — usually takes {expected}"
                ),
                "context": psent.text_window(word, radius=4),
                "treebank_evidence": rule["cases"],
            })

    return issues


def check_constructions(psent: ParsedSentence, rules: list[dict]) -> list[dict]:
    """Detect constructions and flag for cross-reference with English source."""
    detections = []

    for rule in rules:
        detect_type = rule.get("detect")
        if not detect_type:
            continue

        found = []
        if detect_type == "genitive_absolute":
            found = _detect_genitive_absolute(psent)
        elif detect_type == "articular_infinitive":
            found = _detect_articular_infinitive(psent)
        elif detect_type == "acc_inf":
            found = _detect_acc_inf(psent)
        elif detect_type == "relative_clause":
            found = _detect_relative_clause(psent)
        elif detect_type == "conditional":
            found = _detect_conditional(psent)
        elif detect_type == "purpose_clause":
            found = _detect_purpose_clause(psent)
        elif detect_type == "result_clause":
            found = _detect_result_clause(psent)
        elif detect_type == "temporal_clause":
            found = _detect_temporal_clause(psent)

        for item in found:
            detections.append({
                "rule": rule["name"],
                "type": "construction",
                "severity": rule.get("severity", "info"),
                "description": rule["description"],
                "note": rule.get("note", ""),
                "message": item["message"],
                "context": item.get("context", ""),
            })

    return detections


def _detect_genitive_absolute(psent: ParsedSentence) -> list[dict]:
    """Detect genitive absolute: participle in genitive + noun/pronoun in genitive."""
    found = []
    for word in psent.words:
        feats = psent.feats(word)
        if feats.get("VerbForm") != "Part" or feats.get("Case") != "Gen":
            continue
        # Check for a genitive noun/pronoun child
        for child in psent.children(word):
            cfeats = psent.feats(child)
            if child.upos in ("NOUN", "PROPN", "PRON") and cfeats.get("Case") == "Gen":
                found.append({
                    "message": f"Genitive absolute: '{child.text}' + '{word.text}'",
                    "context": psent.text_window(word, radius=5),
                })
                break
    return found


def _detect_articular_infinitive(psent: ParsedSentence) -> list[dict]:
    """Detect articular infinitive: article + infinitive."""
    found = []
    for word in psent.words:
        feats = psent.feats(word)
        if feats.get("VerbForm") != "Inf":
            continue
        for child in psent.children(word):
            if child.upos == "DET" and child.deprel == "det":
                found.append({
                    "message": f"Articular infinitive: '{child.text} {word.text}'",
                    "context": psent.text_window(word),
                })
                break
    return found


def _detect_acc_inf(psent: ParsedSentence) -> list[dict]:
    """Detect accusative + infinitive construction."""
    found = []
    for word in psent.words:
        feats = psent.feats(word)
        if feats.get("VerbForm") != "Inf":
            continue
        head = psent.head_word(word)
        if head is None:
            continue
        # Look for accusative subject of the infinitive
        for child in psent.children(word):
            cfeats = psent.feats(child)
            if child.deprel == "nsubj" and cfeats.get("Case") == "Acc":
                found.append({
                    "message": f"Acc+Inf: '{child.text}' (acc) + '{word.text}' (inf)",
                    "context": psent.text_window(word, radius=5),
                })
                break
    return found


def _detect_relative_clause(psent: ParsedSentence) -> list[dict]:
    found = []
    for word in psent.words:
        if word.deprel in ("acl:relcl", "acl"):
            for child in psent.children(word):
                if child.upos == "PRON" and child.lemma in ("ὅς", "ὅστις", "οἷος", "ὅσος"):
                    found.append({
                        "message": f"Relative clause: '{child.text}...{word.text}'",
                        "context": psent.text_window(word, radius=5),
                    })
                    break
    return found


def _detect_conditional(psent: ParsedSentence) -> list[dict]:
    found = []
    for word in psent.words:
        if word.deprel == "advcl":
            for child in psent.children(word):
                if child.deprel == "mark" and child.lemma in ("εἰ", "ἐάν", "ἤν", "ἄν"):
                    found.append({
                        "message": f"Conditional: '{child.text}...{word.text}'",
                        "context": psent.text_window(word, radius=5),
                    })
                    break
    return found


def _detect_purpose_clause(psent: ParsedSentence) -> list[dict]:
    found = []
    for word in psent.words:
        if word.deprel == "advcl":
            for child in psent.children(word):
                if child.deprel == "mark" and child.lemma in ("ἵνα", "ὅπως", "ὡς"):
                    feats = psent.feats(word)
                    if feats.get("Mood") in ("Sub", "Opt"):
                        found.append({
                            "message": f"Purpose clause: '{child.text}...{word.text}'",
                            "context": psent.text_window(word, radius=5),
                        })
                    break
    return found


def _detect_result_clause(psent: ParsedSentence) -> list[dict]:
    found = []
    for word in psent.words:
        if word.deprel in ("advcl", "xcomp", "ccomp"):
            for child in psent.children(word):
                if child.deprel == "mark" and child.lemma == "ὥστε":
                    found.append({
                        "message": f"Result clause: 'ὥστε...{word.text}'",
                        "context": psent.text_window(word, radius=5),
                    })
                    break
    return found


def _detect_temporal_clause(psent: ParsedSentence) -> list[dict]:
    found = []
    temporal_markers = {"ὅτε", "ἐπεί", "ἐπειδή", "πρίν", "ἕως", "μέχρι", "ἄχρι", "ἡνίκα"}
    for word in psent.words:
        if word.deprel == "advcl":
            for child in psent.children(word):
                if child.deprel == "mark" and child.lemma in temporal_markers:
                    found.append({
                        "message": f"Temporal clause: '{child.text}...{word.text}'",
                        "context": psent.text_window(word, radius=5),
                    })
                    break
    return found


# ====================================================================
# Main driver
# ====================================================================

def check_text(text: str, rules: dict, verbose: bool = False) -> list[dict]:
    """Parse Greek text and check all rules."""
    nlp = _get_grc_nlp()
    doc = nlp(text)
    all_issues = []

    for i, sent in enumerate(doc.sentences):
        psent = ParsedSentence(sent)

        if verbose:
            print(f"\n  Sentence {i+1}: {sent.text[:80]}...")
            for w in sent.words:
                feats = parse_feats(w.feats)
                print(f"    {w.id:3d} {w.text:20s} {w.upos:6s} {w.deprel:12s} "
                      f"head={w.head:3d} lemma={w.lemma:20s} {feats}")

        # 1. Agreement (YAML rules)
        if "agreement" in rules:
            all_issues.extend(check_agreement_rules(psent, rules["agreement"]))

        # 2. Government (YAML rules)
        if "government" in rules:
            all_issues.extend(check_government_rules(psent, rules["government"]))

        # 3. Mined preposition governance (from treebanks — more complete)
        if "_mined" in rules:
            all_issues.extend(check_mined_prep_governance(psent, rules["_mined"]))

        # 4. Mined verb government (gen/dat verbs from treebanks)
        if "_mined" in rules:
            all_issues.extend(check_mined_verb_government(psent, rules["_mined"]))

        # 5. Constructions (detection for cross-reference with English)
        if "constructions" in rules:
            all_issues.extend(check_constructions(psent, rules["constructions"]))

    return all_issues


def check_passage(passage_id: str, rules: dict, verbose: bool = False) -> list[dict]:
    """Check a single passage."""
    draft_path = DRAFTS / passage_id / "primary.txt"
    if not draft_path.exists():
        return []

    text = draft_path.read_text("utf-8").strip()
    print(f"  Checking {passage_id} ({len(text)} chars)...")
    issues = check_text(text, rules, verbose=verbose)

    # Tag issues with passage
    for issue in issues:
        issue["passage"] = passage_id

    return issues


def export_conllu(text: str, passage_id: str = "unknown") -> str:
    """Export stanza parse as CoNLL-U (for future Grew integration)."""
    nlp = _get_grc_nlp()
    doc = nlp(text)
    lines = []
    for i, sent in enumerate(doc.sentences):
        lines.append(f"# sent_id = {passage_id}-{i+1}")
        lines.append(f"# text = {sent.text}")
        for word in sent.words:
            lines.append(
                f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t"
                f"{word.xpos or '_'}\t{word.feats or '_'}\t{word.head}\t"
                f"{word.deprel}\t_\t_"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Grammar rule engine for AG")
    parser.add_argument("passages", nargs="*", help="Passage IDs to check")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--conllu", action="store_true", help="Also export CoNLL-U")
    parser.add_argument("--report", type=str, help="Save JSON report")
    parser.add_argument("--constructions-only", action="store_true",
                        help="Only show construction detections")
    parser.add_argument("--warnings-only", action="store_true",
                        help="Only show warnings and errors")
    args = parser.parse_args()

    rules = load_rules()
    yaml_rules = sum(len(rules.get(k, [])) for k in rules if k != "_mined")
    mined = rules.get("_mined", {})
    n_prep = len(mined.get("preposition_governance", {}))
    n_verb = len(mined.get("verb_government", {}))
    print(f"Loaded {yaml_rules} YAML rules + mined rules "
          f"({n_prep} prepositions, {n_verb} verb government patterns) "
          f"from {mined.get('total_sentences', 0)} treebank sentences")

    if args.passages:
        passage_ids = args.passages
    else:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )

    all_issues = []
    conllu_output = []

    for pid in passage_ids:
        issues = check_passage(pid, rules, verbose=args.verbose)
        all_issues.extend(issues)

        if args.conllu:
            text = (DRAFTS / pid / "primary.txt").read_text("utf-8").strip()
            conllu_output.append(export_conllu(text, pid))

    # Print results
    print(f"\n{'='*70}")
    print(f"Grammar Engine: {len(passage_ids)} passages, {len(all_issues)} findings")
    print(f"{'='*70}")

    # Group by type
    warnings = [i for i in all_issues if i["severity"] == "warning"]
    reviews = [i for i in all_issues if i["severity"] == "review"]
    infos = [i for i in all_issues if i["severity"] == "info"]

    if not args.constructions_only:
        if warnings:
            print(f"\n⚠ WARNINGS ({len(warnings)}):")
            for i in warnings:
                print(f"  [{i.get('passage', '?')}] {i['rule']}: {i['message']}")
                print(f"    context: {i.get('context', '')}")

        if reviews:
            print(f"\n⟐ REVIEW ({len(reviews)}):")
            for i in reviews:
                print(f"  [{i.get('passage', '?')}] {i['rule']}: {i['message']}")
                print(f"    context: {i.get('context', '')}")

    if not args.warnings_only:
        if infos:
            print(f"\nℹ CONSTRUCTIONS DETECTED ({len(infos)}):")
            for i in infos:
                note = f" — {i['note']}" if i.get("note") else ""
                print(f"  [{i.get('passage', '?')}] {i['description']}: {i['message']}{note}")

    if not warnings and not reviews:
        print("\n  ✓ No agreement or government issues found.")

    # Save report
    if args.report:
        with open(args.report, "w") as f:
            json.dump(all_issues, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.report}")

    # Save CoNLL-U
    if args.conllu:
        conllu_path = ROOT / "output" / "parses.conllu"
        conllu_path.parent.mkdir(parents=True, exist_ok=True)
        with open(conllu_path, "w") as f:
            f.write("\n".join(conllu_output))
        print(f"CoNLL-U saved to {conllu_path}")


if __name__ == "__main__":
    main()
