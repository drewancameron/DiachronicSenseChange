#!/usr/bin/env python3
"""
Mechanical Assembler: takes a SentenceSkeleton from the construction dispatcher
and produces an annotated Greek draft.

The output is a sequence of Greek lemmas with morphological annotations that
an LLM can inflect and arrange into natural prose.

Format: lemma[features] lemma[features] ...
Example: ἰδού[] παῖς[acc.sg] . ὠχρός[nom.sg] εἰμί[pres.3sg] καί ἰσχνός[nom.sg]

Usage:
  python3 scripts/mechanical_assembler.py "See the child."
  python3 scripts/mechanical_assembler.py --passage 001_see_the_child_he
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from construction_dispatcher import (
    analyse_sentence, SentenceSkeleton, ClauseSkeleton, GreekTarget,
    lookup_greek, _load_vocab,
)


# ====================================================================
# Feature abbreviation
# ====================================================================

FEAT_ABBREV = {
    # Case
    "nom": "ὀν.", "gen": "γεν.", "dat": "δοτ.", "acc": "αἰτ.", "voc": "κλ.",
    # Number
    "sg": "ἑν.", "pl": "πλ.",
    # Tense
    "pres": "ἐνεστ.", "impf": "πρτ.", "fut": "μέλλ.",
    "aor": "ἀόρ.", "perf": "παρακ.", "plup": "ὑπερσ.",
    # Mood
    "ind": "", "subj": "ὑποτ.", "opt": "εὐκτ.",
    "imp": "προστ.", "inf": "ἀπρ.", "ptcp": "μτχ.",
    # Voice
    "act": "ἐν.", "mid": "μέσ.", "pass": "παθ.",
    # Person
    "1": "α´", "2": "β´", "3": "γ´",
}


def format_features(target: GreekTarget) -> str:
    """Format morphological features as concise annotation."""
    parts = []

    if target.pos == "verb":
        for feat in [target.tense, target.mood, target.voice,
                     target.person, target.number]:
            if feat and feat in FEAT_ABBREV and FEAT_ABBREV[feat]:
                parts.append(FEAT_ABBREV[feat])
    elif target.pos in ("noun", "adj", "pron"):
        for feat in [target.case, target.number]:
            if feat and feat in FEAT_ABBREV and FEAT_ABBREV[feat]:
                parts.append(FEAT_ABBREV[feat])

    return ".".join(parts) if parts else ""


# ====================================================================
# Greek word order rules
# ====================================================================

def order_clause_words(clause: ClauseSkeleton) -> list[GreekTarget]:
    """Reorder words for Greek word order.

    Basic rules for narrative Koine:
    - Verb-initial for narrative (McCarthy's style maps well)
    - Subordinator first in subordinate clauses
    - Adjectives before or after noun (flexible)
    - Particles/connectives first (δέ, γάρ, οὖν in second position)
    """
    if not clause.words:
        return []

    # Separate by role
    verbs = [w for w in clause.words if w.pos == "verb"]
    subjects = [w for w in clause.words if w.role in ("nsubj", "nsubj:pass")]
    objects = [w for w in clause.words if w.role in ("obj", "iobj")]
    obliques = [w for w in clause.words if w.role in ("obl", "nmod")]
    modifiers = [w for w in clause.words if w.role in ("amod", "advmod", "nummod")]
    conj = [w for w in clause.words if w.pos == "conj"]
    preps = [w for w in clause.words if w.pos == "prep"]
    rest = [w for w in clause.words if w not in verbs + subjects + objects +
            obliques + modifiers + conj + preps]

    # For McCarthy's paratactic style, keep roughly English order
    # but put verb early (VSO tendency in Greek narrative)
    ordered = []

    # Connectives first
    ordered.extend(conj)

    # Subordinator
    if clause.subordinator:
        sub_target = GreekTarget(lemma=clause.subordinator, pos="conj",
                                 english="[sub]")
        ordered.append(sub_target)

    # Subject (if pronoun, often omitted in Greek — but keep for now)
    ordered.extend(subjects)

    # Main verb
    ordered.extend(verbs)

    # Objects
    ordered.extend(objects)

    # Obliques (with their prepositions)
    ordered.extend(preps)
    ordered.extend(obliques)

    # Modifiers
    ordered.extend(modifiers)

    # Everything else
    ordered.extend(rest)

    return ordered


# ====================================================================
# Assembly
# ====================================================================

def assemble_skeleton(skeleton: SentenceSkeleton) -> str:
    """Assemble a sentence skeleton into an annotated Greek draft.

    Returns a string of lemmas with morphological annotations.
    """
    parts = []

    for clause in skeleton.clauses:
        ordered = order_clause_words(clause)

        for target in ordered:
            lemma = target.lemma
            if lemma.startswith("["):
                # Unknown word — pass through with marker
                parts.append(f"⟨{target.english}⟩")
                continue

            feats = format_features(target)
            if feats:
                parts.append(f"{lemma}[{feats}]")
            else:
                parts.append(lemma)

    return " ".join(parts)


def assemble_passage(passage_id: str) -> list[tuple[str, str]]:
    """Assemble all sentences in a passage.

    Returns list of (english_sentence, annotated_greek_draft) pairs.
    """
    p_path = ROOT / "passages" / f"{passage_id}.json"
    if not p_path.exists():
        print(f"Passage not found: {passage_id}")
        return []

    en_text = json.load(open(p_path)).get("text", "")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text)
                 if s.strip()]

    results = []
    for sent in sentences:
        skeleton = analyse_sentence(sent)
        draft = assemble_skeleton(skeleton)
        results.append((sent, draft))

    return results


def format_llm_prompt(passage_id: str) -> str:
    """Format the mechanical draft as a prompt for LLM polishing.

    The LLM's job: inflect the lemmas according to the feature annotations,
    arrange words naturally, add articles and particles as needed.
    """
    pairs = assemble_passage(passage_id)
    if not pairs:
        return ""

    annotated_lines = []
    for en, draft in pairs:
        annotated_lines.append(f"EN: {en}")
        annotated_lines.append(f"GR: {draft}")
        annotated_lines.append("")

    draft_block = "\n".join(annotated_lines)

    prompt = f"""You are an Ancient Greek prose stylist. Below is a mechanical translation draft
where Greek lemmas are annotated with target morphological features in brackets.

Your task:
1. INFLECT each lemma according to its annotation (e.g. βάλλω[ἀόρ.γ´.ἑν.] → ἔβαλεν)
2. ADD articles (ὁ, ἡ, τό) where Greek requires them
3. ADD particles (δέ, μέν, γάρ) for natural prose flow
4. ADJUST word order for rhythm and emphasis
5. KEEP the vocabulary — do NOT change the lemmas chosen
6. PRESERVE McCarthy's style: parataxis, asyndeton, fragments

Words in ⟨angle brackets⟩ have no Greek equivalent — transliterate or find the closest term.

Feature key: ὀν.=nominative γεν.=genitive δοτ.=dative αἰτ.=accusative
ἑν.=singular πλ.=plural ἐνεστ.=present ἀόρ.=aorist παρακ.=perfect
ἐν.=active μέσ.=middle παθ.=passive μτχ.=participle ἀπρ.=infinitive

Mechanical draft:
{draft_block}

Output ONLY the polished Greek text. No commentary."""

    return prompt


# ====================================================================
# CLI
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mechanical assembler")
    parser.add_argument("text", nargs="*")
    parser.add_argument("--passage", type=str)
    parser.add_argument("--prompt", action="store_true",
                        help="Output the LLM polish prompt")
    args = parser.parse_args()

    _load_vocab()

    if args.passage:
        if args.prompt:
            prompt = format_llm_prompt(args.passage)
            print(prompt)
        else:
            pairs = assemble_passage(args.passage)
            for en, draft in pairs:
                print(f"EN: {en}")
                print(f"GR: {draft}")
                print()

    elif args.text:
        text = " ".join(args.text)
        skeleton = analyse_sentence(text)
        draft = assemble_skeleton(skeleton)
        print(f"EN: {text}")
        print(f"GR: {draft}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
