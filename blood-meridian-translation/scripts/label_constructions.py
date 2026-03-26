#!/usr/bin/env python3
"""
Unified construction labeller for English and Ancient Greek sentences.

Takes a sentence (text or stanza parse), returns a list of named
construction labels from construction_taxonomy.yaml.

For Greek: uses Grew patterns against a CoNLL-U parse.
For English: uses stanza dependency features + heuristics.

The labels are attached to parallel pairs in the fingerprint index
so the prompt can say "this pair uses a Genitive Absolute" rather
than just "this pair has similar word count."

Usage:
  python3 scripts/label_constructions.py "ἡ μήτηρ τεθνηκυῖα ἐνεθάλπετο" grc
  python3 scripts/label_constructions.py "The mother dead did incubate" en
  python3 scripts/label_constructions.py --test-treebanks          # validate patterns
  python3 scripts/label_constructions.py --label-index             # label all pairs in fingerprint index
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ensure opam env for Grew
OPAM_SWITCH = Path.home() / ".opam" / "4.14.2"
if OPAM_SWITCH.is_dir():
    opam_bin = str(OPAM_SWITCH / "bin")
    if opam_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = opam_bin + ":" + os.environ.get("PATH", "")
    os.environ["CAML_LD_LIBRARY_PATH"] = str(OPAM_SWITCH / "lib" / "stublibs")

sys.path.insert(0, str(ROOT / "config"))
sys.path.insert(0, str(ROOT / "scripts"))

_pipelines = {}


def _get_pipeline(lang: str):
    if lang not in _pipelines:
        import stanza
        _pipelines[lang] = stanza.Pipeline(
            lang, processors='tokenize,pos,lemma,depparse', verbose=False
        )
    return _pipelines[lang]


# ====================================================================
# Greek labelling via Grew patterns
# ====================================================================

_grew_initialized = False


def _init_grew():
    global _grew_initialized
    if not _grew_initialized:
        from grewpy import set_config
        set_config('ud')
        _grew_initialized = True


def _text_to_conllu(text: str, lang: str) -> str:
    """Parse text with stanza, return CoNLL-U string."""
    nlp = _get_pipeline(lang)
    doc = nlp(text)
    lines = []
    for i, sent in enumerate(doc.sentences):
        lines.append(f"# sent_id = s{i+1}")
        lines.append(f"# text = {sent.text}")
        for word in sent.words:
            lines.append(
                f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t"
                f"{word.xpos or '_'}\t{word.feats or '_'}\t{word.head}\t"
                f"{word.deprel}\t_\t_"
            )
        lines.append("")
    return "\n".join(lines)


def label_greek_grew(conllu_text: str) -> list[str]:
    """Label Greek constructions using Grew patterns on CoNLL-U text.
    Only returns high-value labels that affect translation decisions."""
    _init_grew()
    from grewpy import Corpus, Request
    from construction_patterns import CONSTRUCTION_PATTERNS, HIGH_VALUE_PATTERNS

    # Write to temp file for Grew
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.conllu', delete=False,
                                      encoding='utf-8')
    tmp.write(conllu_text)
    tmp.close()

    try:
        corpus = Corpus(tmp.name)
        labels = []
        for name, pattern in CONSTRUCTION_PATTERNS:
            if name not in HIGH_VALUE_PATTERNS:
                continue  # skip low-value / too-common patterns
            try:
                results = corpus.search(Request(pattern))
                if results:
                    labels.append(name)
            except Exception:
                pass
        return labels
    finally:
        os.unlink(tmp.name)


def label_greek(text: str) -> list[str]:
    """Label Greek constructions from raw text."""
    conllu = _text_to_conllu(text, "grc")
    return label_greek_grew(conllu)


# ====================================================================
# English labelling via stanza features
# ====================================================================

def label_english(text: str) -> list[str]:
    """Label English constructions using stanza parse + heuristics."""
    nlp = _get_pipeline("en")
    doc = nlp(text)
    labels = []

    for sent in doc.sentences:
        words = sent.words

        has_root_verb = any(w.deprel == "root" and w.upos in ("VERB", "AUX") for w in words)
        if not has_root_verb:
            labels.append("Fragment")

        for w in words:
            feats = _parse_feats(w.feats)

            # Relative clauses
            if w.deprel == "acl:relcl":
                # Check for modal auxiliary
                has_modal = any(
                    c.head == w.id and c.upos == "AUX" and c.text.lower() in ("would", "could", "might")
                    for c in words
                )
                if has_modal:
                    labels.append("Modal/Future Relative")
                else:
                    # Check for possessive (whose)
                    has_whose = any(
                        c.head == w.id and c.text.lower() == "whose"
                        for c in words
                    )
                    if has_whose:
                        labels.append("Possessive Relative")
                    else:
                        labels.append("Defining Relative")

            # Conditionals
            if w.deprel == "advcl":
                marker = next((c for c in words if c.head == w.id and c.deprel == "mark"), None)
                if marker:
                    mt = marker.text.lower()
                    if mt in ("if", "unless"):
                        aux = next((c for c in words if c.head == w.id and c.upos == "AUX"), None)
                        aux_text = aux.text.lower() if aux else ""
                        tense = feats.get("Tense", "")

                        if aux_text == "had":
                            labels.append("Past Contrafactual")
                        elif aux_text in ("should", "were"):
                            labels.append("Future Less Vivid")
                        elif tense == "Past":
                            labels.append("Present Contrafactual")
                        else:
                            labels.append("Future More Vivid")

                    elif mt in ("when", "while", "after", "before", "until", "once"):
                        tense = feats.get("Tense", "")
                        if mt == "before":
                            labels.append("Prior Temporal")
                        elif mt == "until":
                            labels.append("Terminal Temporal")
                        elif tense == "Past":
                            labels.append("Past Temporal")
                        else:
                            labels.append("General Temporal")

                    elif mt in ("so", "that"):
                        labels.append("Purpose Clause")

            # Purpose infinitive / future participle (after verbs of motion)
            if w.deprel in ("advcl", "xcomp") and w.upos == "VERB":
                has_to = any(
                    c.head == w.id and c.deprel == "mark" and c.text.lower() == "to"
                    for c in words
                )
                if has_to:
                    head = next((h for h in words if h.id == w.head), None)
                    if head and head.lemma in (
                        "come", "go", "descend", "sail", "march", "send",
                        "ride", "walk", "run", "return", "advance", "proceed",
                        "travel", "move", "climb", "cross", "flee", "rush",
                    ):
                        labels.append("Future Participle of Purpose")

            # Complement clauses (indirect speech)
            if w.deprel == "ccomp":
                head = next((h for h in words if h.id == w.head), None)
                if head and head.lemma in ("say", "tell", "claim", "think", "believe",
                                            "know", "declare", "inform", "announce"):
                    labels.append("Indirect Speech")

            # Coordination count
            if w.deprel == "conj":
                if "Coordination Chain" not in labels:
                    coord_count = sum(1 for c in words if c.deprel == "conj")
                    if coord_count >= 2:
                        labels.append("Coordination Chain")
                    else:
                        labels.append("Simple Coordination")

            # Passive
            if w.deprel == "nsubj:pass":
                labels.append("Passive Voice")

            # Imperative (heuristic: first word is base-form verb)
            if w.id == 1 and w.upos == "VERB" and feats.get("VerbForm") == "Inf":
                labels.append("Imperative")
            if w.id == 1 and w.upos == "VERB" and feats.get("Mood") == "Imp":
                labels.append("Imperative")

        # Comma splice detection
        if ", " in sent.text:
            clauses = [c for c in words if c.deprel in ("root", "parataxis", "conj")]
            if len(clauses) >= 2:
                labels.append("Asyndeton (comma splice)")

    # Oath detection (regex)
    import re
    if re.search(r'\bdamn\b.*\bif\b|\bI\s+swear\b|\bby\s+God\b', text, re.I):
        labels.append("Oath-Conditional")

    # Direct speech
    if re.search(r'\bsaid\b|\bcried\b|\bcalled\b|\basked\b|\breplied\b', text, re.I):
        labels.append("Direct Speech")

    # Only keep labels that actually help translation decisions.
    # Drop labels that are too common to be informative.
    HIGH_VALUE_EN = {
        # These change what Greek construction to use:
        "Modal/Future Relative",     # → ὅς + future indicative
        "Possessive Relative",       # → ὧν + subject
        "Defining Relative",         # → ὅς/ἥ/ὅ + indicative (not participle!)
        "Future More Vivid",         # → ἐάν + subjunctive
        "Future Less Vivid",         # → εἰ + optative
        "Present Contrafactual",     # → εἰ + imperfect
        "Past Contrafactual",        # → εἰ + aorist
        "Oath-Conditional",          # → ἦ μήν + εἰ μή
        "Prior Temporal",            # → πρίν + infinitive
        "Terminal Temporal",         # → ἕως + subjunctive
        "Indirect Speech",          # → acc+inf or ὅτι
        "Imperative",               # → aorist imperative
        "Fragment",                  # → preserve as fragment
        "Future Participle of Purpose",  # → future participle (μαχησόμενος)
        # These are useful context but less critical:
        "Passive Voice",            # → passive or middle
        "Asyndeton (comma splice)", # → no δέ
    }

    return sorted(set(labels) & HIGH_VALUE_EN)


def _parse_feats(feats_str):
    if not feats_str:
        return {}
    return dict(p.split("=", 1) for p in feats_str.split("|") if "=" in p)


# ====================================================================
# Label parallel pairs in fingerprint index
# ====================================================================

def label_index():
    """Add construction labels to every pair in the fingerprint index."""
    INDEX_DIR = ROOT / "models" / "fingerprint_index_fast"
    META_PATH = INDEX_DIR / "metadata.jsonl"
    LABELLED_PATH = INDEX_DIR / "metadata_labelled.jsonl"

    if not META_PATH.exists():
        print("No fingerprint index found. Run build_fingerprint_index_fast.py --build first.")
        return

    # Load metadata
    records = []
    with open(META_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Labelling {len(records)} pairs...")

    # For Greek: batch all texts into one CoNLL-U, then run Grew once
    # This is much faster than parsing each sentence individually
    _init_grew()
    from grewpy import Corpus, Request
    from construction_patterns import CONSTRUCTION_PATTERNS

    # Parse all Greek texts into one CoNLL-U file
    nlp_grc = _get_pipeline("grc")
    print("  Parsing Greek texts with stanza...")

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.conllu', delete=False,
                                      encoding='utf-8')
    for i, rec in enumerate(records):
        grc = rec.get("greek", "")
        if not grc or len(grc) < 10:
            tmp.write(f"# sent_id = pair_{i}\n# text = [empty]\n\n")
            continue
        try:
            doc = nlp_grc(grc[:500])
            for j, sent in enumerate(doc.sentences):
                tmp.write(f"# sent_id = pair_{i}\n")
                tmp.write(f"# text = {sent.text}\n")
                for word in sent.words:
                    tmp.write(
                        f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t"
                        f"{word.xpos or '_'}\t{word.feats or '_'}\t{word.head}\t"
                        f"{word.deprel}\t_\t_\n"
                    )
                tmp.write("\n")
                break  # only first sentence per pair
        except Exception:
            tmp.write(f"# sent_id = pair_{i}\n# text = [error]\n\n")

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(records)} parsed...")

    tmp.close()
    print(f"  Loading into Grew ({tmp.name})...")

    corpus = Corpus(tmp.name)
    print(f"  {len(corpus)} sentences in Grew corpus")

    # Run each pattern and record which pair_ids match
    pair_labels = {}  # pair_id → set of labels
    for name, pattern in CONSTRUCTION_PATTERNS:
        try:
            results = corpus.search(Request(pattern))
            for match in results:
                sid = match.get("sent_id", "")
                if sid.startswith("pair_"):
                    pid = int(sid.split("_")[1])
                    if pid not in pair_labels:
                        pair_labels[pid] = set()
                    pair_labels[pid].add(name)
        except Exception as e:
            print(f"    Pattern error '{name}': {e}")

    os.unlink(tmp.name)

    # Also label English side (fast, no Grew needed)
    nlp_en = _get_pipeline("en")
    print("  Labelling English constructions...")

    for i, rec in enumerate(records):
        en = rec.get("english", "")
        if not en or len(en) < 10:
            continue
        try:
            en_labels = label_english(en)
            if i not in pair_labels:
                pair_labels[i] = set()
            pair_labels[i].update(f"EN:{l}" for l in en_labels)
        except Exception:
            pass

        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(records)} labelled...")

    # Write labelled metadata
    n_labelled = 0
    with open(LABELLED_PATH, "w") as f:
        for i, rec in enumerate(records):
            labels = sorted(pair_labels.get(i, set()))
            rec["construction_labels"] = labels
            if labels:
                n_labelled += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone: {n_labelled}/{len(records)} pairs have construction labels")
    print(f"Saved → {LABELLED_PATH}")

    # Print summary
    from collections import Counter
    all_labels = Counter()
    for labels in pair_labels.values():
        all_labels.update(labels)

    print(f"\nTop construction labels:")
    for label, count in all_labels.most_common(20):
        print(f"  {label:45s} {count:6d}")


# ====================================================================
# Main
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("lang", nargs="?", default="en", choices=["en", "grc"])
    parser.add_argument("--test-treebanks", action="store_true")
    parser.add_argument("--label-index", action="store_true")
    args = parser.parse_args()

    if args.label_index:
        label_index()
    elif args.test_treebanks:
        # Quick validation against treebanks
        _init_grew()
        from grewpy import Corpus, Request
        from construction_patterns import CONSTRUCTION_PATTERNS

        corpus = Corpus([
            str(ROOT / "data/treebanks/grc_perseus-ud-train.conllu"),
            str(ROOT / "data/treebanks/grc_proiel-ud-train.conllu"),
        ])
        print(f"Treebank: {len(corpus)} sentences\n")
        for name, pattern in CONSTRUCTION_PATTERNS:
            try:
                results = corpus.search(Request(pattern))
                print(f"  {name:45s} {len(results):6d}")
            except Exception as e:
                print(f"  {name:45s} ERROR: {e}")
    elif args.text:
        if args.lang == "grc":
            labels = label_greek(args.text)
        else:
            labels = label_english(args.text)
        print(f"Labels: {labels}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
