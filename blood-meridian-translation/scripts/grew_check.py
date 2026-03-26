#!/usr/bin/env python3
"""
Grammar checker using Grew pattern matching on stanza-parsed translations.

Patterns are defined in config/grew_rules.py — add new grammar rules there.
Patterns can be validated against the UD AG treebanks before using them on
our translations.

Requires: opam + grewpy_backend + grewpy, stanza (grc model)

Usage:
  python3 scripts/grew_check.py                          # all passages
  python3 scripts/grew_check.py 004_at_fourteen          # one passage
  python3 scripts/grew_check.py --validate               # validate patterns against treebanks
  python3 scripts/grew_check.py --warnings-only          # skip info-level detections
  python3 scripts/grew_check.py --report report.json     # JSON output
"""

import json
import os
import sys
from pathlib import Path

# Ensure opam env is loaded
opam_switch = os.path.expanduser("~/.opam/4.14.2")
if os.path.isdir(opam_switch):
    opam_bin = os.path.join(opam_switch, "bin")
    if opam_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = opam_bin + ":" + os.environ.get("PATH", "")
    os.environ["CAML_LD_LIBRARY_PATH"] = os.path.join(opam_switch, "lib", "stublibs")
    os.environ["OCAML_TOPLEVEL_PATH"] = os.path.join(opam_switch, "lib", "toplevel")

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
TREEBANK_DIR = ROOT / "data" / "treebanks"
CONLLU_PATH = ROOT / "output" / "translation.conllu"

# Import rules
sys.path.insert(0, str(ROOT / "config"))
from grew_rules import ALL_RULES, AGREEMENT_RULES, GOVERNMENT_RULES, CONSTRUCTION_RULES

# Lazy stanza
_grc_nlp = None


def _get_grc_nlp():
    global _grc_nlp
    if _grc_nlp is None:
        import stanza
        _grc_nlp = stanza.Pipeline(
            'grc', processors='tokenize,pos,lemma,depparse', verbose=False
        )
    return _grc_nlp


def _load_morpheus_cache() -> dict:
    """Load the Morpheus API cache for morphological ground truth."""
    cache_path = ROOT / "retrieval" / "data" / "morpheus_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


# Map Morpheus feature values → UD feature values
_MORPHEUS_TO_UD_GENDER = {"masculine": "Masc", "feminine": "Fem", "neuter": "Neut"}
_MORPHEUS_TO_UD_CASE = {
    "nominative": "Nom", "genitive": "Gen", "dative": "Dat",
    "accusative": "Acc", "vocative": "Voc",
}
_MORPHEUS_TO_UD_NUMBER = {"singular": "Sing", "plural": "Plur", "dual": "Dual"}


def _morpheus_consensus(analyses: list[dict], feat: str) -> str | None:
    """Get the consensus value for a morphological feature from Morpheus.
    Returns the value only if ALL analyses agree (no ambiguity)."""
    mapping = {
        "gender": _MORPHEUS_TO_UD_GENDER,
        "case": _MORPHEUS_TO_UD_CASE,
        "number": _MORPHEUS_TO_UD_NUMBER,
    }
    converter = mapping.get(feat, {})

    values = set()
    for a in analyses:
        raw = a.get(feat, "")
        if raw:
            values.add(converter.get(raw, raw))

    if len(values) == 1:
        return values.pop()
    return None  # ambiguous or missing


def _correct_feats_with_morpheus(feats_str: str, word_text: str,
                                  stanza_lemma: str,
                                  morpheus_cache: dict) -> str:
    """Correct stanza's morphological features using Morpheus as ground truth.

    Strategy:
    1. Look up the surface form in Morpheus cache → use if unambiguous
    2. If surface form unknown or ambiguous, look up stanza's LEMMA
       (nominative/dictionary form) and propagate gender from there.
       Gender is a lexical property: κριτής is always masculine regardless
       of the declined form.
    """
    clean = word_text.strip(".,·;:—–«»()[]!\"' *")
    if not clean:
        return feats_str

    # Parse current feats
    feats = {}
    if feats_str and feats_str != "_":
        for part in feats_str.split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                feats[k] = v

    # Try surface form first
    analyses = morpheus_cache.get(clean, [])
    if analyses and not any("error" in a for a in analyses):
        corrected = False
        for morph_feat, ud_feat in [("gender", "Gender"), ("case", "Case"), ("number", "Number")]:
            consensus = _morpheus_consensus(analyses, morph_feat)
            if consensus and feats.get(ud_feat) and feats[ud_feat] != consensus:
                feats[ud_feat] = consensus
                corrected = True
        if corrected:
            return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))
        return feats_str

    # Surface form not in Morpheus (or empty) — try the lemma for GENDER.
    # Gender is a lexical property that doesn't change with declension.
    lemma_clean = stanza_lemma.strip(".,·;:—–«»()[]!\"' *") if stanza_lemma else ""
    if lemma_clean and lemma_clean in morpheus_cache:
        lemma_analyses = morpheus_cache[lemma_clean]
        if lemma_analyses and not any("error" in a for a in lemma_analyses):
            gender_consensus = _morpheus_consensus(lemma_analyses, "gender")
            if gender_consensus and feats.get("Gender") and feats["Gender"] != gender_consensus:
                feats["Gender"] = gender_consensus
                return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))

    return feats_str


def parse_to_conllu(passage_ids: list[str] | None = None) -> str:
    """Parse drafts with stanza, correct morphology with Morpheus, return CoNLL-U."""
    nlp = _get_grc_nlp()
    morpheus_cache = _load_morpheus_cache()
    n_corrected = 0
    lines = []

    if passage_ids is None:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )

    for pid in passage_ids:
        draft_path = DRAFTS / pid / "primary.txt"
        if not draft_path.exists():
            continue

        text = draft_path.read_text("utf-8").strip()
        doc = nlp(text)

        for i, sent in enumerate(doc.sentences):
            lines.append(f"# sent_id = {pid}-{i+1}")
            lines.append(f"# text = {sent.text}")
            for word in sent.words:
                feats = word.feats or "_"
                if feats != "_" and morpheus_cache:
                    new_feats = _correct_feats_with_morpheus(
                        feats, word.text, word.lemma, morpheus_cache
                    )
                    if new_feats != feats:
                        n_corrected += 1
                        feats = new_feats

                lines.append(
                    f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t"
                    f"{word.xpos or '_'}\t{feats}\t{word.head}\t"
                    f"{word.deprel}\t_\t_"
                )
            lines.append("")

    if n_corrected:
        print(f"  Morpheus corrected {n_corrected} stanza feature values")
    return "\n".join(lines)


def _build_conllu_index(conllu_text: str) -> dict:
    """Build a lookup: (sent_id, word_id) → {form, lemma, feats...} from CoNLL-U."""
    index = {}
    current_sid = None
    for line in conllu_text.split("\n"):
        if line.startswith("# sent_id = "):
            current_sid = line[len("# sent_id = "):]
        elif line and not line.startswith("#") and current_sid:
            parts = line.split("\t")
            if len(parts) >= 8 and parts[0].isdigit():
                index[(current_sid, parts[0])] = {
                    "form": parts[1], "lemma": parts[2], "upos": parts[3],
                    "feats": parts[5],
                }
    return index


def _morpheus_gender_for(form: str, lemma: str, cache: dict) -> str | None:
    """Get unambiguous gender from Morpheus for a form or its lemma."""
    clean = form.strip(".,·;:—–«»()[]!\"' *")
    # Try surface form
    analyses = cache.get(clean, [])
    if analyses and not any("error" in a for a in analyses):
        g = _morpheus_consensus(analyses, "gender")
        if g:
            return g
    # Try lemma
    lemma_clean = lemma.strip(".,·;:—–«»()[]!\"' *") if lemma else ""
    if lemma_clean and lemma_clean in cache:
        la = cache[lemma_clean]
        if la and not any("error" in a for a in la):
            g = _morpheus_consensus(la, "gender")
            if g:
                return g
    return None


# Agreement rules where false positives from stanza are common
_AGREEMENT_RULES_TO_VERIFY = {"det_noun_gender", "det_noun_case", "det_noun_number",
                                "amod_gender", "amod_case", "amod_number"}


def run_checks(conllu_text: str, rules_to_run=None, warnings_only=False,
               include_noisy=False):
    """Run Grew pattern checks on CoNLL-U text. Returns list of findings."""
    from grewpy import Corpus, Request, set_config
    set_config('ud')

    # Write temp file for Grew
    CONLLU_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONLLU_PATH.write_text(conllu_text, encoding="utf-8")

    corpus = Corpus(str(CONLLU_PATH))
    print(f"  Loaded {len(corpus)} sentences into Grew")

    # Build index for Morpheus cross-verification
    conllu_index = _build_conllu_index(conllu_text)
    morpheus_cache = _load_morpheus_cache()

    if rules_to_run is None:
        rules_to_run = ALL_RULES

    findings = []
    n_suppressed = 0

    for name, severity, description, pattern_str in rules_to_run:
        if warnings_only and severity == "info":
            continue
        if severity == "noisy" and not include_noisy:
            continue

        try:
            request = Request(pattern_str)
            results = corpus.search(request)
        except Exception as e:
            print(f"  ⚠ Pattern error in '{name}': {e}")
            continue

        for match in results:
            sent_id = match.get("sent_id", "?")
            passage = "-".join(sent_id.rsplit("-", 1)[:-1]) if "-" in sent_id else sent_id
            nodes = match.get("matching", {}).get("nodes", {})

            # For agreement rules, cross-verify with Morpheus to suppress
            # stanza parse errors (the main source of false positives)
            if name in _AGREEMENT_RULES_TO_VERIFY and morpheus_cache:
                # Get both matched node IDs
                node_keys = list(nodes.keys())
                if len(node_keys) >= 2:
                    n_info = conllu_index.get((sent_id, nodes.get("N", "")))
                    d_info = conllu_index.get((sent_id, nodes.get("D", nodes.get("A", ""))))

                    if n_info and d_info and "gender" in name:
                        # Check if Morpheus agrees with stanza about the gender
                        n_gender = _morpheus_gender_for(
                            n_info["form"], n_info["lemma"], morpheus_cache)
                        d_gender = _morpheus_gender_for(
                            d_info["form"], d_info["lemma"], morpheus_cache)
                        if n_gender and d_gender and n_gender == d_gender:
                            n_suppressed += 1
                            continue  # Morpheus says they agree — stanza was wrong

            findings.append({
                "rule": name,
                "severity": severity,
                "description": description,
                "sent_id": sent_id,
                "passage": passage,
                "matched_nodes": nodes,
            })

    if n_suppressed:
        print(f"  Morpheus suppressed {n_suppressed} false-positive agreement warnings")
    return findings


def validate_against_treebanks(rules_to_validate=None):
    """Run patterns against gold treebanks to see baseline rates."""
    from grewpy import Corpus, Request, set_config
    set_config('ud')

    tb_files = sorted(TREEBANK_DIR.glob("*.conllu"))
    if not tb_files:
        print("No treebank files found. Run mine_grammar_rules.py first.")
        return

    train_files = [str(f) for f in tb_files if "train" in f.name]
    corpus = Corpus(train_files)
    print(f"  Treebank corpus: {len(corpus)} sentences")

    if rules_to_validate is None:
        rules_to_validate = ALL_RULES

    print(f"\n{'Rule':<30s} {'Matches':>8s}  {'Rate':>8s}  Description")
    print("-" * 90)

    for name, severity, description, pattern_str in rules_to_validate:
        try:
            request = Request(pattern_str)
            results = corpus.search(request)
            rate = len(results) / len(corpus) * 100
            marker = "⚠" if severity == "warning" and len(results) > 0 else " "
            print(f"{marker} {name:<28s} {len(results):>8d}  {rate:>7.2f}%  {description}")
        except Exception as e:
            print(f"  {name:<28s}    ERROR  {e}")


def enrich_findings(findings: list[dict], conllu_text: str) -> list[dict]:
    """Add sentence text to findings for readable output."""
    # Build sent_id → text map from CoNLL-U
    sent_texts = {}
    current_id = None
    for line in conllu_text.split("\n"):
        if line.startswith("# sent_id = "):
            current_id = line[len("# sent_id = "):]
        elif line.startswith("# text = ") and current_id:
            sent_texts[current_id] = line[len("# text = "):]

    for f in findings:
        f["sentence"] = sent_texts.get(f["sent_id"], "")

    return findings


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Grew-based AG grammar checker")
    parser.add_argument("passages", nargs="*", help="Passage IDs to check")
    parser.add_argument("--validate", action="store_true",
                        help="Validate patterns against treebanks")
    parser.add_argument("--warnings-only", action="store_true",
                        help="Only show warnings, skip info-level")
    parser.add_argument("--noisy", action="store_true",
                        help="Include noisy rules (agreement checks with high FP rate)")
    parser.add_argument("--report", type=str, help="Save JSON report")
    args = parser.parse_args()

    if args.validate:
        print("Validating patterns against UD AG treebanks...")
        validate_against_treebanks()
        return

    passage_ids = args.passages or None
    desc = f"passages {', '.join(args.passages)}" if args.passages else "all passages"
    print(f"Parsing {desc} with stanza...")
    conllu_text = parse_to_conllu(passage_ids)

    print("Running Grew pattern checks...")
    findings = run_checks(conllu_text, warnings_only=args.warnings_only,
                          include_noisy=args.noisy)
    findings = enrich_findings(findings, conllu_text)

    # Print results
    noisy = [f for f in findings if f["severity"] == "noisy"]
    warnings = [f for f in findings if f["severity"] == "warning"]
    reviews = [f for f in findings if f["severity"] == "review"]
    infos = [f for f in findings if f["severity"] == "info"]

    print(f"\n{'='*70}")
    print(f"Grew Grammar Check: {len(findings)} findings")
    print(f"{'='*70}")

    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for f in warnings:
            print(f"  [{f['passage']}] {f['rule']}: {f['description']}")
            if f.get("sentence"):
                print(f"    → {f['sentence'][:80]}...")

    if reviews:
        print(f"\n⟐ REVIEW ({len(reviews)}):")
        for f in reviews:
            print(f"  [{f['passage']}] {f['rule']}: {f['description']}")
            if f.get("sentence"):
                print(f"    → {f['sentence'][:80]}...")

    if noisy:
        print(f"\n~ NOISY ({len(noisy)} — stanza parse may be wrong):")
        for f in noisy:
            print(f"  [{f['passage']}] {f['rule']}: {f['description']}")
            if f.get("sentence"):
                print(f"    → {f['sentence'][:80]}...")

    if not args.warnings_only and infos:
        print(f"\nℹ CONSTRUCTIONS ({len(infos)}):")
        for f in infos:
            print(f"  [{f['passage']}] {f['rule']}: {f['description']}")

    if not warnings and not reviews:
        print("\n  ✓ No grammar violations found.")

    # Save report
    if args.report:
        with open(args.report, "w") as f:
            json.dump(findings, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
