#!/usr/bin/env python3
"""
Structural tree decomposition for English and Ancient Greek sentences.

Converts flat UD dependency parses (from stanza) into nested hierarchical
trees with linguistically meaningful levels:

  Sentence
  ├── Clause (main, relative, temporal, conditional, purpose, ...)
  │   ├── Phrase (NP subject, NP object, PP oblique, VP, ...)
  │   │   ├── Word (with morphological features)
  │   │   └── Word
  │   └── Phrase
  └── Clause (subordinate)
      └── ...

The same tree structure is used for both English and Greek — UD provides
the same dependency relations for both.  The decomposition can be viewed
at any level: sentence, clause, phrase, or word.

Storage: JSON trees, one per sentence.  Each node has:
  - level: "sentence" | "clause" | "phrase" | "word"
  - role: the linguistic function (subject, object, oblique, modifier, ...)
  - type: more specific label (main_clause, relative_clause, np, pp, ...)
  - features: morphological features (for words) or aggregate (for higher levels)
  - children: list of child nodes
  - text: surface text span

Usage:
  python3 scripts/tree_decompose.py "He turned to the man who spoke."  en
  python3 scripts/tree_decompose.py "ἐστράφη πρὸς τὸν ἄνδρα ὃς ἐλάλησεν."  grc
  python3 scripts/tree_decompose.py --passage 010_reverend_dialogue  # decompose a BM passage
  python3 scripts/tree_decompose.py --view clause  # show clause-level only
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Lazy stanza pipelines
_pipelines = {}


def _get_pipeline(lang: str):
    if lang not in _pipelines:
        import stanza
        _pipelines[lang] = stanza.Pipeline(
            lang, processors='tokenize,pos,lemma,depparse', verbose=False
        )
    return _pipelines[lang]


def parse_feats(feats_str: str | None) -> dict:
    if not feats_str:
        return {}
    return dict(p.split("=", 1) for p in feats_str.split("|") if "=" in p)


# ====================================================================
# Core decomposition
# ====================================================================

def decompose_sentence(sent, lang: str = "en") -> dict:
    """Decompose a stanza sentence into a nested tree.

    The algorithm:
    1. Build word nodes from the flat UD parse
    2. Identify clause boundaries (verbs and their dependents)
    3. Group words into phrases (NP, PP, VP)
    4. Nest phrases into clauses, clauses into the sentence
    """
    words = sent.words
    word_map = {w.id: w for w in words}

    # Step 1: Build word-level nodes
    word_nodes = {}
    for w in words:
        if w.upos == "PUNCT":
            continue
        word_nodes[w.id] = {
            "level": "word",
            "id": w.id,
            "text": w.text,
            "lemma": w.lemma,
            "upos": w.upos,
            "deprel": w.deprel,
            "head": w.head,
            "features": parse_feats(w.feats),
        }

    # Step 2: Identify clause heads (verbs that head clauses)
    clause_heads = []
    for w in words:
        if w.upos in ("VERB", "AUX") and w.deprel in (
            "root", "acl:relcl", "acl", "advcl", "ccomp", "xcomp",
            "conj",  # coordinated clauses
            "parataxis",
        ):
            clause_heads.append(w)

    # If no verb found, the whole sentence is a fragment
    if not clause_heads:
        clause_heads = [w for w in words if w.deprel == "root"]

    # Step 3: For each clause head, collect its dependents recursively
    def get_subtree_ids(head_id: int, exclude: set = None) -> set:
        """Get all word IDs in the subtree rooted at head_id."""
        if exclude is None:
            exclude = set()
        ids = {head_id}
        for w in words:
            if w.head == head_id and w.id not in exclude and w.upos != "PUNCT":
                # Don't cross into other clause heads (they form their own clauses)
                if w in clause_heads and w.id != head_id:
                    continue
                ids |= get_subtree_ids(w.id, exclude)
        return ids

    # Step 4: Build clause nodes
    claimed_ids = set()
    clause_nodes = []

    for ch in clause_heads:
        subtree_ids = get_subtree_ids(ch.id)
        # Remove IDs already claimed by a higher clause
        available_ids = subtree_ids - claimed_ids
        claimed_ids |= available_ids

        # Classify clause type
        clause_type = _classify_clause(ch, word_map, lang)

        # Group words in this clause into phrases
        phrase_nodes = _build_phrases(ch, available_ids, word_map, words, lang)

        clause_node = {
            "level": "clause",
            "type": clause_type,
            "role": ch.deprel,
            "head_verb": ch.text,
            "head_lemma": ch.lemma,
            "features": parse_feats(ch.feats),
            "children": phrase_nodes,
            "text": _span_text(available_ids, words),
            "word_ids": sorted(available_ids),
        }
        clause_nodes.append(clause_node)

    # Sentence-level node
    root_verb = next((w for w in words if w.deprel == "root"), None)
    has_root_verb = root_verb and root_verb.upos in ("VERB", "AUX")

    sentence_node = {
        "level": "sentence",
        "type": "fragment" if not has_root_verb else _classify_sentence(clause_nodes),
        "text": sent.text,
        "lang": lang,
        "word_count": len([w for w in words if w.upos != "PUNCT"]),
        "children": clause_nodes,
    }

    return sentence_node


def _classify_clause(verb_word, word_map: dict, lang: str) -> str:
    """Classify a clause by its type based on the verb's deprel and markers."""
    deprel = verb_word.deprel

    if deprel == "root":
        return "main"
    elif deprel == "acl:relcl":
        return "relative"
    elif deprel == "acl":
        feats = parse_feats(verb_word.feats)
        if feats.get("VerbForm") == "Part":
            if lang == "grc" and feats.get("Case") == "Gen":
                return "genitive_absolute"
            return "participial"
        return "adnominal"
    elif deprel == "advcl":
        # Check the marker to determine subtype
        # (marker is a child of this verb with deprel=mark)
        return "adverbial"  # refined below in _refine_advcl
    elif deprel == "ccomp":
        return "complement"
    elif deprel == "xcomp":
        return "open_complement"
    elif deprel == "conj":
        return "coordinated"
    elif deprel == "parataxis":
        return "parataxis"
    else:
        return deprel


def _refine_advcl(verb_word, words: list, lang: str) -> str:
    """Refine adverbial clause type based on its marker."""
    for w in words:
        if w.head == verb_word.id and w.deprel == "mark":
            marker = w.text.lower() if lang == "en" else w.lemma
            if lang == "en":
                if marker in ("if", "unless"):
                    return "conditional"
                elif marker in ("when", "while", "after", "before", "until", "since", "as"):
                    return "temporal"
                elif marker in ("because", "since"):
                    return "causal"
                elif marker in ("although", "though", "even"):
                    return "concessive"
                elif marker == "to":
                    return "purpose"
                elif marker in ("so", "that"):
                    return "result"
            else:  # grc
                if marker in ("εἰ", "ἐάν", "ἤν", "ἄν"):
                    return "conditional"
                elif marker in ("ὅτε", "ἐπεί", "ἐπειδή", "πρίν", "ἕως", "μέχρι"):
                    return "temporal"
                elif marker in ("ὅτι", "διότι"):
                    return "causal"
                elif marker in ("ἵνα", "ὅπως"):
                    return "purpose"
                elif marker == "ὥστε":
                    return "result"
                elif marker in ("εἰ", "καίπερ"):
                    return "concessive"
    return "adverbial"


def _build_phrases(clause_head, available_ids: set, word_map: dict,
                    all_words: list, lang: str) -> list[dict]:
    """Group words within a clause into phrase-level nodes."""
    phrases = []
    used_ids = set()

    # The clause head verb itself
    verb_phrase = {
        "level": "phrase",
        "type": "vp",
        "role": "predicate",
        "children": [{
            "level": "word",
            "text": clause_head.text,
            "lemma": clause_head.lemma,
            "upos": clause_head.upos,
            "features": parse_feats(clause_head.feats),
        }],
        "text": clause_head.text,
    }
    used_ids.add(clause_head.id)

    # Add auxiliaries and adverbs directly modifying the verb
    for w in all_words:
        if w.id in available_ids and w.head == clause_head.id and w.id != clause_head.id:
            if w.upos in ("AUX", "PART") or w.deprel in ("aux", "aux:pass", "advmod", "neg"):
                verb_phrase["children"].append({
                    "level": "word",
                    "text": w.text,
                    "lemma": w.lemma,
                    "upos": w.upos,
                    "features": parse_feats(w.feats),
                })
                verb_phrase["text"] = _span_text(
                    {c.get("id", 0) for c in verb_phrase["children"] if "id" in c} | {clause_head.id},
                    all_words
                ) or verb_phrase["text"]
                used_ids.add(w.id)

    phrases.append(verb_phrase)

    # Collect NP/PP phrases for each argument of the verb
    for w in all_words:
        if w.id not in available_ids or w.id in used_ids:
            continue
        if w.head != clause_head.id:
            continue

        if w.deprel in ("nsubj", "nsubj:pass"):
            np = _build_np(w, available_ids, used_ids, all_words, lang)
            np["role"] = "subject"
            phrases.append(np)
        elif w.deprel in ("obj", "iobj"):
            np = _build_np(w, available_ids, used_ids, all_words, lang)
            np["role"] = "object" if w.deprel == "obj" else "indirect_object"
            phrases.append(np)
        elif w.deprel == "obl":
            # Check if there's a preposition child → PP
            has_prep = any(
                c.head == w.id and c.deprel == "case" and c.upos == "ADP"
                for c in all_words if c.id in available_ids
            )
            if has_prep:
                pp = _build_pp(w, available_ids, used_ids, all_words, lang)
                pp["role"] = "oblique"
                phrases.append(pp)
            else:
                np = _build_np(w, available_ids, used_ids, all_words, lang)
                np["role"] = "oblique"
                phrases.append(np)
        elif w.deprel == "vocative":
            np = _build_np(w, available_ids, used_ids, all_words, lang)
            np["role"] = "vocative"
            phrases.append(np)
        elif w.deprel == "mark":
            phrases.append({
                "level": "phrase",
                "type": "marker",
                "role": "subordinator",
                "text": w.text,
                "children": [{"level": "word", "text": w.text, "lemma": w.lemma,
                              "upos": w.upos, "features": parse_feats(w.feats)}],
            })
            used_ids.add(w.id)

    # Remaining unclaimed words as loose modifiers
    for w in all_words:
        if w.id in available_ids and w.id not in used_ids and w.upos != "PUNCT":
            phrases.append({
                "level": "word",
                "text": w.text,
                "lemma": w.lemma,
                "upos": w.upos,
                "deprel": w.deprel,
                "features": parse_feats(w.feats),
            })
            used_ids.add(w.id)

    return phrases


def _build_np(head_word, available_ids: set, used_ids: set,
               all_words: list, lang: str) -> dict:
    """Build a noun phrase node from a head noun and its dependents."""
    np_ids = {head_word.id}

    children = []
    for w in all_words:
        if w.id in available_ids and w.head == head_word.id and w.id not in used_ids:
            if w.deprel in ("det", "amod", "nummod", "nmod", "appos", "flat", "compound"):
                np_ids.add(w.id)
                children.append({
                    "level": "word",
                    "text": w.text,
                    "lemma": w.lemma,
                    "upos": w.upos,
                    "deprel": w.deprel,
                    "features": parse_feats(w.feats),
                })

    # Head noun itself
    children.append({
        "level": "word",
        "text": head_word.text,
        "lemma": head_word.lemma,
        "upos": head_word.upos,
        "features": parse_feats(head_word.feats),
    })

    used_ids |= np_ids

    return {
        "level": "phrase",
        "type": "np",
        "text": _span_text(np_ids, all_words),
        "head_lemma": head_word.lemma,
        "features": parse_feats(head_word.feats),
        "children": children,
    }


def _build_pp(head_noun, available_ids: set, used_ids: set,
               all_words: list, lang: str) -> dict:
    """Build a prepositional phrase node."""
    pp_ids = set()
    prep_word = None

    # Find the preposition
    for w in all_words:
        if w.head == head_noun.id and w.deprel == "case" and w.upos == "ADP":
            prep_word = w
            pp_ids.add(w.id)
            break

    # Build the NP object of the preposition
    np = _build_np(head_noun, available_ids, used_ids, all_words, lang)
    pp_ids |= set(np.get("word_ids", []))

    children = []
    if prep_word:
        children.append({
            "level": "word",
            "text": prep_word.text,
            "lemma": prep_word.lemma,
            "upos": "ADP",
            "features": parse_feats(prep_word.feats),
        })
        used_ids.add(prep_word.id)

    children.append(np)

    return {
        "level": "phrase",
        "type": "pp",
        "text": _span_text(pp_ids | {head_noun.id}, all_words),
        "prep_lemma": prep_word.lemma if prep_word else "?",
        "governed_case": np.get("features", {}).get("Case", "?"),
        "children": children,
    }


def _span_text(word_ids: set, all_words: list) -> str:
    """Reconstruct text span from word IDs."""
    return " ".join(w.text for w in all_words if w.id in word_ids)


def _classify_sentence(clause_nodes: list) -> str:
    """Classify sentence type from its clause structure."""
    types = [c["type"] for c in clause_nodes]
    if len(types) == 1 and types[0] == "main":
        return "simple"
    elif any(t in ("relative", "temporal", "conditional", "causal", "purpose",
                    "result", "concessive", "complement", "adverbial",
                    "genitive_absolute", "participial")
             for t in types):
        return "complex"
    elif all(t in ("main", "coordinated", "parataxis") for t in types):
        return "compound"
    else:
        return "compound_complex"


# ====================================================================
# Tree views at different levels
# ====================================================================

def view_at_level(tree: dict, level: str) -> list[dict]:
    """Extract all nodes at a given level from the tree.

    level: "sentence", "clause", "phrase", or "word"
    """
    results = []

    if tree.get("level") == level:
        results.append(tree)
    else:
        for child in tree.get("children", []):
            results.extend(view_at_level(child, level))

    return results


def tree_signature(tree: dict, depth: int = 2) -> str:
    """Compact structural signature of a tree, truncated at depth.

    Examples:
      "simple[main[subj:np obj:np pred:vp]]"
      "complex[main[subj:np pred:vp obl:pp] relative[subj:np pred:vp]]"
    """
    if depth <= 0 or "children" not in tree:
        return tree.get("type", tree.get("upos", "?"))

    node_label = tree.get("type", tree.get("level", "?"))
    child_sigs = []
    for child in tree.get("children", []):
        role = child.get("role", child.get("deprel", ""))
        child_sig = tree_signature(child, depth - 1)
        if role:
            child_sigs.append(f"{role}:{child_sig}")
        else:
            child_sigs.append(child_sig)

    return f"{node_label}[{' '.join(child_sigs)}]"


# ====================================================================
# Pretty printing
# ====================================================================

def print_tree(tree: dict, indent: int = 0, max_depth: int = 10):
    """Pretty-print a decomposition tree."""
    prefix = "  " * indent
    level = tree.get("level", "?")
    ttype = tree.get("type", "")
    role = tree.get("role", "")
    text = tree.get("text", tree.get("lemma", ""))

    # Color/format by level
    label_parts = []
    if level == "sentence":
        label_parts.append(f"SENTENCE ({ttype})")
    elif level == "clause":
        label_parts.append(f"CLAUSE:{ttype}")
        if role:
            label_parts.append(f"[{role}]")
    elif level == "phrase":
        label_parts.append(f"{ttype.upper()}")
        if role:
            label_parts.append(f"({role})")
    elif level == "word":
        upos = tree.get("upos", "")
        feats = tree.get("features", {})
        feat_str = ""
        if feats:
            key_feats = {k: v for k, v in feats.items()
                         if k in ("Case", "Number", "Gender", "Tense", "Mood", "Voice", "VerbForm")}
            if key_feats:
                feat_str = " " + "|".join(f"{k}={v}" for k, v in key_feats.items())
        label_parts.append(f'"{text}" {upos}{feat_str}')

    label = " ".join(label_parts)

    # Add text span for non-word levels
    if level != "word" and text:
        display_text = text[:60] + ("..." if len(text) > 60 else "")
        print(f"{prefix}{label}")
        if level in ("sentence", "clause"):
            print(f"{prefix}  \"{display_text}\"")
    else:
        print(f"{prefix}{label}")

    if indent < max_depth:
        for child in tree.get("children", []):
            print_tree(child, indent + 1, max_depth)


# ====================================================================
# Batch decomposition
# ====================================================================

def decompose_text(text: str, lang: str) -> list[dict]:
    """Decompose a text into sentence trees."""
    nlp = _get_pipeline(lang)
    doc = nlp(text)
    trees = []
    for sent in doc.sentences:
        tree = decompose_sentence(sent, lang)
        # Refine adverbial clauses
        for clause in view_at_level(tree, "clause"):
            if clause["type"] == "adverbial":
                # Find the verb word to refine
                for w in sent.words:
                    if w.text == clause.get("head_verb") and w.deprel == "advcl":
                        clause["type"] = _refine_advcl(w, sent.words, lang)
                        break
        trees.append(tree)
    return trees


def decompose_passage(passage_id: str, lang: str = "grc") -> list[dict]:
    """Decompose a BM passage draft."""
    if lang == "grc":
        path = ROOT / "drafts" / passage_id / "primary.txt"
    else:
        path = ROOT / "passages" / f"{passage_id}.json"
        if path.exists():
            text = json.load(open(path)).get("text", "")
            return decompose_text(text, lang)
        return []

    if not path.exists():
        return []
    text = path.read_text("utf-8").strip()
    return decompose_text(text, lang)


# ====================================================================
# Main
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Structural tree decomposition")
    parser.add_argument("text", nargs="?", help="Text to decompose")
    parser.add_argument("lang", nargs="?", default="en", choices=["en", "grc"])
    parser.add_argument("--passage", type=str, help="BM passage ID")
    parser.add_argument("--view", choices=["sentence", "clause", "phrase", "word"],
                        help="Show only nodes at this level")
    parser.add_argument("--signature", action="store_true",
                        help="Show compact structural signatures")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--both", action="store_true",
                        help="Show both EN and GRC decompositions for a passage")
    args = parser.parse_args()

    if args.passage:
        if args.both:
            print(f"=== ENGLISH ===")
            en_trees = decompose_passage(args.passage, "en")
            for tree in en_trees:
                if args.signature:
                    print(f"  {tree_signature(tree, depth=3)}")
                elif args.json:
                    print(json.dumps(tree, ensure_ascii=False, indent=2))
                elif args.view:
                    for node in view_at_level(tree, args.view):
                        print_tree(node, max_depth=2)
                        print()
                else:
                    print_tree(tree)
                    print()

            print(f"\n=== ANCIENT GREEK ===")
            grc_trees = decompose_passage(args.passage, "grc")
            for tree in grc_trees:
                if args.signature:
                    print(f"  {tree_signature(tree, depth=3)}")
                elif args.json:
                    print(json.dumps(tree, ensure_ascii=False, indent=2))
                elif args.view:
                    for node in view_at_level(tree, args.view):
                        print_tree(node, max_depth=2)
                        print()
                else:
                    print_tree(tree)
                    print()
        else:
            lang = args.lang
            trees = decompose_passage(args.passage, lang)
            for tree in trees:
                if args.signature:
                    print(f"  {tree_signature(tree, depth=3)}")
                elif args.json:
                    print(json.dumps(tree, ensure_ascii=False, indent=2))
                elif args.view:
                    for node in view_at_level(tree, args.view):
                        print_tree(node, max_depth=2)
                        print()
                else:
                    print_tree(tree)
                    print()

    elif args.text:
        trees = decompose_text(args.text, args.lang)
        for tree in trees:
            if args.signature:
                print(f"  {tree_signature(tree, depth=3)}")
            elif args.json:
                print(json.dumps(tree, ensure_ascii=False, indent=2))
            elif args.view:
                for node in view_at_level(tree, args.view):
                    print_tree(node, max_depth=2)
                    print()
            else:
                print_tree(tree)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
