#!/usr/bin/env python3
"""
Generate human-readable structural descriptions of English and Greek sentences.

Given a sentence, produces a description like:
  "Simple transitive: subject(pronoun, nominative) + verb(present indicative active)
   + object(article + noun, accusative) with genitive modifier"

This replaces bare labels like "compound, 13w" with actual grammatical descriptions
that an LLM can use as a blueprint for translation.

Usage:
  python3 scripts/describe_structure.py "He stokes the scullery fire." en
  python3 scripts/describe_structure.py "σκαλεύει τὴν ἐσχάραν τοῦ μαγειρείου." grc
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_pipelines = {}


def _get_pipeline(lang: str):
    if lang not in _pipelines:
        import stanza
        _pipelines[lang] = stanza.Pipeline(
            lang, processors='tokenize,pos,lemma,depparse', verbose=False
        )
    return _pipelines[lang]


def _feats(w) -> dict:
    if not w.feats:
        return {}
    return dict(p.split("=", 1) for p in w.feats.split("|") if "=" in p)


def describe_word(w, lang: str = "en") -> str:
    """Compact grammatical description of a word."""
    f = _feats(w)
    parts = [w.text]

    if w.upos == "VERB" or w.upos == "AUX":
        mood = f.get("Mood", "")
        tense = f.get("Tense", "")
        voice = f.get("Voice", "")
        vform = f.get("VerbForm", "")
        if vform == "Inf":
            parts.append("infinitive")
        elif vform == "Part":
            case = f.get("Case", "")
            parts.append(f"participle{'·' + case.lower() if case else ''}")
        elif mood:
            parts.append(mood.lower())
        if tense:
            parts.append(tense.lower())
        if voice and voice != "Act":
            parts.append(voice.lower())
    elif w.upos == "NOUN" or w.upos == "PROPN":
        case = f.get("Case", "")
        num = f.get("Number", "")
        if case:
            parts.append(case.lower())
        if num and num != "Sing":
            parts.append(num.lower())
    elif w.upos == "PRON":
        case = f.get("Case", "")
        if case:
            parts.append(case.lower())
    elif w.upos == "ADJ":
        case = f.get("Case", "")
        if case:
            parts.append(case.lower())
    elif w.upos == "DET":
        case = f.get("Case", "")
        if case:
            parts.append(case.lower())

    return " ".join(parts)


def describe_phrase(head_word, words: list, lang: str) -> str:
    """Describe a phrase rooted at head_word."""
    children = [w for w in words if w.head == head_word.id and w.upos != "PUNCT"]

    parts = []
    # Determiner
    det = next((w for w in children if w.deprel == "det"), None)
    if det:
        parts.append(f"article({describe_word(det, lang)})")

    # Adjective modifiers
    adjs = [w for w in children if w.deprel == "amod"]
    for a in adjs:
        parts.append(f"adj({describe_word(a, lang)})")

    # Head
    parts.append(describe_word(head_word, lang))

    # Genitive/prepositional modifiers
    nmods = [w for w in children if w.deprel in ("nmod", "nmod:poss")]
    for nm in nmods:
        prep = next((w for w in words if w.head == nm.id and w.deprel == "case"), None)
        if prep:
            parts.append(f"+ {prep.text} {describe_word(nm, lang)}")
        else:
            parts.append(f"+ {describe_word(nm, lang)}")

    return " ".join(parts)


def describe_clause(verb_word, words: list, lang: str) -> str:
    """Describe a clause rooted at a verb."""
    children = [w for w in words if w.head == verb_word.id and w.upos != "PUNCT"]

    parts = []

    # Subject
    subj = next((w for w in children if w.deprel in ("nsubj", "nsubj:pass")), None)
    if subj:
        parts.append(f"subject({describe_phrase(subj, words, lang)})")

    # Verb
    parts.append(f"verb({describe_word(verb_word, lang)})")

    # Object
    obj = next((w for w in children if w.deprel == "obj"), None)
    if obj:
        parts.append(f"object({describe_phrase(obj, words, lang)})")

    # Indirect object
    iobj = next((w for w in children if w.deprel == "iobj"), None)
    if iobj:
        parts.append(f"indirect_object({describe_phrase(iobj, words, lang)})")

    # Oblique/PP
    obls = [w for w in children if w.deprel == "obl"]
    for obl in obls:
        prep = next((w for w in words if w.head == obl.id and w.deprel == "case"), None)
        if prep:
            parts.append(f"oblique({prep.text} + {describe_phrase(obl, words, lang)})")
        else:
            parts.append(f"oblique({describe_phrase(obl, words, lang)})")

    # Predicate adjective / complement
    for w in children:
        if w.deprel in ("xcomp", "ccomp") and w.upos == "ADJ":
            parts.append(f"predicate({describe_word(w, lang)})")

    # Adverbial modifiers
    advmods = [w for w in children if w.deprel == "advmod" and w.upos == "ADV"]
    for a in advmods:
        parts.append(f"adverb({a.text})")

    return " + ".join(parts)


def describe_sentence(sent, lang: str = "en") -> str:
    """Full structural description of a sentence."""
    words = sent.words

    # Find root
    root = next((w for w in words if w.deprel == "root"), None)
    if not root:
        return f"[no root found] \"{sent.text[:60]}\""

    has_root_verb = root.upos in ("VERB", "AUX")

    # Find subordinate clauses
    subclauses = []
    for w in words:
        if w.deprel == "acl:relcl":
            head_noun = next((h for h in words if h.id == w.head), None)
            subclauses.append(("relative clause", w, head_noun))
        elif w.deprel == "advcl":
            marker = next((m for m in words if m.head == w.id and m.deprel == "mark"), None)
            marker_text = marker.text.lower() if marker else "?"
            if marker_text in ("if", "unless") or (lang == "grc" and marker and marker.lemma in ("εἰ", "ἐάν")):
                subclauses.append(("conditional", w, None))
            elif marker_text in ("when", "while", "after", "before", "until") or \
                 (lang == "grc" and marker and marker.lemma in ("ὅτε", "ἐπεί", "ἕως", "πρίν")):
                subclauses.append(("temporal", w, None))
            else:
                subclauses.append(("adverbial", w, None))
        elif w.deprel in ("ccomp", "xcomp"):
            subclauses.append(("complement", w, None))

    # Count coordination
    n_coord = sum(1 for w in words if w.deprel == "conj")

    # Build description
    lines = []

    # Sentence type
    if not has_root_verb:
        lines.append("Verbless fragment.")
    elif not subclauses and n_coord <= 1:
        lines.append("Simple sentence.")
    elif not subclauses and n_coord >= 2:
        lines.append(f"Compound sentence ({n_coord} coordinated clauses).")
    elif subclauses and n_coord <= 1:
        lines.append("Complex sentence.")
    else:
        lines.append(f"Compound-complex sentence ({n_coord} coordinations).")

    # Main clause
    lines.append(f"Main clause: {describe_clause(root, words, lang)}")

    # Subordinate clauses
    for clause_type, verb, head_noun in subclauses:
        clause_desc = describe_clause(verb, words, lang)
        if clause_type == "relative clause" and head_noun:
            lines.append(
                f"Relative clause on '{head_noun.text}': {clause_desc}"
            )
        elif clause_type == "conditional":
            marker = next((m for m in words if m.head == verb.id and m.deprel == "mark"), None)
            lines.append(f"Conditional ({marker.text if marker else '?'}): {clause_desc}")
        elif clause_type == "temporal":
            marker = next((m for m in words if m.head == verb.id and m.deprel == "mark"), None)
            lines.append(f"Temporal ({marker.text if marker else '?'}): {clause_desc}")
        else:
            lines.append(f"{clause_type.capitalize()}: {clause_desc}")

    # Coordination detail
    if n_coord >= 2:
        conj_words = [w for w in words if w.deprel == "conj"]
        lines.append(f"Coordinated elements: {', '.join(w.text for w in conj_words)}")

    return "\n  ".join(lines)


def describe_text(text: str, lang: str) -> str:
    """Describe all sentences in a text."""
    nlp = _get_pipeline(lang)
    doc = nlp(text)
    descriptions = []
    for i, sent in enumerate(doc.sentences, 1):
        desc = describe_sentence(sent, lang)
        descriptions.append(f"{i}. \"{sent.text[:70]}{'...' if len(sent.text) > 70 else ''}\"\n  {desc}")
    return "\n\n".join(descriptions)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 describe_structure.py \"sentence\" [en|grc]")
        sys.exit(1)

    text = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    print(describe_text(text, lang))
