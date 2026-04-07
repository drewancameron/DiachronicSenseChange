#!/usr/bin/env python3
"""
Construction Dispatcher: maps English sentence structures to Greek construction skeletons.

Takes a stanza-parsed English sentence and produces:
1. Clause decomposition (main clause, subordinate clauses)
2. Construction type per clause (conditional, temporal, relative, etc.)
3. Greek construction choice with morphological targets (mood, tense, voice)
4. Word-level vocabulary via Woodhouse → LSJ fallback chain

This is Phase 3 of the mechanical-first translation pipeline.

Usage:
  python3 scripts/construction_dispatcher.py "See the child."
  python3 scripts/construction_dispatcher.py --passage 001_see_the_child_he
  python3 scripts/construction_dispatcher.py --test
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = ROOT / "glossary"
MODELS = ROOT / "models" / "construction_model"

# ====================================================================
# Data structures
# ====================================================================

@dataclass
class GreekTarget:
    """Target morphological features for a Greek word."""
    lemma: str          # Greek lemma (from Woodhouse/LSJ)
    pos: str            # noun, verb, adj, adv, prep, conj, particle
    case: str = ""      # nom, gen, dat, acc, voc
    number: str = ""    # sg, pl
    gender: str = ""    # masc, fem, neut
    tense: str = ""     # pres, impf, fut, aor, perf, plup
    mood: str = ""      # ind, subj, opt, imp, inf, ptcp
    voice: str = ""     # act, mid, pass
    person: str = ""    # 1, 2, 3
    role: str = ""      # subj, obj, iobj, pred, attr, adv
    english: str = ""   # original English word


@dataclass
class ClauseSkeleton:
    """A clause-level translation skeleton."""
    clause_type: str          # main, relative, conditional, temporal, purpose, result
    construction: str         # specific Greek construction name
    words: list = field(default_factory=list)  # list of GreekTarget
    subordinator: str = ""    # ἐάν, ὅτε, ἵνα, ὅς, etc.
    notes: str = ""           # translation notes


@dataclass
class SentenceSkeleton:
    """Complete sentence-level translation plan."""
    english: str
    clauses: list = field(default_factory=list)  # list of ClauseSkeleton
    style: str = "paratactic"  # paratactic, periodic, fragmentary


# ====================================================================
# Vocabulary lookup chain
# ====================================================================

_woodhouse = None
_lsj_en = None
_locked_glossary = None


def _load_vocab():
    global _woodhouse, _lsj_en, _locked_glossary
    if _woodhouse is not None:
        return

    wh_path = GLOSSARY / "woodhouse.json"
    lsj_path = GLOSSARY / "en_to_grc.json"
    glossary_path = GLOSSARY / "idf_glossary.json"

    _woodhouse = json.load(open(wh_path)) if wh_path.exists() else {}
    _lsj_en = json.load(open(lsj_path)) if lsj_path.exists() else {}

    _locked_glossary = {}
    if glossary_path.exists():
        data = json.load(open(glossary_path))
        for cat, entries in data.items():
            if cat.startswith("_") or not isinstance(entries, dict):
                continue
            for key, entry in entries.items():
                if isinstance(entry, dict) and entry.get("status") == "locked":
                    en = entry.get("english", "").lower()
                    ag = entry.get("ancient_greek", "").replace("*", "")
                    if en and ag:
                        _locked_glossary[en] = ag

    print(f"  Vocab: {len(_locked_glossary)} locked, "
          f"{len(_woodhouse):,} Woodhouse, {len(_lsj_en):,} LSJ")


def lookup_greek(english_word: str, pos_hint: str = "") -> str | None:
    """Look up the Greek lemma for an English word.

    Priority: locked glossary → Woodhouse → inverted LSJ.
    Returns the Greek lemma or None.
    """
    _load_vocab()
    en = english_word.lower().strip()

    # 1. Locked glossary (highest priority)
    if en in _locked_glossary:
        return _locked_glossary[en]

    # 2. Woodhouse (composition dictionary — best for primary meanings)
    if en in _woodhouse:
        greek_words = _woodhouse[en].get("greek", [])
        if greek_words:
            return greek_words[0]  # first = most common

    # 3. Inverted LSJ (fallback)
    if en in _lsj_en:
        entries = _lsj_en[en]
        if entries:
            return entries[0]["lemma"]

    return None


# ====================================================================
# Construction distributions
# ====================================================================

_distributions = None


def _load_distributions():
    global _distributions
    if _distributions is not None:
        return
    dist_path = MODELS / "cond_distributions.json"
    if dist_path.exists():
        _distributions = json.load(open(dist_path))
    else:
        _distributions = {}


def get_greek_construction(en_construction: str, period: str = "koine") -> str:
    """Given an English construction type, return the most likely Greek construction.

    Uses P(GRC | EN) from the parallel corpus.
    """
    _load_distributions()

    # Map period names
    period_map = {"koine": "imperial", "attic": "classical",
                  "classical": "classical", "imperial": "imperial"}
    p = period_map.get(period, "overall")

    # Look up in period-specific distributions first
    dists = _distributions.get("by_period", {}).get(p, {})
    if en_construction not in dists:
        dists = _distributions.get("overall", {})

    if en_construction not in dists:
        return en_construction  # passthrough

    dist = dists[en_construction].get("distribution", {})
    # Return highest probability non-"none" construction
    best = None
    best_prob = 0
    for grc_name, info in dist.items():
        if grc_name == "none":
            continue
        prob = info.get("probability", 0)
        if prob > best_prob:
            best_prob = prob
            best = grc_name

    return best or en_construction


# ====================================================================
# English sentence analysis
# ====================================================================

_en_nlp = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        import stanza
        _en_nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse",
                                  verbose=False)
    return _en_nlp


# Tense mapping: English verb features → Greek target tense/mood
TENSE_MAP = {
    # English tense → (Greek tense, Greek mood)
    ("Past", "Ind"): ("aor", "ind"),      # "he ran" → aorist indicative
    ("Pres", "Ind"): ("pres", "ind"),     # "he runs" → present indicative
    ("Past", "Part"): ("aor", "ptcp"),    # "having run" → aorist participle
    ("Pres", "Part"): ("pres", "ptcp"),   # "running" → present participle
    ("Inf",): ("pres", "inf"),            # "to run" → present infinitive
}

# McCarthy-specific style rules
MCCARTHY_RULES = {
    "comma_splice": "asyndeton",    # McCarthy comma splice → Greek asyndeton
    "and_chain": "kai_chain",       # "and...and...and" → καί...καί...καί
    "fragment": "fragment",         # "Dark woods." → no added verb
    "present_tense": "present",     # McCarthy's historical present → Greek present
}


def analyse_sentence(text: str) -> SentenceSkeleton:
    """Analyse an English sentence and produce a translation skeleton.

    Returns a SentenceSkeleton with clauses, construction choices,
    and vocabulary targets.
    """
    _load_vocab()
    _load_distributions()

    skeleton = SentenceSkeleton(english=text)

    # Quick checks for fragments and simple structures
    words = text.split()
    if len(words) <= 3 and not any(w.lower() in ("is", "was", "are", "were",
                                                    "has", "had", "does", "did")
                                   for w in words):
        skeleton.style = "fragmentary"

    # Parse with stanza
    nlp = _get_en_nlp()
    doc = nlp(text)

    for sent in doc.sentences:
        clause = _analyse_clause(sent)
        skeleton.clauses.append(clause)

    return skeleton


def _analyse_clause(sent) -> ClauseSkeleton:
    """Analyse a single stanza sentence/clause."""
    clause = ClauseSkeleton(clause_type="main", construction="simple_declarative")

    # Identify clause type from dependency structure
    has_subordinator = False
    for word in sent.words:
        # Check for subordinating conjunctions
        if word.deprel == "mark":
            has_subordinator = True
            if word.text.lower() in ("if", "unless"):
                clause.clause_type = "conditional"
                clause.construction = _identify_conditional(sent)
                clause.subordinator = _conditional_subordinator(clause.construction)
            elif word.text.lower() in ("when", "while", "after", "before",
                                        "until", "once"):
                clause.clause_type = "temporal"
                clause.construction = "temporal_clause"
                clause.subordinator = _temporal_subordinator(word.text.lower())
            elif word.text.lower() in ("that", "because", "since"):
                clause.clause_type = "causal"
                clause.construction = "causal_clause"
            elif word.text.lower() in ("so", "lest"):
                clause.clause_type = "purpose"
                clause.construction = "purpose_clause"
                clause.subordinator = "ἵνα"

        # Check for relative pronouns
        if word.deprel in ("nsubj", "obj", "obl") and word.text.lower() in (
                "who", "whom", "whose", "which", "that"):
            clause.clause_type = "relative"
            clause.construction = "relative_clause"
            clause.subordinator = "ὅς"

    # Process each word
    for word in sent.words:
        target = _word_to_target(word, clause)
        if target:
            clause.words.append(target)

    return clause


def _word_to_target(word, clause: ClauseSkeleton) -> GreekTarget | None:
    """Convert a stanza word to a Greek translation target."""
    # Skip punctuation
    if word.upos == "PUNCT":
        return None

    # Skip function words that'll be handled structurally
    if word.upos in ("DET",) and word.text.lower() in ("the", "a", "an"):
        # Articles handled separately in assembly
        return None

    # Look up Greek lemma
    lemma = word.lemma or word.text
    greek = lookup_greek(lemma)

    # Determine POS mapping
    pos_map = {
        "NOUN": "noun", "PROPN": "noun", "VERB": "verb", "AUX": "verb",
        "ADJ": "adj", "ADV": "adv", "ADP": "prep", "CCONJ": "conj",
        "SCONJ": "conj", "PRON": "pron", "NUM": "num",
    }
    pos = pos_map.get(word.upos, "other")

    target = GreekTarget(
        lemma=greek or f"[{lemma}]",
        pos=pos,
        english=word.text,
        role=word.deprel,
    )

    # Set verb features
    if pos == "verb":
        feats = _parse_feats(word.feats)
        tense = feats.get("Tense", "Pres")
        mood = feats.get("Mood", "Ind")
        voice = feats.get("Voice", "Act")
        person = feats.get("Person", "3")
        number = feats.get("Number", "Sing")

        # Map English tense to Greek
        tense_key = (tense, mood)
        if tense_key in TENSE_MAP:
            target.tense, target.mood = TENSE_MAP[tense_key]
        else:
            target.tense = tense.lower()[:4]
            target.mood = mood.lower()[:3]

        target.voice = {"Act": "act", "Pass": "pass", "Mid": "mid"}.get(voice, "act")
        target.person = person
        target.number = {"Sing": "sg", "Plur": "pl"}.get(number, "sg")

        # McCarthy's historical present → keep as present
        if target.tense == "pres" and target.mood == "ind":
            pass  # good as is

    # Set noun/adj features
    elif pos in ("noun", "adj", "pron"):
        feats = _parse_feats(word.feats)
        target.number = {"Sing": "sg", "Plur": "pl"}.get(
            feats.get("Number", "Sing"), "sg")
        # Case from dependency role
        target.case = _role_to_case(word.deprel)

    return target


def _parse_feats(feats_str: str | None) -> dict:
    """Parse stanza feature string like 'Number=Sing|Person=3|Tense=Pres'."""
    if not feats_str:
        return {}
    return dict(f.split("=") for f in feats_str.split("|") if "=" in f)


def _role_to_case(deprel: str) -> str:
    """Map dependency relation to likely Greek case."""
    case_map = {
        "nsubj": "nom",
        "nsubj:pass": "nom",
        "obj": "acc",
        "iobj": "dat",
        "obl": "dat",      # oblique — could be gen/dat depending on preposition
        "nmod": "gen",
        "vocative": "voc",
    }
    return case_map.get(deprel, "")


def _identify_conditional(sent) -> str:
    """Identify the type of conditional from the English parse."""
    # Check verb tenses to classify
    from conditional_guide import identify_constructions
    text = " ".join(w.text for w in sent.words)
    constructions = identify_constructions(text)
    if constructions:
        return constructions[0].get("construction", "conditional_real")
    return "conditional_real"


def _conditional_subordinator(construction: str) -> str:
    """Return the Greek subordinator for a conditional type."""
    sub_map = {
        "conditional_real": "εἰ",
        "conditional_fv": "ἐάν",
        "future_more_vivid": "ἐάν",
        "present_contrafactual": "εἰ",
        "past_contrafactual": "εἰ",
        "future_less_vivid": "εἰ",
        "present_general": "ἐάν",
    }
    return sub_map.get(construction, "εἰ")


def _temporal_subordinator(english_word: str) -> str:
    """Return the Greek temporal subordinator."""
    sub_map = {
        "when": "ὅτε",
        "while": "ἕως",
        "after": "ἐπεί",
        "before": "πρίν",
        "until": "ἕως",
        "once": "ἐπεί",
        "whenever": "ὅταν",
    }
    return sub_map.get(english_word, "ὅτε")


# ====================================================================
# Output formatting
# ====================================================================

def skeleton_to_dict(skeleton: SentenceSkeleton) -> dict:
    """Convert skeleton to a JSON-serializable dict."""
    return {
        "english": skeleton.english,
        "style": skeleton.style,
        "clauses": [
            {
                "type": c.clause_type,
                "construction": c.construction,
                "subordinator": c.subordinator,
                "words": [
                    {
                        "english": w.english,
                        "lemma": w.lemma,
                        "pos": w.pos,
                        "case": w.case,
                        "number": w.number,
                        "tense": w.tense,
                        "mood": w.mood,
                        "voice": w.voice,
                        "person": w.person,
                        "role": w.role,
                    }
                    for w in c.words
                ],
            }
            for c in skeleton.clauses
        ],
    }


def print_skeleton(skeleton: SentenceSkeleton):
    """Pretty-print a translation skeleton."""
    print(f"  English: {skeleton.english}")
    print(f"  Style: {skeleton.style}")
    for i, clause in enumerate(skeleton.clauses):
        print(f"  Clause {i}: {clause.clause_type} ({clause.construction})")
        if clause.subordinator:
            print(f"    Subordinator: {clause.subordinator}")
        for w in clause.words:
            features = []
            if w.tense: features.append(w.tense)
            if w.mood: features.append(w.mood)
            if w.voice: features.append(w.voice)
            if w.case: features.append(w.case)
            if w.number: features.append(w.number)
            feat_str = f" ({', '.join(features)})" if features else ""
            found = "✓" if not w.lemma.startswith("[") else "✗"
            print(f"    {found} {w.english:15s} → {w.lemma:20s} {w.pos}{feat_str}")


# ====================================================================
# CLI
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Construction dispatcher")
    parser.add_argument("text", nargs="*", help="English text to analyse")
    parser.add_argument("--passage", type=str, help="Passage ID")
    parser.add_argument("--test", action="store_true", help="Run test sentences")
    args = parser.parse_args()

    if args.test:
        test_sentences = [
            "See the child.",
            "He is pale and thin, he wears a thin and ragged linen shirt.",
            "He stokes the scullery fire.",
            "Outside lie dark turned fields with rags of snow.",
            "His folk are known for hewers of wood and drawers of water.",
            "The boy crouches by the fire and watches him.",
            "Why damn my eyes if I wont shoot the son of a bitch.",
            "Night of your birth.",
        ]
        for sent in test_sentences:
            print(f"\n{'='*60}")
            skeleton = analyse_sentence(sent)
            print_skeleton(skeleton)

    elif args.passage:
        p_path = ROOT / "passages" / f"{args.passage}.json"
        if not p_path.exists():
            print(f"Passage not found: {args.passage}")
            return
        en_text = json.load(open(p_path)).get("text", "")
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text)
                     if s.strip()]
        for sent in sentences[:5]:  # first 5
            print(f"\n{'='*60}")
            skeleton = analyse_sentence(sent)
            print_skeleton(skeleton)

    elif args.text:
        text = " ".join(args.text)
        skeleton = analyse_sentence(text)
        print_skeleton(skeleton)
        print()
        print(json.dumps(skeleton_to_dict(skeleton), indent=2, ensure_ascii=False))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
