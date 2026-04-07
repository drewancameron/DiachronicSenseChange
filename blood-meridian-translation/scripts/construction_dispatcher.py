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

    Priority: locked glossary → Woodhouse (POS-aware) → inverted LSJ.
    Returns the Greek lemma or None.
    """
    return _lookup_pos_aware(english_word, english_word, pos_hint or "")


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
    # Check for multi-word expressions in locked glossary FIRST
    sent_text = " ".join(w.text.lower() for w in sent.words)
    _load_vocab()
    mwe_spans = set()  # word indices consumed by MWE matches
    for en_phrase, grc in _locked_glossary.items():
        if en_phrase in sent_text and " " in en_phrase:
            # Find which word indices this phrase covers
            phrase_words = en_phrase.split()
            for i in range(len(sent.words) - len(phrase_words) + 1):
                window = [sent.words[i + j].text.lower() for j in range(len(phrase_words))]
                if window == phrase_words:
                    # Emit the Greek as a single unit at the first word's position
                    target = GreekTarget(lemma=grc, pos="noun",
                                         english=en_phrase, role="obj")
                    clause.words.append(target)
                    for j in range(len(phrase_words)):
                        mwe_spans.add(i + j)
                    break

    for i, word in enumerate(sent.words):
        if i in mwe_spans:
            continue  # consumed by MWE
        target = _word_to_target(word, clause)
        if target:
            clause.words.append(target)

    return clause


# Function words: don't look up in Woodhouse/LSJ, handle structurally
_SKIP_WORDS = {"the", "a", "an"}  # articles: added by assembler

# McCarthy-specific vocabulary not in Woodhouse/LSJ
_MCCARTHY_VOCAB = {
    "stoke": "σκαλεύειν",       # poke/stir a fire
    "stokes": "σκαλεύειν",
    "scullery": "μαγειρεῖον",    # kitchen
    "saloon": "καπηλεῖον",       # bar/tavern
    "bartender": "κάπηλος",
    "barman": "κάπηλος",
    "pistol": "πιστόλιον",
    "gun": "πιστόλιον",
    "rifle": "τόξον",
    "saddle": "ἐφίππιον",
    "boots": "ἐμβάδες",
    "hat": "πέτασος",
    "felt": "πῖλος",
    "mud": "πηλός",
    "mule": "ἡμίονος",
    "flatboat": "σχεδία",
    "steamboat": "ἀτμόπλοιον",
    "tent": "σκηνή",
    "preacher": "ἱερεύς",
    "reverend": "αἰδέσιμος",
    "judge": "κριτής",
    "nigger": "μέλας",           # McCarthy's language, not endorsed
    "whiskey": "οἶνος",          # closest Greek equivalent
    "tobacco": "καπνός",
}

# Pronouns: map directly to Greek
_PRONOUN_MAP = {
    # Subject pronouns — usually DROPPED in Greek (pro-drop language)
    "he": None, "she": None, "it": None, "they": None,
    "i": "ἐγώ", "we": "ἡμεῖς", "you": "σύ",
    # Object pronouns
    "him": "αὐτός", "her": "αὐτός", "them": "αὐτός",
    "me": "ἐγώ", "us": "ἡμεῖς",
    # Possessives → genitive of pronoun
    "his": "αὐτός", "her": "αὐτός", "its": "αὐτός", "their": "αὐτός",
    "my": "ἐμός", "our": "ἡμέτερος", "your": "σός",
    # Demonstratives
    "this": "οὗτος", "that": "ἐκεῖνος", "these": "οὗτος", "those": "ἐκεῖνος",
    # Reflexive
    "himself": "ἑαυτοῦ", "herself": "ἑαυτοῦ", "themselves": "ἑαυτῶν",
    # Interrogative
    "who": "τίς", "what": "τί", "which": "τίς",
    # Relative
    "who": "ὅς", "whom": "ὅς", "whose": "ὅς", "which": "ὅς",
}

# Prepositions: map English prep + dependency context → Greek prep + case
_PREPOSITION_MAP = {
    "in": ("ἐν", "dat"),
    "into": ("εἰς", "acc"),
    "to": ("πρός", "acc"),
    "from": ("ἀπό", "gen"),
    "out of": ("ἐκ", "gen"),
    "with": ("μετά", "gen"),     # accompaniment
    "by": ("παρά", "dat"),       # agent or location
    "for": ("ὑπέρ", "gen"),      # on behalf of
    "through": ("διά", "gen"),
    "against": ("πρός", "acc"),
    "about": ("περί", "gen"),
    "upon": ("ἐπί", "gen"),
    "on": ("ἐπί", "dat"),
    "over": ("ὑπέρ", "acc"),
    "under": ("ὑπό", "dat"),
    "before": ("πρό", "gen"),
    "after": ("μετά", "acc"),
    "behind": ("ὄπισθεν", "gen"),
    "between": ("μεταξύ", "gen"),
    "across": ("διά", "gen"),
    "along": ("παρά", "acc"),
    "toward": ("πρός", "acc"),
    "beside": ("παρά", "dat"),
    "without": ("ἄνευ", "gen"),
    "among": ("ἐν", "dat"),
    "beyond": ("πέρα", "gen"),
    "until": ("μέχρι", "gen"),
    "above": ("ὑπέρ", "gen"),
    "below": ("ὑπό", "acc"),
    "around": ("περί", "acc"),
    "outside": ("ἔξω", "gen"),
    "inside": ("ἐντός", "gen"),
    "near": ("ἐγγύς", "gen"),
    "down": ("κατά", "gen"),
    "up": ("ἀνά", "acc"),
    "at": ("ἐν", "dat"),         # location
    "of": None,                  # → genitive case, not a preposition
}

# Conjunctions/particles: direct mapping
_CONJUNCTION_MAP = {
    "and": "καί",
    "but": "ἀλλά",
    "or": "ἤ",
    "nor": "μηδέ",
    "yet": "ὅμως",
    "so": "οὖν",
    "for": "γάρ",       # as conjunction (causal), not as preposition
    "because": "ὅτι",
    "although": "καίπερ",
    "if": "εἰ",
    "unless": "εἰ μή",
    "when": "ὅτε",
    "while": "ἕως",
    "since": "ἐπεί",
    "not": "οὐ",
    "never": "οὐδέποτε",
    "no": "οὐ",
    "now": "νῦν",
    "then": "τότε",
    "still": "ἔτι",
    "already": "ἤδη",
    "also": "καί",
    "even": "καί",
}

# Adverbs with direct Greek equivalents
_ADVERB_MAP = {
    "outside": "ἔξω",
    "inside": "ἔνδον",
    "here": "ἐνταῦθα",
    "there": "ἐκεῖ",
    "where": "ποῦ",
    "always": "ἀεί",
    "never": "οὐδέποτε",
    "again": "πάλιν",
    "away": "ἀπό",
    "back": "πάλιν",
    "down": "κάτω",
    "up": "ἄνω",
    "out": "ἔξω",
    "far": "πόρρω",
    "perhaps": "ἴσως",
    "only": "μόνον",
    "thus": "οὕτως",
    "very": "μάλα",
    "much": "πολύ",
    "well": "εὖ",
}


def _word_to_target(word, clause: ClauseSkeleton) -> GreekTarget | None:
    """Convert a stanza word to a Greek translation target."""
    # Skip punctuation
    if word.upos == "PUNCT":
        return None

    en = word.text.lower()
    lemma = (word.lemma or word.text).lower()

    # Skip articles (handled by assembler)
    if en in _SKIP_WORDS:
        return None

    # Determine POS
    pos_map = {
        "NOUN": "noun", "PROPN": "propn", "VERB": "verb", "AUX": "verb",
        "ADJ": "adj", "ADV": "adv", "ADP": "prep", "CCONJ": "conj",
        "SCONJ": "conj", "PRON": "pron", "NUM": "num", "PART": "particle",
    }
    pos = pos_map.get(word.upos, "other")

    # === PRONOUNS ===
    if pos == "pron":
        greek = _PRONOUN_MAP.get(en)
        if greek is None:
            # Subject pronoun → drop (Greek is pro-drop)
            return None
        target = GreekTarget(lemma=greek, pos="pron", english=word.text,
                             role=word.deprel)
        feats = _parse_feats(word.feats)
        target.number = {"Sing": "sg", "Plur": "pl"}.get(
            feats.get("Number", "Sing"), "sg")
        target.case = _role_to_case(word.deprel)
        # Possessives → genitive
        if word.deprel == "nmod:poss":
            target.case = "gen"
        return target

    # === PREPOSITIONS ===
    if pos == "prep":
        if en == "of":
            # "of" → genitive case on the dependent noun, not a Greek preposition
            return None
        prep_info = _PREPOSITION_MAP.get(en)
        if prep_info:
            greek_prep, governed_case = prep_info
            target = GreekTarget(lemma=greek_prep, pos="prep",
                                 english=word.text, role=word.deprel)
            target.case = governed_case  # case the prep governs
            return target
        # Unknown preposition — try Woodhouse
        greek = lookup_greek(en)
        return GreekTarget(lemma=greek or f"[{en}]", pos="prep",
                           english=word.text, role=word.deprel)

    # === CONJUNCTIONS ===
    if pos == "conj":
        # Distinguish "for" as conjunction (causal) vs preposition
        if en == "for" and word.deprel == "mark":
            greek = "γάρ"
        else:
            greek = _CONJUNCTION_MAP.get(en)
        if greek:
            return GreekTarget(lemma=greek, pos="conj",
                               english=word.text, role=word.deprel)
        return None  # skip unknown conjunctions

    # === ADVERBS ===
    if pos == "adv":
        greek = _ADVERB_MAP.get(en) or _CONJUNCTION_MAP.get(en)
        if greek:
            return GreekTarget(lemma=greek, pos="adv",
                               english=word.text, role=word.deprel)
        # Try Woodhouse for content adverbs
        greek = lookup_greek(lemma, pos_hint="adv")
        if greek:
            return GreekTarget(lemma=greek, pos="adv",
                               english=word.text, role=word.deprel)
        return None

    # === PARTICLES ===
    if pos == "particle":
        greek = _CONJUNCTION_MAP.get(en)
        if greek:
            return GreekTarget(lemma=greek, pos="particle",
                               english=word.text, role=word.deprel)
        return None

    # === PROPER NOUNS ===
    if pos == "propn":
        # Check locked glossary first, then transliterate
        greek = lookup_greek(word.text) or lookup_greek(lemma)
        if not greek:
            # Transliterate: keep as-is with asterisk marker
            greek = f"*{word.text}"
        target = GreekTarget(lemma=greek, pos="propn", english=word.text,
                             role=word.deprel)
        target.case = _role_to_case(word.deprel)
        feats = _parse_feats(word.feats)
        target.number = {"Sing": "sg", "Plur": "pl"}.get(
            feats.get("Number", "Sing"), "sg")
        return target

    # === CONTENT WORDS (nouns, verbs, adjectives) ===
    # POS-aware Woodhouse lookup
    greek = _lookup_pos_aware(lemma, word.text, pos)

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

    # Set noun/adj features
    elif pos in ("noun", "adj"):
        feats = _parse_feats(word.feats)
        target.number = {"Sing": "sg", "Plur": "pl"}.get(
            feats.get("Number", "Sing"), "sg")
        target.case = _role_to_case(word.deprel)

    return target


def _lookup_pos_aware(lemma: str, word_form: str, target_pos: str) -> str | None:
    """POS-aware vocabulary lookup via Woodhouse.

    Woodhouse entries have sense labels that indicate POS (v. trans., adj., subs.).
    When looking up 'pale', prefer the adjective entry over the verb entry.
    """
    _load_vocab()
    en = lemma.lower().strip()

    # 0. McCarthy-specific vocabulary
    if en in _MCCARTHY_VOCAB:
        return _MCCARTHY_VOCAB[en]

    # 1. Locked glossary (always takes priority)
    if en in _locked_glossary:
        return _locked_glossary[en]

    # 2. Woodhouse with POS awareness
    if en in _woodhouse:
        entry = _woodhouse[en]
        greek_words = entry.get("greek", [])
        senses = entry.get("senses", [])

        if not greek_words:
            pass
        elif target_pos == "verb":
            # Prefer entries from verb senses
            # Woodhouse verbs end in -ειν, -ειν, -αν, -ναι (infinitive)
            for gw in greek_words:
                if any(gw.endswith(s) for s in ("ειν", "εῖν", "αν", "ᾶν",
                                                 "ναι", "σθαι", "ειν")):
                    return gw
            return greek_words[0]
        elif target_pos in ("noun", "propn"):
            # Prefer entries that are NOT verbs (don't end in infinitive)
            for gw in greek_words:
                if not any(gw.endswith(s) for s in ("ειν", "εῖν", "αν", "ᾶν",
                                                     "ναι", "σθαι")):
                    return gw
            return greek_words[0]
        elif target_pos == "adj":
            # Prefer adjective forms — but distinguish from nouns in -ος
            # True adjectives: -ός/-ή/-όν (2nd decl), -ύς/-εῖα/-ύ (3rd decl),
            #                  -ης/-ες (3rd decl), -ων/-ον (3rd decl)
            # Use Woodhouse senses to help: if sense says "adj." prefer it
            adj_candidates = []
            for gw in greek_words:
                # Skip articles, verbs, obvious nouns
                if gw in ("τό", "τά", "ὁ", "ἡ", "τοῦ"):
                    continue
                if any(gw.endswith(s) for s in ("ειν", "εῖν", "ᾶν", "σθαι",
                                                 "ναι")):
                    continue
                # Prefer words ending in typical adj suffixes
                if any(gw.endswith(s) for s in ("ός", "ή", "όν", "ές",
                                                 "ύς", "εῖα", "ής",
                                                 "αῖος", "ινος", "ικός",
                                                 "ωδης", "ώδης")):
                    adj_candidates.append(gw)

            # Skip the first candidate if it matches a common noun pattern
            # (words like σκότος, θάνατος, λόγος are nouns not adjectives)
            # Before using Woodhouse's adjective candidates (which may be rare),
            # check if synonym lookup gives a more common word
            _ADJECTIVE_SYNONYMS_INNER = {
                "dark": ["gloomy", "murky", "dim"],
                "pale": ["pallid", "wan", "sallow"],
                "thin": ["lean", "slender", "spare"],
                "old": ["aged", "ancient", "elderly"],
                "big": ["large", "great"], "small": ["little", "tiny"],
                "fast": ["swift", "quick"], "slow": ["sluggish", "tardy"],
                "hot": ["warm", "burning"], "cold": ["chill", "frigid"],
                "wet": ["damp", "moist"], "dry": ["arid", "parched"],
                "hard": ["firm", "tough"], "soft": ["gentle", "tender"],
            }
            for syn in _ADJECTIVE_SYNONYMS_INNER.get(en, []):
                if syn in _woodhouse:
                    for gw in _woodhouse[syn].get("greek", []):
                        if any(gw.endswith(s) for s in ("ός", "ος", "ή", "ης",
                                                         "ύς", "υς", "ον")):
                            return gw

            if adj_candidates:
                return adj_candidates[0]
            # Don't return a verb/noun when we need an adjective — fall through
        else:
            return greek_words[0]

    # 3. If we need an adjective but Woodhouse only had verbs, try synonyms FIRST
    #    (before LSJ, which gives less natural primary translations)
    #    try English synonyms in Woodhouse
    if target_pos == "adj" and en in _woodhouse:
        # Woodhouse might have the adjective under a synonym
        _ADJECTIVE_SYNONYMS = {
            "pale": ["pallid", "wan", "sallow"],
            "thin": ["lean", "slender", "spare"],
            "dark": ["gloomy", "murky", "dim", "obscure"],
            "old": ["aged", "ancient", "elderly"],
            "big": ["large", "great"],
            "small": ["little", "tiny"],
            "fast": ["swift", "quick", "rapid"],
            "slow": ["sluggish", "tardy"],
            "hot": ["warm", "burning"],
            "cold": ["chill", "frigid"],
            "wet": ["damp", "moist"],
            "dry": ["arid", "parched"],
            "hard": ["firm", "tough"],
            "soft": ["gentle", "tender"],
        }
        for syn in _ADJECTIVE_SYNONYMS.get(en, []):
            if syn in _woodhouse:
                for gw in _woodhouse[syn].get("greek", []):
                    if any(gw.endswith(s) for s in ("ός", "ος", "ή", "ης",
                                                     "ύς", "υς", "ον")):
                        return gw

    # 4. WordNet synonym expansion — find a synonym that IS in Woodhouse
    try:
        from nltk.corpus import wordnet as wn
        for syn in wn.synsets(en):
            for lemma_name in syn.lemma_names():
                candidate = lemma_name.replace("_", " ").lower()
                if candidate == en:
                    continue
                if candidate in _woodhouse:
                    greek_words = _woodhouse[candidate].get("greek", [])
                    if target_pos == "adj":
                        for gw in greek_words:
                            if any(gw.endswith(s) for s in ("ός", "ος", "ή",
                                                             "ης", "ύς", "ον")):
                                return gw
                    elif target_pos == "verb":
                        for gw in greek_words:
                            if any(gw.endswith(s) for s in ("ειν", "εῖν", "αν",
                                                             "ᾶν", "ναι", "σθαι")):
                                return gw
                    elif greek_words:
                        # For nouns, take the first non-verb entry
                        for gw in greek_words:
                            if not any(gw.endswith(s) for s in ("ειν", "εῖν",
                                                                 "ᾶν", "σθαι")):
                                return gw
                        return greek_words[0]
    except Exception:
        pass

    # 5. Inverted LSJ fallback
    if en in _lsj_en:
        entries = _lsj_en[en]
        if entries:
            return entries[0]["lemma"]

    # 6. Try the word form itself (not just lemma)
    word_lower = word_form.lower().strip()
    if word_lower != en:
        result = _lookup_pos_aware(word_lower, word_lower, target_pos)
        if result:
            return result

    return None


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
        "nmod": "gen",      # "of X" → genitive
        "nmod:poss": "gen", # possessive → genitive
        "vocative": "voc",
        "appos": "nom",     # apposition matches head case
        "conj": "",         # conjunction — inherits case from head
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
