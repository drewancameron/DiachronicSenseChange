#!/usr/bin/env python3
"""
Mechanical glosser: fully automated Ørberg-style marginal glosses.

Pipeline:
1. IDF → identify rare surface forms (threshold-based)
2. Morpheus API → lemma + morphological parsing
3. LSJ (local JSON) → English short definition + antonyms
4. Wiktionary API → fallback for LSJ gaps
5. Formatting → choose Ørberg notation type based on word properties

No LLM calls. Uses the English source text for contextual disambiguation.

Usage:
  python3 scripts/mechanical_glosser.py 001_see_the_child_he
  python3 scripts/mechanical_glosser.py --all
  python3 scripts/mechanical_glosser.py --all --dry-run
  python3 scripts/mechanical_glosser.py --all --threshold 8.5
"""

import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DRAFTS = ROOT / "drafts"
PASSAGES = ROOT / "passages"
APPARATUS = ROOT / "apparatus"
GLOSSARY = ROOT / "glossary"

sys.path.insert(0, str(SCRIPTS))

DEFAULT_THRESHOLD = 9.0

# ====================================================================
# Semantic similarity for sense disambiguation
# ====================================================================

_embedder = None


def _get_embedder():
    """Lazy-load sentence-transformers model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _find_english_sentence(greek_word: str, greek_text: str, en_text: str) -> str:
    """Find the English sentence that most likely corresponds to a Greek word.

    Uses position-based alignment: if the Greek word appears at position 40%
    through the Greek text, return the English sentence at roughly 40% through
    the English text.
    """
    # Find position of word in Greek text (as fraction)
    pos = greek_text.find(greek_word)
    if pos < 0:
        return en_text  # fallback to full text
    frac = pos / max(len(greek_text), 1)

    # Split English into sentences
    en_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]
    if not en_sentences:
        return en_text

    # Find the sentence at the corresponding position
    target_idx = min(int(frac * len(en_sentences)), len(en_sentences) - 1)

    # Return a window of 1-3 sentences around the target
    start = max(0, target_idx - 1)
    end = min(len(en_sentences), target_idx + 2)
    return " ".join(en_sentences[start:end])


def _ranked_best_sense(definitions: list[str], en_context: str) -> str:
    """Pick the best definition combining primary-sense position with semantic similarity.

    Primary senses (first 3 in LSJ) get a position bonus. Semantic similarity
    acts as a tiebreaker and can promote a later sense only if it's much more
    relevant to the context.
    """
    if not definitions:
        return ""
    if len(definitions) == 1:
        return definitions[0]

    SKIP = {"repeated", "habitual", "habitually", "in motion", "conveyed",
            "carrying about", "served as a messenger"}

    # Filter
    valid = [(i, d) for i, d in enumerate(definitions)
             if d.lower().strip() not in SKIP and len(d) > 1]
    if not valid:
        return definitions[0]

    # Compute semantic similarity for each candidate
    model = _get_embedder()
    texts = [en_context] + [d for _, d in valid]
    embeddings = model.encode(texts, convert_to_numpy=True)

    from numpy import dot
    from numpy.linalg import norm

    context_emb = embeddings[0]
    scores = []
    for j, (orig_idx, defn) in enumerate(valid):
        def_emb = embeddings[j + 1]
        sim = float(dot(context_emb, def_emb) / (norm(context_emb) * norm(def_emb) + 1e-8))

        # Position bonus: mild preference for primary senses
        position_bonus = max(0, (3 - orig_idx) * 0.05)

        combined = sim + position_bonus
        scores.append((combined, defn))

    scores.sort(key=lambda x: -x[0])
    return scores[0][1]


def semantic_best_sense(definitions: list[str], en_context: str) -> str:
    """Pick the definition most semantically similar to the English context.

    Uses sentence-transformers embeddings + cosine similarity.
    """
    if not definitions:
        return ""
    if len(definitions) == 1:
        return definitions[0]

    model = _get_embedder()

    # Encode context and all definitions
    texts = [en_context] + definitions
    embeddings = model.encode(texts, convert_to_numpy=True)

    context_emb = embeddings[0]
    def_embs = embeddings[1:]

    # Cosine similarity
    from numpy import dot
    from numpy.linalg import norm

    best_idx = 0
    best_sim = -1
    for i, de in enumerate(def_embs):
        sim = dot(context_emb, de) / (norm(context_emb) * norm(de) + 1e-8)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    return definitions[best_idx]

# ====================================================================
# LSJ dictionary
# ====================================================================

_lsj: dict | None = None
_lsj_multi: dict | None = None
_lsj_stripped: dict | None = None
_lsj_antonyms: dict | None = None


def _load_lsj():
    global _lsj, _lsj_multi, _lsj_stripped, _lsj_antonyms
    if _lsj is not None:
        return

    lsj_path = GLOSSARY / "lsj_short.json"
    multi_path = GLOSSARY / "lsj_multi.json"
    stripped_path = GLOSSARY / "lsj_stripped_index.json"
    antonyms_path = GLOSSARY / "lsj_antonyms.json"

    if not lsj_path.exists():
        print("ERROR: LSJ data not found. Run the LSJ parser first.")
        sys.exit(1)

    with open(lsj_path) as f:
        _lsj = json.load(f)
    with open(stripped_path) as f:
        _lsj_stripped = json.load(f)
    _lsj_multi = {}
    if multi_path.exists():
        with open(multi_path) as f:
            _lsj_multi = json.load(f)
    _lsj_antonyms = {}
    if antonyms_path.exists():
        with open(antonyms_path) as f:
            _lsj_antonyms = json.load(f)

    print(f"  LSJ: {len(_lsj):,} entries, {len(_lsj_multi):,} multi-def, "
          f"{len(_lsj_antonyms):,} antonyms")


def _resolve_lsj_key(word: str) -> str | None:
    """Find the LSJ key for a word, handling normalization."""
    _load_lsj()
    w = unicodedata.normalize("NFC", word)
    if w in _lsj:
        return w
    nfd = unicodedata.normalize("NFD", w)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()
    if stripped in _lsj_stripped:
        return _lsj_stripped[stripped]
    return None


def lsj_lookup(word: str, en_context: str = "") -> str | None:
    """Look up a word in LSJ. If en_context is given, pick the best sense."""
    key = _resolve_lsj_key(word)
    if not key:
        return None

    # If we have multi-definitions and context, pick the best one
    if en_context and key in _lsj_multi:
        defs = _lsj_multi[key]
        if len(defs) > 1:
            return _pick_best_sense(defs, en_context)
        return defs[0] if defs else None

    return _lsj.get(key)


def _pick_best_sense(definitions: list[str], en_context: str) -> str:
    """Pick the definition that best matches the English context.

    Uses keyword overlap with crude stemming (first 3-4 chars).
    """
    STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at",
            "is", "it", "he", "she", "his", "her", "by", "for", "with",
            "as", "was", "not", "but", "be", "had", "have", "from", "that",
            "this", "they", "them", "no", "so", "if", "up", "out", "into",
            "one", "man", "men", "set", "run", "get", "got", "put", "let",
            "see", "saw", "may", "can", "old", "new", "long", "last", "first",
            "come", "came", "take", "took", "make", "made", "give", "gave",
            "go", "went", "way", "back", "just", "over", "down", "after",
            "said", "say", "like", "well", "still", "also", "then", "now",
            "about", "upon", "been", "being", "some", "much", "only",
            "through", "even", "most", "other", "before", "more", "such"}

    def _en_stem(word: str) -> str:
        w = word.lower().rstrip(".,;:!?'\"")
        for suffix in ["ing", "tion", "ness", "ment", "ous", "ive",
                        "ed", "er", "est", "ly", "es", "s"]:
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[:-len(suffix)]
        return w

    context_lower = en_context.lower()
    context_words = {w for w in context_lower.split() if w not in STOP and len(w) > 2}
    context_stemmed = {_en_stem(w) for w in context_words}

    # Skip trivially unhelpful definitions
    SKIP = {"repeated", "habitual", "habitually", "in motion", "conveyed",
            "carrying about", "served as a messenger"}

    best_def = None
    best_score = -1

    for defn in definitions:
        if defn.lower().strip() in SKIP:
            continue
        if len(defn) < 2:
            continue

        def_words = {w for w in re.sub(r"[,;.()]", "", defn.lower()).split()
                     if w not in STOP and len(w) > 2}
        def_stemmed = {_en_stem(w) for w in def_words}

        # Score: exact word matches (×5) + stemmed matches (×3)
        score = len(context_words & def_words) * 5
        score += len(context_stemmed & def_stemmed) * 3

        # Bonus for informative definitions (not too short, not too long)
        if 5 < len(defn) < 40:
            score += 1
        # Penalize very short generic definitions (likely fragments)
        if len(defn.split()) <= 1 and score == 0:
            score -= 2
        # Strong bonus for primary senses (first 3 in LSJ are most important)
        idx = definitions.index(defn)
        if idx < 3:
            score += (3 - idx) * 2  # +6, +4, +2 for first three
        else:
            score -= idx * 0.2

        if score > best_score:
            best_score = score
            best_def = defn

    # Fallback: first non-skipped definition
    if best_def is None:
        for defn in definitions:
            if defn.lower().strip() not in SKIP and len(defn) > 1:
                return defn
        return definitions[0]

    return best_def


def _match_sentence_context(lemma: str, word: str, en_text: str) -> str | None:
    """Try to find the best definition by matching against individual English sentences.

    For each English sentence, check if it plausibly corresponds to the Greek word
    (via keyword overlap with the definition candidates), then use that narrow
    context for sense selection.
    """
    key = _resolve_lsj_key(lemma) or _resolve_lsj_key(word)
    if not key or key not in _lsj_multi:
        return None
    defs = _lsj_multi[key]
    if len(defs) <= 1:
        return None

    def en_stem(word: str) -> str:
        """Crude English stemmer — strip common suffixes."""
        w = word.lower().rstrip(".,;:!?'\"")
        for suffix in ["ing", "tion", "ness", "ment", "ous", "ive",
                        "ed", "er", "est", "ly", "es", "s"]:
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[:-len(suffix)]
        return w

    # Split English into sentences
    en_sentences = re.split(r'[.!?]+', en_text)

    # For each definition, find the sentence that best matches it
    # Then score the definition against just that sentence
    best_def = None
    best_score = 0

    STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at",
            "is", "it", "he", "she", "his", "her", "by", "for", "with",
            "as", "was", "not", "but", "be", "had", "have", "from", "that",
            "this", "they", "them", "no", "so", "if", "up", "out", "into",
            # High-frequency English words that cause false positive matches
            "one", "man", "men", "set", "run", "get", "got", "put", "let",
            "see", "saw", "may", "can", "old", "new", "long", "last", "first",
            "come", "came", "take", "took", "make", "made", "give", "gave",
            "go", "went", "way", "back", "just", "over", "down", "after",
            "said", "say", "like", "well", "still", "also", "then", "now",
            "about", "upon", "been", "being", "some", "much", "only",
            "through", "even", "most", "other", "before", "more", "such"}
    SKIP_DEFS = {"repeated", "habitual", "habitually", "in motion"}

    for defn in defs:
        if defn.lower().strip() in SKIP_DEFS or len(defn) < 2:
            continue
        def_raw = {w for w in re.sub(r"[,;.()]", "", defn.lower()).split()
                   if w not in STOP and len(w) > 2}
        def_stemmed = {en_stem(w) for w in def_raw}

        for sent in en_sentences:
            sent_raw = {w for w in sent.lower().split()
                       if w not in STOP and len(w) > 2}
            sent_stemmed = {en_stem(w) for w in sent_raw}

            # Exact word overlap (high confidence)
            exact = def_raw & sent_raw
            # Stemmed overlap (catches wear/wears, lie/lies, etc.)
            stemmed = def_stemmed & sent_stemmed
            score = len(exact) * 5 + len(stemmed) * 3

            if score > best_score:
                best_score = score
                best_def = defn

    # Return if we found any meaningful match
    if best_score >= 3:
        return best_def
    return None


def lsj_antonym(lemma: str) -> str | None:
    """Look up antonym from LSJ 'opp.' cross-references."""
    _load_lsj()
    w = unicodedata.normalize("NFC", lemma)
    if w in _lsj_antonyms:
        return _lsj_antonyms[w]
    nfd = unicodedata.normalize("NFD", w)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()
    if stripped in _lsj_stripped:
        real_lemma = _lsj_stripped[stripped]
        return _lsj_antonyms.get(real_lemma)
    return None


# ====================================================================
# Wiktionary fallback
# ====================================================================

_wiki_cache_path = GLOSSARY / "wiktionary_cache.json"
_wiki_cache: dict | None = None


def _load_wiki_cache():
    global _wiki_cache
    if _wiki_cache is not None:
        return _wiki_cache
    if _wiki_cache_path.exists():
        with open(_wiki_cache_path) as f:
            _wiki_cache = json.load(f)
    else:
        _wiki_cache = {}
    return _wiki_cache


def _save_wiki_cache():
    if _wiki_cache is not None:
        _wiki_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_wiki_cache_path, "w") as f:
            json.dump(_wiki_cache, f, ensure_ascii=False)


def wiktionary_lookup(lemma: str) -> str | None:
    """Look up a lemma on Wiktionary, returning the first English definition."""
    cache = _load_wiki_cache()
    if lemma in cache:
        return cache[lemma] if cache[lemma] else None

    encoded = urllib.parse.quote(lemma)
    url = f"https://en.wiktionary.org/w/api.php?action=parse&page={encoded}&prop=wikitext&format=json"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "MechanicalGlosser/1.0")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")

        # Find Ancient Greek section
        if "Ancient Greek" not in wikitext:
            cache[lemma] = None
            return None

        # Extract definitions: lines starting with # that contain English
        defn = None
        in_greek = False
        for line in wikitext.split("\n"):
            if "==Ancient Greek==" in line:
                in_greek = True
                continue
            if in_greek and line.startswith("==") and "Ancient Greek" not in line:
                break
            if in_greek and line.startswith("# "):
                # Clean wikitext markup
                clean = re.sub(r"\{\{[^}]*\}\}", "", line[2:])
                clean = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", clean)
                clean = clean.strip().strip(".")
                if clean and len(clean) > 1:
                    defn = clean
                    break

        cache[lemma] = defn
        time.sleep(0.2)  # rate limit
        return defn

    except Exception:
        cache[lemma] = None
        return None


# ====================================================================
# Morphological analysis
# ====================================================================

def get_morpheus(word: str, en_context: str = "") -> dict | None:
    """Get Morpheus parse for a word. Returns best analysis or None.

    When multiple lemmas are returned, uses English context + IDF + LSJ
    definition count to pick the most likely one.
    """
    from morpheus_check import parse_word
    results = parse_word(word)
    if not results:
        return None

    # Filter to non-error results with lemmas
    valid = [r for r in results if "error" not in r and r.get("lemma")]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    # Deduplicate by lemma
    seen = {}
    for r in valid:
        lemma = r.get("lemma", "")
        if lemma not in seen:
            seen[lemma] = r
    valid = list(seen.values())
    if len(valid) == 1:
        return valid[0]

    # Multiple lemmas — score each
    _load_lsj()

    def en_stem(w):
        w = w.lower().rstrip(".,;:!?'\"")
        for suffix in ["ing", "tion", "ness", "ment", "ous", "ive",
                        "ed", "er", "est", "ly", "es", "s"]:
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[:-len(suffix)]
        return w

    STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at",
            "is", "it", "he", "she", "his", "her", "by", "for", "with",
            "as", "was", "not", "but", "be", "had", "have", "from", "that",
            "this", "they", "them", "no", "so", "if", "up", "out", "into"}

    context_stemmed = set()
    if en_context:
        context_stemmed = {en_stem(w) for w in en_context.lower().split()
                          if w not in STOP and len(w) > 2}

    def lemma_score(r):
        lemma = unicodedata.normalize("NFC", r.get("lemma", ""))
        score = 0.0

        # 1. English context match via semantic similarity
        key = _resolve_lsj_key(lemma)
        if key and key in _lsj_multi and en_context:
            defs = _lsj_multi[key]
            if defs:
                best_def = semantic_best_sense(defs, en_context)
                # Use the similarity as a score bonus
                model = _get_embedder()
                embs = model.encode([en_context, best_def], convert_to_numpy=True)
                from numpy import dot
                from numpy.linalg import norm
                sim = dot(embs[0], embs[1]) / (norm(embs[0]) * norm(embs[1]) + 1e-8)
                score -= float(sim) * 10  # higher similarity = lower score = better

        # 2. Definition count (more = more common)
        if key and key in _lsj_multi:
            score -= min(len(_lsj_multi[key]), 10)

        # 3. Prefer nouns over adjectives (nouns are more often the gloss target)
        pofs = r.get("pofs", "")
        if pofs == "noun":
            score -= 2
        elif pofs == "verb":
            score -= 1  # slight preference over adjectives

        # 4. IDF if available
        try:
            from auto_gloss import load_idf
            lemma_idf, _ = load_idf()
            idf = lemma_idf.get(lemma)
            if idf is not None:
                score += idf  # lower IDF = more common
        except Exception:
            pass

        return score

    valid.sort(key=lemma_score)
    return valid[0]


# ====================================================================
# Compound detection
# ====================================================================

# Common Greek combining forms (first elements of compounds)
COMPOUND_PREFIXES = [
    "ξυλο", "ὑδρο", "λιθο", "ἱππο", "ἀνδρο", "γυναικο", "παιδο",
    "οἰνο", "σιδηρο", "χρυσο", "ἀργυρο", "πυρο", "αἱμο", "νεκρο",
    "ὀστεο", "δερμο", "πατρο", "μητρο", "αὐτο", "παντο", "πολυ",
    "μονο", "κακο", "ἀγαθο", "μεγαλο", "μικρο", "νεο", "παλαιο",
    "ὁμο", "ἑτερο", "ἰσο", "ὀρθο", "στρατο", "ναυ",
]

# Common second elements
COMPOUND_SUFFIXES = [
    "κόπος", "φόρος", "λόγος", "γράφος", "φάγος", "πότης",
    "μαχος", "κτόνος", "γενής", "ειδής", "ώδης",
    "πωλεῖον", "δοκεῖον", "τήριον",
]


def detect_compound(lemma: str) -> tuple[str, str] | None:
    """Try to split a compound word. Returns (first, second) or None."""
    lemma_lower = lemma.lower()
    for prefix in COMPOUND_PREFIXES:
        if lemma_lower.startswith(prefix) and len(lemma_lower) > len(prefix) + 2:
            rest = lemma[len(prefix):]
            return lemma[:len(prefix)], rest
    return None


# ====================================================================
# Derivation detection
# ====================================================================

DERIVATION_SUFFIXES = [
    ("ώδης", "full of, resembling"),
    ("ικός", "pertaining to"),
    ("ινος", "made of"),
    ("ειος", "belonging to"),
    ("αῖος", "of or from"),
    ("ιστής", "one who"),
    ("εύς", "one who"),
    ("τήρ", "agent"),
    ("τής", "agent"),
    ("σις", "act of"),
    ("μα", "result of"),
    ("τρον", "instrument"),
]


def detect_derivation(lemma: str) -> tuple[str, str] | None:
    """Detect if a word is derived from a simpler root via a known suffix.
    Returns (root_hint, suffix_meaning) or None."""
    for suffix, meaning in DERIVATION_SUFFIXES:
        if lemma.endswith(suffix) and len(lemma) > len(suffix) + 2:
            return suffix, meaning
    return None


# ====================================================================
# Ørberg gloss formatting
# ====================================================================

# Parsing abbreviations (Greek)
PARSING_ABBREVS = {
    "present": "ἐνεστ.",
    "imperfect": "πρτ.",
    "future": "μέλλ.",
    "aorist": "ἀόρ.",
    "perfect": "παρακ.",
    "pluperfect": "ὑπερσ.",
    "indicative": "",  # default, omit
    "subjunctive": "ὑποτ.",
    "optative": "εὐκτ.",
    "imperative": "προστ.",
    "infinitive": "ἀπρ.",
    "participle": "μτχ.",
    "active": "ἐν.",
    "mediopassive": "μέσ.",
    "middle": "μέσ.",
    "passive": "παθ.",
    "singular": "ἑν.",
    "plural": "πλ.",
    "dual": "δυϊκ.",
    "masculine": "ἀρσ.",
    "feminine": "θηλ.",
    "neuter": "οὐδ.",
    "nominative": "ὀν.",
    "genitive": "γεν.",
    "dative": "δοτ.",
    "accusative": "αἰτ.",
    "vocative": "κλ.",
    "1st": "α´",
    "2nd": "β´",
    "3rd": "γ´",
}


def format_parsing(morph: dict) -> str:
    """Format morphological parsing as Greek abbreviations."""
    parts = []
    # For verbs: tense, mood, voice
    if morph.get("pofs") == "verb":
        for key in ["tense", "mood", "voice"]:
            val = morph.get(key, "")
            if val and val in PARSING_ABBREVS and PARSING_ABBREVS[val]:
                parts.append(PARSING_ABBREVS[val])
    # For nouns/adjectives: unusual case/number
    elif morph.get("pofs") in ("noun", "adjective"):
        # Only add parsing if the form is notably different from lemma
        case = morph.get("case", "")
        number = morph.get("number", "")
        if case and case != "nominative":
            parts.append(PARSING_ABBREVS.get(case, ""))
        if number and number != "singular":
            parts.append(PARSING_ABBREVS.get(number, ""))

    return " ".join(p for p in parts if p)


def is_proper_noun(word: str) -> bool:
    """Check if word is likely a proper noun (starts with uppercase)."""
    return bool(word) and word[0].isupper()


def format_gloss(word: str, morph: dict | None, definition: str | None,
                 antonym: str | None, en_context: str = "") -> str | None:
    """Format a single Ørberg-style gloss entry.

    Chooses the best notation type based on available information.
    Returns the formatted gloss string, or None if we can't gloss this word.
    """
    if not morph and not definition:
        return None

    lemma = morph.get("lemma", word) if morph else word
    pofs = morph.get("pofs", "") if morph else ""

    parts = []

    # 1. Compound decomposition
    compound = detect_compound(lemma)
    if compound:
        first, second = compound
        parts.append(f"{first}·{second}")

    # 2. Derivation from root
    deriv = detect_derivation(lemma)
    if deriv and not compound:
        suffix, meaning = deriv
        parts.append(f"< {lemma}")

    # 3. Show lemma if form differs significantly from the surface word
    #    (skip for simple inflections where the stem is recognizable)
    if morph and lemma:
        # Check if the form shares at least 4 chars with the lemma
        shared = min(len(word), len(lemma))
        common_prefix = 0
        for i in range(min(shared, 10)):
            w_nfd = unicodedata.normalize("NFD", word.lower())
            l_nfd = unicodedata.normalize("NFD", lemma.lower())
            if i < len(w_nfd) and i < len(l_nfd) and w_nfd[i] == l_nfd[i]:
                common_prefix += 1
            else:
                break

        form_differs = common_prefix < 3
        is_participle = morph.get("mood") == "participle"
        is_unusual_tense = morph.get("tense") in ("perfect", "pluperfect", "aorist")

        if form_differs or is_participle or is_unusual_tense:
            parsing = format_parsing(morph)
            if parsing and not compound and not deriv:
                parts.append(f"< {lemma} ({parsing})")
            elif form_differs and not compound and not deriv:
                parts.append(f"< {lemma}")

    # 4. Definition — clean up noise
    if definition:
        # Strip leading articles/prepositions
        clean_def = re.sub(r"^(the|a|an|to|of)\s+", "", definition.strip(), flags=re.I)
        # Strip trailing fragments (ending with "of a", "with the", etc.)
        clean_def = re.sub(r"\s+(of|with|for|from|in|on|at|to)(\s+(a|an|the|one|its|his|her))?\s*$",
                           "", clean_def)
        # Strip trailing ellipsis dots and spaces
        clean_def = clean_def.rstrip(". ")
        # Skip if too short after cleaning (fragments)
        if clean_def and len(clean_def) > 1:
            parts.append(f"= {clean_def}")

    # 5. Antonym
    if antonym:
        # Validate antonym is a real Greek word (not citation garbage)
        ant_clean = antonym.strip()
        if (len(ant_clean) > 1 and len(ant_clean) < 30 and
                any("GREEK" in unicodedata.name(c, "") for c in ant_clean[:3]
                    if c.isalpha())):
            parts.append(f"↔ {ant_clean}")

    if not parts:
        return None

    return " ".join(parts)


# ====================================================================
# IDF word selection
# ====================================================================

def load_idf():
    """Load IDF scores."""
    from auto_gloss import load_idf as _load
    return _load()


def get_words_to_gloss(passage_id: str, lemma_idf: dict, form_idf: dict,
                       threshold: float) -> list[dict]:
    """Get words to gloss based on IDF threshold."""
    from auto_gloss import propose_glosses
    return propose_glosses(passage_id, lemma_idf, form_idf, threshold,
                           threshold + 2)


# ====================================================================
# Main glossing pipeline
# ====================================================================

def gloss_passage(passage_id: str, lemma_idf: dict, form_idf: dict,
                  threshold: float, dry_run: bool = False) -> bool:
    """Generate mechanical glosses for a passage.

    Returns True if glosses were written.
    """
    primary_path = DRAFTS / passage_id / "primary.txt"
    if not primary_path.exists():
        print(f"  {passage_id}: no primary.txt")
        return False

    # Check if primary.txt is Greek
    text = primary_path.read_text("utf-8").strip()
    if not text or "GREEK" not in unicodedata.name(text[0], ""):
        print(f"  {passage_id}: not Greek text")
        return False

    # Load English source for context
    p_path = PASSAGES / f"{passage_id}.json"
    en_text = ""
    if p_path.exists():
        en_text = json.load(open(p_path)).get("text", "")

    # Get words to gloss via IDF
    words = get_words_to_gloss(passage_id, lemma_idf, form_idf, threshold)
    if not words:
        print(f"  {passage_id}: no words need glossing")
        return False

    # Filter out proper nouns
    words = [w for w in words if not is_proper_noun(w["word"])]

    print(f"  {passage_id}: {len(words)} words to gloss")

    if dry_run:
        for w in words:
            print(f"    {w['word']:>25}  IDF={w['idf']:.1f}")
        return False

    # Split Greek into sentences
    greek = primary_path.read_text("utf-8").strip()
    sentences = [s.strip() for s in re.split(r'(?<=[.;·!])\s+', greek)
                 if s.strip()]

    # Process each word: Morpheus → LSJ → format
    gloss_lookup = {}
    for w in words:
        word = w["word"]

        # Morpheus — pass English context for lemma disambiguation
        morph = get_morpheus(word, en_text)
        lemma = morph.get("lemma", word) if morph else word

        # LSJ definition — combine primary-sense bias with semantic similarity
        # Use sentence-level English context for better disambiguation
        definition = None
        key = _resolve_lsj_key(lemma) or _resolve_lsj_key(word)
        if key and key in _lsj_multi and en_text:
            defs = _lsj_multi[key]
            if len(defs) > 1:
                # Find the best matching English sentence for this word
                en_sentence = _find_english_sentence(word, greek, en_text)
                definition = _ranked_best_sense(defs, en_sentence)
            elif defs:
                definition = defs[0]
        if not definition:
            definition = lsj_lookup(word) or lsj_lookup(lemma)

        # For compounds without a definition, try the second element
        if not definition:
            comp = detect_compound(lemma)
            if comp:
                _, second = comp
                definition = lsj_lookup(second, en_text)
                if definition:
                    definition = f"({second}) {definition}"

        # Wiktionary fallback
        if not definition:
            definition = wiktionary_lookup(lemma)

        # Antonym (from LSJ)
        antonym = lsj_antonym(lemma)

        # Format the gloss
        gloss = format_gloss(word, morph, definition, antonym, en_text)

        if gloss:
            gloss_lookup[word] = gloss

    print(f"    {len(gloss_lookup)} glosses generated")

    # Build marginal_glosses.json structure
    mg_sentences = []
    for i, sent in enumerate(sentences):
        sent_glosses = []
        for w in words:
            word = w["word"]
            if word in sent and word in gloss_lookup:
                sent_glosses.append({
                    "anchor": word,
                    "note": gloss_lookup[word],
                    "rank": 1,
                })
        mg_sentences.append({
            "index": i,
            "greek": sent,
            "glosses": sent_glosses,
        })

    # Write output
    out_dir = APPARATUS / passage_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "marginal_glosses.json"
    with open(out_path, "w") as f:
        json.dump({"sentences": mg_sentences}, f, ensure_ascii=False, indent=2)

    total_glosses = sum(len(s["glosses"]) for s in mg_sentences)
    print(f"    Wrote {out_path} ({total_glosses} glosses)")
    return True


# ====================================================================
# CLI
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Mechanical Ørberg-style glosser (no LLM)")
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )
    elif args.passages:
        passage_ids = args.passages
    else:
        parser.print_help()
        return

    _load_lsj()
    lemma_idf, form_idf = load_idf()
    print(f"IDF: {len(lemma_idf):,} lemmas, threshold={args.threshold}")
    print()

    t0 = time.time()
    glossed = 0
    for pid in passage_ids:
        if gloss_passage(pid, lemma_idf, form_idf, args.threshold,
                         args.dry_run):
            glossed += 1
        print()

    _save_wiki_cache()
    from morpheus_check import _save_cache
    _save_cache()

    print(f"Done in {time.time()-t0:.0f}s — {glossed} passages glossed")


if __name__ == "__main__":
    main()
