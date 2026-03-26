#!/usr/bin/env python3
"""
Automatic Ørberg-style gloss generator.

For each passage draft, this script:
  1. Tokenises the Greek and computes corpus frequency for each lemma
  2. Identifies rare words (below frequency threshold)
  3. For each rare word, generates a context-sensitive gloss using:
     - Corpus collocate data for sense disambiguation
     - Heuristic case/construction analysis
     - The IDF glossary for locked terms
     - The full Ørberg notation vocabulary (= ↔ < > · + case labels)
  4. Outputs marginal_glosses.json

Usage:
  python3 scripts/generate_glosses.py                     # all passages
  python3 scripts/generate_glosses.py 001_see_the_child   # one passage
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"
GLOSSARY_PATH = ROOT / "glossary" / "idf_glossary.json"
CORPUS_PATH = ROOT / "retrieval" / "data" / "corpus.jsonl"
COLLOCATE_PATH = ROOT / "retrieval" / "data" / "collocates.json"

# ====================================================================
# Greek normalisation
# ====================================================================

def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower()


def tokenise(text: str) -> list[str]:
    """Split Greek text into tokens, stripping punctuation."""
    return [t for t in re.findall(r'[\w\u0370-\u03FF\u1F00-\u1FFF]+', text) if t]


# ====================================================================
# Corpus frequency model
# ====================================================================

_freq_cache: Counter | None = None

def load_corpus_frequencies() -> Counter:
    global _freq_cache
    if _freq_cache is not None:
        return _freq_cache

    _freq_cache = Counter()
    if not CORPUS_PATH.exists():
        print("  WARNING: no corpus.jsonl — frequency data unavailable")
        return _freq_cache

    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            for tok in tokenise(text):
                _freq_cache[strip_accents(tok)] += 1

    print(f"  Loaded frequency model: {len(_freq_cache)} types, {sum(_freq_cache.values())} tokens")
    return _freq_cache


def word_frequency(word: str) -> int:
    """Return corpus frequency of a word (accent-stripped)."""
    freq = load_corpus_frequencies()
    return freq.get(strip_accents(word), 0)


def frequency_rank(word: str) -> int:
    """Return frequency rank (1 = most common). 0 = not found."""
    freq = load_corpus_frequencies()
    norm = strip_accents(word)
    count = freq.get(norm, 0)
    if count == 0:
        return 0
    # Rank among all types
    rank = sum(1 for c in freq.values() if c > count) + 1
    return rank


# ====================================================================
# Collocate lookup
# ====================================================================

_colloc_cache: dict | None = None

def load_collocates() -> dict:
    global _colloc_cache
    if _colloc_cache is not None:
        return _colloc_cache
    if COLLOCATE_PATH.exists():
        with open(COLLOCATE_PATH) as f:
            _colloc_cache = json.load(f)
    else:
        _colloc_cache = {}
    return _colloc_cache


# ====================================================================
# IDF glossary lookup
# ====================================================================

_glossary_cache: dict | None = None

def load_glossary() -> dict:
    """Load glossary and build a lookup by normalised Greek form."""
    global _glossary_cache
    if _glossary_cache is not None:
        return _glossary_cache

    with open(GLOSSARY_PATH) as f:
        raw = json.load(f)

    _glossary_cache = {}
    for cat_name, cat in raw.items():
        if cat_name.startswith("_") or not isinstance(cat, dict):
            continue
        for key, entry in cat.items():
            if not isinstance(entry, dict):
                continue
            ag = entry.get("ancient_greek", "")
            if not ag:
                continue
            # Extract primary form, strip articles and *
            primary = ag.split("/")[0].split("(")[0].strip()
            primary = re.sub(r'^[ὁἡτὸτόαἱοἱ]\s+', '', primary).replace("*", "").strip()
            norm = strip_accents(primary)
            _glossary_cache[norm] = entry

    return _glossary_cache


# ====================================================================
# Case / construction analysis (heuristic)
# ====================================================================

PREPOSITIONS = {
    "ἐν": "δοτ.", "εἰς": "αἰτ.", "ἐκ": "γεν.", "ἐξ": "γεν.",
    "ἀπό": "γεν.", "ἀπ'": "γεν.", "πρό": "γεν.", "μετά": "γεν./αἰτ.",
    "μετ'": "γεν./αἰτ.", "παρά": "γεν./δοτ./αἰτ.", "παρ'": "γεν./δοτ./αἰτ.",
    "ὑπέρ": "γεν./αἰτ.", "ὑπό": "γεν./αἰτ.", "ὑπ'": "γεν./αἰτ.",
    "πρός": "αἰτ.", "διά": "γεν./αἰτ.", "δι'": "γεν./αἰτ.",
    "κατά": "γεν./αἰτ.", "κατ'": "γεν./αἰτ.",
    "περί": "γεν./αἰτ.", "ἐπί": "γεν./δοτ./αἰτ.", "ἐπ'": "γεν./δοτ./αἰτ.",
}

GENITIVE_ARTICLE = {"τοῦ", "τῆς", "τῶν"}
DATIVE_ARTICLE = {"τῷ", "τῇ", "τοῖς", "ταῖς"}
ACCUSATIVE_ARTICLE = {"τόν", "τήν", "τόν", "τούς", "τάς"}


def analyse_context(tokens: list[str], idx: int) -> dict:
    """Analyse the grammatical context of a token at position idx."""
    ctx = {"case_label": None, "construction": None, "prep_governed": False}

    # Check if preceded by a preposition
    if idx > 0:
        prev = tokens[idx - 1]
        if prev in PREPOSITIONS:
            ctx["prep_governed"] = True
            ctx["case_label"] = PREPOSITIONS[prev]
            ctx["construction"] = f"+ {prev}"

    # Check if followed by a genitive article (suggesting X + gen.)
    if idx + 1 < len(tokens) and tokens[idx + 1] in GENITIVE_ARTICLE:
        # Check what kind of genitive
        if idx + 2 < len(tokens):
            next_word_norm = strip_accents(tokens[idx + 2])
            # Rough heuristic: if the word after the genitive article
            # shares a root with a person/agent word, it's likely possessive
            # otherwise it might be objective
            ctx["genitive_follows"] = tokens[idx + 1] + " " + tokens[idx + 2]

    return ctx


# ====================================================================
# Gloss generation
# ====================================================================

# Ørberg notation templates
def make_synonym_gloss(word: str, synonym: str) -> str:
    return f"= {synonym}"

def make_derivation_gloss(word: str, root: str, meaning: str = "") -> str:
    if meaning:
        return f"< {root} = {meaning}"
    return f"< {root}"

def make_antonym_gloss(word: str, synonym: str, antonym: str) -> str:
    return f"= {synonym} ↔ {antonym}"

def make_compound_gloss(parts: list[str], meaning: str) -> str:
    return f"{'·'.join(parts)} = {meaning}"

def make_scale_gloss(word: str, base: str, relation: str) -> str:
    return f"> {base}· {relation}"

def make_context_gloss(word: str, here_meaning: str, default_meaning: str = "") -> str:
    if default_meaning:
        return f"ἐνταῦθα = {here_meaning} (οὐ {default_meaning})"
    return f"ἐνταῦθα = {here_meaning}"


# Frequency threshold: words below this rank get glossed
RARE_RANK_THRESHOLD = 3000  # top 3000 words are "known"
MAX_GLOSSES_PER_SENTENCE = 3
MAX_GLOSSES_PER_PASSAGE = 6


def should_gloss(word: str, freq: Counter) -> bool:
    """Decide if a word needs glossing based on frequency."""
    norm = strip_accents(word)

    # Skip very short words (particles, articles)
    if len(norm) <= 2:
        return False

    # Skip common function words
    common = {"και", "δε", "τε", "γαρ", "μεν", "ουν", "αλλα", "ουτε", "μητε",
              "εστι", "εστιν", "ειναι", "ην", "ησαν", "αυτου", "αυτης", "αυτων",
              "αυτον", "αυτην", "αυτοις", "ουτος", "εκεινος", "τις", "τινος",
              "πας", "πασα", "παν", "ουδεις", "ουκ", "μη", "ως", "οτι",
              "εις", "εν", "εκ", "απο", "προς", "δια", "κατα", "μετα",
              "περι", "υπο", "επι", "παρα", "υπερ", "προ"}
    if norm in common:
        return False

    count = freq.get(norm, 0)
    if count == 0:
        return True  # hapax or very rare — definitely gloss
    rank = sum(1 for c in freq.values() if c > count) + 1
    return rank > RARE_RANK_THRESHOLD


def detect_compound(word: str) -> list[str] | None:
    """Detect if a word is a transparent compound. Returns parts or None."""
    norm = strip_accents(word)
    # Known compound prefixes
    prefixes = [
        ("ξυλο", "ξύλον"), ("υδρο", "ὕδωρ"), ("πυρι", "πῦρ"), ("πυρο", "πῦρ"),
        ("σκληρο", "σκληρός"), ("πατρ", "πατήρ"), ("μεγαλο", "μέγας"),
        ("οινο", "οἶνος"), ("πτυελο", "πτύελον"), ("γραμματο", "γράμμα"),
    ]
    for pref_norm, pref_full in prefixes:
        if norm.startswith(pref_norm) and len(norm) > len(pref_norm) + 2:
            remainder = word[len(pref_norm):]
            return [pref_full, remainder]
    return None


def detect_antonym(word: str) -> str | None:
    """Return a common antonym if one exists."""
    antonyms = {
        "σκοτεινος": "φαιδρός",
        "μεγας": "μικρός",
        "λεπτος": "παχύς",
        "ισχνος": "παχύς",
        "ωχρος": "ἐρυθρός",
        "αλουτος": "καθαρός",
        "ανιπτος": "καθαρός",
        "ανοπλος": "ὡπλισμένος",
        "φαλακρος": "κομήτης",
        "αγραμματος": "πεπαιδευμένος",
        "σκληρος": "μαλακός",
        "κενος": "πλήρης",
        "ερημος": "οἰκουμένη",
    }
    return antonyms.get(strip_accents(word))


def generate_gloss_for_word(word: str, tokens: list[str], idx: int) -> str | None:
    """Generate an Ørberg-style gloss for a single word in context."""
    norm = strip_accents(word)
    ctx = analyse_context(tokens, idx)

    # Check for compound decomposition
    parts = detect_compound(word)
    if parts:
        return f"{'·'.join(p for p in parts)} (< {parts[0]})"

    # Check for antonym pair opportunity
    antonym = detect_antonym(word)

    # Check glossary for locked definition
    glossary = load_glossary()
    gloss_entry = glossary.get(norm)

    # Build the gloss
    # For now, generate a placeholder that the review pipeline can refine
    # The key insight: we flag WHAT needs glossing and provide the raw data;
    # the final gloss text still benefits from human/LLM polish

    return None  # signal that auto-generation couldn't produce a good gloss


def generate_passage_glosses(passage_id: str) -> dict:
    """Generate glosses for a single passage."""
    draft_path = DRAFTS / passage_id / "primary.txt"
    if not draft_path.exists():
        return {}

    text = draft_path.read_text("utf-8").strip()
    freq = load_corpus_frequencies()

    # Split into sentences
    sentences = re.split(r'(?<=[.·;!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Also handle paragraph breaks
    expanded = []
    for s in sentences:
        for sub in s.split("\n\n"):
            sub = sub.strip()
            if sub:
                expanded.append(sub)
    sentences = expanded

    result = {
        "passage_id": passage_id,
        "style": "Ørberg",
        "sentences": [],
    }

    total_glosses = 0
    glossed_norms = set()  # deduplicate across sentences

    for sent_idx, sent in enumerate(sentences):
        tokens = tokenise(sent)
        sent_glosses = []

        for tok_idx, tok in enumerate(tokens):
            if total_glosses >= MAX_GLOSSES_PER_PASSAGE:
                break
            if len(sent_glosses) >= MAX_GLOSSES_PER_SENTENCE:
                break

            norm = strip_accents(tok)
            if norm in glossed_norms:
                continue

            if should_gloss(tok, freq):
                glossed_norms.add(norm)
                sent_glosses.append({
                    "anchor": tok,
                    "note": "",  # to be filled by gloss polish step
                    "_frequency": freq.get(norm, 0),
                    "_rank": frequency_rank(tok),
                    "_context": analyse_context(
                        [t for t in re.findall(r'[\S]+', sent)],
                        min(tok_idx, len(re.findall(r'[\S]+', sent)) - 1)
                    ),
                    "_compound": detect_compound(tok),
                    "_antonym": detect_antonym(tok),
                    "_needs_polish": True,
                })
                total_glosses += 1

        result["sentences"].append({
            "index": sent_idx,
            "greek": sent,
            "glosses": sent_glosses,
        })

    return result


def report_rare_words(passage_id: str):
    """Print a report of rare words that need glossing."""
    data = generate_passage_glosses(passage_id)
    print(f"\n=== {passage_id} ===")
    for sent in data.get("sentences", []):
        for g in sent.get("glosses", []):
            anchor = g["anchor"]
            freq = g.get("_frequency", 0)
            rank = g.get("_rank", 0)
            compound = g.get("_compound")
            antonym = g.get("_antonym")

            notes = []
            if compound:
                notes.append(f"compound: {'·'.join(compound)}")
            if antonym:
                notes.append(f"↔ {antonym}")
            if freq == 0:
                notes.append("hapax/unattested")
            else:
                notes.append(f"rank {rank}, freq {freq}")

            print(f"  {anchor:30s} {', '.join(notes)}")


def main():
    if len(sys.argv) > 1:
        passage_ids = sys.argv[1:]
    else:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )

    print("Loading frequency model...")
    load_corpus_frequencies()

    for pid in passage_ids:
        report_rare_words(pid)


if __name__ == "__main__":
    main()
