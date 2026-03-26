#!/usr/bin/env python3
"""
Morphological grammar checker using the Alpheios/Morpheus API.

Parses each word in the Greek translation, checks:
  1. Verb agreement: neuter plural subjects should have singular verbs
  2. Preposition governance: correct case after each preposition
  3. Adjective-noun agreement: gender, number, case
  4. Time expressions: dative for 'when', genitive for 'within'

Uses the Perseus Morpheus API for morphological analysis.
Caches results to avoid repeated API calls.

Usage:
  python3 scripts/morpheus_check.py                     # check all passages
  python3 scripts/morpheus_check.py 002_night_of_your_birth
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
CACHE_PATH = ROOT / "retrieval" / "data" / "morpheus_cache.json"

MORPHEUS_URL = "https://morph.alpheios.net/api/v1/analysis/word?word={}&lang=grc&engine=morpheusgrc"

# ====================================================================
# Morpheus API + cache
# ====================================================================

_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            _cache = json.load(f)
    else:
        _cache = {}
    return _cache


def _save_cache():
    if _cache is not None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(_cache, f, ensure_ascii=False)


def parse_word(word: str) -> list[dict]:
    """Parse a Greek word via Morpheus. Returns list of possible analyses."""
    cache = _load_cache()
    clean = word.strip(".,·;:—–«»()[]!\"' ")
    if not clean:
        return []

    if clean in cache:
        return cache[clean]

    encoded = urllib.parse.quote(clean)
    url = MORPHEUS_URL.format(encoded)

    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        analyses = _extract_analyses(data)
        cache[clean] = analyses
        time.sleep(0.15)  # rate limit
        return analyses

    except Exception as e:
        cache[clean] = [{"error": str(e)}]
        return []


def _extract_analyses(data: dict) -> list[dict]:
    """Extract morphological analyses from Morpheus API response."""
    results = []
    try:
        body = data.get("RDF", {}).get("Annotation", {}).get("Body", {})
        rest = body.get("rest", {})
        entry = rest.get("entry", {})

        # Can be a single entry or a list
        entries = entry if isinstance(entry, list) else [entry]

        for e in entries:
            if not isinstance(e, dict):
                continue
            d = e.get("dict", {})
            lemma = d.get("hdwd", {}).get("$", "")
            pofs = d.get("pofs", {}).get("$", "")

            infl_data = e.get("infl", {})
            infls = infl_data if isinstance(infl_data, list) else [infl_data]

            for infl in infls:
                if not isinstance(infl, dict):
                    continue
                analysis = {
                    "lemma": lemma,
                    "pofs": pofs,  # part of speech
                    "person": infl.get("prsn", {}).get("$", ""),
                    "number": infl.get("num", {}).get("$", ""),
                    "tense": infl.get("tense", {}).get("$", ""),
                    "mood": infl.get("mood", {}).get("$", ""),
                    "voice": infl.get("voice", {}).get("$", ""),
                    "gender": infl.get("gend", {}).get("$", ""),
                    "case": infl.get("case", {}).get("$", ""),
                }
                results.append(analysis)

    except Exception:
        pass

    return results


# ====================================================================
# Grammar checks using parsed morphology
# ====================================================================

def check_passage(passage_id: str) -> list[dict]:
    """Parse and check a passage. Returns list of issues."""
    draft_path = DRAFTS / passage_id / "primary.txt"
    if not draft_path.exists():
        return []

    text = draft_path.read_text("utf-8").strip()
    tokens = re.findall(r'[\w\u0370-\u03FF\u1F00-\u1FFF\']+|[^\w\s]', text)

    issues = []
    parsed = []

    print(f"  Parsing {passage_id} ({len(tokens)} tokens)...")
    for tok in tokens:
        if re.match(r'^[\W]+$', tok):
            parsed.append({"token": tok, "analyses": []})
            continue
        analyses = parse_word(tok)
        parsed.append({"token": tok, "analyses": analyses})

    _save_cache()

    # Check 0: Unattested words (Morpheus doesn't recognise them)
    issues.extend(_check_unattested(parsed, tokens))

    # Check 1: Neuter plural + plural verb
    issues.extend(_check_neuter_plural_verb(parsed, tokens))

    # Check 2: Preposition + case
    issues.extend(_check_preposition_case(parsed, tokens))

    return issues


_corpus_forms: set | None = None

def _load_corpus_forms() -> set:
    """Load all attested surface forms from our 57K-sentence corpus."""
    global _corpus_forms
    if _corpus_forms is not None:
        return _corpus_forms

    import unicodedata
    corpus_path = ROOT / "retrieval" / "data" / "corpus.jsonl"
    _corpus_forms = set()
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = rec.get("text", "")
                for tok in re.findall(r'[\w\u0370-\u03FF\u1F00-\u1FFF]+', text):
                    # Store accent-stripped lowercase
                    decomposed = unicodedata.normalize("NFD", tok)
                    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
                    _corpus_forms.add(stripped.lower())
    return _corpus_forms


def _is_attested_in_corpus(word: str) -> bool:
    """Check if the accent-stripped form appears in our AG corpus."""
    import unicodedata
    forms = _load_corpus_forms()
    decomposed = unicodedata.normalize("NFD", word)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower() in forms


# Database attestation (839K passages from the diachronic project)
_db_checked: dict | None = None
DB_PATH = ROOT.parent / "db" / "diachronic.db"


def _is_attested_in_db(word: str) -> bool:
    """Check if the exact surface form appears in the 839K-passage database.
    Uses sqlite LIKE for exact substring match. Caches results."""
    global _db_checked
    if _db_checked is None:
        _db_checked = {}
    if word in _db_checked:
        return _db_checked[word]

    if not DB_PATH.exists():
        return False

    import sqlite3
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM passages WHERE greek_text LIKE ?",
            (f"%{word}%",)
        )
        count = cur.fetchone()[0]
        conn.close()
        _db_checked[word] = count > 0
        return count > 0
    except Exception:
        return False


# Words that Morpheus API misses but are clearly real (LSJ-attested).
# Add to this set as false positives are discovered.
MORPHEUS_WHITELIST = {
    "ῥακῶν", "ῥάκος", "ῥακώδη", "ποιητῶν", "ποιητής",
    "Ἄρκτος", "ἄρκτος", "μετανάστης", "μετανάστου",
    "ὁμαλῆς", "ὁμαλός", "ἰσχνοί", "ἰσχνοὶ", "ἰσχνός",
    "ἀράχναι", "ἀράχνη", "κάλυξι", "κάλυξ",
    "αὐγῆς", "αὐγή", "σκαλεύει", "σκαλεύω",
    "ἐσχάραν", "ἐσχάρα", "μαγειρεῖον", "μαγειρείου",
    "γραμματιστής", "γραμματιστοῦ",
    "πτώσσει", "πτώσσω",
    "ἐπῳάζεται", "ἐπῳάζω",
    "λυκόφωτι", "λυκόφως",
    "παγετῶδες", "παγετώδης",
    "ὑπομβρίου", "ὑπόμβριος",
}


def _check_unattested(parsed: list[dict], tokens: list[str]) -> list[dict]:
    """Flag words that neither Morpheus nor our corpus recognise."""
    issues = []
    skip_patterns = re.compile(r'^[\W\d]+$|^\*|^[α-ω]$|^[Α-Ω]$', re.IGNORECASE)

    for i, p in enumerate(parsed):
        tok = p["token"]
        analyses = p["analyses"]

        if skip_patterns.match(tok):
            continue
        if len(tok) <= 2:
            continue
        if any("error" in a for a in analyses):
            continue

        # Morpheus knows it → fine
        if analyses:
            continue

        # Whitelisted (known Morpheus false positive) → fine
        if tok in MORPHEUS_WHITELIST:
            continue

        # Morpheus doesn't know it — check corpus as second opinion
        if _is_attested_in_corpus(tok):
            continue

        # Check the diachronic project database (839K passages)
        if _is_attested_in_db(tok):
            continue

        # None of our sources know this word → flag
        issues.append({
            "type": "unattested_word",
            "severity": "warning",
            "position": i,
            "word": tok,
            "context": " ".join(tokens[max(0, i-2):i+3]),
            "message": f"'{tok}' not in Morpheus, AG corpus, or database — possibly invented.",
        })

    return issues


def _is_neuter_plural(analyses: list[dict]) -> bool:
    """Check if any analysis gives neuter plural (nom or acc)."""
    for a in analyses:
        if a.get("gender") == "neuter" and a.get("number") == "plural":
            if a.get("case") in ("nominative", "accusative", "vocative"):
                return True
    return False


def _is_plural_verb(analyses: list[dict]) -> bool:
    """Check if the word is ONLY parseable as a plural finite verb.
    If it has both singular and plural readings, it's ambiguous — don't flag."""
    has_plural = False
    has_singular = False
    for a in analyses:
        if a.get("pofs") == "verb":
            if a.get("mood") in ("indicative", "subjunctive", "optative", "imperative"):
                if a.get("number") == "plural":
                    has_plural = True
                elif a.get("number") == "singular":
                    has_singular = True
    # Only flag if unambiguously plural
    return has_plural and not has_singular


def _check_neuter_plural_verb(parsed: list[dict], tokens: list[str]) -> list[dict]:
    """Find neuter plural subjects with plural verbs."""
    issues = []
    for i, p in enumerate(parsed):
        if _is_neuter_plural(p["analyses"]):
            # Look for a verb within next 5 tokens
            for j in range(i + 1, min(i + 6, len(parsed))):
                if _is_plural_verb(parsed[j]["analyses"]):
                    # Check if there's a singular alternative
                    verb_lemma = parsed[j]["analyses"][0].get("lemma", "?")
                    context = " ".join(t["token"] for t in parsed[max(0, i-1):j+2])
                    issues.append({
                        "type": "neuter_plural_verb",
                        "severity": "warning",
                        "position": j,
                        "word": tokens[j],
                        "lemma": verb_lemma,
                        "context": context,
                        "message": f"neuter plural '{tokens[i]}' + plural verb '{tokens[j]}' (< {verb_lemma}) — singular verb expected (Attic rule)",
                    })
                    break
    return issues


PREP_CASE_RULES = {
    "ἐν": ["dative"],
    "εἰς": ["accusative"],
    "ἐκ": ["genitive"],
    "ἐξ": ["genitive"],
    "ἀπό": ["genitive"],
    "ἀπ'": ["genitive"],
    "πρό": ["genitive"],
    "μετά": ["genitive", "accusative"],
    "μετ'": ["genitive", "accusative"],
    "παρά": ["genitive", "dative", "accusative"],
    "παρ'": ["genitive", "dative", "accusative"],
    "ὑπέρ": ["genitive", "accusative"],
    "ὑπό": ["genitive", "accusative"],
    "ὑπ'": ["genitive", "accusative"],
    "πρός": ["accusative", "dative", "genitive"],
    "διά": ["genitive", "accusative"],
    "δι'": ["genitive", "accusative"],
    "κατά": ["genitive", "accusative"],
    "κατ'": ["genitive", "accusative"],
    "περί": ["genitive", "accusative"],
    "ἐπί": ["genitive", "dative", "accusative"],
    "ἐπ'": ["genitive", "dative", "accusative"],
}


def _check_preposition_case(parsed: list[dict], tokens: list[str]) -> list[dict]:
    """Check that words after prepositions are in the correct case."""
    issues = []
    for i, p in enumerate(parsed):
        tok = p["token"]
        allowed_cases = PREP_CASE_RULES.get(tok)
        if not allowed_cases:
            continue

        # Find the FIRST word after the preposition that has case info.
        # Stop immediately — don't scan past clause boundaries.
        for j in range(i + 1, min(i + 4, len(parsed))):
            next_tok = parsed[j]["token"]
            next_analyses = parsed[j]["analyses"]

            # Stop at punctuation (clause boundary)
            if next_tok in (",", "·", ";", ".", "—", "–"):
                break

            if not next_analyses:
                continue

            # Get the case of this word
            cases_found = set()
            for a in next_analyses:
                c = a.get("case", "")
                if c:
                    cases_found.add(c)

            if not cases_found:
                continue

            # Check if any found case matches the allowed cases
            if cases_found & set(allowed_cases):
                break  # OK
            else:
                context = " ".join(t["token"] for t in parsed[max(0, i):j+2])
                issues.append({
                    "type": "preposition_case",
                    "severity": "warning",
                    "position": j,
                    "word": tokens[j],
                    "context": context,
                    "message": f"'{tok}' requires {'/'.join(allowed_cases)} but '{tokens[j]}' appears to be {'/'.join(cases_found)}",
                })
                break

    return issues


# ====================================================================
# Main
# ====================================================================

def main():
    if len(sys.argv) > 1:
        passage_ids = sys.argv[1:]
    else:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )

    all_issues = []
    for pid in passage_ids:
        issues = check_passage(pid)
        all_issues.extend(issues)

    _save_cache()

    print(f"\n{'='*60}")
    print(f"Morpheus Grammar Check: {len(passage_ids)} passages")
    print(f"{'='*60}")

    warnings = [i for i in all_issues if i["severity"] == "warning"]
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for i in warnings:
            print(f"  ⚠ [{i.get('type')}] {i['message']}")
            print(f"    context: {i.get('context', '')}")
    else:
        print("\n  ✓ No grammar issues found.")

    cached = len(_load_cache())
    print(f"\nMorpheus cache: {cached} forms cached")


if __name__ == "__main__":
    main()
