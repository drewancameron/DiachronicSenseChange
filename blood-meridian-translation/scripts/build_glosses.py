#!/usr/bin/env python3
"""
Build Ørberg-style marginal glosses for each passage.

1. Identifies rare/noteworthy words using corpus frequency
2. Calls LLM to generate concise Ørberg-notation glosses
3. Writes marginal_glosses.json to the apparatus directory

The LLM receives the Greek sentence, the English source, the target word,
and frequency/compound/antonym metadata. It produces glosses in Ørberg
notation: = synonym, < derivation, ↔ antonym, · compound boundary, etc.

Usage:
  python3 scripts/build_glosses.py              # all passages
  python3 scripts/build_glosses.py 001_see_the_child
  python3 scripts/build_glosses.py --dry-run    # show candidates only
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DRAFTS = ROOT / "drafts"
PASSAGES = ROOT / "passages"
APPARATUS = ROOT / "apparatus"

sys.path.insert(0, str(SCRIPTS))
from generate_glosses import (
    load_corpus_frequencies, strip_accents,
    tokenise, detect_compound, detect_antonym, analyse_context,
    frequency_rank,
)

LEDGER_PATH = APPARATUS / "_gloss_ledger.json"
MORPHEUS_CACHE_PATH = ROOT / "retrieval" / "data" / "morpheus_cache.json"
LEMMA_IDF_PATH = ROOT / "data" / "lemma_idf.json"

# Limits
MAX_GLOSSES_PER_SENTENCE = 3
MIN_GLOSSES_PER_PASSAGE = 6
MAX_GLOSSES_PER_PASSAGE = 40  # hard ceiling
GLOSSES_PER_100_WORDS = 8     # target density: ~8 glosses per 100 words
MAX_PER_LEMMA_PER_CHAPTER = 2
MAX_PER_LEMMA_BOOK = 5

# Document-frequency threshold: a lemma appearing in more than this fraction
# of the corpus documents is "known" and should not be glossed.
# 0.5% of 57K = ~288 documents. This captures the top ~1500 lemmas.
KNOWN_DOC_FREQ_THRESHOLD = 0.005

# ====================================================================
# Known-lemma filter: words any intermediate Greek reader knows
# ====================================================================

# IDF-based known vocabulary — loaded from corpus at runtime.
# Replaces hand-curated list with empirical document frequency.
_known_lemmas_cache = None

def _load_known_lemmas() -> set:
    """Load known lemmas from corpus IDF. A lemma is 'known' if it appears
    in more than KNOWN_DOC_FREQ_THRESHOLD of all corpus documents."""
    global _known_lemmas_cache
    if _known_lemmas_cache is not None:
        return _known_lemmas_cache

    # Always start with the hand-curated set (reliable for common words)
    _known_lemmas_cache = set(_FALLBACK_KNOWN)

    # Supplement with IDF-derived high-frequency lemmas
    if LEMMA_IDF_PATH.exists():
        data = json.load(open(LEMMA_IDF_PATH))
        doc_count = data["doc_count"]
        threshold = int(doc_count * KNOWN_DOC_FREQ_THRESHOLD)
        idf_known = {
            lemma for lemma, freq in data["lemma_doc_freq"].items()
            if freq >= threshold
        }
        _known_lemmas_cache |= idf_known

    return _known_lemmas_cache


# Fallback if lemma_idf.json doesn't exist yet
_FALLBACK_KNOWN = {
    # Articles, pronouns, demonstratives
    "ο", "η", "το", "ος", "ης", "ον", "αυτος", "ουτος", "εκεινος",
    "τις", "τι", "εγω", "συ", "ημεις", "υμεις", "σφεις",
    # Common verbs (lemma forms)
    "ειμι", "εχω", "λεγω", "ποιεω", "γιγνομαι", "γινομαι", "ερχομαι",
    "φημι", "δοκεω", "βουλομαι", "δει", "χρη", "οιδα", "δυναμαι",
    "βαινω", "βαλλω", "φερω", "τιθημι", "διδωμι", "ιστημι",
    "λαμβανω", "ευρισκω", "αγω", "πεμπω", "γραφω", "ακουω",
    "ορωω", "οραω", "κελευω", "πασχω", "θνησκω", "αποθνησκω",
    "μανθανω", "πιπτω", "τυγχανω", "αρχω", "παυω", "εθελω",
    "πινω", "εσθιω", "καθιζω", "κειμαι", "μενω", "φευγω",
    "τρεχω", "πιστευω", "κρινω", "νομιζω", "καλεω",
    "αιρεω", "γιγνωσκω", "γινωσκω", "δεικνυμι", "ζητεω",
    "μαχομαι", "πειθω", "στρατευω", "φαινω", "φοβεω",
    # Common nouns
    "ανηρ", "γυνη", "παις", "ανθρωπος", "θεος", "πολις", "γη",
    "θαλασσα", "θαλαττα", "ημερα", "νυξ", "οδος", "πολεμος",
    "λογος", "εργον", "πραγμα", "χρονος", "τοπος", "χωρα",
    "οικια", "οικος", "πατηρ", "μητηρ", "υιος", "αδελφος",
    "βασιλευς", "στρατηγος", "πλοιον", "ναυς", "ιππος",
    "χειρ", "κεφαλη", "οφθαλμος", "ποδ", "πους", "σωμα",
    "ψυχη", "ονομα", "υδωρ", "πυρ", "ξυλον", "λιθος",
    "αγρος", "δενδρον", "ζωον", "χρυσος", "αργυρος",
    "σιτος", "οινος", "αρτος", "πολιτης", "δουλος",
    "δεσποτης", "διδασκαλος", "ιερευς", "στρατιωτης",
    "ναυτης", "ποιητης", "κριτης", "βιος", "τεκνον",
    "παιδιον", "δημος", "νομος", "αρχη", "δικη", "ειρηνη",
    # Common adjectives
    "αγαθος", "κακος", "καλος", "μεγας", "μικρος", "πολυς",
    "ολιγος", "αλλος", "πας", "ουδεις", "μηδεις", "εκαστος",
    "μονος", "σοφος", "δικαιος", "αληθης", "νεος", "παλαιος",
    "πρωτος", "δευτερος", "τριτος", "εσχατος", "μεσος",
    "λευκος", "μελας", "σκοτεινος", "ετερος", "αυτος",
    "ικανος", "δεινος", "ισχυρος", "ταχυς", "βαρυς",
    "ελευθερος", "φιλος", "εχθρος", "αξιος", "δυνατος",
    "φανερος", "ομοιος", "ιδιος", "κοινος", "ιερος",
    # Common adverbs and spatial/temporal words
    "εγγυς", "πορρω", "οπισθεν", "εμπροσθεν", "πλησιον",
    "αρτι", "παλιν", "μαλα", "σφοδρα", "ταχεως", "ευθυς",
    "ποτε", "που", "πως", "πωποτε", "ποθεν", "οπου",
    "μηποτε", "ουπω", "ουποτε", "δευρο", "εντευθεν",
    # Additional common verbs (including compounds)
    "επιτιθημι", "αποδιδωμι", "παραδιδωμι", "αφιημι",
    "αποκτεινω", "γραφω", "πεμπω", "αιτεω", "ερωταω",
    "κελευω", "πραττω", "πασχω", "μανθανω", "αναβαινω",
    "καταβαινω", "εισερχομαι", "εξερχομαι", "απερχομαι",
    "προσερχομαι", "αποστελλω", "κηρυσσω", "διδασκω",
    "θεραπευω", "σωζω", "κρατεω", "αγαπαω", "μισεω",
    "τρεπω", "στρεφω", "ανιστημι", "καθημαι", "κλαιω",
    "αποκρινομαι", "προσκυνεω", "εκβαλλω", "παρεχω",
    "αναγινωσκω", "ανοιγω", "κλειω", "κρυπτω",
    "αισθανομαι", "εφορεω", "φορεω", "καταπιπτω",
    "αγγελλω", "απαγγελλω", "συλλαμβανω", "αποκαλυπτω",
    "καταλαμβανω", "επιστρεφω", "αναστρεφω", "αφαιρεω",
    "προστιθημι", "εκπιπτω", "αποπιπτω", "επιβαινω",
    "παρερχομαι", "διερχομαι", "περιερχομαι", "κατερχομαι",
    "καθιστημι", "αποκαθιστημι", "ανατιθημι", "συντιθημι",
    "εγειρω", "κοιμαομαι", "αποθνησκω", "γενναω",
    "θαυμαζω", "φιλεω", "δεχομαι", "αποδεχομαι",
    "πυνθανομαι", "ηγεομαι", "παυομαι", "κωλυω",
    "αρπαζω", "διωκω", "κρυπτω", "θαπτω",
    "ωθεω", "ελκω", "σειω", "ριπτω",
    # Common forms the Morpheus cache misses (augmented forms etc.)
    "εφορει", "κατεπεσεν", "αισθανομαι", "αγγειλαι",
    "επεθηκεν", "κατεπεσε", "εισηλθεν", "εξηλθεν",
    "απηλθεν", "ανεστη", "κατεβη", "ανεβη", "επεστρεψεν",
    "προσηλθεν", "συνηλθον", "παρηλθεν", "εγενετο",
    "απεθανεν", "ειδεν", "ειπεν", "ηλθεν", "ελαβεν",
    "ευρεν", "εδωκεν", "εθηκεν", "εγραψεν", "ηκουσεν",
    # Additional common nouns
    "αληθεια", "δυναμις", "εξουσια", "δοξα", "χαρις",
    "πιστις", "ελπις", "αγαπη", "ειρηνη", "σοφια",
    "φωνη", "γλωσσα", "γνωμη", "θυρα", "κλινη",
    "γονυ", "ους", "στομα", "αιμα", "προσωπον",
    "τειχος", "πυλη", "αγορα", "πλατεια", "ιματιον",
    "χλαινα", "πιλος", "μαχαιρα", "ασπις", "δορυ",
    "πλοιον", "ναυς", "αμαξα", "ζυγον",
    "ηλιος", "σεληνη", "αστηρ", "ανεμος", "ομβρος",
    "ποταμος", "ορος", "πετρα", "θαλασσα",
    "κυριος", "κυρια", "κυριαι",
    "αισχρος", "πονηρος", "φαυλος",
    "εκκλησια", "συναγωγη", "σκηνη",
    "θυγατηρ", "κορη", "παρθενος",
    "ξενος", "φυλαξ", "κλεπτης",
    "κριτης", "ιερευς", "βασιλευς", "αρχιερευς",
    "πληθος", "οχλος", "λαος", "δημος",
    "αιτια", "αρχη", "τελος", "μεσον",
    # Particles, conjunctions, prepositions, adverbs
    "και", "δε", "τε", "γαρ", "μεν", "ουν", "αλλα",
    "ουτε", "μητε", "ουδε", "μηδε", "ει", "εαν", "αν",
    "οτι", "ως", "ινα", "οπως", "μη", "ου", "ουκ",
    "εις", "εν", "εκ", "απο", "προς", "δια", "κατα", "μετα",
    "περι", "υπο", "υπερ", "επι", "παρα", "προ", "συν", "αντι",
    "νυν", "τοτε", "ηδη", "ετι", "ουτως", "ουτω", "εκει",
    "ενταυθα", "εξω", "εισω", "ανω", "κατω", "μαλιστα",
    "πρωτον", "επειτα", "ομως", "μαλλον", "καλως",
    # Numbers
    "εις", "δυο", "τρεις", "τεσσαρες", "πεντε", "εξ", "επτα",
    "οκτω", "εννεα", "δεκα", "εκατον", "χιλιοι",
}


def _load_morpheus_cache() -> dict:
    """Load Morpheus cache for lemma lookups."""
    if MORPHEUS_CACHE_PATH.exists():
        return json.load(open(MORPHEUS_CACHE_PATH))
    return {}


_morph_cache = None

def get_lemma(word: str) -> str:
    """Get lemma for a word from Morpheus cache. Falls back to accent-stripped form."""
    global _morph_cache
    if _morph_cache is None:
        _morph_cache = _load_morpheus_cache()
    clean = word.strip(".,·;:—–«»()[]!\"' *")
    analyses = _morph_cache.get(clean, [])
    if analyses and isinstance(analyses, list):
        for a in analyses:
            if isinstance(a, dict) and a.get("lemma"):
                return strip_accents(a["lemma"])
    return strip_accents(clean)


def load_ledger() -> dict:
    """Load the cross-passage gloss ledger."""
    if LEDGER_PATH.exists():
        return json.load(open(LEDGER_PATH))
    return {"lemma_counts": {}, "chapter_counts": {}}


def save_ledger(ledger: dict):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)


def _has_interesting_morphology(word: str) -> bool:
    """Check if Morpheus positively identifies this as a rare form worth glossing.

    Returns True only if Morpheus can parse the word AND the analysis shows
    a non-trivial form (perfect, pluperfect, aorist passive, optative, future).
    Returns False if Morpheus can't parse it (no data = assume boring).
    """
    global _morph_cache
    if _morph_cache is None:
        _morph_cache = _load_morpheus_cache()
    clean = word.strip(".,·;:—–«»()[]!\"' *")
    analyses = _morph_cache.get(clean, [])
    if not analyses or not isinstance(analyses, list):
        return False  # No Morpheus data — cannot confirm interesting

    for a in analyses:
        if not isinstance(a, dict):
            continue
        tense = a.get("tense", "").lower()
        mood = a.get("mood", "").lower()
        voice = a.get("voice", "").lower()
        # These forms are genuinely worth noting
        if tense in ("perfect", "pluperfect", "future perfect"):
            return True
        if mood in ("optative",):
            return True
        if tense == "aorist" and voice == "passive":
            return True
    return False


def _is_boring_form(word: str) -> bool:
    """Check if this is a present/imperfect form of a known verb — not worth glossing.

    Rare forms (perfect, aorist passive, optative, future) of common verbs ARE
    worth glossing because the form itself teaches something. But a present
    indicative of κεῖμαι or φέρω is not interesting.
    """
    global _morph_cache
    if _morph_cache is None:
        _morph_cache = _load_morpheus_cache()
    clean = word.strip(".,·;:—–«»()[]!\"' *")
    analyses = _morph_cache.get(clean, [])
    if not analyses or not isinstance(analyses, list):
        return False

    for a in analyses:
        if not isinstance(a, dict):
            continue
        lemma = strip_accents(a.get("lemma", ""))
        if lemma not in _load_known_lemmas():
            continue
        # This is a form of a known lemma — check if the form is boring
        tense = a.get("tense", "").lower()
        mood = a.get("mood", "").lower()
        # Present and imperfect indicative/subjunctive are boring
        if tense in ("present", "imperfect") and mood in ("indicative", "subjunctive", ""):
            return True
        # Infinitives and basic participles of common verbs are boring too
        if mood in ("infinitive",) and tense in ("present",):
            return True
    return False


def should_gloss_word(word: str, freq, ledger: dict, chapter: str = "I") -> bool:
    """Decide if a word needs glossing.

    Strategy:
      1. Check Morpheus lemma against hand-curated known set — catches common
         verbs/nouns even when the surface form is rare (ἐφόρει → φορέω → known)
      2. For known lemmas, only gloss truly rare morphological forms (perfect, etc.)
      3. For unknown lemmas, use surface-form corpus frequency as fallback
      4. Respect chapter/book caps from ledger
    """
    norm = strip_accents(word)
    lemma = get_lemma(word)

    # Skip very short words
    if len(norm) <= 2:
        return False

    # Check hand-curated known lemmas (reliable despite broken lemmatisation
    # because the Morpheus cache does resolve ~2500 common forms)
    known = _load_known_lemmas()
    if lemma in known or norm in known:
        # Known lemma — only gloss if Morpheus positively identifies a rare form
        # (perfect, aorist passive, optative, etc.). If Morpheus can't parse it
        # at all, assume it's a boring form and skip — err on side of not glossing.
        if _has_interesting_morphology(word):
            count = freq.get(norm, 0)
            if count == 0:
                return True
        return False

    # Check book-level cap
    book_count = ledger.get("lemma_counts", {}).get(lemma, 0)
    if book_count >= MAX_PER_LEMMA_BOOK:
        return False

    # Check chapter-level cap
    ch_key = f"{chapter}:{lemma}"
    ch_count = ledger.get("chapter_counts", {}).get(ch_key, 0)
    if ch_count >= MAX_PER_LEMMA_PER_CHAPTER:
        return False

    # Surface form frequency check
    count = freq.get(norm, 0)
    if count == 0:
        return True  # hapax — definitely gloss

    rank = sum(1 for c in freq.values() if c > count) + 1
    return rank > 8000


def record_gloss(lemma: str, ledger: dict, chapter: str = "I"):
    """Record that a lemma was glossed."""
    if "lemma_counts" not in ledger:
        ledger["lemma_counts"] = {}
    if "chapter_counts" not in ledger:
        ledger["chapter_counts"] = {}
    ledger["lemma_counts"][lemma] = ledger["lemma_counts"].get(lemma, 0) + 1
    ch_key = f"{chapter}:{lemma}"
    ledger["chapter_counts"][ch_key] = ledger["chapter_counts"].get(ch_key, 0) + 1


def load_english(passage_id: str) -> str:
    p = PASSAGES / f"{passage_id}.json"
    if p.exists():
        return json.load(open(p)).get("text", "")
    return ""


def split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.·;!])\s+', text)
    result = []
    for chunk in raw:
        for sub in chunk.split("\n\n"):
            sub = sub.strip()
            if sub:
                result.append(sub)
    return result


def identify_candidates(passage_id: str, ledger: dict, chapter: str = "I") -> dict:
    """Identify words needing glosses, distributed evenly across the passage."""
    draft_path = DRAFTS / passage_id / "primary.txt"
    if not draft_path.exists():
        return {}

    text = draft_path.read_text("utf-8").strip()
    english = load_english(passage_id)
    freq = load_corpus_frequencies()
    sentences = split_sentences(text)

    result = {
        "passage_id": passage_id,
        "english": english,
        "sentences": [],
    }

    # --- Phase 1: collect ALL candidate words across all sentences ---
    all_candidates = []  # list of (sent_idx, candidate_dict, priority)
    seen_norms = set()

    for sent_idx, sent in enumerate(sentences):
        raw_tokens = re.findall(r'\S+', sent)
        tokens = tokenise(sent)

        # Loanwords (*-marked) — highest priority
        for raw_tok in raw_tokens:
            if raw_tok.startswith("*"):
                clean = raw_tok.lstrip("*").rstrip(".,·;:—–«»()[]!\"' ")
                norm = strip_accents(clean)
                lemma = get_lemma(clean)
                if norm in seen_norms or len(norm) <= 2:
                    continue
                book_count = ledger.get("lemma_counts", {}).get(lemma, 0)
                if book_count >= MAX_PER_LEMMA_BOOK:
                    continue
                seen_norms.add(norm)
                all_candidates.append((sent_idx, {
                    "anchor": clean,
                    "frequency": 0,
                    "rank": 0,
                    "compound": detect_compound(clean),
                    "antonym": detect_antonym(clean),
                    "loanword": True,
                }, 0))  # priority 0 = highest

        # Rare words by frequency
        for tok in tokens:
            norm = strip_accents(tok)
            if norm in seen_norms:
                continue
            if should_gloss_word(tok, freq, ledger, chapter):
                seen_norms.add(norm)
                f = freq.get(norm, 0)
                r = frequency_rank(tok) if f > 0 else 0  # hapax = highest priority
                all_candidates.append((sent_idx, {
                    "anchor": tok,
                    "frequency": f,
                    "rank": r,
                    "compound": detect_compound(tok),
                    "antonym": detect_antonym(tok),
                }, r))  # priority = rank (lower = rarer = more important)

    # --- Phase 2: running-window distribution ---
    # Guarantee minimum gloss density throughout the passage.
    # Walk through sentences tracking a word-count window; whenever
    # the window exceeds MAX_WORDS_WITHOUT_GLOSS, force-pick the best
    # available candidate in that region.

    n_sents = len(sentences)
    word_count = len(text.split())
    target = max(MIN_GLOSSES_PER_PASSAGE,
                 int(word_count * GLOSSES_PER_100_WORDS / 100))
    budget = min(MAX_GLOSSES_PER_PASSAGE, target, len(all_candidates))

    # Max words between glosses — ensures no long barren stretches
    MAX_WORDS_WITHOUT_GLOSS = max(15, word_count // max(1, budget))

    # Build per-sentence word counts and candidate lookup
    sent_word_counts = [len(tokenise(s)) for s in sentences]
    cands_by_sent = {}
    for sent_idx, cand, pri in all_candidates:
        cands_by_sent.setdefault(sent_idx, []).append((cand, pri))
    # Sort each sentence's candidates by priority (rarest first)
    for s in cands_by_sent:
        cands_by_sent[s].sort(key=lambda x: x[1])

    selected = {}  # sent_idx → [candidates]
    total_selected = 0
    words_since_gloss = 0

    for sent_idx in range(n_sents):
        words_since_gloss += sent_word_counts[sent_idx]

        # Check if this sentence has candidates and we need a gloss
        available = cands_by_sent.get(sent_idx, [])
        if not available:
            continue

        sent_list = selected.setdefault(sent_idx, [])

        # Pick candidates when the gap since last gloss is large enough.
        # Loanwords (*-marked, priority 0 with loanword flag) get a lower
        # threshold since they always need glossing.
        has_loanword = any(cand.get("loanword") for cand, _ in available)
        threshold = MAX_WORDS_WITHOUT_GLOSS // 2 if has_loanword else MAX_WORDS_WITHOUT_GLOSS

        if words_since_gloss >= threshold:
            # Pick just ONE candidate per trigger to spread budget evenly
            cand, pri = available[0]
            if total_selected < budget:
                sent_list.append(cand)
                lemma = get_lemma(cand["anchor"])
                record_gloss(lemma, ledger, chapter)
                total_selected += 1
                words_since_gloss = 0

    # Second pass: if we still have budget, fill in remaining best candidates
    # (sorted globally by priority) to reach target density
    if total_selected < budget:
        remaining = []
        for sent_idx, cand, pri in all_candidates:
            already = selected.get(sent_idx, [])
            if cand in already:
                continue
            if len(already) >= MAX_GLOSSES_PER_SENTENCE:
                continue
            remaining.append((sent_idx, cand, pri))
        remaining.sort(key=lambda x: x[2])

        for sent_idx, cand, pri in remaining:
            if total_selected >= budget:
                break
            sent_list = selected.setdefault(sent_idx, [])
            if len(sent_list) >= MAX_GLOSSES_PER_SENTENCE:
                continue
            sent_list.append(cand)
            lemma = get_lemma(cand["anchor"])
            record_gloss(lemma, ledger, chapter)
            total_selected += 1

    # --- Phase 3: build output structure ---
    glossed_norms = set()

    for sent_idx, sent in enumerate(sentences):
        candidates = selected.get(sent_idx, [])
        result["sentences"].append({
            "index": sent_idx,
            "greek": sent,
            "candidates": candidates,
        })

    return result


def build_gloss_prompt(data: dict) -> str:
    """Build LLM prompt to generate Ørberg-style glosses."""
    candidates_text = []
    for sent in data["sentences"]:
        if not sent["candidates"]:
            continue
        candidates_text.append(f'Sentence: "{sent["greek"]}"')
        for c in sent["candidates"]:
            meta = []
            if c.get("loanword"):
                meta.append("LOANWORD/NEOLOGISM — must gloss")
            elif c["frequency"] == 0:
                meta.append("hapax/unattested in corpus")
            else:
                meta.append(f"rank {c['rank']}, freq {c['frequency']}")
            if c["compound"]:
                meta.append(f"compound: {'·'.join(c['compound'])}")
            if c["antonym"]:
                meta.append(f"possible antonym: {c['antonym']}")
            candidates_text.append(f"  → {c['anchor']} ({', '.join(meta)})")

    if not candidates_text:
        return ""

    prompt = f"""You are generating Ørberg-style marginal glosses for an Ancient Greek text (a translation of McCarthy's Blood Meridian). These appear in the margin of a reader-edition and help an intermediate Greek reader understand rare or noteworthy words in context.

## Ørberg Notation System
The notation is terse, information-dense, and ENTIRELY in Greek (no English). Each symbol has a precise function:

- `=` synonym/definition: `= λεπτός, ἀσθενὴς τὸ σῶμα`
- `<` derivation from root: `< λίνον· = ἐκ λίνου πεποιημένον`
- `↔` antonym: `↔ παχύς, εὔρωστος`
- `·` compound boundary: `ξυλο·κόπος · ὑδρο·φόρος`
- `+` case/construction: `σκαλεύω + αἰτ.`
- `()` supplementary info: `(ὕφασμα φυτικόν)`, `(Ὅμ.)`, `(παρακ. μτχ.)`
- `ἐνταῦθα` = contextual meaning differing from default sense

Grammatical abbreviations: αἰτ. γεν. δοτ. ὀν. κλ. = cases; ἑν. δυ. πλ. = number; ἐν. μέσ. παθ. = voice; παρακ. ἀόρ. μέλλ. ὑπερσ. = tense; μτχ. ἀπρ. ὑποτ. εὐκτ. προστ. = mood; ῥ. = verb

## Quality Exemplars (match this level)
- ἰσχνός → `= λεπτός, ἀσθενὴς τὸ σῶμα ↔ παχύς, εὔρωστος`
- λίνεον → `< λίνον· = ἐκ λίνου πεποιημένον (ὕφασμα φυτικόν)`
- ῥακώδη → `< ῥάκος = κουρέλιον· ἐνταῦθα τετριμμένον ↔ ὁλόκληρον`
- σκαλεύει → `σκαλεύω + αἰτ. = κινεῖ τοὺς ἄνθρακας σιδήρῳ ↔ σβέννυμι`
- ἐσχάραν → `ἐσχάρα = ἑστία, τόπος ἐν ᾧ καίεται πῦρ (αἰτ.)`
- ἐσκαμμένοι → `< σκάπτω (παρακ. μτχ.)· = ἠροτριασμένοι, ἀνεσκαμμένοι`
- ξυλοκόπων καὶ ὑδροφόρων → `ξυλο·κόπος · ὑδρο·φόρος = δοῦλοι ταπεινοί (ΙΗΣ. ΝΑΥ. θ´ 21)`
- ἀπόλωλεν → `< ἀπόλλυμι (παρακ.)· οὐδ. πλ. + ἑν. ῥ. κατ᾽ Ἀττ. ἔθος`
- πτώσσει → `↔ ἵσταται ὀρθός· = ὀκλάζει ὥσπερ θηρίον δεδοικός (Ὅμ.)`

## Rules
1. ENTIRELY in Greek — no English words, no Latin script.
2. Dense and terse — under 80 characters per gloss. No full sentences.
3. Multi-word anchors are allowed when a phrase is a unit (e.g. ξυλοκόπων καὶ ὑδροφόρων).
4. For compound words, ALWAYS show component parts with · boundary.
5. For derived forms, show the root with < and the tense/mood/voice in parentheses.
6. Include grammatical notes where they illuminate: case governance, Attic rules, voice.
7. Note biblical/Septuagintal/Homeric resonances with abbreviated references.
8. For verbs, include case governance if non-obvious (+ γεν., + δοτ.).
9. For metaphorical or contextual uses, use ἐνταῦθα to distinguish.
10. Prioritise information density: every character should teach something.

## English source
{data['english']}

## Words to gloss
{chr(10).join(candidates_text)}

## Output format
Return a JSON array of objects, one per word, in the same order as listed above.
The "anchor" field must exactly match the word(s) from the Greek text.
[
  {{"anchor": "word", "note": "Ørberg gloss"}},
  ...
]

Output ONLY the JSON array, no other text."""

    return prompt


def call_llm(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def build_passage_glosses(passage_id: str, ledger: dict,
                          chapter: str = "I", dry_run: bool = False) -> bool:
    """Generate glosses for one passage."""
    data = identify_candidates(passage_id, ledger, chapter)
    if not data:
        return False

    total_candidates = sum(len(s["candidates"]) for s in data["sentences"])
    if total_candidates == 0:
        print(f"  {passage_id}: no words need glossing")
        # Still write the file so renderer has sentence boundaries
        write_bare_glosses(passage_id, data)
        return True

    print(f"  {passage_id}: {total_candidates} words to gloss")

    if dry_run:
        for sent in data["sentences"]:
            for c in sent["candidates"]:
                print(f"    {c['anchor']:30s} rank={c['rank']}, freq={c['frequency']}")
        return False

    prompt = build_gloss_prompt(data)
    if not prompt:
        write_bare_glosses(passage_id, data)
        return True

    print(f"    Calling LLM for glosses...")
    raw = call_llm(prompt)

    # Parse JSON response
    try:
        # Strip markdown code block if present
        clean = re.sub(r'^```json\s*', '', raw)
        clean = re.sub(r'\s*```$', '', clean)
        glosses = json.loads(clean)
    except json.JSONDecodeError:
        print(f"    WARNING: could not parse LLM response, writing bare glosses")
        write_bare_glosses(passage_id, data)
        return True

    # Merge LLM glosses into sentence structure
    gloss_map = {g["anchor"]: g["note"] for g in glosses}

    output = {"passage_id": passage_id, "style": "Ørberg", "sentences": []}
    for sent in data["sentences"]:
        sent_glosses = []
        for c in sent["candidates"]:
            note = gloss_map.get(c["anchor"], "")
            sent_glosses.append({"anchor": c["anchor"], "note": note})
        output["sentences"].append({
            "index": sent["index"],
            "greek": sent["greek"],
            "glosses": sent_glosses,
        })

    # Write to apparatus
    out_dir = APPARATUS / passage_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "marginal_glosses.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"    ✓ Wrote {total_candidates} glosses")
    return True


def write_bare_glosses(passage_id: str, data: dict):
    """Write gloss file with sentence boundaries but no annotations."""
    output = {"passage_id": passage_id, "style": "Ørberg", "sentences": []}
    for sent in data["sentences"]:
        output["sentences"].append({
            "index": sent["index"],
            "greek": sent["greek"],
            "glosses": [],
        })
    out_dir = APPARATUS / passage_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "marginal_glosses.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-ledger", action="store_true",
                        help="Reset the cross-passage gloss ledger")
    parser.add_argument("--chapter", default="I",
                        help="Chapter number for ledger tracking")
    args = parser.parse_args()

    # Load or reset the cross-passage ledger
    if args.reset_ledger:
        ledger = {"lemma_counts": {}, "chapter_counts": {}}
    else:
        ledger = load_ledger()

    if args.passages:
        passage_ids = args.passages
    else:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )

    for pid in passage_ids:
        build_passage_glosses(pid, ledger, chapter=args.chapter,
                              dry_run=args.dry_run)

    # Save ledger after all passages
    if not args.dry_run:
        save_ledger(ledger)
        total = sum(ledger["lemma_counts"].values())
        print(f"\n  Ledger: {len(ledger['lemma_counts'])} lemmas glossed, {total} total glosses")


if __name__ == "__main__":
    main()
