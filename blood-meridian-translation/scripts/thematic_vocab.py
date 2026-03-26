#!/usr/bin/env python3
"""
Thematic vocabulary extraction from the full 809K-passage classical corpus.

For a given English passage, detects themes, searches the parent database
for Greek passages on those themes, and extracts a curated vocabulary palette
of attested content words with source attribution.

Two retrieval strategies:
  1. Greek keyword search (fast, works on full 809K corpus)
  2. English embedding search (richer, works on 267K aligned passages)

Usage:
  python3 scripts/thematic_vocab.py "He walked through the dark forest..."
  python3 scripts/thematic_vocab.py --passage-id 007_divested_of_all
"""

import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT.parent / "db" / "diachronic.db"
PASSAGES = ROOT / "passages"

# ====================================================================
# Theme detection from English text
# ====================================================================

# Thematic clusters: English keywords → Greek search terms + theme label
THEMES = {
    "landscape_mountain": {
        "en_triggers": ["mountain", "mountains", "gorge", "canyon", "ridge",
                        "cliff", "rock", "rocks", "boulder", "peak", "slope",
                        "precipice", "crag", "bluff", "butte"],
        "grc_search": ["φάραγξ", "κρημνός", "σπήλαιον", "λόφος", "ὄρος",
                        "πέτρα", "σκόπελος", "κλιτύς", "κορυφή"],
        "label": "mountains and rocky landscape",
    },
    "landscape_plain": {
        "en_triggers": ["plain", "plains", "prairie", "desert", "waste",
                        "barren", "dust", "sand", "horizon", "mesa"],
        "grc_search": ["πεδίον", "ἔρημος", "ψάμμος", "ἄμμος", "κόνις",
                        "ἐρημία", "στέππα"],
        "label": "plains and desert",
    },
    "landscape_water": {
        "en_triggers": ["river", "stream", "torrent", "creek", "waterfall",
                        "cascade", "spring", "flood", "ford", "crossing"],
        "grc_search": ["ποταμός", "χείμαρρος", "καταρράκτης", "πηγή",
                        "ῥεῦμα", "ῥέω", "νᾶμα", "κρήνη"],
        "label": "rivers and water",
    },
    "landscape_forest": {
        "en_triggers": ["forest", "woods", "trees", "oak", "pine", "cedar",
                        "thicket", "grove", "timber", "lumber"],
        "grc_search": ["ὕλη", "δρυμός", "δρῦς", "πεύκη", "δένδρον",
                        "ἄλσος", "νάπη", "λόχμη"],
        "label": "forest and trees",
    },
    "weather": {
        "en_triggers": ["rain", "storm", "lightning", "thunder", "wind",
                        "hail", "snow", "cloud", "fog", "mist"],
        "grc_search": ["ὄμβρος", "χειμών", "κεραυνός", "βροντή", "ἄνεμος",
                        "χάλαζα", "χιών", "νέφος", "ὀμίχλη", "θύελλα"],
        "label": "weather and storms",
    },
    "celestial": {
        "en_triggers": ["sun", "moon", "stars", "sky", "heaven", "dawn",
                        "dusk", "sunset", "night", "darkness"],
        "grc_search": ["ἥλιος", "σελήνη", "ἀστήρ", "οὐρανός", "ἠώς",
                        "σκότος", "ζόφος", "φῶς", "αὐγή"],
        "label": "sky, light and darkness",
    },
    "violence_combat": {
        "en_triggers": ["fight", "fought", "kill", "killed", "blood",
                        "wound", "struck", "hit", "attack", "battle",
                        "slaughter", "murder", "scalp"],
        "grc_search": ["μάχη", "φόνος", "αἷμα", "τραῦμα", "πληγή",
                        "σφαγή", "ξίφος", "μάχαιρα", "πόλεμος", "ἀναιρέω"],
        "label": "fighting and violence",
    },
    "violence_weapons": {
        "en_triggers": ["knife", "blade", "pistol", "gun", "rifle", "arrow",
                        "sword", "lance", "spear", "club"],
        "grc_search": ["μάχαιρα", "ξίφος", "φάσγανον", "δόρυ", "λόγχη",
                        "τόξον", "βέλος", "ῥόπαλον", "σάρισσα"],
        "label": "weapons",
    },
    "horses_riding": {
        "en_triggers": ["horse", "horses", "rode", "riding", "saddle",
                        "stirrup", "rein", "mount", "mule", "donkey"],
        "grc_search": ["ἵππος", "ἱππεύω", "ἐλαύνω", "ἡνίοχος", "ἡμίονος",
                        "ὄνος", "ἔφιππος", "πῶλος"],
        "label": "horses and riding",
    },
    "drinking": {
        "en_triggers": ["drunk", "drink", "drank", "whiskey", "saloon",
                        "tavern", "bottle", "bottles", "wine", "liquor"],
        "grc_search": ["μέθη", "μεθύω", "πίνω", "οἶνος", "κύπελλον",
                        "κρατήρ", "ποτήριον", "καπηλεῖον", "κῶμος"],
        "label": "drinking",
    },
    "religion": {
        "en_triggers": ["god", "lord", "reverend", "preacher", "sermon",
                        "church", "bible", "prayer", "devil", "soul",
                        "hell", "heaven", "sin", "damn", "holy", "revival"],
        "grc_search": ["θεός", "κύριος", "ἱερεύς", "διάβολος", "ψυχή",
                        "ἁμαρτία", "γέεννα", "προσευχή", "εὐσέβεια"],
        "label": "religion and the sacred",
    },
    "death_corpses": {
        "en_triggers": ["dead", "death", "corpse", "body", "bones", "skull",
                        "carcass", "carrion", "vulture", "rot"],
        "grc_search": ["νεκρός", "θάνατος", "πτῶμα", "ὀστέον", "κρανίον",
                        "σῆψις", "γύψ", "σκελετός"],
        "label": "death and corpses",
    },
    "fire": {
        "en_triggers": ["fire", "flame", "burn", "burning", "smoke", "ash",
                        "ember", "charcoal", "blaze", "conflagration"],
        "grc_search": ["πῦρ", "φλόξ", "καίω", "κατακαίω", "καπνός",
                        "τέφρα", "ἄνθραξ", "πυρκαϊά", "ἐμπίπρημι"],
        "label": "fire and burning",
    },
    "money_trade": {
        "en_triggers": ["dollar", "dollars", "money", "coin", "pay", "wage",
                        "earn", "sell", "sold", "bought", "price", "trade"],
        "grc_search": ["ἀργύριον", "χρῆμα", "νόμισμα", "μισθός", "ὠνέομαι",
                        "πωλέω", "ἐμπορία", "κέρδος"],
        "label": "money and trade",
    },
    "clothing": {
        "en_triggers": ["hat", "shirt", "coat", "boots", "cloak", "dress",
                        "wore", "wearing", "leather", "buckskin", "cotton"],
        "grc_search": ["χιτών", "ἱμάτιον", "χλαμύς", "χλαῖνα", "πῖλος",
                        "ἐμβάς", "δέρμα", "ἐσθής", "στολή"],
        "label": "clothing and dress",
    },
    "philosophy_nature": {
        "en_triggers": ["nature", "existence", "truth", "reason", "creation",
                        "suzerain", "autonomous", "will", "mystery"],
        "grc_search": ["φύσις", "ὕπαρξις", "ἀλήθεια", "λόγος", "κτίσις",
                        "αὐτονομία", "βούλησις", "μυστήριον"],
        "label": "philosophy and nature",
    },
}


def detect_themes(en_text: str) -> list[str]:
    """Detect which themes are present in an English passage."""
    en_lower = en_text.lower()
    active = []
    for theme_id, spec in THEMES.items():
        hits = sum(1 for t in spec["en_triggers"] if t in en_lower)
        if hits >= 2 or (hits >= 1 and len(spec["en_triggers"]) <= 5):
            active.append(theme_id)
    return active


# ====================================================================
# Greek corpus search
# ====================================================================

def search_corpus(theme_ids: list[str], max_per_theme: int = 15,
                  max_per_author: int = 3) -> dict[str, list[dict]]:
    """Search the full 809K corpus for passages matching the given themes.

    Returns dict of theme_id → list of {author, work, period, greek, reference}.
    Source diversity enforced: max_per_author passages per author per theme.
    """
    if not DB_PATH.exists():
        return {}

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    results = {}

    for theme_id in theme_ids:
        spec = THEMES[theme_id]
        grc_terms = spec["grc_search"]

        # Build OR query across all Greek search terms
        conditions = " OR ".join(
            f"p.greek_text LIKE '%{term}%'" for term in grc_terms
        )

        cur.execute(f"""
            SELECT a.name, d.title, d.period, p.greek_text, p.reference
            FROM passages p
            JOIN documents d ON p.document_id = d.document_id
            JOIN authors a ON d.author_id = a.author_id
            WHERE ({conditions})
              AND LENGTH(p.greek_text) BETWEEN 40 AND 300
            ORDER BY RANDOM()
            LIMIT {max_per_theme * 5}
        """)

        # Enforce author diversity
        author_counts = Counter()
        theme_results = []
        for author, title, period, greek, ref in cur.fetchall():
            if author_counts[author] >= max_per_author:
                continue
            author_counts[author] += 1
            theme_results.append({
                "author": author,
                "work": title,
                "period": period or "?",
                "greek": greek.strip()[:250],
                "reference": ref or "",
            })
            if len(theme_results) >= max_per_theme:
                break

        results[theme_id] = theme_results

    conn.close()
    return results


# ====================================================================
# Vocabulary extraction
# ====================================================================

def normalize(word: str) -> str:
    d = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in d if unicodedata.category(c) not in ("Mn",))


# Function words to skip
STOP = {
    "και", "δε", "τε", "γαρ", "μεν", "ουν", "αλλα", "αλλ", "ου", "ουκ",
    "ουχ", "μη", "μηδ", "ουδ", "εν", "εις", "εκ", "εξ", "απο", "προς",
    "δια", "κατα", "μετα", "περι", "υπο", "υπερ", "επι", "παρα", "προ",
    "συν", "αντι", "ως", "οτι", "ινα", "οπως", "ωστε", "ειτε", "ειτα",
    "τις", "αυτος", "ουτος", "εκεινος", "οστις", "ειμι", "εστι", "εστιν",
    "ειναι", "ησαν", "εγενετο", "εχω", "εχει", "ειπε", "ειπεν", "λεγει",
    "τον", "την", "τοις", "ταις", "του", "της", "των", "τους", "τας",
    "αυτου", "αυτης", "αυτων", "αυτον", "αυτοις",
    "μεν", "ουν", "δη", "γε", "αν", "αρα", "τοι", "που",
    "ουτω", "ουτως", "ουδε", "μηδε", "παντα", "πασαν", "πασης",
    "ειχε", "ειχεν", "ηλθε", "ηλθεν", "ελαβε", "ελαβεν",
}


def extract_vocabulary(corpus_results: dict[str, list[dict]],
                       max_words_per_theme: int = 15) -> dict[str, list[dict]]:
    """Extract interesting content words from corpus search results.

    Returns dict of theme_id → list of {word, normalized, author, work, period}.
    Words are deduplicated by normalized form within each theme.
    """
    theme_vocab = {}

    for theme_id, passages in corpus_results.items():
        word_sources = defaultdict(list)  # normalized → [(word, author, work, period)]

        for p in passages:
            author = p["author"]
            work = p["work"]
            period = p["period"]
            for tok in p["greek"].split():
                clean = tok.strip(".,·;:!?()[]\"'«»—–·")
                if not clean:
                    continue
                n = normalize(clean)
                if len(n) <= 3 or n in STOP:
                    continue
                # Skip likely proper nouns (capitalized in non-initial position)
                # and Latin/transliterated words
                if re.match(r'^[A-Za-z]', clean):
                    continue
                # Skip Hebrew/Semitic proper nouns common in LXX
                if any(c in clean for c in "ΔΒΖ") and len(clean) > 6:
                    # Heuristic: long words with rare Greek letter combos
                    pass  # keep — could be legitimate Greek
                word_sources[n].append((clean, author, work, period))

        # Rank by frequency (more attestations = more useful) and diversity
        ranked = []
        for n, sources in word_sources.items():
            # Prefer words attested by multiple authors
            authors = set(s[1] for s in sources)
            # Pick the "best" form (most common surface form)
            form_counts = Counter(s[0] for s in sources)
            best_form = form_counts.most_common(1)[0][0]
            # Pick a representative source
            rep = sources[0]
            ranked.append({
                "word": best_form,
                "normalized": n,
                "author": rep[1],
                "work": rep[2],
                "period": rep[3],
                "n_authors": len(authors),
                "n_attestations": len(sources),
            })

        # Sort: multi-author words first, then by attestation count
        ranked.sort(key=lambda x: (-x["n_authors"], -x["n_attestations"]))
        theme_vocab[theme_id] = ranked[:max_words_per_theme]

    return theme_vocab


# ====================================================================
# Format for prompt
# ====================================================================

def format_for_prompt(theme_vocab: dict[str, list[dict]]) -> str:
    """Format thematic vocabulary as a prompt section."""
    if not theme_vocab:
        return ""

    lines = ["## Thematic Vocabulary (attested in the classical corpus)",
             "These words appear in Greek passages on similar themes. Use them",
             "if they fit naturally — they are suggestions, not requirements.\n"]

    for theme_id, words in theme_vocab.items():
        if not words:
            continue
        label = THEMES[theme_id]["label"]
        lines.append(f"### {label}")
        for w in words:
            attr = f"{w['author']}, {w['work']}"
            if w["n_authors"] > 1:
                attr += f" (+{w['n_authors']-1} others)"
            lines.append(f"  {w['word']:25s}  [{attr}]")
        lines.append("")

    return "\n".join(lines)


# ====================================================================
# Public API
# ====================================================================

def get_thematic_vocabulary(en_text: str) -> str:
    """Full pipeline: detect themes → search corpus → extract vocab → format.

    Returns a formatted string ready to insert into a translation prompt.
    """
    themes = detect_themes(en_text)
    if not themes:
        return ""

    corpus = search_corpus(themes)
    vocab = extract_vocabulary(corpus)
    return format_for_prompt(vocab)


# ====================================================================
# CLI
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="")
    parser.add_argument("--passage-id", default=None)
    args = parser.parse_args()

    if args.passage_id:
        p_path = PASSAGES / f"{args.passage_id}.json"
        en_text = json.load(open(p_path)).get("text", "")
    elif args.text:
        en_text = args.text
    else:
        parser.print_help()
        return

    themes = detect_themes(en_text)
    print(f"Detected themes: {themes}\n")

    corpus = search_corpus(themes)
    for tid, passages in corpus.items():
        print(f"\n=== {THEMES[tid]['label']} ({len(passages)} passages) ===")
        for p in passages[:3]:
            print(f"  [{p['author']}, {p['work']} ({p['period']})]")
            print(f"    {p['greek'][:100]}")

    vocab = extract_vocabulary(corpus)
    formatted = format_for_prompt(vocab)
    print(f"\n{'='*60}")
    print(formatted)


if __name__ == "__main__":
    main()
