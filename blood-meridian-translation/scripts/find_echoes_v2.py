#!/usr/bin/env python3
"""
V2 echo detection: corpus-first, then Sonnet curates.

1. Extract rare/distinctive words and phrases from the Greek translation
2. Search the 809K-passage corpus for exact matches
3. Also search by embedding similarity for thematic matches
4. Present candidates to Sonnet: "which of these are genuine literary echoes?"
5. Hard-verify Sonnet's selections against the database

Usage:
  python3 scripts/find_echoes_v2.py 001_see_the_child
  python3 scripts/find_echoes_v2.py --all
"""

import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DRAFTS = ROOT / "drafts"
APPARATUS = ROOT / "apparatus"
DB_PATH = ROOT.parent / "db" / "diachronic.db"

sys.path.insert(0, str(SCRIPTS))


def normalize(text: str) -> str:
    d = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in d if unicodedata.category(c) not in ("Mn",))


# Common words to skip when searching for distinctive vocabulary
STOP_NORMS = {
    "και", "δε", "τε", "γαρ", "μεν", "ουν", "αλλα", "ου", "ουκ", "ουχ",
    "μη", "εν", "εις", "εκ", "εξ", "απο", "προς", "δια", "κατα", "μετα",
    "περι", "υπο", "υπερ", "επι", "παρα", "προ", "συν", "αντι",
    "ως", "οτι", "ινα", "τις", "αυτος", "ουτος", "εκεινος",
    "ειμι", "εστι", "εστιν", "ειναι", "ησαν", "ην",
    "εχω", "εχει", "ειπε", "ειπεν", "λεγει", "εφη",
    "τον", "την", "τοις", "ταις", "του", "της", "των", "τους", "τας",
    "αυτου", "αυτης", "αυτων", "αυτον", "αυτοις",
    "ουτω", "ουτως", "ουδε", "μηδε", "πας", "πασα", "παν",
    "ουδεις", "τοτε", "νυν", "ηδη", "ετι", "παλιν",
}


def extract_distinctive_words(greek: str) -> list[str]:
    """Extract words that are rare enough to be worth searching the corpus for."""
    tokens = re.findall(r'[\w\u0370-\u03FF\u1F00-\u1FFF]+', greek)
    words = []
    seen = set()
    for tok in tokens:
        n = normalize(tok)
        if len(n) <= 3 or n in STOP_NORMS or n in seen:
            continue
        seen.add(n)
        words.append(tok)
    return words


def extract_phrases(greek: str) -> list[str]:
    """Extract 2-3 word phrases that might be distinctive collocations."""
    tokens = re.findall(r'[\w\u0370-\u03FF\u1F00-\u1FFF]+', greek)
    phrases = []
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if len(bigram) > 10:  # skip very short
            phrases.append(bigram)
    return phrases


def search_corpus_for_word(word: str, cur, max_results: int = 3) -> list[dict]:
    """Search the corpus for passages containing this word."""
    cur.execute("""
        SELECT a.name, d.title, d.period, p.reference,
               SUBSTR(p.greek_text, 1, 200)
        FROM passages p
        JOIN documents d ON p.document_id = d.document_id
        JOIN authors a ON d.author_id = a.author_id
        WHERE p.greek_text LIKE ?
          AND LENGTH(p.greek_text) BETWEEN 20 AND 300
        ORDER BY RANDOM()
        LIMIT ?
    """, (f"%{word}%", max_results))
    return [
        {"author": r[0], "work": r[1], "period": r[2],
         "reference": r[3] or "", "greek": r[4].strip()}
        for r in cur.fetchall()
    ]


def search_corpus_for_phrase(phrase: str, cur, max_results: int = 3) -> list[dict]:
    """Search for an exact phrase match — much more distinctive than single words."""
    cur.execute("""
        SELECT a.name, d.title, d.period, p.reference,
               SUBSTR(p.greek_text, 1, 200)
        FROM passages p
        JOIN documents d ON p.document_id = d.document_id
        JOIN authors a ON d.author_id = a.author_id
        WHERE p.greek_text LIKE ?
          AND LENGTH(p.greek_text) BETWEEN 20 AND 300
        LIMIT ?
    """, (f"%{phrase}%", max_results))
    return [
        {"author": r[0], "work": r[1], "period": r[2],
         "reference": r[3] or "", "greek": r[4].strip()}
        for r in cur.fetchall()
    ]


def search_english_alignments(en_text: str, cur, max_results: int = 5) -> list[dict]:
    """Search the English translations for phrases from McCarthy's source.

    This catches biblical and literary allusions that the translator
    deliberately echoed — e.g. "hewers of wood" → Joshua 9:21.
    """
    # Extract distinctive English phrases (3+ words, skip very common)
    phrases = []
    sents = [s.strip() for s in re.split(r'[.!?]', en_text) if s.strip()]
    for sent in sents:
        words = sent.split()
        # Bigrams and trigrams
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i+3]).lower()
            # Skip very common trigrams
            if not any(w in trigram for w in ["the", "and the", "of the", "in the", "he was"]):
                phrases.append(trigram)
        for i in range(len(words) - 1):
            bigram = " ".join(words[i:i+2]).lower()
            if len(bigram) > 8:
                phrases.append(bigram)

    results = []
    seen_keys = set()
    for phrase in phrases[:30]:  # limit queries
        cur.execute("""
            SELECT a.name, d.title, d.period, p.reference,
                   SUBSTR(p.greek_text, 1, 200),
                   SUBSTR(al.aligned_text, 1, 200)
            FROM alignments al
            JOIN passages p ON al.passage_id = p.passage_id
            JOIN documents d ON p.document_id = d.document_id
            JOIN authors a ON d.author_id = a.author_id
            WHERE LOWER(al.aligned_text) LIKE ?
              AND LENGTH(p.greek_text) BETWEEN 20 AND 300
            LIMIT 2
        """, (f"%{phrase}%",))
        for r in cur.fetchall():
            key = (r[0], r[4][:50])
            if key not in seen_keys:
                seen_keys.add(key)
                results.append({
                    "author": r[0], "work": r[1], "period": r[2],
                    "reference": r[3] or "", "greek": r[4].strip(),
                    "english": r[5].strip() if r[5] else "",
                    "match_type": "english",
                    "match_term": phrase,
                })
        if len(results) >= max_results:
            break
    return results


def search_greek_reverse(greek: str, cur, max_results: int = 5) -> list[dict]:
    """Search for distinctive Greek phrases from our translation in the corpus.

    This catches cases where the translator used a phrase that exists
    verbatim in a classical text — the strongest type of echo.
    """
    # Extract 3-word Greek sequences
    tokens = re.findall(r'[\w\u0370-\u03FF\u1F00-\u1FFF]+', greek)
    results = []
    seen_keys = set()

    for i in range(len(tokens) - 2):
        trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        # Skip if all three words are very common
        norms = [normalize(t) for t in tokens[i:i+3]]
        if all(n in STOP_NORMS or len(n) <= 3 for n in norms):
            continue
        # At least one word should be distinctive
        if not any(len(n) > 4 and n not in STOP_NORMS for n in norms):
            continue

        cur.execute("""
            SELECT a.name, d.title, d.period, p.reference,
                   SUBSTR(p.greek_text, 1, 200)
            FROM passages p
            JOIN documents d ON p.document_id = d.document_id
            JOIN authors a ON d.author_id = a.author_id
            WHERE p.greek_text LIKE ?
              AND LENGTH(p.greek_text) BETWEEN 20 AND 300
            LIMIT 2
        """, (f"%{trigram}%",))
        for r in cur.fetchall():
            key = (r[0], r[4][:50])
            if key not in seen_keys:
                seen_keys.add(key)
                results.append({
                    "author": r[0], "work": r[1], "period": r[2],
                    "reference": r[3] or "", "greek": r[4].strip(),
                    "match_type": "greek_trigram",
                    "match_term": trigram,
                })
        if len(results) >= max_results:
            break
    return results


def find_corpus_candidates(greek: str, cur, en_text: str = "") -> list[dict]:
    """Find all corpus passages that share vocabulary or phrases with the translation."""
    candidates = []
    seen_keys = set()

    # Phase 1: exact phrase matches (most valuable)
    phrases = extract_phrases(greek)
    for phrase in phrases:
        hits = search_corpus_for_phrase(phrase, cur, max_results=2)
        for hit in hits:
            key = (hit["author"], hit["greek"][:50])
            if key not in seen_keys:
                seen_keys.add(key)
                hit["match_type"] = "phrase"
                hit["match_term"] = phrase
                candidates.append(hit)

    # Phase 2: Greek trigram reverse search (verbatim phrases from our text in corpus)
    trigram_hits = search_greek_reverse(greek, cur, max_results=5)
    for hit in trigram_hits:
        key = (hit["author"], hit["greek"][:50])
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(hit)

    # Phase 3: English-side alignment search (catches biblical/literary allusions)
    if en_text:
        en_hits = search_english_alignments(en_text, cur, max_results=5)
        for hit in en_hits:
            key = (hit["author"], hit["greek"][:50])
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append(hit)

    # Phase 4: distinctive single words (lowest priority, fill remaining slots)
    words = extract_distinctive_words(greek)
    for word in words[:15]:
        if len(candidates) >= 40:
            break
        hits = search_corpus_for_word(word, cur, max_results=1)
        for hit in hits:
            key = (hit["author"], hit["greek"][:50])
            if key not in seen_keys:
                seen_keys.add(key)
                hit["match_type"] = "word"
                hit["match_term"] = word
                candidates.append(hit)

    return candidates


CURATE_PROMPT = """You are a classicist preparing a scholarly apparatus for a Greek translation of Cormac McCarthy's Blood Meridian. Below is the Greek translation, followed by real passages from a classical corpus that share vocabulary or phrases with it.

Every candidate below is a VERIFIED match from our database — the Greek text really does appear in that source. Your task: select which matches are worth noting in a scholarly apparatus. Include a match if:

- A multi-word phrase appears verbatim in a classical text (ALWAYS include these — they are direct verbal echoes)
- The shared vocabulary creates a meaningful intertextual resonance (e.g. biblical language in a context of sin/violence, Homeric language in a context of journey/combat)
- The classical passage illuminates the translation's register choice

Reject ONLY matches where a single common word happens to appear in an unrelated context.

Be GENEROUS — a scholarly reader will appreciate even modest connections. If the match type is "greek_trigram" or "phrase", it is almost certainly worth including.

## Our Greek translation
{greek}

## Corpus candidates (all verified from our database)
{candidates}

For each echo worth noting, return:
- greek: the COMPLETE meaningful phrase from OUR translation that echoes the source. This must be a full, self-contained phrase — not a truncated fragment. For example: "ξυλοκόποι καὶ ὑδροφόροι" not "ξυλοκόποι καὶ"; "γραμματοδιδάσκαλος γέγονεν" not "τῇ ἀληθείᾳ ὁ". Quote the whole expression that a reader would recognise as an echo.
- source: Author, Work (and reference from the candidate if given)
- source_quote: copy the relevant Greek FROM THE CORPUS PASSAGE exactly as shown above
- note: brief scholarly note on the connection

Return ONLY a JSON array. Include at least the phrase/trigram matches unless truly trivial.
[{{"greek": "complete phrase from translation", "source": "Author, Work ref", "source_quote": "Greek from corpus", "note": "connection"}}]"""


def curate_echoes(greek: str, candidates: list[dict]) -> list[dict]:
    """Have Sonnet select genuine echoes from corpus candidates."""
    if not candidates:
        return []

    # Format candidates for the prompt
    cand_text = []
    for c in candidates:
        cand_text.append(
            f"  [{c['match_type']} match: '{c['match_term']}'] "
            f"{c['author']}, {c['work']} ({c['period']})"
            f"{' — ' + c['reference'] if c['reference'] else ''}\n"
            f"    {c['greek'][:180]}"
        )

    prompt = CURATE_PROMPT.format(
        greek=greek,
        candidates="\n\n".join(cand_text)
    )

    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    try:
        clean = re.sub(r'^```json\s*', '', raw)
        clean = re.sub(r'\s*```$', '', clean)
        echoes = json.loads(clean)
    except json.JSONDecodeError:
        return []

    # Validate: each greek phrase must appear in our translation
    # and must be a complete phrase (at least 2 words, ending on a word boundary)
    validated = []
    for echo in echoes:
        phrase = echo.get("greek", "").strip()
        if not phrase:
            continue
        # Check it actually appears in our text
        if phrase not in greek:
            # Try to find the closest match — maybe minor whitespace difference
            # If first 10 chars match, accept but truncate to what's in the text
            if phrase[:10] in greek:
                # Find the full extent in our text
                pos = greek.find(phrase[:10])
                # Extend to the end of the last word
                end = pos + len(phrase)
                if end > len(greek):
                    end = len(greek)
                # Snap to word boundary
                while end < len(greek) and greek[end] not in ' .,·;:!?':
                    end += 1
                echo["greek"] = greek[pos:end].strip()
            else:
                continue  # phrase not in our text at all — skip
        validated.append(echo)

    return validated


def hard_verify(echoes: list[dict], cur) -> list[dict]:
    """Final verification: check each source_quote actually exists in the corpus."""
    verified = []
    for echo in echoes:
        quote = echo.get("source_quote", "")
        if not quote or len(quote) < 5:
            continue
        # Check first 20 chars of quote
        cur.execute("""
            SELECT COUNT(*) FROM passages
            WHERE greek_text LIKE ?
        """, (f"%{quote[:25]}%",))
        count = cur.fetchone()[0]
        if count > 0:
            echo["_verified"] = True
            verified.append(echo)
    return verified


def find_echoes_for_passage(passage_id: str) -> list[dict]:
    """Full pipeline: corpus search → Sonnet curate → hard verify."""
    draft_path = DRAFTS / passage_id / "primary.txt"
    if not draft_path.exists():
        return []

    greek = draft_path.read_text("utf-8").strip()

    # Load English source for alignment search
    en_text = ""
    passage_path = ROOT / "passages" / f"{passage_id}.json"
    if passage_path.exists():
        en_text = json.load(open(passage_path)).get("text", "")

    if not DB_PATH.exists():
        print(f"    Database not found")
        return []

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Step 1: find corpus candidates (Greek phrases + English alignments + single words)
    candidates = find_corpus_candidates(greek, cur, en_text)
    print(f"    {len(candidates)} corpus candidates found")

    if not candidates:
        conn.close()
        return []

    # Step 2: Sonnet curates
    echoes = curate_echoes(greek, candidates)
    print(f"    Sonnet selected {len(echoes)} echoes")

    # Step 3: hard verify
    verified = hard_verify(echoes, cur)
    removed = len(echoes) - len(verified)
    if removed:
        print(f"    Removed {removed} unverified after hard check")
    print(f"    {len(verified)} verified echoes")

    conn.close()

    # Save
    echo_path = APPARATUS / passage_id
    echo_path.mkdir(parents=True, exist_ok=True)
    json.dump(verified, open(echo_path / "echoes.json", "w"),
              ensure_ascii=False, indent=2)

    return verified


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
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

    total = 0
    for pid in passage_ids:
        print(f"\n  {pid}:")
        echoes = find_echoes_for_passage(pid)
        for e in echoes:
            print(f"    ✓ {e.get('greek','')[:30]} ← {e.get('source','')}")
        total += len(echoes)

    print(f"\n  Total: {total} verified echoes")


if __name__ == "__main__":
    main()
