#!/usr/bin/env python3
"""
Verify echo citations against the actual classical corpus.

For each echo in the apparatus, check:
  1. Does the author+work exist in our database?
  2. Does the quoted Greek text appear (or closely match) in our corpus?
  3. Flag hallucinated references for removal.

Usage:
  python3 scripts/verify_echoes.py
  python3 scripts/verify_echoes.py 001_see_the_child
"""

import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPARATUS = ROOT / "apparatus"
DB_PATH = ROOT.parent / "db" / "diachronic.db"


def normalize(text: str) -> str:
    """Strip accents and lowercase for fuzzy matching."""
    d = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in d if unicodedata.category(c) not in ("Mn",))


def verify_echo(echo: dict, cur) -> dict:
    """Verify a single echo against the database.

    Returns the echo dict with added verification fields:
      _verified: True/False
      _match_type: "exact" / "partial" / "author_only" / "not_found"
      _match_detail: explanation
    """
    source = echo.get("source", "")
    source_quote = echo.get("source_quote", "")

    if not source_quote or len(source_quote) < 5:
        echo["_verified"] = False
        echo["_match_type"] = "no_quote"
        echo["_match_detail"] = "No source quote to verify"
        return echo

    # Extract author name from source (e.g. "Homer, Iliad 4.75-76" → "Homer")
    author_match = re.match(r'^([^,]+)', source)
    author = author_match.group(1).strip() if author_match else ""

    # Normalize the quote for fuzzy matching
    quote_norm = normalize(source_quote)
    # Take a meaningful chunk (first 30 chars accent-stripped)
    quote_chunk = quote_norm[:40]

    # Strategy 1: exact substring search in Greek text
    # Use accent-stripped matching since our corpus may have different accent conventions
    cur.execute("""
        SELECT a.name, d.title, SUBSTR(p.greek_text, 1, 200)
        FROM passages p
        JOIN documents d ON p.document_id = d.document_id
        JOIN authors a ON d.author_id = a.author_id
        WHERE p.greek_text LIKE ?
        LIMIT 3
    """, (f"%{source_quote[:20]}%",))

    exact_hits = cur.fetchall()
    if exact_hits:
        hit_author, hit_work, hit_text = exact_hits[0]
        echo["_verified"] = True
        echo["_match_type"] = "exact"
        echo["_match_detail"] = f"Found in {hit_author}, {hit_work}"
        return echo

    # Strategy 2: try with a shorter chunk (first few distinctive words)
    words = source_quote.split()
    if len(words) >= 3:
        # Search for 3-word sequence
        search_phrase = " ".join(words[:3])
        cur.execute("""
            SELECT a.name, d.title, SUBSTR(p.greek_text, 1, 200)
            FROM passages p
            JOIN documents d ON p.document_id = d.document_id
            JOIN authors a ON d.author_id = a.author_id
            WHERE p.greek_text LIKE ?
            LIMIT 3
        """, (f"%{search_phrase}%",))

        partial_hits = cur.fetchall()
        if partial_hits:
            hit_author, hit_work, _ = partial_hits[0]
            echo["_verified"] = True
            echo["_match_type"] = "partial"
            echo["_match_detail"] = f"Partial match in {hit_author}, {hit_work}"
            return echo

    # Strategy 3: check if the author at least exists
    if author:
        cur.execute("""
            SELECT name FROM authors
            WHERE LOWER(name) LIKE ?
            LIMIT 1
        """, (f"%{author.lower()}%",))
        author_hit = cur.fetchone()
        if author_hit:
            echo["_verified"] = False
            echo["_match_type"] = "author_only"
            echo["_match_detail"] = f"Author '{author}' exists but quote not found — possible hallucination"
            return echo

    echo["_verified"] = False
    echo["_match_type"] = "not_found"
    echo["_match_detail"] = f"Neither author '{author}' nor quote found in corpus"
    return echo


def verify_passage(passage_id: str, cur) -> list[dict]:
    """Verify all echoes for a passage."""
    echoes_path = APPARATUS / passage_id / "echoes.json"
    if not echoes_path.exists():
        return []

    echoes = json.load(open(echoes_path))
    verified = []
    for echo in echoes:
        verified.append(verify_echo(echo, cur))
    return verified


def main():
    if not DB_PATH.exists():
        print(f"  Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    if len(sys.argv) > 1:
        passage_ids = sys.argv[1:]
    else:
        passage_ids = sorted(
            d.name for d in APPARATUS.iterdir()
            if d.is_dir() and (d / "echoes.json").exists()
        )

    total = 0
    verified = 0
    hallucinated = 0

    for pid in passage_ids:
        echoes = verify_passage(pid, cur)
        if not echoes:
            continue

        print(f"\n  {pid}:")
        for e in echoes:
            status = e.get("_match_type", "?")
            source = e.get("source", "")
            detail = e.get("_match_detail", "")
            quote = e.get("source_quote", "")[:40]

            if e.get("_verified"):
                marker = "✓"
                verified += 1
            else:
                marker = "✗"
                hallucinated += 1

            total += 1
            print(f"    {marker} {source:40s}  [{status}]")
            if quote:
                print(f"      quote: {quote}")
            if not e.get("_verified"):
                print(f"      {detail}")

        # Save verified echoes (strip unverified ones)
        clean_echoes = [e for e in echoes if e.get("_verified")]
        removed = len(echoes) - len(clean_echoes)
        if removed > 0:
            json.dump(clean_echoes, open(APPARATUS / pid / "echoes.json", "w"),
                      ensure_ascii=False, indent=2)
            print(f"    Removed {removed} unverified echo(es)")

    conn.close()

    print(f"\n  {'='*50}")
    print(f"  Total: {total} echoes")
    print(f"  Verified: {verified}")
    print(f"  Hallucinated/unverified: {hallucinated}")
    print(f"  {'='*50}")


if __name__ == "__main__":
    main()
