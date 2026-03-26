#!/usr/bin/env python3
"""
Fetch morphological data and definitions from Wiktionary for Ancient Greek.

For each pilot lemma, retrieves:
1. All inflected forms (from declension/conjugation tables)
2. Definitions (senses listed in Wiktionary)

Uses these to:
- Validate occurrence matching (reject false positives like ψυχρός for ψυχή)
- Enrich the sense inventory with Wiktionary's sense list
"""

import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"

WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"

PILOT_LEMMATA = [
    ("κόσμος", "kosmos"), ("λόγος", "logos"), ("ψυχή", "psyche"),
    ("ἀρετή", "arete"), ("δίκη", "dike"), ("τέχνη", "techne"),
    ("νόμος", "nomos"), ("φύσις", "physis"), ("δαίμων", "daimon"),
    ("σῶμα", "soma"), ("θεός", "theos"), ("χάρις", "charis"),
]


def normalize_greek(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower()


def fetch_wiktionary(word: str) -> dict:
    """Fetch parsed HTML sections for an Ancient Greek word from Wiktionary."""
    encoded = quote(word)

    # Get section list
    url = f"{WIKTIONARY_API}?action=parse&page={encoded}&prop=sections&format=json"
    req = Request(url, headers={"User-Agent": "DiachronicSenseChange/0.1"})
    with urlopen(req) as resp:
        sections = json.loads(resp.read()).get("parse", {}).get("sections", [])

    # Find Ancient Greek noun/verb section and declension
    ag_sections = {}
    in_ag = False
    for s in sections:
        if s["line"] == "Ancient Greek":
            in_ag = True
            continue
        if s["toclevel"] == 1 and s["line"] != "Ancient Greek":
            in_ag = False
        if in_ag:
            ag_sections[s["line"]] = s["index"]

    result = {"word": word, "forms": [], "definitions": [], "pos": ""}

    # Get definitions from Noun/Verb/Adjective section
    for pos in ["Noun", "Verb", "Adjective", "Participle"]:
        if pos in ag_sections:
            result["pos"] = pos
            url = f"{WIKTIONARY_API}?action=parse&page={encoded}&prop=text&format=json&section={ag_sections[pos]}"
            req = Request(url, headers={"User-Agent": "DiachronicSenseChange/0.1"})
            with urlopen(req) as resp:
                html = json.loads(resp.read()).get("parse", {}).get("text", {}).get("*", "")

            # Extract definition list items
            items = re.findall(r'<li>(.*?)</li>', html)
            for item in items:
                clean = re.sub(r'<[^>]+>', '', item).strip()
                # Skip items that are just links or very short
                if len(clean) > 3 and not clean.startswith("Appendix:"):
                    result["definitions"].append(clean)
            break

    # Get declension/conjugation forms
    for decl_name in ["Declension", "Conjugation", "Inflection"]:
        if decl_name in ag_sections:
            url = f"{WIKTIONARY_API}?action=parse&page={encoded}&prop=text&format=json&section={ag_sections[decl_name]}"
            req = Request(url, headers={"User-Agent": "DiachronicSenseChange/0.1"})
            with urlopen(req) as resp:
                html = json.loads(resp.read()).get("parse", {}).get("text", {}).get("*", "")

            # Extract Greek forms from title attributes and lang="grc" spans
            forms_from_titles = re.findall(r'title="([^"]+)"', html)
            forms_from_spans = re.findall(r'lang="grc"[^>]*>([^<]+)<', html)

            all_forms = set()
            for f in forms_from_titles + forms_from_spans:
                # Check if it's actually Greek
                if any("\u0370" <= c <= "\u03FF" or "\u1F00" <= c <= "\u1FFF" for c in f):
                    # Skip articles and common particles
                    if f not in ("ὁ", "ἡ", "τό", "οἱ", "αἱ", "τά", "τοῦ", "τῆς",
                                 "τῷ", "τῇ", "τόν", "τήν", "τούς", "τάς", "τῶν",
                                 "τοῖς", "ταῖς", "τοῖν", "τώ"):
                        all_forms.add(f)

            result["forms"] = sorted(all_forms)
            break

    return result


def build_form_index(lemmata_data: dict) -> dict:
    """
    Build a reverse index: normalized_form -> lemma.
    This allows fast validation of whether a surface form belongs to a lemma.
    """
    index = {}
    for lemma, data in lemmata_data.items():
        for form in data["forms"]:
            norm = normalize_greek(form)
            if norm not in index:
                index[norm] = set()
            index[norm].add(lemma)
        # Also add the lemma itself
        norm_lemma = normalize_greek(lemma)
        if norm_lemma not in index:
            index[norm_lemma] = set()
        index[norm_lemma].add(lemma)
    return index


def validate_occurrences(conn, form_index: dict) -> dict:
    """
    Check existing occurrences against the morphological form index.
    Flag occurrences where the surface form doesn't match any known
    inflection of the assigned lemma.
    """
    rows = conn.execute("""
        SELECT o.occurrence_id, o.lemma, t.surface_form
        FROM occurrences o
        JOIN tokens t ON o.token_id = t.token_id
        WHERE o.lemma IN ({})
    """.format(",".join(f"'{l}'" for l in form_index.values()
                        if isinstance(l, str)) or
               ",".join(f"'{l}'" for l, _ in PILOT_LEMMATA))).fetchall()

    # Actually get all occurrences for pilot lemmata
    rows = conn.execute("""
        SELECT o.occurrence_id, o.lemma, t.surface_form
        FROM occurrences o
        JOIN tokens t ON o.token_id = t.token_id
    """).fetchall()

    valid = 0
    invalid = 0
    invalid_examples = []

    for occ_id, lemma, surface in rows:
        norm_surface = normalize_greek(surface.strip(".,;:·!?\"'()[] "))
        # Check if this normalized form maps to the assigned lemma
        matched_lemmata = form_index.get(norm_surface, set())
        if lemma in matched_lemmata:
            valid += 1
        else:
            # Check with more aggressive normalization (strip final sigmas etc.)
            # σ/ς normalization
            norm2 = norm_surface.replace("ς", "σ")
            matched2 = form_index.get(norm2, set())
            if lemma in matched2:
                valid += 1
            else:
                invalid += 1
                if len(invalid_examples) < 20:
                    invalid_examples.append((occ_id, lemma, surface, norm_surface))

    return {
        "valid": valid,
        "invalid": invalid,
        "total": valid + invalid,
        "examples": invalid_examples,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wiktionary morphology integration")
    parser.add_argument("--fetch", action="store_true", help="Fetch from Wiktionary API")
    parser.add_argument("--validate", action="store_true", help="Validate occurrences")
    parser.add_argument("--enrich", action="store_true", help="Add Wiktionary senses to inventory")
    parser.add_argument("--all", action="store_true", help="Do everything")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent.parent / "config" / "wiktionary_forms.json")
    args = parser.parse_args()

    if args.all:
        args.fetch = args.validate = args.enrich = True

    # Step 1: Fetch from Wiktionary
    if args.fetch:
        print("Fetching morphological data from Wiktionary...", flush=True)
        lemmata_data = {}
        for lemma, translit in PILOT_LEMMATA:
            print(f"  {translit} ({lemma})...", end=" ", flush=True)
            try:
                data = fetch_wiktionary(lemma)
                lemmata_data[lemma] = data
                print(f"{len(data['forms'])} forms, {len(data['definitions'])} definitions")
            except Exception as e:
                print(f"ERROR: {e}")
                lemmata_data[lemma] = {"word": lemma, "forms": [], "definitions": []}
            time.sleep(1)  # Rate limiting

        # Save to JSON
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(lemmata_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")

    else:
        # Load existing
        if args.output.exists():
            with open(args.output) as f:
                lemmata_data = json.load(f)
        else:
            print("No cached data. Run with --fetch first.")
            return

    # Step 2: Validate occurrences
    if args.validate:
        print("\nValidating occurrences against morphological forms...", flush=True)
        form_index = build_form_index(lemmata_data)
        print(f"  Form index: {len(form_index)} unique normalized forms")

        conn = sqlite3.connect(DB_PATH)
        result = validate_occurrences(conn, form_index)
        conn.close()

        print(f"\n  Valid: {result['valid']:,} ({result['valid']*100//result['total']}%)")
        print(f"  Invalid: {result['invalid']:,} ({result['invalid']*100//result['total']}%)")
        print(f"\n  Sample invalid matches:")
        for occ_id, lemma, surface, norm in result["examples"]:
            print(f"    occ {occ_id}: {lemma} matched '{surface}' (norm: {norm})")

    # Step 3: Enrich sense inventory
    if args.enrich:
        print("\nEnriching sense inventory with Wiktionary definitions...", flush=True)
        conn = sqlite3.connect(DB_PATH)

        added = 0
        for lemma, data in lemmata_data.items():
            for defn in data.get("definitions", []):
                # Check if a similar sense already exists
                existing = conn.execute(
                    "SELECT COUNT(*) FROM sense_inventory WHERE lemma = ? AND sense_label = ?",
                    (lemma, defn[:100]),
                ).fetchone()[0]
                if not existing:
                    conn.execute(
                        """INSERT INTO sense_inventory (lemma, sense_label, sense_description, notes)
                           VALUES (?, ?, ?, 'from_wiktionary')""",
                        (lemma, defn[:100], defn, ),
                    )
                    added += 1

        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM sense_inventory").fetchone()[0]
        print(f"  Added {added} senses from Wiktionary")
        print(f"  Total sense inventory: {total}")
        conn.close()


if __name__ == "__main__":
    main()
