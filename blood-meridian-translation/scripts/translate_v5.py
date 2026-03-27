#!/usr/bin/env python3
"""
V5: Single Opus conversation — translator and editor in one context window.

The translator writes, then we inject editorial feedback as a user turn,
and the translator revises in the same conversation. The model remembers
its own choices and reasoning. Loop until the editor is satisfied.

Usage:
  python3 scripts/translate_v5.py 001_see_the_child
  python3 scripts/translate_v5.py --all
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PASSAGES = ROOT / "passages"
DRAFTS = ROOT / "drafts"
GLOSSARY = ROOT / "glossary"

sys.path.insert(0, str(SCRIPTS))


def fix_missing_spaces(text: str) -> str:
    """Fix missing spaces between Greek words — an LLM tokenization artifact.

    Inserts a space where a Greek word-ending character is immediately
    followed by an article, preposition, or other function word that
    should start a new word.
    """
    # Pattern: Greek letter followed immediately by a known word-starter
    # that should be a separate word. Only fire when the preceding char
    # is clearly the end of a different word (lowercase letter followed
    # by an uppercase or a known article/preposition pattern).
    import re

    # Fix: lowercase Greek letter directly followed by τό/τήν/τῆς/τοῦ/τῷ etc.
    # These articles almost never appear mid-word
    articles = r'(τό|τήν|τῆς|τοῦ|τῷ|τά|τοί|τῶν|τούς|τάς|τοῖς|ταῖς)'
    text = re.sub(r'([α-ωά-ώϊϋΐΰᾶῆῖῦῶ])(' + articles + r')\b', r'\1 \2', text)

    # Fix: lowercase Greek followed by μετά/ἐν/εἰς/ἐκ/πρός/διά/κατά/παρά/ὑπό/ἐπί
    preps = r'(μετά|μετὰ|ἐν|εἰς|ἐκ|ἐξ|πρός|πρὸς|διά|διὰ|κατά|κατὰ|παρά|παρὰ|ὑπό|ὑπὸ|ἐπί|ἐπὶ|ἀπό|ἀπὸ|σύν|σὺν)'
    text = re.sub(r'([α-ωά-ώϊϋΐΰᾶῆῖῦῶ])(' + preps + r')\s', r'\1 \2 ', text)

    return text


def load_living_glossary(en_text: str = "") -> str:
    glossary_path = GLOSSARY / "idf_glossary.json"
    if not glossary_path.exists():
        return ""
    data = json.load(open(glossary_path))
    en_lower = en_text.lower()
    lines = []
    for category, entries in data.items():
        if category.startswith("_") or not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if not isinstance(entry, dict) or entry.get("status") != "locked":
                continue
            en_term = entry.get("english", "")
            ag = entry.get("ancient_greek", "").replace("*", "")
            justification = entry.get("justification", "")
            keywords = [en_term.lower()] + key.replace("_", " ").lower().split()
            if any(kw in en_lower for kw in keywords if len(kw) > 2):
                line = f"  {en_term} → {ag}"
                if justification:
                    line += f"  ({justification[:120]})"
                lines.append(line)
    if not lines:
        return ""
    return "## Locked Translation Decisions\n" + "\n".join(lines)


def load_thematic_vocab(en_text: str) -> str:
    try:
        from thematic_vocab import get_thematic_vocabulary
        return get_thematic_vocabulary(en_text)
    except Exception:
        return ""


def load_previous_greek(passage_id: str) -> str:
    all_ids = sorted(p.stem for p in PASSAGES.glob("*.json")
                     if not p.stem.startswith("exp_") and not p.stem.startswith("905"))
    try:
        idx = all_ids.index(passage_id)
    except ValueError:
        return ""
    if idx == 0:
        return ""
    prev_id = all_ids[idx - 1]
    prev_path = DRAFTS / prev_id / "primary.txt"
    if prev_path.exists():
        prev = prev_path.read_text("utf-8").strip()
        if prev:
            return f"## Previous passage (match this style)\n{prev}"
    return ""


def load_corpus_evidence(en_text: str, greek: str = "") -> str:
    """Load thematic vocab + corpus echo candidates for the editor."""
    parts = []
    thematic = load_thematic_vocab(en_text)
    if thematic:
        parts.append(thematic)

    if greek:
        try:
            from find_echoes_v2 import find_corpus_candidates
            import sqlite3
            db_path = ROOT.parent / "db" / "diachronic.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cur = conn.cursor()
                candidates = find_corpus_candidates(greek, cur, en_text)
                conn.close()
                if candidates:
                    lines = []
                    for c in candidates[:12]:
                        lines.append(
                            f"  [{c['match_type']}] {c['author']}, {c['work']}: "
                            f"{c['greek'][:100]}"
                        )
                    parts.append(
                        "## Corpus passages sharing vocabulary with the translation\n"
                        + "\n".join(lines)
                    )
        except Exception:
            pass

    return "\n\n".join(parts)


def translate_passage(passage_id: str, max_rounds: int = 5,
                      dry_run: bool = False) -> dict | None:
    p_path = PASSAGES / f"{passage_id}.json"
    if not p_path.exists():
        print(f"  {passage_id}: not found")
        return None
    en_text = json.load(open(p_path)).get("text", "")
    if not en_text:
        return None

    draft_path = DRAFTS / passage_id / "primary.txt"
    if draft_path.exists() and not dry_run:
        existing = draft_path.read_text("utf-8").strip()
        if existing:
            print(f"  {passage_id}: draft exists. Skipping.")
            return None

    draft_path.parent.mkdir(parents=True, exist_ok=True)

    glossary = load_living_glossary(en_text)
    previous = load_previous_greek(passage_id)
    thematic = load_thematic_vocab(en_text)

    # Build the initial system/translate prompt
    translate_msg = f"""You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine register with Attic vocabulary, polytonic orthography).

McCarthy's prose is paratactic, asyndetic, and fragment-heavy. Preserve his voice:
- "and...and...and" coordination chains → preserve as καί chains. Do NOT subordinate.
- Comma splices → asyndeton. But use particles (δέ, γάρ, οὖν) where Greek genuinely needs them.
- Fragments remain fragments. Direct speech preserves its structure.
- Neuter plural subjects take SINGULAR verbs (Attic rule).

Every word must be attestable in Morpheus/LSJ. Mark modern coinages with *.

{glossary}

{thematic}

{previous}

## English Source
{en_text}

Translate this passage into Ancient Greek. Output ONLY the Greek text."""

    # Start the conversation
    import anthropic
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": translate_msg}]

    print(f"\n  [{passage_id}] Round 1: TRANSLATE")
    if dry_run:
        print(f"    [DRY RUN] {len(translate_msg)} chars")
        return None

    t0 = time.time()
    response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        messages=messages,
    )
    greek = response.content[0].text.strip()
    print(f"    {time.time()-t0:.0f}s ({len(greek)} chars)")
    print(f"    {greek[:100]}...")

    # Add translator's response to conversation
    messages.append({"role": "assistant", "content": greek})

    # Save stage 0
    (draft_path.parent / "stage0_draft.txt").write_text(greek)
    draft_path.write_text(greek + "\n")

    # === Editorial loop ===
    for round_num in range(2, max_rounds + 1):
        # Run mechanical checks on current greek BEFORE the editor sees it
        mech_report = ""
        try:
            # Write current draft for mechanical checker
            draft_path.write_text(greek + "\n")

            all_issues = []

            # Grammar checks (agreement, preposition governance, constructions)
            from translate_v3 import run_mechanical_sanity
            mech_issues = run_mechanical_sanity(passage_id, en_text, greek)
            for m in mech_issues:
                all_issues.append(f"  - GRAMMAR: {m.get('greek','')}: {m.get('problem','')} → {m.get('fix','')}")

            # Attestation check: two levels
            # 1. Completely unattested (not even Morpheus) — definitely fabricated
            # 2. Morpheus-only (valid morphology but no corpus/DB attestation) — suspect
            try:
                from morpheus_check import (
                    check_passage, _save_cache, parse_word,
                    _is_attested_in_corpus, _is_attested_in_db,
                    _load_corpus_forms, MORPHEUS_WHITELIST,
                )
                _load_corpus_forms()
                morph = check_passage(passage_id)
                _save_cache()
                for m in morph:
                    if m["type"] == "unattested_word":
                        all_issues.append(
                            f"  - UNATTESTED: {m['word']} — not found in Morpheus, "
                            f"corpus, or database. Fabricated — replace with an attested word."
                        )

                # Check for Morpheus-only words (valid morphology, no ancient usage)
                tokens = re.findall(r'[\w\u0370-\u03FF\u1F00-\u1FFF]+', greek)
                for tok in tokens:
                    if len(tok) <= 3:
                        continue
                    analyses = parse_word(tok)
                    if not analyses:
                        continue  # already caught above
                    # Morpheus knows it — but is it in the corpus or DB?
                    in_corpus = _is_attested_in_corpus(tok)
                    in_db = _is_attested_in_db(tok)
                    if not in_corpus and not in_db and tok not in MORPHEUS_WHITELIST:
                        lemma = analyses[0].get("lemma", tok) if analyses else tok
                        all_issues.append(
                            f"  - MORPHEUS-ONLY: {tok} (lemma: {lemma}) — Morpheus "
                            f"parses this but it appears in neither our 57K corpus "
                            f"nor 839K database. Possibly a valid but unattested form. "
                            f"Prefer a well-attested alternative."
                        )
            except Exception:
                pass

            # Polysemy sense check
            try:
                from translate_v2 import check_polysemy
                poly = check_polysemy(en_text, greek)
                for p in poly:
                    all_issues.append(f"  - WRONG SENSE: {p}")
            except Exception:
                pass

            if all_issues:
                mech_report = (
                    "\n## Automated Check Results\n"
                    "These issues were flagged by our mechanical checkers (grammar, attestation, sense):\n"
                    + "\n".join(all_issues)
                )
                print(f"    Mechanical pre-check: {len(all_issues)} issues")
                for a in all_issues:
                    print(f"      {a[:70]}")
            else:
                print(f"    Mechanical pre-check: clean")
        except Exception as e:
            print(f"    Mechanical pre-check error: {e}")

        # Build editorial feedback with corpus evidence + mechanical results
        corpus = load_corpus_evidence(en_text, greek)

        edit_msg = f"""Now review your translation as a scholarly editor. You have the English source, corpus evidence, and automated grammar check results below.

Give your review in TWO parts:
1. STRENGTHS: List specific Greek words and phrases that are excellent. Quote the exact Greek.
2. PROBLEMS: List specific issues. Check especially:
   - Subject-verb agreement: plural subjects need plural verbs. The neuter plural + singular verb rule ONLY applies to purely neuter plural subjects, NOT to mixed or non-neuter plurals.
   - Morphology: is every word form real? Check participles and contracted verbs carefully — fabricated forms are common.
   - Preposition governance: μετά + genitive, σύν + dative, etc.
   - Word sense: does each Greek word actually mean what the English says?

IMPORTANT: If the automated grammar check flagged issues, you MUST include them in your PROBLEMS list.

If there are NO problems (including no grammar check issues), say "APPROVED" and nothing else.

{mech_report}

{corpus}

## English Source (for reference)
{en_text}

Review your translation above."""

        messages.append({"role": "user", "content": edit_msg})

        print(f"\n  [{passage_id}] Round {round_num}: EDIT")
        t0 = time.time()
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            messages=messages,
        )
        review = response.content[0].text.strip()
        print(f"    {time.time()-t0:.0f}s")
        messages.append({"role": "assistant", "content": review})

        # Save review
        (draft_path.parent / f"stage{round_num-1}_review.txt").write_text(review)

        # Check if approved
        if "APPROVED" in review.upper() and len(review) < 200:
            print(f"    ✓ APPROVED — editor satisfied")
            break

        # Show summary
        lines = review.split("\n")
        for l in lines[:10]:
            if l.strip():
                print(f"    {l[:80]}")

        # Ask for revision
        revise_msg = """Now revise your translation based on your editorial feedback above. Fix every problem you identified while preserving every strength you praised.

Output ONLY the complete revised Greek text."""

        messages.append({"role": "user", "content": revise_msg})

        print(f"\n  [{passage_id}] Round {round_num}: REVISE")
        t0 = time.time()
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            messages=messages,
        )
        greek = response.content[0].text.strip()
        print(f"    {time.time()-t0:.0f}s ({len(greek)} chars)")
        print(f"    {greek[:100]}...")
        messages.append({"role": "assistant", "content": greek})

        # Save
        (draft_path.parent / f"stage{round_num}_revised.txt").write_text(greek)
        draft_path.write_text(greek + "\n")

    # === FINAL MECHANICAL BACKSTOP ===
    print(f"\n  [{passage_id}] FINAL CHECK")
    try:
        draft_path.write_text(greek + "\n")
        from translate_v3 import run_mechanical_sanity
        final_mech = run_mechanical_sanity(passage_id, en_text, greek)
        if final_mech:
            print(f"    ⚠ {len(final_mech)} residual issues (for human review):")
            for m in final_mech:
                print(f"      {m.get('problem','')[:70]}")
        else:
            print(f"    ✓ Clean")
    except Exception as e:
        print(f"    Error: {e}")

    # === ECHOES ===
    print(f"\n  [{passage_id}] ECHOES")
    try:
        from find_echoes_v2 import find_echoes_for_passage
        echoes = find_echoes_for_passage(passage_id)
        print(f"    {len(echoes)} verified")
    except Exception as e:
        print(f"    Error: {e}")
        echoes = []

    return {
        "passage_id": passage_id,
        "greek": greek,
        "rounds": round_num,
        "echoes": echoes,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        passage_ids = sorted(p.stem for p in PASSAGES.glob("*.json")
                            if not p.stem.startswith("exp_") and not p.stem.startswith("905"))
    elif args.passages:
        passage_ids = args.passages
    else:
        parser.print_help()
        return

    print(f"╔{'═'*60}╗")
    print(f"║  V5: Opus conversation (translate ↔ edit ↔ revise)       ║")
    print(f"╚{'═'*60}╝")

    start = time.time()
    for pid in passage_ids:
        translate_passage(pid, max_rounds=args.max_rounds, dry_run=args.dry_run)

    print(f"\n{'═'*60}")
    print(f"  Done in {time.time()-start:.0f}s")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
