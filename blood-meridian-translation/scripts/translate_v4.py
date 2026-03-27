#!/usr/bin/env python3
"""
V4 translation pipeline: Opus translator ↔ Opus editor conversation.

Step 1: TRANSLATE — Opus produces first draft (bare prompt + glossary + thematic vocab)
Step 2: EDIT — Opus-as-editor reviews: praises strengths, flags problems,
        suggests classical vocabulary from our corpus
Step 3: REVISE — Opus-as-translator revises in light of editorial feedback,
        preserving praised elements
Step 4: MECHANICAL CHECK — cheap safety net for anything both missed
Step 5: ECHOES — corpus-verified allusion detection

The key insight: the editor and translator are a conversation, not a
one-shot pipeline. The editor's praise protects good choices from
being destroyed in revision.

Usage:
  python3 scripts/translate_v4.py 001_see_the_child
  python3 scripts/translate_v4.py --all
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


# ====================================================================
# Shared: glossary + thematic vocab
# ====================================================================

def load_living_glossary(en_text: str = "") -> str:
    """Load standardised names and locked translation decisions."""
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
            keywords = [en_term.lower()] + key.replace("_", " ").lower().split()
            if any(kw in en_lower for kw in keywords if len(kw) > 2):
                lines.append(f"  {en_term} → {ag}")
    if not lines:
        return ""
    return "## Translation Decisions (for consistency)\n" + "\n".join(lines)


def load_thematic_vocab(en_text: str) -> str:
    """Load thematic vocabulary from the 809K corpus."""
    try:
        from thematic_vocab import get_thematic_vocabulary
        return get_thematic_vocabulary(en_text)
    except Exception:
        return ""


def load_previous_greek(passage_id: str) -> str:
    """Load the Greek from the immediately preceding passage for style continuity."""
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
        prev_greek = prev_path.read_text("utf-8").strip()
        if prev_greek:
            return (
                "## Previous passage (for style continuity — match this register and tone)\n"
                + prev_greek
            )
    return ""


# ====================================================================
# Step 1: TRANSLATE (Opus as translator)
# ====================================================================

TRANSLATE_PROMPT = """You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine register with Attic vocabulary, polytonic orthography).

McCarthy's prose is paratactic, asyndetic, and fragment-heavy. Preserve his voice:
- "and...and...and" coordination chains → preserve as καί chains. Do NOT subordinate.
- Comma splices → asyndeton. But use particles (δέ, γάρ, οὖν) where Greek genuinely needs them.
- Fragments remain fragments.
- Direct speech preserves its structure and line breaks.
- Neuter plural subjects take SINGULAR verbs (Attic rule).

Draw on whatever classical register suits the context naturally.
Every word must be attestable in Morpheus/LSJ. Mark modern coinages with *.

{glossary}

{thematic_vocab}

{previous_greek}

## English Source
{english}

Output ONLY the Greek text, matching McCarthy's paragraph formatting."""


# ====================================================================
# Step 2: EDIT (Opus as scholarly editor)
# ====================================================================

EDIT_PROMPT = """You are a scholarly editor reviewing an Ancient Greek translation of Cormac McCarthy's Blood Meridian. You are working with a talented translator — your job is to help them produce the best possible Greek, not to rewrite from scratch.

Your review has TWO equally important parts:

## Part 1: PRAISE — What MUST be preserved
List specific Greek words and phrases that are excellent and MUST survive any revision. For each one, quote the EXACT GREEK TEXT and explain why it works. These form a protection list — the translator may not remove or change any of these in revision.

Pay special attention to:
- Words that echo the Septuagint, Homer, or other classical sources (e.g. ξυλοκόποι echoing LXX Joshua)
- The opening word/construction — if it sets the right tone, name it
- Relative clauses that preserve McCarthy's sentence structure
- Biblical/Homeric register choices that suit the context

## Part 2: CRITIQUE — What needs fixing
For each problem, give:
- The EXACT Greek text that is wrong
- What's wrong (wrong sense, calque, lost construction, register mismatch, unattested word)
- Your suggested fix, with a classical attestation if you can cite one
- Check: is the grammar correct? Compound subjects need plural verbs. Neuter plurals need singular verbs (Attic rule) BUT only when the subject is purely neuter — mixed-gender compound subjects take plural.

Focus on: wrong word senses, destroyed McCarthy constructions (especially relative clauses converted to separate sentences, lost coordination chains, expanded fragments), register mismatches, unattested vocabulary, and agreement errors.

{corpus_evidence}

## English Source
{english}

## Greek Translation to Review
{greek}

Return a JSON object:
{{
  "praise": ["specific praise 1", "specific praise 2", ...],
  "fixes": [{{"greek": "problem text", "problem": "what's wrong", "fix": "suggestion"}}]
}}"""


# ====================================================================
# Step 3: REVISE (Opus as translator, responding to editor)
# ====================================================================

REVISE_PROMPT = """You are revising your Ancient Greek translation of McCarthy's Blood Meridian, responding to your editor's feedback.

## English Source
{english}

## Your Current Translation
{greek}

## PROTECTED — these exact Greek words/phrases MUST appear in your revision:
{protected_words}

## Editor's Full Praise (context for the protection list):
{praise}

## Editor's Corrections (FIX these, but never at the cost of a protected element):
{fixes}

Revise the translation. Your revision MUST contain every word listed in the PROTECTED section. Fix the listed problems without removing or altering any protected element.

Output ONLY the complete revised Greek text."""


# ====================================================================
# LLM
# ====================================================================

def call_opus(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    collected = []
    with client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)
    return "".join(collected).strip()


# ====================================================================
# Full pipeline
# ====================================================================

def translate_passage(passage_id: str, dry_run: bool = False) -> dict | None:
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

    # === Step 1: TRANSLATE ===
    print(f"\n  [{passage_id}] Step 1: TRANSLATE")
    glossary = load_living_glossary(en_text)
    thematic = load_thematic_vocab(en_text)
    previous = load_previous_greek(passage_id)
    prompt = TRANSLATE_PROMPT.format(
        glossary=glossary, thematic_vocab=thematic,
        previous_greek=previous, english=en_text,
    )
    print(f"    Prompt: {len(prompt)} chars")

    if dry_run:
        print(f"    [DRY RUN]")
        return None

    t0 = time.time()
    greek = call_opus(prompt)
    print(f"    Done in {time.time()-t0:.0f}s ({len(greek)} chars)")

    # Save stage 0
    (draft_path.parent / "stage0_draft.txt").write_text(greek)
    draft_path.write_text(greek + "\n")

    # === Step 2: EDIT ===
    print(f"\n  [{passage_id}] Step 2: OPUS EDITOR")

    # Build corpus evidence for the editor: thematic vocab + real echo candidates
    corpus_evidence_parts = []
    if thematic:
        corpus_evidence_parts.append(
            f"## Classical vocabulary on similar themes\n{thematic}")

    # Flag locked glossary terms that appear in the translation
    if glossary:
        corpus_evidence_parts.append(
            "## Locked translation decisions (these are DELIBERATE choices)\n"
            "If these words appear in the translation, they were chosen for\n"
            "consistency across the book. Do NOT suggest alternatives.\n"
            + glossary
        )

    # Run echo search so editor can reference real corpus matches
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
                echo_lines = []
                for c in candidates[:15]:
                    echo_lines.append(
                        f"  [{c['match_type']}] {c['author']}, {c['work']}: "
                        f"{c['greek'][:100]}"
                    )
                corpus_evidence_parts.append(
                    "## Corpus passages sharing vocabulary with this translation\n"
                    "Use these to identify genuine echoes worth praising, and to\n"
                    "suggest attested alternatives for problematic words.\n\n"
                    + "\n".join(echo_lines)
                )
    except Exception:
        pass

    corpus_evidence = "\n\n".join(corpus_evidence_parts)

    edit_prompt = EDIT_PROMPT.format(
        english=en_text, greek=greek, corpus_evidence=corpus_evidence
    )

    t0 = time.time()
    raw = call_opus(edit_prompt)
    print(f"    Review in {time.time()-t0:.0f}s")

    try:
        clean = re.sub(r'^```json\s*', '', raw)
        clean = re.sub(r'\s*```$', '', clean)
        review = json.loads(clean)
    except json.JSONDecodeError:
        print(f"    Parse error — skipping revision")
        review = {"praise": [], "fixes": []}

    praise = review.get("praise", [])
    fixes = review.get("fixes", [])
    print(f"    Praise: {len(praise)} | Fixes: {len(fixes)}")

    for p in praise:
        print(f"      ✓ {p[:70]}")
    for f in fixes:
        print(f"      ✗ {f.get('greek','')[:25]}: {f.get('problem','')[:45]}")

    # Save review
    (draft_path.parent / "stage1_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2))

    # Extract specific Greek words/phrases from praise for protection list
    protected = []
    for p in praise:
        match = re.match(r'^([^\-—]+?)(?:\s*[-—])', p)
        if match:
            word = match.group(1).strip()
            if re.search(r'[\u0370-\u03FF\u1F00-\u1FFF]', word):
                protected.append(word)
    protected_text = "\n".join(f"  • {w}" for w in protected) if protected else "  (none specified)"

    # === Step 3: REVISE ===
    if fixes:
        print(f"\n  [{passage_id}] Step 3: REVISE")
        praise_text = "\n".join(f"  ✓ {p}" for p in praise)
        fixes_text = "\n".join(
            f"  ✗ {f.get('greek','')}: {f.get('problem','')}\n    → {f.get('fix','')}"
            for f in fixes
        )
        revise_prompt = REVISE_PROMPT.format(
            english=en_text, greek=greek,
            protected_words=protected_text,
            praise=praise_text, fixes=fixes_text,
        )
        t0 = time.time()
        revised = call_opus(revise_prompt)
        print(f"    Revised in {time.time()-t0:.0f}s")

        # Sanity: revision shouldn't be wildly different length
        if len(revised) > len(greek) * 0.5 and len(revised) < len(greek) * 1.5:
            greek = revised
            draft_path.write_text(greek + "\n")
        else:
            print(f"    ⚠ Revision length suspicious ({len(revised)} vs {len(greek)}), keeping original")

        (draft_path.parent / "stage2_revised.txt").write_text(greek)
    else:
        print(f"\n  [{passage_id}] No fixes needed — draft is clean")

    # === Step 4: MECHANICAL CHECK ===
    print(f"\n  [{passage_id}] Step 4: MECHANICAL CHECK")
    try:
        from translate_v3 import run_mechanical_sanity
        mech = run_mechanical_sanity(passage_id, en_text, greek)
        if mech:
            print(f"    {len(mech)} issues")
            for m in mech:
                print(f"      {m.get('greek','')[:25]}: {m.get('problem','')[:50]}")
            # One more revision if needed
            mech_fixes = "\n".join(
                f"  ✗ {m.get('greek','')}: {m.get('problem','')}\n    → {m.get('fix','')}"
                for m in mech
            )
            mech_prompt = REVISE_PROMPT.format(
                english=en_text, greek=greek,
                protected_words=protected_text if 'protected_text' in dir() else "  (preserve all praised elements)",
                praise="\n".join(f"  ✓ {p}" for p in praise),
                fixes=mech_fixes,
            )
            greek = call_opus(mech_prompt)
            draft_path.write_text(greek + "\n")
        else:
            print(f"    Clean")
    except Exception as e:
        print(f"    Mechanical check error: {e}")

    # === Step 5: ECHOES ===
    print(f"\n  [{passage_id}] Step 5: ECHOES")
    try:
        from find_echoes_v2 import find_echoes_for_passage
        echoes = find_echoes_for_passage(passage_id)
        print(f"    {len(echoes)} verified echoes")
        for e in echoes:
            print(f"      {e.get('greek','')[:25]} ← {e.get('source','')}")
    except Exception as e:
        print(f"    Echo error: {e}")
        echoes = []

    return {
        "passage_id": passage_id,
        "greek": greek,
        "praise": praise,
        "fixes": fixes,
        "echoes": echoes,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
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
    print(f"║  V4: Opus translator ↔ Opus editor                      ║")
    print(f"╚{'═'*60}╝")

    start = time.time()
    for pid in passage_ids:
        translate_passage(pid, dry_run=args.dry_run)

    print(f"\n{'═'*60}")
    print(f"  Done in {time.time()-start:.0f}s")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
