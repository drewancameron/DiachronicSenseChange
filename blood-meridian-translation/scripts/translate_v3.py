#!/usr/bin/env python3
"""
V3 translation pipeline: bare translate → Sonnet cascade review → Opus revise.

Step 1: TRANSLATE — minimal prompt (register + living glossary of decisions)
Step 2: SONNET STYLE REVIEW → Opus revise (idiom, calques, over-subordination)
Step 3: SONNET VOCAB REVIEW → Opus revise if needed (sense, attestation, period)
Step 4: UPDATE GLOSSARY — lock any new coinages/decisions for future consistency
Step 5: FIND ECHOES — scan output for classical allusions to note in apparatus

Usage:
  python3 scripts/translate_v3.py 001_see_the_child
  python3 scripts/translate_v3.py --all
  python3 scripts/translate_v3.py 005_saint_louis --dry-run
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
# Living glossary: all locked decisions (names + coinages + periphrases)
# ====================================================================

def load_living_glossary(en_text: str = "") -> str:
    """Load the full living glossary of translation decisions.

    Includes: standardised names, coinages for modern concepts, periphrastic
    solutions, and any other locked vocabulary. These ensure consistency
    across the book. Listed as preferences with justification.
    """
    glossary_path = GLOSSARY / "idf_glossary.json"
    if not glossary_path.exists():
        return ""
    data = json.load(open(glossary_path))

    relevant = []
    all_locked = []
    en_lower = en_text.lower()

    for category, entries in data.items():
        if category.startswith("_") or not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if not isinstance(entry, dict) or entry.get("status") != "locked":
                continue
            en_term = entry.get("english", "")
            ag = entry.get("ancient_greek", "").replace("*", "")
            register = entry.get("register", "")
            justification = entry.get("justification", "")

            line = f"  {en_term} → {ag}"
            if register and register not in ("classical", "koine"):
                line += f"  [{register}]"
            all_locked.append(line)

            # Flag terms relevant to this passage
            keywords = [en_term.lower()] + key.replace("_", " ").lower().split()
            if any(kw in en_lower for kw in keywords if len(kw) > 2):
                detail = f"  {en_term} → {ag}"
                if justification:
                    detail += f"  ({justification[:100]})"
                relevant.append(detail)

    if not all_locked:
        return ""

    sections = ["## Translation Decisions (use these for consistency)"]
    if relevant:
        sections.append("Relevant to this passage:")
        sections.extend(relevant)
        sections.append("")
    sections.append("Full list:")
    sections.extend(all_locked)
    return "\n".join(sections)


# ====================================================================
# Step 1: TRANSLATE — bare prompt
# ====================================================================

TRANSLATE_PROMPT = """You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine register with Attic vocabulary, polytonic orthography).

McCarthy's prose is paratactic, asyndetic, and fragment-heavy. Preserve his voice:
- "and...and...and" coordination chains → preserve as καί chains. Do NOT subordinate.
- Comma splices → asyndeton. But use particles (δέ, γάρ, οὖν) where Greek genuinely needs them for readability.
- Fragments remain fragments. Do not expand.
- Direct speech preserves its structure and line breaks.
- Neuter plural subjects take SINGULAR verbs (Attic rule).

Draw on whatever classical register suits the context naturally: Septuagintal for religious, Platonic for philosophical, Thucydidean for military narration.

Every word must be attestable in Morpheus/LSJ. For modern concepts with no classical equivalent, use a transparent Greek compound or Hellenised transliteration marked with * (e.g. *πιστόλιον).

{glossary}

## English Source
{english}

Output ONLY the Greek text, matching McCarthy's paragraph formatting."""


def build_translate_prompt(en_text: str) -> str:
    glossary = load_living_glossary(en_text)
    return TRANSLATE_PROMPT.format(glossary=glossary, english=en_text)


# ====================================================================
# Step 2: SONNET STYLE REVIEW
# ====================================================================

STYLE_REVIEW_PROMPT = """You are reviewing an Ancient Greek translation of Cormac McCarthy's Blood Meridian. Target register: literary Koine with Attic vocabulary.

Review for these issues ONLY. Do not comment on correct choices.

1. OVER-SUBORDINATION: McCarthy's parataxis collapsed into Greek subordination
2. CALQUES: English idiom wearing Greek inflections — unidiomatic word order or constructions
3. WRONG SENSE: polysemous words where the wrong meaning was chosen
4. REGISTER MISMATCH: vocabulary from the wrong period or genre for the context
5. RELATIVE PRONOUN ERRORS: wrong gender/number (must agree with TRUE antecedent, not nearest noun)
6. LOST FRAGMENTS: McCarthy fragments expanded into full sentences
7. PARTICLE ISSUES: δέ/γάρ missing where Greek needs them, or inserted where McCarthy is asyndetic

## English Source
{english}

## Greek Translation
{greek}

Return ONLY a JSON array:
[{{"greek": "problematic text", "problem": "description", "fix": "suggested replacement"}}]
Empty array [] if no issues."""


def sonnet_style_review(en_text: str, greek: str) -> list[dict]:
    prompt = STYLE_REVIEW_PROMPT.format(english=en_text, greek=greek)
    raw = call_sonnet(prompt)
    try:
        clean = re.sub(r'^```json\s*', '', raw)
        clean = re.sub(r'\s*```$', '', clean)
        return json.loads(clean)
    except json.JSONDecodeError:
        return []


# ====================================================================
# Step 3: SONNET VOCAB REVIEW
# ====================================================================

VOCAB_REVIEW_PROMPT = """You are doing a FINAL vocabulary check on an Ancient Greek translation of Blood Meridian. Focus ONLY on:

1. WRONG SENSE: words where the Greek means something different from the English intent
2. UNATTESTED: words probably not in LSJ (LLM inventions by analogy)
3. ANACHRONISTIC: post-classical vocabulary where classical alternatives exist
4. INCONSISTENCY: words that should use the standardised forms from the glossary below

Do NOT comment on style, register, or construction — those have been reviewed already.

{glossary_note}

## English Source
{english}

## Greek Translation
{greek}

Return ONLY a JSON array:
[{{"greek": "word", "problem": "description", "fix": "suggestion"}}]
Empty array [] if no issues."""


def sonnet_vocab_review(en_text: str, greek: str) -> list[dict]:
    # Include relevant glossary terms for consistency checking
    glossary_path = GLOSSARY / "idf_glossary.json"
    glossary_note = ""
    if glossary_path.exists():
        data = json.load(open(glossary_path))
        en_lower = en_text.lower()
        relevant = []
        for category, entries in data.items():
            if category.startswith("_") or not isinstance(entries, dict):
                continue
            for key, entry in entries.items():
                if not isinstance(entry, dict) or entry.get("status") != "locked":
                    continue
                keywords = [entry.get("english", "").lower()] + key.replace("_", " ").lower().split()
                if any(kw in en_lower for kw in keywords if len(kw) > 2):
                    en_term = entry.get("english", "")
                    ag = entry.get("ancient_greek", "").replace("*", "")
                    relevant.append(f"  {en_term} → {ag}")
        if relevant:
            glossary_note = "## Standardised terms (check consistency):\n" + "\n".join(relevant)

    prompt = VOCAB_REVIEW_PROMPT.format(
        english=en_text, greek=greek, glossary_note=glossary_note
    )
    raw = call_sonnet(prompt)
    try:
        clean = re.sub(r'^```json\s*', '', raw)
        clean = re.sub(r'\s*```$', '', clean)
        return json.loads(clean)
    except json.JSONDecodeError:
        return []


# ====================================================================
# Opus revision from Sonnet findings
# ====================================================================

def build_revision_prompt(en_text: str, greek: str, issues: list[dict],
                          evidence: str = "") -> str | None:
    """Build targeted revision prompt from Sonnet findings."""
    if not issues:
        return None

    issue_lines = []
    for si in issues:
        g = si.get("greek", "")
        p = si.get("problem", "")
        f = si.get("fix", "")
        issue_lines.append(f"  - {g}: {p}")
        if f:
            issue_lines.append(f"    → {f}")

    prompt = f"""You are revising an Ancient Greek translation of McCarthy's Blood Meridian (Koine register, Attic vocabulary).

Fix ONLY the specific issues listed below. Do not change anything that is already correct.

## English Source
{en_text}

## Current Greek Translation
{greek}

## Issues to Fix
{chr(10).join(issue_lines)}

{evidence}

## Instructions
1. Fix each listed issue while preserving everything correct.
2. Every word must be attestable in Morpheus/LSJ.
3. Output ONLY the complete revised Greek text."""

    return prompt


# ====================================================================
# Step 5: FIND ECHOES — detect classical allusions in the output
# ====================================================================

ECHO_PROMPT = """You are a classicist examining an Ancient Greek translation of Cormac McCarthy. Identify any phrases or word choices that echo specific classical texts — whether deliberate allusions or natural affinities.

For each echo, give:
- The Greek phrase from the translation
- The classical source (author, work, approximate reference)
- What the echo evokes

Only note genuine, recognisable echoes — not generic vocabulary.

## Greek Translation
{greek}

Return ONLY a JSON array:
[{{"greek": "phrase", "source": "Author, Work ref", "note": "what it evokes"}}]
Empty array [] if nothing notable."""


def find_echoes(greek: str) -> list[dict]:
    raw = call_sonnet(ECHO_PROMPT.format(greek=greek))
    try:
        clean = re.sub(r'^```json\s*', '', raw)
        clean = re.sub(r'\s*```$', '', clean)
        return json.loads(clean)
    except json.JSONDecodeError:
        return []


# ====================================================================
# LLM calls
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


def call_sonnet(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ====================================================================
# Full pipeline
# ====================================================================

def translate_passage(passage_id: str, dry_run: bool = False) -> dict | None:
    """Full v3 pipeline for one passage."""

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

    # === Step 1: TRANSLATE ===
    print(f"\n  [{passage_id}] Step 1: TRANSLATE")
    prompt = build_translate_prompt(en_text)
    print(f"    Prompt: {len(prompt)} chars")

    if dry_run:
        print(f"    [DRY RUN]\n{prompt[:500]}...")
        return None

    t0 = time.time()
    greek = call_opus(prompt)
    print(f"    Done in {time.time()-t0:.0f}s ({len(greek)} chars)")
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(greek + "\n", encoding="utf-8")

    # === Step 2: SONNET STYLE REVIEW → Opus revise ===
    print(f"\n  [{passage_id}] Step 2: SONNET STYLE REVIEW")
    t0 = time.time()
    style_issues = sonnet_style_review(en_text, greek)
    print(f"    {len(style_issues)} issues ({time.time()-t0:.0f}s)")
    for si in style_issues:
        print(f"      {si.get('greek','')[:30]}: {si.get('problem','')[:60]}")

    if style_issues:
        rev = build_revision_prompt(en_text, greek, style_issues)
        if rev:
            t0 = time.time()
            greek = call_opus(rev)
            print(f"    Revised in {time.time()-t0:.0f}s")
            draft_path.write_text(greek + "\n", encoding="utf-8")

    # === Step 3: SONNET VOCAB REVIEW → Opus revise if needed ===
    print(f"\n  [{passage_id}] Step 3: SONNET VOCAB REVIEW")
    t0 = time.time()
    vocab_issues = sonnet_vocab_review(en_text, greek)
    print(f"    {len(vocab_issues)} issues ({time.time()-t0:.0f}s)")
    for si in vocab_issues:
        print(f"      {si.get('greek','')[:30]}: {si.get('problem','')[:60]}")

    if vocab_issues:
        rev = build_revision_prompt(en_text, greek, vocab_issues)
        if rev:
            t0 = time.time()
            greek = call_opus(rev)
            print(f"    Revised in {time.time()-t0:.0f}s")
            draft_path.write_text(greek + "\n", encoding="utf-8")

    # === Step 4: FIND ECHOES ===
    print(f"\n  [{passage_id}] Step 4: FIND ECHOES")
    t0 = time.time()
    echoes = find_echoes(greek)
    print(f"    {len(echoes)} echoes ({time.time()-t0:.0f}s)")
    for e in echoes:
        print(f"      {e.get('greek','')[:30]} ← {e.get('source','')}")

    # Save echoes for apparatus
    if echoes:
        echo_path = ROOT / "apparatus" / passage_id
        echo_path.mkdir(parents=True, exist_ok=True)
        json.dump(echoes, open(echo_path / "echoes.json", "w"),
                  ensure_ascii=False, indent=2)

    result = {
        "passage_id": passage_id,
        "greek": greek,
        "style_issues": style_issues,
        "vocab_issues": vocab_issues,
        "echoes": echoes,
    }
    return result


# ====================================================================
# Main
# ====================================================================

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
    print(f"║  V3: translate → Sonnet cascade → echoes                ║")
    print(f"╚{'═'*60}╝")

    start = time.time()
    for pid in passage_ids:
        translate_passage(pid, dry_run=args.dry_run)

    print(f"\n{'═'*60}")
    print(f"  Done in {time.time()-start:.0f}s")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
