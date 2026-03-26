#!/usr/bin/env python3
"""
V2 translation pipeline: generate bare, diagnose, revise surgically.

Step 1: TRANSLATE — minimal prompt, let Opus use natural instincts
Step 2: DIAGNOSE — mechanical checks + optional Sonnet review
Step 3: REVISE — targeted fixes for actual problems found
Step 4: RE-DIAGNOSE — verify, optional second pass

Usage:
  python3 scripts/translate_v2.py 001_see_the_child
  python3 scripts/translate_v2.py --all
  python3 scripts/translate_v2.py --all --max-passes 3
  python3 scripts/translate_v2.py 005_saint_louis --dry-run  # show prompts only
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PASSAGES = ROOT / "passages"
DRAFTS = ROOT / "drafts"
CONFIG = ROOT / "config"
GLOSSARY = ROOT / "glossary"

sys.path.insert(0, str(SCRIPTS))

# ====================================================================
# Step 1: TRANSLATE — bare prompt
# ====================================================================

def load_name_glossary() -> str:
    """Load standardised names and essential coinages from IDF glossary.

    This is the ONLY vocabulary guidance in the initial prompt — just
    names, transliterations, and mandatory coinages for consistency.
    """
    glossary_path = GLOSSARY / "idf_glossary.json"
    if not glossary_path.exists():
        return ""
    data = json.load(open(glossary_path))

    lines = []
    for category, entries in data.items():
        if category.startswith("_") or not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            status = entry.get("status", "")
            register = entry.get("register", "")
            if status != "locked":
                continue
            en = entry.get("english", "")
            ag = entry.get("ancient_greek", "").replace("*", "")
            # Only include names, transliterations, and essential coinages
            if register in ("transliteration", "modern_loan") or category in ("characters", "place_names"):
                lines.append(f"  {en} = {ag}")

    if not lines:
        return ""
    return "## Standardised Names and Coinages\n" + "\n".join(lines)


TRANSLATE_PROMPT = """You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine register with Attic vocabulary, polytonic orthography).

McCarthy's prose is paratactic, asyndetic, and fragment-heavy. Preserve his voice:
- "and...and...and" coordination chains → καί chains. Do NOT subordinate.
- Comma splices → asyndeton. But use particles (δέ, γάρ, οὖν) where Greek genuinely needs them.
- Fragments remain fragments. Do not expand into full sentences.
- Direct speech preserves its structure and line breaks.
- Neuter plural subjects take SINGULAR verbs (Attic rule).

Draw on whatever classical register suits the context naturally: Septuagintal for religious, Platonic for philosophical, Thucydidean for military/political narration.

Every word must be attestable in Morpheus/LSJ. For modern concepts with no classical equivalent, use a transparent Greek compound or Hellenised transliteration marked with * (e.g. *πιστόλιον).

{name_glossary}

## English Source
{english}

Output ONLY the Greek text, matching McCarthy's paragraph formatting."""


def build_translate_prompt(en_text: str) -> str:
    names = load_name_glossary()
    return TRANSLATE_PROMPT.format(name_glossary=names, english=en_text)


# ====================================================================
# Step 2: DIAGNOSE — mechanical + Sonnet review
# ====================================================================

def run_mechanical_checks(passage_id: str) -> dict:
    """Run Morpheus, construction checker, and Grew. Return structured findings."""
    findings = {
        "unattested_words": [],
        "grammar_violations": [],
        "construction_mismatches": [],
    }

    # Morpheus
    try:
        from morpheus_check import check_passage, _save_cache
        issues = check_passage(passage_id)
        _save_cache()
        for i in issues:
            if i["type"] == "unattested_word":
                findings["unattested_words"].append(i["word"])
            elif i["type"] == "neuter_plural_verb":
                findings["grammar_violations"].append(
                    f"Neuter plural + plural verb: {i.get('word', '')} in '{i.get('context', '')[:60]}'"
                )
    except Exception as e:
        findings["_morpheus_error"] = str(e)

    # Construction checker
    try:
        from check_constructions import check_passage as check_constr
        constr_issues = check_constr(passage_id)
        for c in constr_issues:
            if isinstance(c, dict):
                findings["construction_mismatches"].append(c.get("message", str(c)))
    except Exception as e:
        findings["_construction_error"] = str(e)

    # Grew
    try:
        from grew_check import parse_to_conllu, run_checks
        conllu = parse_to_conllu([passage_id])
        grew_issues = run_checks(conllu, warnings_only=True)
        for g in grew_issues:
            if g.get("passage") == passage_id:
                findings["grammar_violations"].append(
                    f"{g.get('rule', '')}: {g.get('description', '')} in '{g.get('sentence', '')[:60]}'"
                )
    except Exception as e:
        findings["_grew_error"] = str(e)

    return findings


def run_sonnet_review(en_text: str, greek_text: str) -> dict:
    """Have Sonnet review for register, idiom, and sense errors."""
    import anthropic

    prompt = f"""You are reviewing an Ancient Greek translation of Cormac McCarthy's Blood Meridian. The target register is literary Koine with Attic vocabulary.

Review the translation for these specific issues ONLY. Do not comment on things that are correct.

1. WRONG SENSE: words where the wrong meaning was chosen (e.g. οἴδημα for sea-swell instead of κῦμα)
2. REGISTER MISMATCH: words or constructions that belong to the wrong period or genre
3. OVER-SUBORDINATION: places where McCarthy's parataxis was collapsed into Greek subordination
4. UNIDIOMATIC GREEK: calques from English that a native Greek prose author would not write
5. RELATIVE PRONOUN ERRORS: relative pronouns that don't agree with their true antecedent

For each issue found, give:
- The specific Greek text
- What's wrong
- A suggested fix

## English Source
{en_text}

## Greek Translation
{greek_text}

Return ONLY a JSON array of issues:
[
  {{"greek": "problematic text", "problem": "description", "fix": "suggested replacement"}},
  ...
]
If no issues found, return an empty array: []"""

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
        return {"issues": json.loads(clean)}
    except json.JSONDecodeError:
        return {"issues": [], "raw": raw}


def check_name_consistency(greek_text: str) -> list[str]:
    """Check that standardised names from the glossary are used consistently."""
    glossary_path = GLOSSARY / "idf_glossary.json"
    if not glossary_path.exists():
        return []
    data = json.load(open(glossary_path))

    issues = []
    for category, entries in data.items():
        if category.startswith("_") or not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if not isinstance(entry, dict) or entry.get("status") != "locked":
                continue
            en = entry.get("english", "").lower()
            ag = entry.get("ancient_greek", "").replace("*", "").strip()
            register = entry.get("register", "")

            # Check for common violations
            if register in ("transliteration", "modern_loan") or category in ("characters", "place_names"):
                # Skip if not relevant to this text
                if not ag:
                    continue
                # Check if a variant spelling appears
                # (This is a simplified check — could be expanded)

    return issues


def check_polysemy(en_text: str, greek_text: str) -> list[str]:
    """Check for known polysemy errors by comparing against our table."""
    from translate import POLYSEMOUS

    en_lower = en_text.lower()
    en_words = set(re.findall(r'\b\w+\b', en_lower))
    issues = []

    for word, senses in POLYSEMOUS.items():
        matched = word in en_words or any(w.startswith(word) for w in en_words)
        if not matched:
            continue

        # Find best sense
        best_sense = None
        best_score = -1
        for clues, sense_desc, greek_suggestion in senses:
            if not clues:
                best_sense = (sense_desc, greek_suggestion)
                break
            score = sum(1 for c in clues if c in en_lower)
            if score > best_score:
                best_score = score
                best_sense = (sense_desc, greek_suggestion)

        if best_sense:
            sense_desc, greek_suggestion = best_sense
            # Check if the WRONG sense was used
            # Extract the "NOT" part if present
            not_match = re.search(r'NOT (\S+)', greek_suggestion)
            if not_match:
                wrong_word = not_match.group(1).rstrip(")")
                if wrong_word in greek_text:
                    issues.append(
                        f"'{word}' here = {sense_desc}: replace {wrong_word} with appropriate form from: {greek_suggestion}"
                    )

    return issues


# ====================================================================
# Step 3: REVISE — targeted prompt
# ====================================================================

def build_revision_prompt(en_text: str, greek_text: str, diagnosis: dict) -> str | None:
    """Build a revision prompt addressing ONLY the issues found in diagnosis.

    Returns None if no issues worth revising.
    """
    sections = []
    has_issues = False

    # Unattested words
    unattested = diagnosis.get("unattested_words", [])
    if unattested:
        has_issues = True
        sections.append(
            "## Unattested Words (replace with attested alternatives)\n"
            + "\n".join(f"  - {w}" for w in unattested[:10])
        )

    # Grammar violations
    grammar = diagnosis.get("grammar_violations", [])
    if grammar:
        has_issues = True
        sections.append(
            "## Grammar Violations (fix these)\n"
            + "\n".join(f"  - {g}" for g in grammar[:10])
        )

    # Construction mismatches
    constr = diagnosis.get("construction_mismatches", [])
    if constr:
        has_issues = True
        sections.append(
            "## Construction Issues\n"
            + "\n".join(f"  - {c}" for c in constr[:5])
        )

    # Sonnet review issues
    sonnet_issues = diagnosis.get("sonnet_review", {}).get("issues", [])
    if sonnet_issues:
        has_issues = True
        issue_lines = []
        for si in sonnet_issues[:8]:
            greek = si.get("greek", "")
            problem = si.get("problem", "")
            fix = si.get("fix", "")
            issue_lines.append(f"  - {greek}: {problem}")
            if fix:
                issue_lines.append(f"    → Suggested: {fix}")
        sections.append(
            "## Style and Sense Issues (from review)\n"
            + "\n".join(issue_lines)
        )

    # Polysemy issues
    polysemy = diagnosis.get("polysemy_issues", [])
    if polysemy:
        has_issues = True
        sections.append(
            "## Wrong Word Sense\n"
            + "\n".join(f"  - {p}" for p in polysemy)
        )

    # Name consistency
    name_issues = diagnosis.get("name_issues", [])
    if name_issues:
        has_issues = True
        sections.append(
            "## Name Standardisation\n"
            + "\n".join(f"  - {n}" for n in name_issues)
        )

    if not has_issues:
        return None

    issues_text = "\n\n".join(sections)

    # === Supporting evidence: only include what's relevant to the issues found ===
    evidence_sections = []

    # Style models: include when Sonnet flagged register/idiom issues
    has_register_issue = any(
        "register" in si.get("problem", "").lower() or
        "idiom" in si.get("problem", "").lower() or
        "calque" in si.get("problem", "").lower()
        for si in sonnet_issues
    )
    if has_register_issue:
        try:
            sys.path.insert(0, str(ROOT))
            from retrieval.search import lexical_inspiration, Scale
            hits = lexical_inspiration(en_text, Scale.SENTENCE, period_filter=None, top_k=12)
            seen_authors = set()
            models = []
            for hit in hits:
                author = getattr(hit.chunk, 'author', '') or ''
                if author in seen_authors:
                    continue
                seen_authors.add(author)
                greek_ex = hit.chunk.text[:200]
                source = f"{author}, {getattr(hit.chunk, 'work', '')}"
                models.append(f"  [{source}]: {greek_ex}")
                if len(models) >= 3:
                    break
            if models:
                evidence_sections.append(
                    "## Style Reference (thematically matched passages — guide your register)\n"
                    + "\n\n".join(models)
                )
        except Exception:
            pass

    # Construction guides: include when construction mismatches found
    if constr:
        try:
            from label_constructions import label_english
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]
            label_lines = []
            for i, sent in enumerate(sents, 1):
                labels = label_english(sent)
                if labels:
                    short = sent[:60] + ("..." if len(sent) > 60 else "")
                    label_lines.append(f'  {i}. "{short}" — {", ".join(labels)}')
            if label_lines:
                evidence_sections.append(
                    "## Construction Guide (for the flagged structural issues)\n"
                    + "\n".join(label_lines)
                )
        except Exception:
            pass

        try:
            from conditional_guide import identify_constructions, format_for_prompt
            findings = identify_constructions(en_text)
            if findings:
                seen = set()
                unique = []
                for f in findings:
                    key = (f["type"], f["text"][:40])
                    if key not in seen:
                        seen.add(key)
                        unique.append(f)
                evidence_sections.append(format_for_prompt(unique))
        except Exception:
            pass

    # Polysemy: include full sense table for flagged words
    from translate import POLYSEMOUS
    if polysemy:
        en_lower = en_text.lower()
        en_words = set(re.findall(r'\b\w+\b', en_lower))
        poly_notes = []
        for word, senses in POLYSEMOUS.items():
            matched = word in en_words or any(w.startswith(word) for w in en_words)
            if not matched:
                continue
            best_sense = None
            best_score = -1
            for clues, sense_desc, greek_sug in senses:
                if not clues:
                    best_sense = (sense_desc, greek_sug)
                    break
                score = sum(1 for c in clues if c in en_lower)
                if score > best_score:
                    best_score = score
                    best_sense = (sense_desc, greek_sug)
            if best_sense:
                poly_notes.append(f"  '{word}' here = {best_sense[0]} → {best_sense[1]}")
        if poly_notes:
            evidence_sections.append(
                "## Correct Word Senses\n" + "\n".join(poly_notes)
            )

    evidence_text = "\n\n".join(evidence_sections) if evidence_sections else ""

    prompt = f"""You are revising an Ancient Greek translation of McCarthy's Blood Meridian (Koine register, Attic vocabulary).

Fix ONLY the specific issues listed below. Do not change anything that is already correct.

## English Source
{en_text}

## Current Greek Translation
{greek_text}

{issues_text}

{evidence_text}

## Instructions
1. Fix each listed issue while preserving everything that is correct.
2. Use the supporting evidence above to guide your corrections where provided.
3. Every word must be attestable in Morpheus/LSJ.
4. Output ONLY the complete revised Greek text — no explanations."""

    return prompt


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


# ====================================================================
# Full pipeline
# ====================================================================

def translate_passage(passage_id: str, max_passes: int = 2,
                      dry_run: bool = False) -> bool:
    """Full v2 pipeline for one passage."""

    # Load English
    p_path = PASSAGES / f"{passage_id}.json"
    if not p_path.exists():
        print(f"  {passage_id}: passage file not found")
        return False
    en_text = json.load(open(p_path)).get("text", "")
    if not en_text:
        print(f"  {passage_id}: no text")
        return False

    # Check if draft already exists
    draft_path = DRAFTS / passage_id / "primary.txt"
    if draft_path.exists() and not dry_run:
        existing = draft_path.read_text("utf-8").strip()
        if existing:
            print(f"  {passage_id}: draft exists ({len(existing)} chars). Skipping.")
            return False

    # === Step 1: TRANSLATE ===
    print(f"\n  [{passage_id}] Step 1: TRANSLATE (bare prompt)")
    prompt = build_translate_prompt(en_text)
    print(f"    Prompt: {len(prompt)} chars")

    if dry_run:
        print(f"    [DRY RUN] Would call Opus with {len(prompt)} char prompt")
        print(f"    Prompt preview:\n{prompt[:500]}...")
        return False

    t0 = time.time()
    greek = call_opus(prompt)
    print(f"    Translated in {time.time()-t0:.0f}s ({len(greek)} chars)")

    # Save initial draft
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(greek + "\n", encoding="utf-8")

    # === Steps 2-3: DIAGNOSE + REVISE loop ===
    for pass_num in range(1, max_passes + 1):
        print(f"\n  [{passage_id}] Step 2: DIAGNOSE (pass {pass_num}/{max_passes})")

        diagnosis = {}

        # Mechanical checks
        t0 = time.time()
        mech = run_mechanical_checks(passage_id)
        diagnosis.update(mech)
        n_mech = (len(mech.get("unattested_words", [])) +
                  len(mech.get("grammar_violations", [])) +
                  len(mech.get("construction_mismatches", [])))
        print(f"    Mechanical: {n_mech} issues ({time.time()-t0:.0f}s)")

        # Polysemy check
        greek = draft_path.read_text("utf-8").strip()
        poly = check_polysemy(en_text, greek)
        diagnosis["polysemy_issues"] = poly
        if poly:
            print(f"    Polysemy: {len(poly)} wrong-sense words")

        # Sonnet review
        t0 = time.time()
        sonnet = run_sonnet_review(en_text, greek)
        diagnosis["sonnet_review"] = sonnet
        n_sonnet = len(sonnet.get("issues", []))
        print(f"    Sonnet review: {n_sonnet} issues ({time.time()-t0:.0f}s)")

        # Total issues
        total_issues = n_mech + len(poly) + n_sonnet
        if total_issues == 0:
            print(f"    ✓ No issues found. Done.")
            break

        # === Step 3: REVISE ===
        print(f"\n  [{passage_id}] Step 3: REVISE ({total_issues} issues)")

        revision_prompt = build_revision_prompt(en_text, greek, diagnosis)
        if revision_prompt is None:
            print(f"    No actionable issues. Done.")
            break

        print(f"    Revision prompt: {len(revision_prompt)} chars")
        t0 = time.time()
        revised = call_opus(revision_prompt)
        print(f"    Revised in {time.time()-t0:.0f}s ({len(revised)} chars)")

        # Sanity check
        if len(revised) < len(greek) * 0.5 or len(revised) > len(greek) * 1.5:
            print(f"    ⚠ Revision length suspicious ({len(revised)} vs {len(greek)}). Keeping original.")
            break

        # Save backup and update
        bak_path = draft_path.with_suffix(f".pass{pass_num-1}.bak")
        if not bak_path.exists():
            bak_path.write_text(greek + "\n", encoding="utf-8")
        draft_path.write_text(revised + "\n", encoding="utf-8")
        print(f"    ✓ Updated draft ({len(greek)} → {len(revised)} chars)")

    return True


# ====================================================================
# Main
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-passes", type=int, default=2)
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
    print(f"║  V2 Pipeline: translate → diagnose → revise              ║")
    print(f"╚{'═'*60}╝")
    print(f"  Passages: {len(passage_ids)}")
    print(f"  Max revision passes: {args.max_passes}")

    start = time.time()
    for pid in passage_ids:
        translate_passage(pid, max_passes=args.max_passes, dry_run=args.dry_run)

    elapsed = time.time() - start
    print(f"\n{'═'*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
