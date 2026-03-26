#!/usr/bin/env python3
"""
Automated revision pipeline: detect → prompt → revise → verify.

Collects findings from all checkers (Morpheus, constructions, Grew),
builds a targeted revision prompt for each passage with issues, calls
the LLM to revise only the flagged problems, writes back the corrected
draft, and re-runs verification.

Usage:
  python3 scripts/auto_revise.py                    # revise all passages with issues
  python3 scripts/auto_revise.py 013_toadvine_fight  # revise one passage
  python3 scripts/auto_revise.py --dry-run           # show prompts without calling LLM
  python3 scripts/auto_revise.py --max-passes 3      # limit revision rounds
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DRAFTS = ROOT / "drafts"
PASSAGES = ROOT / "passages"
CONFIG = ROOT / "config"
GLOSSARY = ROOT / "glossary"

# Ensure opam env for Grew
OPAM_SWITCH = Path.home() / ".opam" / "4.14.2"
if OPAM_SWITCH.is_dir():
    opam_bin = str(OPAM_SWITCH / "bin")
    if opam_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = opam_bin + ":" + os.environ.get("PATH", "")
    os.environ["CAML_LD_LIBRARY_PATH"] = str(OPAM_SWITCH / "lib" / "stublibs")


# ====================================================================
# 1. Collect findings from all checkers
# ====================================================================

def collect_morpheus_findings(passage_id: str) -> list[dict]:
    """Run Morpheus checker on one passage, return findings."""
    sys.path.insert(0, str(SCRIPTS))
    from morpheus_check import check_passage, _save_cache
    issues = check_passage(passage_id)
    _save_cache()
    return issues


def collect_construction_findings(passage_id: str) -> list[dict]:
    """Run construction checker on one passage, return findings."""
    sys.path.insert(0, str(SCRIPTS))
    from check_constructions import check_passage
    return check_passage(passage_id)


def collect_grew_findings(passage_id: str) -> list[dict]:
    """Run Grew checker on one passage, return findings."""
    try:
        sys.path.insert(0, str(SCRIPTS))
        from grew_check import parse_to_conllu, run_checks

        conllu = parse_to_conllu([passage_id])
        findings = run_checks(conllu, warnings_only=True)
        return findings
    except Exception as e:
        print(f"    Grew: {e}")
        return []


def collect_all_findings(passage_id: str) -> dict:
    """Collect findings from all checkers for a passage."""
    print(f"\n  Collecting findings for {passage_id}...")

    findings = {
        "morpheus": [],
        "constructions": [],
        "grew": [],
    }

    # Morpheus (word-level)
    try:
        morpheus = collect_morpheus_findings(passage_id)
        # Filter to actionable (skip false positives we know about)
        findings["morpheus"] = [
            i for i in morpheus
            if i["type"] in ("neuter_plural_verb", "unattested_word")
            and "παρ᾽" not in i.get("word", "")  # known elision FP
        ]
    except Exception as e:
        print(f"    Morpheus error: {e}")

    # Constructions (structural)
    try:
        constr = collect_construction_findings(passage_id)
        findings["constructions"] = constr
    except Exception as e:
        print(f"    Construction check error: {e}")

    # Grew (treebank-backed)
    try:
        grew = collect_grew_findings(passage_id)
        findings["grew"] = [f for f in grew if f.get("passage") == passage_id]
    except Exception as e:
        print(f"    Grew error: {e}")

    total = sum(len(v) for v in findings.values())
    print(f"    Found: {len(findings['morpheus'])} morpheus, "
          f"{len(findings['constructions'])} construction, "
          f"{len(findings['grew'])} grew ({total} total)")

    return findings


# ====================================================================
# 2. Build targeted revision prompt
# ====================================================================

def load_english(passage_id: str) -> str:
    p = PASSAGES / f"{passage_id}.json"
    if p.exists():
        return json.load(open(p)).get("text", "")
    return ""


def load_greek(passage_id: str) -> str:
    p = DRAFTS / passage_id / "primary.txt"
    if p.exists():
        return p.read_text("utf-8").strip()
    return ""


def load_rules_text() -> str:
    rules = (CONFIG / "translation_prompt_rules.md").read_text("utf-8")
    particles = (CONFIG / "particle_guide.md").read_text("utf-8")
    return rules + "\n\n" + particles


def load_glossary_locks() -> str:
    """Load locked glossary terms as a concise reference."""
    glossary_path = GLOSSARY / "idf_glossary.json"
    if not glossary_path.exists():
        return ""
    data = json.load(open(glossary_path))
    lines = ["Locked term translations (MUST use these):"]
    for entry in data.get("terms", []):
        if entry.get("status") == "locked":
            lines.append(f"  {entry['english']} = {entry['greek']}")
    return "\n".join(lines)


def format_findings_for_prompt(findings: dict) -> str:
    """Format findings into clear instructions for the LLM."""
    sections = []

    # Neuter plural violations
    neut_pl = [f for f in findings["morpheus"] if f["type"] == "neuter_plural_verb"]
    neut_pl += [f for f in findings["grew"]
                if f.get("rule", "").startswith("neut_pl")]
    if neut_pl:
        items = []
        for f in neut_pl:
            ctx = f.get("context", f.get("sentence", ""))[:100]
            word = f.get("word", "")
            items.append(f"  - {word}: {ctx}")
        sections.append(
            "NEUTER PLURAL + PLURAL VERB (must fix — Attic rule: singular verb):\n"
            + "\n".join(items)
        )

    # Unattested words
    unattested = [f for f in findings["morpheus"] if f["type"] == "unattested_word"]
    if unattested:
        words = list(set(f["word"] for f in unattested))
        sections.append(
            f"UNATTESTED WORDS (check if real; replace if invented):\n"
            f"  {', '.join(words)}"
        )

    # Lost relative clauses
    relcl = [f for f in findings["constructions"]
             if isinstance(f, dict) and f.get("type") == "relative_clause_lost"]
    if relcl:
        for f in relcl:
            sections.append(
                f"RELATIVE CLAUSES CONVERTED TO PARTICIPLES (restore where natural):\n"
                f"  {f['message']}"
            )

    # Lost coordination
    coord = [f for f in findings["constructions"]
             if isinstance(f, dict) and f.get("type") == "coordination_lost"]
    if coord:
        for f in coord:
            sections.append(
                f"COORDINATION CHAINS LOST (McCarthy's 'and...and...and' style):\n"
                f"  {f['message']}"
            )

    # Lost fragments
    frags = [f for f in findings["constructions"]
             if isinstance(f, dict) and f.get("type") == "fragments_expanded"]
    if frags:
        for f in frags:
            sections.append(
                f"FRAGMENTS EXPANDED INTO FULL SENTENCES (keep McCarthy's fragments):\n"
                f"  {f['message']}"
            )

    # Preposition governance
    prep = [f for f in findings["grew"]
            if f.get("rule", "").startswith(("en_", "ek_", "eis_", "apo_", "mined_prep_"))]
    if prep:
        items = [f"  - {f.get('description', '')}: {f.get('sentence', '')[:80]}"
                 for f in prep]
        sections.append(
            "PREPOSITION GOVERNANCE VIOLATIONS:\n" + "\n".join(items)
        )

    if not sections:
        return ""

    return "\n\n".join(sections)


def build_revision_prompt(passage_id: str, english: str, greek: str,
                          findings: dict, rules: str, glossary: str) -> str:
    """Build revision prompt using the same four-layer guidance as translate.py."""
    issues_text = format_findings_for_prompt(findings)

    if not issues_text:
        return ""  # nothing to fix

    # Build structural + construction + vocabulary guidance (same as translate.py)
    try:
        sys.path.insert(0, str(SCRIPTS))
        from translate import (
            _load_best_index, _fingerprint_query, build_sentence_guidance,
            build_vocab_guidance
        )
        features_arr, metadata = _load_best_index()

        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', english) if s.strip()]
        guidance_parts = []
        for i, sent in enumerate(sents, 1):
            guidance_parts.append(build_sentence_guidance(sent, i, features_arr, metadata))
        guidance = "\n".join(guidance_parts)
        vocab = build_vocab_guidance(english)
    except Exception as e:
        guidance = ""
        vocab = ""
        print(f"    Warning: could not build guidance: {e}")

    # Build construction guide
    construction_guide = ""
    try:
        from conditional_guide import identify_constructions, format_for_prompt
        cfindings = identify_constructions(english)
        if cfindings:
            seen = set()
            unique = []
            for f in cfindings:
                key = (f["type"], f["text"][:40])
                if key not in seen:
                    seen.add(key)
                    unique.append(f)
            construction_guide = format_for_prompt(unique)
    except Exception:
        pass

    prompt = f"""You are revising an Ancient Greek (Koine/Attic) translation of McCarthy's Blood Meridian.

## Translation Rules
{rules}

## {glossary}

## English Source
{english}

## Current Greek Translation
{greek}

## Issues to Fix
{issues_text}

## Structural Guidance
{guidance}

{construction_guide}

{vocab}

## Instructions
1. Fix each flagged issue in the current Greek translation.
2. Follow the structural guidance and construction labels for construction choices.
3. Use the vocabulary guidance for word choices.
4. Output ONLY the complete revised Greek text — no explanations.
5. Do NOT change anything that isn't flagged.
6. Every word must be attestable in Morpheus/LSJ.
"""
    return prompt


# ====================================================================
# 3. Call LLM
# ====================================================================

def call_llm(prompt: str, model: str = "claude-opus-4-20250514") -> str:
    """Call Anthropic API to revise the text. Uses streaming for long outputs."""
    import anthropic

    client = anthropic.Anthropic()
    collected = []
    with client.messages.stream(
        model=model,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)
    return "".join(collected).strip()


# ====================================================================
# 4. Write back and verify
# ====================================================================

def write_draft(passage_id: str, text: str, backup: bool = True):
    """Write revised draft, optionally backing up the original."""
    draft_path = DRAFTS / passage_id / "primary.txt"

    if backup:
        backup_path = DRAFTS / passage_id / "primary.txt.bak"
        if draft_path.exists() and not backup_path.exists():
            backup_path.write_text(draft_path.read_text("utf-8"), encoding="utf-8")

    draft_path.write_text(text + "\n", encoding="utf-8")


def verify_revision(passage_id: str) -> dict:
    """Re-run checkers on the revised passage."""
    return collect_all_findings(passage_id)


# ====================================================================
# 5. Main loop
# ====================================================================

def revise_passage(passage_id: str, dry_run: bool = False,
                   model: str = "claude-opus-4-20250514") -> bool:
    """Full revision cycle for one passage. Returns True if changes made."""
    english = load_english(passage_id)
    greek = load_greek(passage_id)
    if not english or not greek:
        print(f"  Skipping {passage_id} — missing source or draft")
        return False

    findings = collect_all_findings(passage_id)
    total = sum(len(v) for v in findings.values())

    if total == 0:
        print(f"  ✓ {passage_id}: no issues found")
        return False

    rules = load_rules_text()
    glossary = load_glossary_locks()
    prompt = build_revision_prompt(passage_id, english, greek, findings,
                                   rules, glossary)

    if not prompt:
        print(f"  ✓ {passage_id}: no actionable issues")
        return False

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {passage_id}")
        print(f"{'='*60}")
        print(f"Issues: {total}")
        issues_text = format_findings_for_prompt(findings)
        print(issues_text)
        print(f"\nPrompt length: {len(prompt)} chars")
        return False

    print(f"  Calling LLM to revise {passage_id}...")
    revised = call_llm(prompt, model=model)

    # Basic sanity: revised text should be similar length (±50%)
    if len(revised) < len(greek) * 0.8 or len(revised) > len(greek) * 1.3:
        print(f"  ⚠ Revision length suspicious ({len(revised)} vs {len(greek)})")
        print(f"    Saving to primary.txt.revision for manual review")
        (DRAFTS / passage_id / "primary.txt.revision").write_text(
            revised, encoding="utf-8"
        )
        return False

    write_draft(passage_id, revised)
    print(f"  ✓ Revised {passage_id} ({len(greek)} → {len(revised)} chars)")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-revise translations")
    parser.add_argument("passages", nargs="*", help="Passage IDs to revise")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without calling LLM")
    parser.add_argument("--max-passes", type=int, default=2,
                        help="Maximum revision passes per passage")
    parser.add_argument("--model", type=str, default="claude-opus-4-20250514",
                        help="Model to use for revision")
    args = parser.parse_args()

    if args.passages:
        passage_ids = args.passages
    else:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )

    print(f"╔{'═'*50}╗")
    print(f"║  Auto-revision pipeline                         ║")
    print(f"╚{'═'*50}╝")

    rules = load_rules_text()
    glossary = load_glossary_locks()

    for passage_id in passage_ids:
        for pass_num in range(1, args.max_passes + 1):
            print(f"\n{'─'*50}")
            print(f"  {passage_id} — pass {pass_num}/{args.max_passes}")
            print(f"{'─'*50}")

            changed = revise_passage(
                passage_id,
                dry_run=args.dry_run,
                model=args.model,
            )

            if not changed:
                break

            # Verify
            print(f"  Verifying revision...")
            post_findings = collect_all_findings(passage_id)
            post_total = sum(len(v) for v in post_findings.values())

            if post_total == 0:
                print(f"  ✓ {passage_id}: all issues resolved")
                break
            else:
                print(f"  {post_total} issues remain — "
                      f"{'will retry' if pass_num < args.max_passes else 'manual review needed'}")

    print(f"\n{'═'*50}")
    print(f"  Done")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
