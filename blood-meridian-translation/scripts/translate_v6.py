#!/usr/bin/env python3
"""
V6: Two-call pipeline — Sonnet draft → Opus improve → mechanical checks.

Much cheaper than V5 ($0.05 vs $0.31 per paragraph) with equal or better
quality, driven by optimised prompts with specific vocabulary guidance.

Usage:
  python3 scripts/translate_v6.py 001_see_the_child
  python3 scripts/translate_v6.py --all
  python3 scripts/translate_v6.py 007_divested_of_all --dry-run
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


def load_glossary_hints(en_text: str) -> str:
    """Load locked glossary entries relevant to this passage as prompt hints."""
    glossary_path = GLOSSARY / "idf_glossary.json"
    if not glossary_path.exists():
        return ""
    data = json.load(open(glossary_path))
    en_lower = en_text.lower()
    hints = []
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
                hint = f'- "{en_term}" → {ag}'
                if justification:
                    hint += f" ({justification[:80]})"
                hints.append(hint)
    if not hints:
        return ""
    return "Vocabulary decisions (use these exact words):\n" + "\n".join(hints)


def load_previous_greek(passage_id: str) -> str:
    """Load the previous passage's Greek for stylistic continuity."""
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
            return f"Previous passage (match this style):\n{prev}"
    return ""


def run_checks(passage_id: str, en_text: str, greek: str) -> list[str]:
    """Run mechanical checks and return list of issue strings."""
    issues = []

    # Morpheus attestation
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
                issues.append(f"UNATTESTED: {m['word']}")

        # Morpheus-only check
        tokens = re.findall(r'[\w\u0370-\u03FF\u1F00-\u1FFF]+', greek)
        seen = set()
        for tok in tokens:
            if len(tok) <= 4:
                continue
            analyses = parse_word(tok)
            if not analyses:
                continue
            in_corpus = _is_attested_in_corpus(tok)
            in_db = _is_attested_in_db(tok)
            if in_corpus or in_db or tok in MORPHEUS_WHITELIST:
                continue
            lemma = analyses[0].get("lemma", "") if analyses else ""
            if lemma and (_is_attested_in_corpus(lemma) or _is_attested_in_db(lemma)):
                continue
            if tok not in seen:
                seen.add(tok)
                issues.append(f"MORPHEUS-ONLY: {tok} (lemma: {lemma})")
    except Exception as e:
        issues.append(f"Morpheus error: {e}")

    # Grew grammar
    try:
        from translate_v3 import run_mechanical_sanity
        mech = run_mechanical_sanity(passage_id, en_text, greek)
        for m in mech:
            issues.append(f"GRAMMAR: {m.get('problem', '')[:80]}")
    except Exception as e:
        issues.append(f"Grammar error: {e}")

    return issues


def translate_passage(passage_id: str, dry_run: bool = False,
                      force: bool = False) -> dict | None:
    p_path = PASSAGES / f"{passage_id}.json"
    if not p_path.exists():
        print(f"  {passage_id}: not found")
        return None
    en_text = json.load(open(p_path)).get("text", "")
    if not en_text:
        return None

    draft_path = DRAFTS / passage_id / "primary.txt"
    if draft_path.exists() and not force and not dry_run:
        existing = draft_path.read_text("utf-8").strip()
        if existing:
            print(f"  {passage_id}: draft exists. Use --force to overwrite.")
            return None

    draft_path.parent.mkdir(parents=True, exist_ok=True)

    glossary_hints = load_glossary_hints(en_text)
    previous = load_previous_greek(passage_id)

    # === CALL 1: Sonnet draft ===
    draft_prompt = f"""Translate this passage from Cormac McCarthy's Blood Meridian into Ancient Greek (Koine register, Attic vocabulary, polytonic orthography).

Key rules:
- Mirror McCarthy's sentence structure exactly: parataxis, asyndeton, fragments stay as they are
- "and" → καί chains, do NOT subordinate or convert to participles
- Comma splices → asyndeton
- Neuter plural subjects take SINGULAR verbs (Attic rule)
- Every word must exist in LSJ/Morpheus — do not invent words
- Fragments remain fragments (no added verbs)
- "swells" (ocean) → κύματα (waves), NOT οἰδήματα (medical swelling)
- "gawking" → κεχηνότες (gaping, from χαίνω)

{glossary_hints}

{previous}

Output ONLY the Greek text, no commentary.

English:
{en_text}"""

    print(f"\n  [{passage_id}] CALL 1: Sonnet draft")
    if dry_run:
        print(f"    [DRY RUN] {len(draft_prompt)} chars")
        print(f"    Prompt preview:\n{draft_prompt[:500]}...")
        return None

    import anthropic
    client = anthropic.Anthropic()

    t0 = time.time()
    r1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": draft_prompt}],
    )
    draft = r1.content[0].text.strip()
    print(f"    {time.time()-t0:.0f}s ({r1.usage.input_tokens} in, {r1.usage.output_tokens} out)")
    print(f"    {draft[:100]}...")

    # Save draft
    (draft_path.parent / "stage0_sonnet_draft.txt").write_text(draft)

    # === CALL 2: Opus improve ===
    improve_prompt = f"""You are a scholar of Ancient Greek prose composition. Below is a first-draft translation of Cormac McCarthy into Ancient Greek (Koine/Attic register). Produce an IMPROVED version.

Priorities:
1. Every word must be attested in LSJ. Replace any invented or Byzantine forms.
2. Preserve McCarthy's paratactic, asyndetic, fragment-heavy style. Do not subordinate what he coordinates. Do not add particles he doesn't signal.
3. Choose the most precise classical word for each concept. Prefer Attic over Hellenistic where both exist.
4. Neuter plural subjects take singular verbs.
5. "swells" (ocean waves) = κύματα or κῦμα, NEVER οἰδήματα (which means medical swelling)
6. Fragments stay as fragments — no added finite verbs.
7. Check every participle and contracted verb form — fabricated morphology is common in drafts.

{glossary_hints}

English source:
{en_text}

First draft to improve:
{draft}

Output ONLY the improved Greek text. No commentary."""

    print(f"\n  [{passage_id}] CALL 2: Opus improve")
    t0 = time.time()
    r2 = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": improve_prompt}],
    )
    improved = r2.content[0].text.strip()
    print(f"    {time.time()-t0:.0f}s ({r2.usage.input_tokens} in, {r2.usage.output_tokens} out)")
    print(f"    {improved[:100]}...")

    # Save improved
    (draft_path.parent / "stage1_opus_improved.txt").write_text(improved)
    draft_path.write_text(improved + "\n")

    # === MECHANICAL CHECKS ===
    print(f"\n  [{passage_id}] MECHANICAL CHECKS")
    check_issues = run_checks(passage_id, en_text, improved)
    final = improved

    if check_issues:
        print(f"    ⚠ {len(check_issues)} issues:")
        for issue in check_issues:
            print(f"      {issue[:80]}")

        # === CALL 3: Sonnet targeted fix (cheap, surgical) ===
        # Only ask for specific, actionable changes — not a rewrite
        actionable = []
        for issue in check_issues:
            if issue.startswith("UNATTESTED:"):
                word = issue.split("UNATTESTED:")[1].strip()
                actionable.append(
                    f"Replace '{word}' — this word does not exist in Ancient Greek. "
                    f"Substitute a real word from LSJ that fits the same meaning in context."
                )
            elif issue.startswith("GRAMMAR:") and "neuter plural" in issue.lower():
                actionable.append(
                    "Fix neuter plural agreement: neuter plural subjects take SINGULAR verbs in Attic Greek. "
                    "Find the neuter plural subject and change its verb to singular."
                )
            # Skip MORPHEUS-ONLY (word exists in Morpheus = probably real, just rare)
            # Skip relative clause mismatches (often valid Greek choices)

        if not actionable:
            print(f"    No actionable issues — skipping fix call")
            r3_cost = 0.0
        else:
            changes = "\n".join(f"{i+1}. {a}" for i, a in enumerate(actionable))
            fix_prompt = f"""Make ONLY these specific changes to the Greek text below. Do not change anything else — keep every other word exactly as it is.

Changes needed:
{changes}

Greek text:
{improved}

Output the corrected Greek text with ONLY the above changes applied."""

            print(f"\n  [{passage_id}] CALL 3: Sonnet fix ({len(actionable)} changes)")
            t0 = time.time()
            r3 = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": fix_prompt}],
            )
            fixed = r3.content[0].text.strip()
            r3_cost = r3.usage.input_tokens / 1e6 * 3 + r3.usage.output_tokens / 1e6 * 15
            print(f"    {time.time()-t0:.0f}s ({r3.usage.input_tokens} in, {r3.usage.output_tokens} out, ${r3_cost:.4f})")

            # Save fixed version
            (draft_path.parent / "stage2_sonnet_fixed.txt").write_text(fixed)
            draft_path.write_text(fixed + "\n")
            final = fixed

            # Re-run checks on fixed version
            print(f"\n  [{passage_id}] RE-CHECK")
            recheck = run_checks(passage_id, en_text, fixed)
            if recheck:
                print(f"    {len(recheck)} residual issues (for human review):")
                for issue in recheck:
                    print(f"      {issue[:80]}")
            else:
                print(f"    ✓ All clear after fix")

        print(f"\n  [{passage_id}] CALL 3: Sonnet targeted fix")
        t0 = time.time()
        r3 = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": fix_prompt}],
        )
        fixed = r3.content[0].text.strip()
        r3_cost = r3.usage.input_tokens / 1e6 * 3 + r3.usage.output_tokens / 1e6 * 15
        print(f"    {time.time()-t0:.0f}s ({r3.usage.input_tokens} in, {r3.usage.output_tokens} out, ${r3_cost:.4f})")
        print(f"    {fixed[:100]}...")

        # Save fixed version
        (draft_path.parent / "stage2_sonnet_fixed.txt").write_text(fixed)
        draft_path.write_text(fixed + "\n")
        final = fixed

        # Re-run checks on fixed version
        print(f"\n  [{passage_id}] RE-CHECK")
        recheck = run_checks(passage_id, en_text, fixed)
        if recheck:
            print(f"    ⚠ {len(recheck)} residual issues (for human review):")
            for issue in recheck:
                print(f"      {issue[:80]}")
        else:
            print(f"    ✓ All clear after fix")
    else:
        print(f"    ✓ All clear")
        r3_cost = 0.0

    # === COST ===
    s_cost = r1.usage.input_tokens / 1e6 * 3 + r1.usage.output_tokens / 1e6 * 15
    o_cost = r2.usage.input_tokens / 1e6 * 15 + r2.usage.output_tokens / 1e6 * 75
    total_cost = s_cost + o_cost + r3_cost
    calls = "3" if r3_cost > 0 else "2"
    print(f"\n  [{passage_id}] COST: ${total_cost:.4f} ({calls} calls)")

    return {
        "passage_id": passage_id,
        "greek": final,
        "cost": total_cost,
        "issues": check_issues,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing translations")
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
    print(f"║  V6: Sonnet draft → Opus improve (2 calls)              ║")
    print(f"╚{'═'*60}╝")

    start = time.time()
    total_cost = 0
    for pid in passage_ids:
        result = translate_passage(pid, dry_run=args.dry_run, force=args.force)
        if result:
            total_cost += result["cost"]

    print(f"\n{'═'*60}")
    print(f"  Done in {time.time()-start:.0f}s — total cost: ${total_cost:.4f}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
