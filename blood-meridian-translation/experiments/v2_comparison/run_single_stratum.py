#!/usr/bin/env python3
"""Run three-tier comparison on a single stratum with full diagnostic output."""
import json, sys, time, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from translate_v2 import (
    build_translate_prompt, run_sonnet_review,
    run_mechanical_checks, check_polysemy, build_revision_prompt, call_opus,
)

OUT = Path(__file__).resolve().parent / 'long_subordinated'
OUT.mkdir(parents=True, exist_ok=True)

en_text = (OUT / 'english.txt').read_text().strip()
print(f"PASSAGE ({len(en_text.split())}w): {en_text}\n")

# === TIER A ===
print("=" * 60)
print("  TIER A: Raw Opus")
print("=" * 60)
prompt_a = build_translate_prompt(en_text)
(OUT / 'tier_a_prompt.txt').write_text(prompt_a)
print(f"  Prompt: {len(prompt_a)} chars")
t0 = time.time()
greek_a = call_opus(prompt_a)
print(f"  Done in {time.time()-t0:.0f}s\n  {greek_a}\n")
(OUT / 'tier_a_output.txt').write_text(greek_a)

# === TIER B ===
print("=" * 60)
print("  TIER B: Opus + Sonnet review only")
print("=" * 60)
t0 = time.time()
greek_b_draft = call_opus(prompt_a)  # same bare prompt
print(f"  Draft in {time.time()-t0:.0f}s\n  {greek_b_draft}\n")
(OUT / 'tier_b_draft.txt').write_text(greek_b_draft)

print("  Sonnet reviewing...")
t0 = time.time()
sonnet = run_sonnet_review(en_text, greek_b_draft)
issues = sonnet.get("issues", [])
print(f"  {len(issues)} issues in {time.time()-t0:.0f}s:")
for si in issues:
    print(f"    - {si.get('greek','')[:40]}: {si.get('problem','')[:60]}")
    if si.get('fix'):
        print(f"      → {si['fix'][:60]}")

diagnosis_b = {"sonnet_review": sonnet}
rev_b = build_revision_prompt(en_text, greek_b_draft, diagnosis_b)
if rev_b:
    (OUT / 'tier_b_revision_prompt.txt').write_text(rev_b)
    print(f"\n  Revision prompt: {len(rev_b)} chars")
    t0 = time.time()
    greek_b = call_opus(rev_b)
    print(f"  Revised in {time.time()-t0:.0f}s\n  {greek_b}\n")
else:
    greek_b = greek_b_draft
    print("  No revision needed.\n")
(OUT / 'tier_b_output.txt').write_text(greek_b)

# === TIER C ===
print("=" * 60)
print("  TIER C: Opus + full pipeline")
print("=" * 60)
t0 = time.time()
greek_c_draft = call_opus(prompt_a)
print(f"  Draft in {time.time()-t0:.0f}s\n  {greek_c_draft}\n")
(OUT / 'tier_c_draft.txt').write_text(greek_c_draft)

# Temp passage for mechanical checks
tmp_id = "_exp_long_sub"
tmp_dir = ROOT / "drafts" / tmp_id
tmp_dir.mkdir(parents=True, exist_ok=True)
(tmp_dir / "primary.txt").write_text(greek_c_draft + "\n")

print("  Mechanical checks...")
t0 = time.time()
mech = run_mechanical_checks(tmp_id)
print(f"    Unattested: {mech.get('unattested_words', [])}")
print(f"    Grammar: {mech.get('grammar_violations', [])[:3]}")
print(f"    Construction: {mech.get('construction_mismatches', [])[:3]}")
print(f"    ({time.time()-t0:.0f}s)")

print("  Sonnet reviewing...")
t0 = time.time()
sonnet_c = run_sonnet_review(en_text, greek_c_draft)
issues_c = sonnet_c.get("issues", [])
print(f"  {len(issues_c)} issues in {time.time()-t0:.0f}s:")
for si in issues_c:
    print(f"    - {si.get('greek','')[:40]}: {si.get('problem','')[:60]}")

poly = check_polysemy(en_text, greek_c_draft)
if poly:
    print(f"  Polysemy: {poly}")

diagnosis_c = {}
diagnosis_c.update(mech)
diagnosis_c["sonnet_review"] = sonnet_c
diagnosis_c["polysemy_issues"] = poly

rev_c = build_revision_prompt(en_text, greek_c_draft, diagnosis_c)
if rev_c:
    (OUT / 'tier_c_revision_prompt.txt').write_text(rev_c)
    print(f"\n  Revision prompt: {len(rev_c)} chars")
    # Show just the section headers
    for line in rev_c.split('\n'):
        if line.startswith('##'):
            print(f"    {line}")
    t0 = time.time()
    greek_c = call_opus(rev_c)
    print(f"  Revised in {time.time()-t0:.0f}s\n  {greek_c}\n")
else:
    greek_c = greek_c_draft
    print("  No revision needed.\n")
(OUT / 'tier_c_output.txt').write_text(greek_c)

shutil.rmtree(tmp_dir, ignore_errors=True)

# === Summary ===
print("=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"\n  A (raw):    {greek_a}\n")
print(f"  B (sonnet): {greek_b}\n")
print(f"  C (full):   {greek_c}\n")

# Save summary
json.dump({
    "passage": en_text,
    "tier_a": greek_a,
    "tier_b": greek_b,
    "tier_b_issues": sonnet.get("issues", []),
    "tier_c": greek_c,
    "tier_c_mechanical": mech,
    "tier_c_sonnet_issues": sonnet_c.get("issues", []),
    "tier_c_polysemy": poly,
}, open(OUT / 'summary.json', 'w'), ensure_ascii=False, indent=2)
