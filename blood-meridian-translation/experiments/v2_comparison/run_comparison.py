#!/usr/bin/env python3
"""
Three-tier V2 pipeline comparison.

For each of the 6 fixed stratified passages, compare:
  Tier A: Raw Opus — bare prompt, no revision
  Tier B: Opus + Sonnet review — bare prompt, then Sonnet diagnoses, Opus revises
  Tier C: Opus + full pipeline — bare prompt, then mechanical + Sonnet diagnosis, Opus revises

Saves all prompts, intermediate outputs, and final translations.

Usage:
  python3 experiments/v2_comparison/run_comparison.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "experiments" / "prompt_optimisation"))

from passage_sampler import sample_passages
import random


def load_fixed_passages() -> dict[str, str]:
    cache = OUT / "fixed_passages.json"
    if cache.exists():
        return json.load(open(cache))
    # Use same seed as the optimisation experiment
    random.seed(42)
    passages = sample_passages(target_words=100)
    cache.write_text(json.dumps(passages, ensure_ascii=False, indent=2))
    return passages


def call_opus(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    collected = []
    with client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=8192,
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
# Tier A: Raw Opus
# ====================================================================

def tier_a_raw(en_text: str) -> dict:
    from translate_v2 import build_translate_prompt
    prompt = build_translate_prompt(en_text)
    greek = call_opus(prompt)
    return {"prompt": prompt, "greek": greek, "revision_prompt": None}


# ====================================================================
# Tier B: Opus + Sonnet review only
# ====================================================================

def tier_b_sonnet_only(en_text: str) -> dict:
    from translate_v2 import build_translate_prompt, run_sonnet_review, build_revision_prompt

    # Translate
    prompt = build_translate_prompt(en_text)
    greek = call_opus(prompt)

    # Sonnet review only
    sonnet = run_sonnet_review(en_text, greek)
    diagnosis = {"sonnet_review": sonnet}

    # Revise if issues found
    rev_prompt = build_revision_prompt(en_text, greek, diagnosis)
    if rev_prompt:
        revised = call_opus(rev_prompt)
    else:
        revised = greek
        rev_prompt = "(no revision needed)"

    return {
        "prompt": prompt,
        "first_draft": greek,
        "sonnet_issues": sonnet.get("issues", []),
        "revision_prompt": rev_prompt,
        "greek": revised,
    }


# ====================================================================
# Tier C: Opus + full pipeline (mechanical + Sonnet)
# ====================================================================

def tier_c_full_pipeline(en_text: str, passage_id: str = "_exp_tmp") -> dict:
    from translate_v2 import (
        build_translate_prompt, run_sonnet_review,
        run_mechanical_checks, check_polysemy, build_revision_prompt,
    )
    import shutil

    # Translate
    prompt = build_translate_prompt(en_text)
    greek = call_opus(prompt)

    # Write temp draft for mechanical checkers
    tmp_dir = ROOT / "drafts" / passage_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "primary.txt").write_text(greek + "\n", encoding="utf-8")

    # Full diagnosis
    mech = run_mechanical_checks(passage_id)
    sonnet = run_sonnet_review(en_text, greek)
    poly = check_polysemy(en_text, greek)

    diagnosis = {}
    diagnosis.update(mech)
    diagnosis["sonnet_review"] = sonnet
    diagnosis["polysemy_issues"] = poly

    # Revise
    rev_prompt = build_revision_prompt(en_text, greek, diagnosis)
    if rev_prompt:
        revised = call_opus(rev_prompt)
    else:
        revised = greek
        rev_prompt = "(no revision needed)"

    # Clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "prompt": prompt,
        "first_draft": greek,
        "mechanical": mech,
        "sonnet_issues": sonnet.get("issues", []),
        "polysemy_issues": poly,
        "revision_prompt": rev_prompt,
        "greek": revised,
    }


# ====================================================================
# Scoring (reuse the quality scorer)
# ====================================================================

def score_output(en_text: str, greek_text: str) -> dict:
    from quality_scorer import llm_judge_score
    return llm_judge_score(en_text, greek_text)


# ====================================================================
# Main
# ====================================================================

def main():
    passages = load_fixed_passages()

    tiers = [
        ("A_raw_opus", tier_a_raw),
        ("B_sonnet_review", tier_b_sonnet_only),
        ("C_full_pipeline", tier_c_full_pipeline),
    ]

    all_results = {}

    for stratum, en_text in passages.items():
        print(f"\n{'='*70}")
        print(f"  STRATUM: {stratum} ({len(en_text.split())} words)")
        print(f"{'='*70}")

        stratum_results = {}

        for tier_name, tier_fn in tiers:
            print(f"\n  --- {tier_name} ---")
            t0 = time.time()

            if tier_name == "C_full_pipeline":
                result = tier_fn(en_text, f"_exp_{stratum}")
            else:
                result = tier_fn(en_text)

            elapsed = time.time() - t0
            print(f"    Translated in {elapsed:.0f}s")

            # Score
            print(f"    Scoring...")
            scores = score_output(en_text, result["greek"])
            result["scores"] = scores
            print(f"    Total: {scores.get('total', '?')}/60")

            stratum_results[tier_name] = result

            # Save individual output
            out_dir = OUT / stratum
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{tier_name}_greek.txt").write_text(result["greek"], encoding="utf-8")
            (out_dir / f"{tier_name}_detail.json").write_text(
                json.dumps({k: v for k, v in result.items() if k != "prompt"},
                           indent=2, ensure_ascii=False, default=str),
                encoding="utf-8"
            )

        all_results[stratum] = stratum_results

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Stratum':<25s}  {'A (raw)':>8s}  {'B (sonnet)':>10s}  {'C (full)':>10s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*10}")

    totals = {"A_raw_opus": 0, "B_sonnet_review": 0, "C_full_pipeline": 0}
    for stratum, results in all_results.items():
        scores = []
        for tier_name in ["A_raw_opus", "B_sonnet_review", "C_full_pipeline"]:
            s = results[tier_name]["scores"].get("total", 0)
            scores.append(s)
            totals[tier_name] += s
        print(f"  {stratum:<25s}  {scores[0]:>8.0f}  {scores[1]:>10.0f}  {scores[2]:>10.0f}")

    print(f"  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*10}")
    print(f"  {'TOTAL':<25s}  {totals['A_raw_opus']:>8.0f}  {totals['B_sonnet_review']:>10.0f}  {totals['C_full_pipeline']:>10.0f}")

    # Save full results
    summary = {
        "experiment": "V2 pipeline three-tier comparison",
        "model_translate": "claude-opus-4-20250514",
        "model_review": "claude-sonnet-4-20250514",
        "model_judge": "claude-opus-4-20250514",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "totals": totals,
        "per_stratum": {
            stratum: {
                tier: {
                    "score": res["scores"].get("total", 0),
                    "voice": res["scores"].get("voice_preservation", 0),
                    "idiom": res["scores"].get("greek_idiom", 0),
                    "register": res["scores"].get("register_match", 0),
                    "grammar": res["scores"].get("grammar", 0),
                    "vocabulary": res["scores"].get("vocabulary", 0),
                    "imagery": res["scores"].get("imagery", 0),
                    "errors": res["scores"].get("errors", []),
                }
                for tier, res in results.items()
            }
            for stratum, results in all_results.items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Saved to {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
