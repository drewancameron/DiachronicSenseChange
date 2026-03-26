#!/usr/bin/env python3
"""
Greedy forward search for optimal prompt configuration.

Fix 6 stratified passages. Start with bare minimum prompt. At each round,
test adding the first level of each unused dimension, one at a time.
Keep the one that improves score most. Repeat until no improvement.

Then optionally: for each active dimension, test upgrading to higher levels.

Usage:
  python3 experiments/prompt_optimisation/greedy_search.py
  python3 experiments/prompt_optimisation/greedy_search.py --dry-run  # show passages only
"""

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_config import DIMENSIONS, BARE_CONFIG, config_to_key
from prompt_builder import build_prompt
from quality_scorer import score_translation

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# ====================================================================
# Fixed test passages (seed=42)
# ====================================================================

def load_fixed_passages() -> dict[str, str]:
    """Load or generate the fixed test passages."""
    cache_path = OUT / "fixed_passages.json"
    if cache_path.exists():
        return json.load(open(cache_path))

    from passage_sampler import sample_passages
    random.seed(42)
    passages = sample_passages(target_words=100)
    cache_path.write_text(json.dumps(passages, ensure_ascii=False, indent=2))
    return passages


# ====================================================================
# Translation
# ====================================================================

def call_opus(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    collected = []
    with client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)
    return "".join(collected).strip()


def evaluate_config(config: dict, passages: dict[str, str], label: str = "") -> dict:
    """Translate all passages with config, score each, return aggregate."""
    results = {}
    total = 0.0

    for stratum, en_text in passages.items():
        prompt = build_prompt(en_text, config)
        t0 = time.time()
        greek = call_opus(prompt)
        translate_time = time.time() - t0

        t0 = time.time()
        scores = score_translation(en_text, greek)
        score_time = time.time() - t0

        combined = scores["combined"]
        results[stratum] = {
            "combined": combined,
            "llm_total": scores["llm_judge"].get("total", 0),
            "mech_penalty": scores["mechanical"].get("penalty", 0),
            "errors": scores["llm_judge"].get("errors", []),
            "greek": greek[:200],
            "prompt_chars": len(prompt),
            "translate_s": round(translate_time, 1),
            "score_s": round(score_time, 1),
        }
        total += combined

    results["aggregate"] = round(total, 1)
    if label:
        print(f"    {label}: aggregate={total:.1f}  "
              f"({', '.join(f'{k}={v['combined']:.0f}' for k, v in results.items() if k != 'aggregate')})")
    return results


# ====================================================================
# Greedy forward search
# ====================================================================

def greedy_forward(passages: dict[str, str]):
    """Add features one at a time, keeping whatever improves score most."""
    current = deepcopy(BARE_CONFIG)
    history = []

    # First: evaluate the bare baseline
    print(f"\n{'='*70}")
    print(f"  BASELINE: bare config")
    print(f"{'='*70}")
    baseline = evaluate_config(current, passages, "BARE")
    baseline_score = baseline["aggregate"]

    history.append({
        "round": 0,
        "action": "baseline",
        "config": deepcopy(current),
        "score": baseline_score,
        "detail": baseline,
    })

    round_num = 0
    improved = True

    while improved:
        round_num += 1
        improved = False
        best_dim = None
        best_level = None
        best_score = baseline_score
        best_eval = None

        print(f"\n{'='*70}")
        print(f"  ROUND {round_num}: testing each unused/upgradeable dimension")
        print(f"  Current score: {baseline_score:.1f}")
        print(f"  Current config: {config_to_key(current)[:80]}...")
        print(f"{'='*70}")

        # For each dimension, try the next level up (or first level if off)
        candidates = []
        for dim, levels in DIMENSIONS.items():
            current_level = current[dim]
            current_idx = levels.index(current_level)

            # Try each level above current
            for target_idx in range(current_idx + 1, len(levels)):
                target_level = levels[target_idx]
                candidates.append((dim, target_level))

        if not candidates:
            print("  No more upgrades possible.")
            break

        for dim, target_level in candidates:
            test_config = deepcopy(current)
            test_config[dim] = target_level

            print(f"\n  Testing: {dim} = {current[dim]} → {target_level}")
            t0 = time.time()
            result = evaluate_config(test_config, passages, f"{dim}={target_level}")
            elapsed = time.time() - t0
            score = result["aggregate"]
            delta = score - baseline_score

            print(f"    Δ = {delta:+.1f}  (took {elapsed:.0f}s)")

            if score > best_score:
                best_score = score
                best_dim = dim
                best_level = target_level
                best_eval = result

        if best_dim is not None:
            improved = True
            current[best_dim] = best_level
            baseline_score = best_score
            print(f"\n  ✓ BEST: {best_dim} = {best_level}  (score {best_score:.1f}, Δ = {best_score - history[-1]['score']:+.1f})")

            history.append({
                "round": round_num,
                "action": f"{best_dim} → {best_level}",
                "config": deepcopy(current),
                "score": best_score,
                "detail": best_eval,
            })
        else:
            print(f"\n  No improvement found. Stopping.")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Rounds: {round_num}")
    print(f"  Final score: {baseline_score:.1f} (started at {history[0]['score']:.1f})")
    print(f"  Final config:")
    for dim, level in sorted(current.items()):
        marker = "  " if level == "off" else "✓ "
        print(f"    {marker}{dim}: {level}")
    print(f"\n  History:")
    for h in history:
        print(f"    Round {h['round']}: {h['action']:40s} score={h['score']:.1f}")

    # Save
    result = {
        "final_config": current,
        "final_score": baseline_score,
        "history": history,
        "passages": {k: v[:100] + "..." for k, v in passages.items()},
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    out_path = OUT / f"greedy_forward_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  Saved to {out_path.name}")

    return current, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show passages only")
    args = parser.parse_args()

    passages = load_fixed_passages()

    if args.dry_run:
        for stratum, text in passages.items():
            print(f"\n=== {stratum} ({len(text.split())} words) ===")
            print(text[:300])

        # Cost estimate: baseline (12 calls) + up to 8 dims × 2 avg levels = 16 tests × 12 calls
        # Round 1: test 8 dims × ~1.5 levels avg = ~12 tests
        # Round 2: test remaining ~11 options
        # Round 3: maybe ~8 options
        # Total: ~35 tests × 12 calls = 420 Opus calls
        # Plus baseline = 432 calls
        # Cost: 432 × (564+1800)/1M × $15 + 432 × (250+400)/1M × $75 ≈ $36
        n_tests = sum(len(levels) - 1 for levels in DIMENSIONS.values()) + 1  # baseline
        calls = n_tests * 12
        cost = calls * ((564 + 1800) * 15 + (250 + 400) * 75) / 1_000_000
        print(f"\n  === Cost estimate ===")
        print(f"  Max configs to test: {n_tests} (baseline + {n_tests-1} upgrades)")
        print(f"  Max Opus calls: {calls} (12 per config: 6 translate + 6 judge)")
        print(f"  Estimated cost: ${cost:.0f}")
        print(f"  But greedy stops early — likely ~60% of max = ~${cost*0.6:.0f}")
        return

    greedy_forward(passages)


if __name__ == "__main__":
    main()
