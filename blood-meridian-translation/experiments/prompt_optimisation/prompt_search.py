#!/usr/bin/env python3
"""
Stochastic prompt optimisation via hill-climbing.

Algorithm:
  1. Start with a 'current' prompt configuration
  2. Each step:
     a. Draw 6 stratified random passages (~100 words each)
     b. Propose a new config (flip one random dimension)
     c. Translate all 6 passages with BOTH current and proposed configs
     d. Score all 12 translations (mechanical + LLM judge)
     e. If proposed aggregate score > current, accept the proposal
  3. Log the full history for post-hoc analysis

Usage:
  python3 experiments/prompt_optimisation/prompt_search.py --steps 10
  python3 experiments/prompt_optimisation/prompt_search.py --steps 100 --start bare
  python3 experiments/prompt_optimisation/prompt_search.py --steps 1000 --start current_best
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_config import (
    DIMENSIONS, BARE_CONFIG, CURRENT_BEST_CONFIG,
    propose, config_to_key,
)
from prompt_builder import build_prompt
from passage_sampler import sample_passages
from quality_scorer import score_translation

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


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


def evaluate_config(config: dict, passages: dict[str, str]) -> dict:
    """Translate and score all passages with a given config.

    Returns dict with per-stratum scores and aggregate.
    """
    prompt_text = build_prompt  # just to check it's importable
    results = {}
    total = 0

    for stratum, en_text in passages.items():
        prompt = build_prompt(en_text, config)
        greek = call_opus(prompt)
        scores = score_translation(en_text, greek)

        results[stratum] = {
            "english": en_text[:100] + "...",
            "greek": greek[:100] + "...",
            "greek_full": greek,
            "prompt_chars": len(prompt),
            "scores": scores,
            "combined": scores["combined"],
        }
        total += scores["combined"]

    results["aggregate"] = total
    return results


def run_search(n_steps: int, start_config: dict, run_id: str):
    """Run the hill-climbing search."""
    current = start_config.copy()
    current_score = None  # evaluated lazily on first step
    history = []

    print(f"╔{'═'*60}╗")
    print(f"║  Prompt optimisation: {n_steps} steps, 6 strata per step")
    print(f"╚{'═'*60}╝")
    print(f"  Start config: {config_to_key(current)[:80]}...")
    print(f"  Dimensions: {len(DIMENSIONS)}, total levels: {sum(len(v) for v in DIMENSIONS.values())}")

    for step in range(n_steps):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  Step {step + 1}/{n_steps}")
        print(f"{'─'*60}")

        # Draw stratified passages
        passages = sample_passages(target_words=100)
        strata_summary = {k: len(v.split()) for k, v in passages.items()}
        print(f"  Passages: {strata_summary}")

        # Propose
        proposed = propose(current)
        changed_dim = [k for k in DIMENSIONS if current[k] != proposed[k]][0]
        print(f"  Proposal: {changed_dim}: {current[changed_dim]} → {proposed[changed_dim]}")

        # Evaluate BOTH on the same passages for fair comparison
        print(f"  Evaluating current config...")
        current_eval = evaluate_config(current, passages)
        current_score_this_step = current_eval["aggregate"]

        print(f"  Evaluating proposed config...")
        proposed_eval = evaluate_config(proposed, passages)
        proposed_score = proposed_eval["aggregate"]

        # Accept/reject based on this step's head-to-head
        accepted = proposed_score > current_score_this_step
        current_score = current_score_this_step  # track for logging
        if accepted:
            print(f"  ✓ ACCEPTED: {current_score:.1f} → {proposed_score:.1f} (+{proposed_score - current_score:.1f})")
            current = proposed
            current_score = proposed_score
        else:
            print(f"  ✗ Rejected: {current_score:.1f} vs {proposed_score:.1f} ({proposed_score - current_score:.1f})")

        elapsed = time.time() - t0

        # Log
        entry = {
            "step": step + 1,
            "current_config": current.copy(),
            "proposed_config": proposed.copy(),
            "changed_dimension": changed_dim,
            "current_score": current_score,
            "proposed_score": proposed_score,
            "accepted": accepted,
            "passages": {k: v[:80] + "..." for k, v in passages.items()},
            "elapsed_s": round(elapsed, 1),
        }
        # Include per-stratum scores for proposed
        entry["proposed_strata"] = {
            k: v["combined"] for k, v in proposed_eval.items() if k != "aggregate"
        }
        history.append(entry)

        # Save incrementally
        log_path = OUT / f"{run_id}_history.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  Time: {elapsed:.0f}s | Current best: {current_score:.1f}")
        print(f"  Config: {config_to_key(current)[:80]}...")

    # Save final state
    final = {
        "run_id": run_id,
        "n_steps": n_steps,
        "start_config": start_config,
        "final_config": current,
        "final_score": current_score,
        "history_length": len(history),
    }
    (OUT / f"{run_id}_final.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'═'*60}")
    print(f"  Done: {n_steps} steps")
    print(f"  Final score: {current_score:.1f}")
    print(f"  Final config: {json.dumps(current, indent=2)}")
    print(f"  Results: {OUT / run_id}_*.json")
    print(f"{'═'*60}")

    return current, current_score, history


# ====================================================================
# Cost estimation
# ====================================================================

def estimate_cost(n_steps: int):
    """Estimate Opus API cost for a search run.

    Per step:
      - 6 translations with current config (6 Opus calls, ~500 input + ~200 output tokens each)
      - 6 translations with proposed config (same)
      - 6 LLM judge calls for current (6 Opus calls, ~800 input + ~300 output tokens each)
      - 6 LLM judge calls for proposed (same)
      - First step only: extra 6+6 for initial eval
    Total per step: 24 Opus calls (12 translate + 12 judge)
    First step: 36 calls

    But we only evaluate current on step 1, then reuse. So:
      Step 1: 24 calls (proposed) + 24 calls (current) = 48
      Steps 2+: 24 calls (proposed only, current score carried forward)

    Wait — current score is from different passages each step. We need to
    re-evaluate current on the new passages too. So it's 24 calls per step
    for proposed, plus 24 for current = 48 per step.

    Actually: we evaluate current lazily on step 1 only, then carry the score.
    But passages change each step! The current implementation only evaluates
    current on step 1. For fairness we should evaluate both on the same passages.
    """
    # Let me recalculate based on actual implementation:
    # Step 1: evaluate current (6 translate + 6 judge = 12 Opus calls)
    #        + evaluate proposed (6 translate + 6 judge = 12 calls) = 24 total
    # Step 2+: evaluate proposed only (12 calls) — BUT this is unfair since
    #          current wasn't evaluated on these passages.
    #
    # For a fair comparison we need both evaluated on same passages = 24 per step.
    # Let me note this as a design issue.

    # Per-step with fair comparison:
    calls_per_step = 24  # 12 current + 12 proposed

    # Token estimates (Opus pricing: $15/M input, $75/M output)
    avg_translate_input = 3000   # tokens (prompt)
    avg_translate_output = 300   # tokens (Greek text)
    avg_judge_input = 1500       # tokens (rubric + texts)
    avg_judge_output = 400       # tokens (JSON scores)

    translate_calls = 12 * n_steps
    judge_calls = 12 * n_steps

    input_tokens = (
        translate_calls * avg_translate_input +
        judge_calls * avg_judge_input
    )
    output_tokens = (
        translate_calls * avg_translate_output +
        judge_calls * avg_judge_output
    )

    input_cost = input_tokens * 15 / 1_000_000
    output_cost = output_tokens * 75 / 1_000_000
    total_cost = input_cost + output_cost

    # Time estimate: ~30s per Opus call
    time_s = calls_per_step * n_steps * 15  # ~15s average per call
    time_h = time_s / 3600

    print(f"  Steps: {n_steps}")
    print(f"  Opus calls per step: {calls_per_step} (12 translate + 12 judge)")
    print(f"  Total Opus calls: {calls_per_step * n_steps}")
    print(f"  Estimated input tokens: {input_tokens:,}")
    print(f"  Estimated output tokens: {output_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.2f} (input ${input_cost:.2f} + output ${output_cost:.2f})")
    print(f"  Estimated time: {time_h:.1f} hours")

    return {"calls": calls_per_step * n_steps, "cost_usd": round(total_cost, 2),
            "time_hours": round(time_h, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--start", choices=["bare", "current_best"], default="bare")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    start = BARE_CONFIG if args.start == "bare" else CURRENT_BEST_CONFIG
    run_id = args.run_id or f"run_{args.start}_{args.steps}steps_{int(time.time())}"

    if args.estimate_only:
        print(f"\n  Cost estimate for {args.steps} steps:")
        estimate_cost(args.steps)
        return

    run_search(args.steps, start, run_id)


if __name__ == "__main__":
    main()
