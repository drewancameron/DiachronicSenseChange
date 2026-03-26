#!/usr/bin/env python3
"""
Step 2: Build conditional distribution model from extracted construction pairs.

Reads construction_pairs.jsonl (from extract_parallel_constructions.py) and
computes P(GRC_construction | EN_construction) at each scale, optionally
conditioned on author/period.

Also builds a tree-edit-distance index for kernel smoothing of rare patterns.

Usage:
  python3 scripts/build_construction_model.py
  python3 scripts/build_construction_model.py --by-period
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "construction_model"
PAIRS_PATH = MODEL_DIR / "construction_pairs.jsonl"
MODEL_PATH = MODEL_DIR / "cond_distributions.json"


def load_pairs() -> list[dict]:
    records = []
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_distributions(records: list[dict], by_period: bool = False) -> dict:
    """Build P(GRC_type | EN_type) from aligned pairs."""

    # Count (en_type, grc_type) co-occurrences
    # Optionally keyed by period
    joint_counts = defaultdict(Counter)    # en_type → Counter of grc_type
    en_totals = Counter()
    phrase_counts = defaultdict(Counter)   # For phrase-level patterns

    # Period-conditioned counts
    period_joint = defaultdict(lambda: defaultdict(Counter))

    for rec in records:
        source = rec.get("source", "")
        period = rec.get("period", "unknown") or "unknown"

        for pair in rec.get("pairs", []):
            en_type = pair["en_type"]
            grc_type = pair["grc_type"]

            if en_type == "none":
                continue  # GRC-only insertions (interesting but separate)

            joint_counts[en_type][grc_type] += 1
            en_totals[en_type] += 1

            if by_period:
                period_joint[period][en_type][grc_type] += 1

        # Phrase-level: collect GRC verb forms, PP patterns, etc.
        for c in rec.get("grc_constructions", []):
            if c["scale"] == "phrase" and c["type"] == "verb_form":
                key = f"verb_{c.get('tense', '?')}_{c.get('mood', '?')}_{c.get('voice', '?')}"
                phrase_counts["verb_forms"][key] += 1
            elif c["scale"] == "phrase" and c["type"] == "pp":
                key = f"{c.get('prep', '?')}+{c.get('case', '?')}"
                phrase_counts["pp_patterns"][key] += 1

    # Convert to probability distributions
    distributions = {}
    for en_type, grc_counter in joint_counts.items():
        total = en_totals[en_type]
        dist = {}
        for grc_type, count in grc_counter.most_common():
            dist[grc_type] = {
                "count": count,
                "probability": round(count / total, 4),
            }
        distributions[en_type] = {
            "total": total,
            "distribution": dist,
        }

    # Period-conditioned distributions
    period_dists = {}
    if by_period:
        for period, en_types in period_joint.items():
            period_dists[period] = {}
            for en_type, grc_counter in en_types.items():
                total = sum(grc_counter.values())
                dist = {}
                for grc_type, count in grc_counter.most_common():
                    dist[grc_type] = {
                        "count": count,
                        "probability": round(count / total, 4),
                    }
                period_dists[period][en_type] = {
                    "total": total,
                    "distribution": dist,
                }

    return {
        "overall": distributions,
        "by_period": period_dists,
        "phrase_patterns": {k: dict(v.most_common(50)) for k, v in phrase_counts.items()},
        "total_records": len(records),
        "total_pairs": sum(en_totals.values()),
    }


def print_summary(model: dict):
    """Print a readable summary of the conditional distributions."""
    print(f"\n{'='*70}")
    print(f"Construction Model: {model['total_records']} records, {model['total_pairs']} pairs")
    print(f"{'='*70}")

    for en_type, info in sorted(model["overall"].items(), key=lambda x: -x[1]["total"]):
        print(f"\n  P(GRC | EN={en_type})  [n={info['total']}]")
        for grc_type, stats in info["distribution"].items():
            bar = "█" * int(stats["probability"] * 40)
            print(f"    {grc_type:30s} {stats['probability']:5.1%} {bar}  ({stats['count']})")

    # Period breakdown for relative clauses
    if model.get("by_period"):
        print(f"\n{'─'*70}")
        print("  P(GRC | EN=relative_clause) by period:")
        for period, types in sorted(model["by_period"].items()):
            if "relative_clause" in types:
                info = types["relative_clause"]
                top = list(info["distribution"].items())[:3]
                top_str = ", ".join(f"{t}: {s['probability']:.0%}" for t, s in top)
                print(f"    {period:20s} [n={info['total']:4d}] {top_str}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--by-period", action="store_true")
    args = parser.parse_args()

    if not PAIRS_PATH.exists():
        print(f"No construction pairs found at {PAIRS_PATH}")
        print("Run extract_parallel_constructions.py first.")
        return

    print("Loading construction pairs...")
    records = load_pairs()
    print(f"  {len(records)} records")

    print("Building conditional distributions...")
    model = build_distributions(records, by_period=args.by_period)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "w") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(f"Model saved to {MODEL_PATH}")

    print_summary(model)


if __name__ == "__main__":
    main()
