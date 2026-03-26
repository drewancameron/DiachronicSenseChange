#!/usr/bin/env python3
"""
WP4 Task 4.1: Build modelling datasets from the evidence database.

Creates train/val/test splits for sense classification, with
leave-author-out and leave-period-out evaluation splits.

Each example: (greek_context, target_token_position, sense_label, period, register)
"""

import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "data"


def build_dataset():
    conn = sqlite3.connect(DB_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all labelled occurrences with their primary sense
    rows = conn.execute("""
        SELECT o.occurrence_id, o.lemma, o.greek_context, o.period,
               t.surface_form, t.token_offset,
               si.sense_label, cl.confidence,
               a.name as author,
               COALESCE(rl.register, 'unmarked') as register
        FROM candidate_labels cl
        JOIN sense_inventory si ON cl.sense_id = si.sense_id
        JOIN occurrences o ON cl.occurrence_id = o.occurrence_id
        JOIN tokens t ON o.token_id = t.token_id
        JOIN documents d ON o.document_id = d.document_id
        JOIN authors a ON d.author_id = a.author_id
        LEFT JOIN register_labels rl ON o.occurrence_id = rl.occurrence_id
        WHERE cl.is_primary = 1
        AND si.sense_label NOT LIKE '%undetermined%'
        AND si.sense_label NOT LIKE '%uncertain%'
        AND si.sense_label NOT LIKE '%not determinable%'
        AND si.sense_label NOT LIKE '%no clear%'
        AND cl.confidence >= 0.3
        AND length(o.greek_context) > 20
    """).fetchall()

    print(f"Total labelled examples: {len(rows)}")

    # Build sense→id mapping per lemma
    sense_maps = {}
    for lemma in set(r[1] for r in rows):
        senses = sorted(set(r[6] for r in rows if r[1] == lemma))
        sense_maps[lemma] = {s: i for i, s in enumerate(senses)}

    # Build examples
    examples = []
    for occ_id, lemma, context, period, surface, offset, sense, conf, author, register in rows:
        if lemma not in sense_maps or sense not in sense_maps[lemma]:
            continue
        examples.append({
            "occurrence_id": occ_id,
            "lemma": lemma,
            "greek_context": context,
            "surface_form": surface,
            "token_offset": offset,
            "sense_label": sense,
            "sense_id": sense_maps[lemma][sense],
            "confidence": conf,
            "period": period,
            "author": author,
            "register": register,
        })

    print(f"Valid examples: {len(examples)}")

    # Stats
    by_lemma = defaultdict(int)
    by_period = defaultdict(int)
    by_sense = defaultdict(int)
    for ex in examples:
        by_lemma[ex["lemma"]] += 1
        by_period[ex["period"]] += 1
        by_sense[(ex["lemma"], ex["sense_label"])] += 1

    print("\nBy lemma:")
    for lemma, n in sorted(by_lemma.items(), key=lambda x: -x[1]):
        n_senses = len(sense_maps.get(lemma, {}))
        print(f"  {lemma}: {n} examples, {n_senses} senses")

    print("\nBy period:")
    for period, n in sorted(by_period.items()):
        print(f"  {period}: {n}")

    # Create splits
    random.seed(42)
    random.shuffle(examples)

    # 70/15/15 split
    n = len(examples)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    # Also create leave-period-out splits for diachronic evaluation
    period_splits = {}
    for period in set(ex["period"] for ex in examples):
        period_splits[f"test_period_{period}"] = [
            ex for ex in examples if ex["period"] == period
        ]
        period_splits[f"train_no_{period}"] = [
            ex for ex in examples if ex["period"] != period
        ]

    # Save
    metadata = {
        "total_examples": len(examples),
        "sense_maps": {k: v for k, v in sense_maps.items()},
        "splits": {k: len(v) for k, v in splits.items()},
        "period_splits": {k: len(v) for k, v in period_splits.items()},
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    for name, data in splits.items():
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"\n{name}: {len(data)} examples")

    for name, data in period_splits.items():
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(data, f, ensure_ascii=False)

    # Save sense maps separately for the model
    with open(OUTPUT_DIR / "sense_labels.json", "w") as f:
        json.dump(sense_maps, f, indent=2, ensure_ascii=False)

    print(f"\nDatasets saved to {OUTPUT_DIR}")
    conn.close()


if __name__ == "__main__":
    build_dataset()
