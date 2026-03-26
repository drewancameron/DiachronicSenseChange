#!/usr/bin/env python3
"""
WP2: Fast stratified sampling of occurrences for evidence extraction.

Selects a balanced sample per lemma, stratified by period,
prioritizing occurrences that have aligned translations.
"""

import random
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
TARGET_PER_LEMMA = 80
MIN_PER_PERIOD = 5


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_PER_LEMMA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    conn = sqlite3.connect(DB_PATH)

    # Pre-compute: which passages have aligned translations?
    print("Building translation index...", flush=True)
    passages_with_trans = set()
    rows = conn.execute("""
        SELECT DISTINCT passage_id FROM alignments
        WHERE aligned_text NOT LIKE '[awaiting%'
        AND length(aligned_text) > 20
    """).fetchall()
    for (pid,) in rows:
        passages_with_trans.add(pid)
    print(f"  {len(passages_with_trans):,} passages have translations", flush=True)

    # Pre-compute: which passages have linked notes?
    passages_with_notes = set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT passage_id FROM note_alignments WHERE passage_id IS NOT NULL"
        ).fetchall()
        for (pid,) in rows:
            passages_with_notes.add(pid)
    except Exception:
        pass
    print(f"  {len(passages_with_notes):,} passages have notes", flush=True)

    # Clear previous sample
    conn.execute("DELETE FROM model_splits WHERE split_name = 'wp2_extraction_sample'")

    pilot_lemmata = [
        ("κόσμος", "kosmos"), ("λόγος", "logos"), ("ψυχή", "psyche"),
        ("ἀρετή", "arete"), ("δίκη", "dike"), ("τέχνη", "techne"),
        ("νόμος", "nomos"), ("φύσις", "physis"), ("δαίμων", "daimon"),
        ("σῶμα", "soma"), ("θεός", "theos"), ("χάρις", "charis"),
    ]

    total = 0
    for lemma, translit in pilot_lemmata:
        # Get all occurrences with metadata
        rows = conn.execute("""
            SELECT o.occurrence_id, o.period, a.name, o.passage_id
            FROM occurrences o
            JOIN documents d ON o.document_id = d.document_id
            JOIN authors a ON d.author_id = a.author_id
            WHERE o.lemma = ?
            AND o.greek_context IS NOT NULL
            AND length(o.greek_context) > 20
        """, (lemma,)).fetchall()

        # Score: prefer occurrences with translations and notes
        scored = []
        for occ_id, period, author, pid in rows:
            score = 1.0
            if pid in passages_with_trans:
                score += 3.0
            if pid in passages_with_notes:
                score += 2.0
            scored.append({
                "occurrence_id": occ_id,
                "period": period or "unknown",
                "author": author,
                "score": score,
            })

        # Group by period
        by_period = defaultdict(list)
        for occ in scored:
            by_period[occ["period"]].append(occ)

        selected = []
        target = args.target

        # First: ensure minimum per period, preferring high-score
        for period in sorted(by_period.keys()):
            pool = sorted(by_period[period], key=lambda x: -x["score"])
            n = min(MIN_PER_PERIOD, len(pool))
            selected.extend(pool[:n])

        # Then fill remaining quota from highest-scoring across all periods
        selected_ids = {s["occurrence_id"] for s in selected}
        remaining = target - len(selected)
        if remaining > 0:
            candidates = [o for o in scored if o["occurrence_id"] not in selected_ids]
            candidates.sort(key=lambda x: -x["score"])
            # Add some randomness within the top tier
            top = [c for c in candidates if c["score"] >= 4.0]
            rest = [c for c in candidates if c["score"] < 4.0]
            if len(top) > remaining:
                random.shuffle(top)
                selected.extend(top[:remaining])
            else:
                selected.extend(top)
                still_need = remaining - len(top)
                if still_need > 0 and rest:
                    random.shuffle(rest)
                    selected.extend(rest[:still_need])

        # Store
        for occ in selected:
            conn.execute(
                """INSERT INTO model_splits (occurrence_id, split_name, split_version)
                   VALUES (?, 'wp2_extraction_sample', 1)""",
                (occ["occurrence_id"],),
            )

        # Stats
        period_counts = defaultdict(int)
        with_trans = sum(1 for o in selected if o["score"] >= 4.0)
        for occ in selected:
            period_counts[occ["period"]] += 1

        print(f"\n{translit} ({lemma}): {len(selected)} sampled "
              f"({with_trans} with translations)", flush=True)
        print(f"  Periods: {dict(period_counts)}", flush=True)

        total += len(selected)

    conn.commit()

    # Cost estimate
    est_cost = (total * 2000 * 0.15 + total * 500 * 0.60) / 1_000_000
    print(f"\n{'='*60}")
    print(f"Total sampled: {total}")
    print(f"Estimated extraction cost (mini): ${est_cost:.2f}")

    conn.close()


if __name__ == "__main__":
    main()
