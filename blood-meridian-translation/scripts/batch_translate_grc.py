#!/usr/bin/env python3
"""
Batch-translate Ancient Greek sentences to English using Claude Haiku.

Uses the Anthropic Message Batches API for cost-efficient bulk translation.
The translations are rough sense-glosses — we only need them to be close
enough for EN(auto) ↔ EN(human) embedding similarity to work.

Pipeline:
  1. Extract Greek sentences from the database (all passages with text)
  2. Build batch request JSONL
  3. Submit to Anthropic Batch API
  4. Poll for completion
  5. Parse results and store EN translations alongside Greek
  6. Compute EN(auto) ↔ EN(human) embedding similarity to verify alignments

Usage:
  python3 scripts/batch_translate_grc.py --prepare       # build batch JSONL
  python3 scripts/batch_translate_grc.py --submit        # submit to API
  python3 scripts/batch_translate_grc.py --check BATCH_ID  # check status
  python3 scripts/batch_translate_grc.py --retrieve BATCH_ID  # download results
  python3 scripts/batch_translate_grc.py --verify        # compute embedding similarities
  python3 scripts/batch_translate_grc.py --pilot 200     # small test run (non-batch, direct API)
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT.parent / "db" / "diachronic.db"
OUTPUT_DIR = ROOT / "models" / "translations"
BATCH_INPUT = OUTPUT_DIR / "batch_input.jsonl"
BATCH_OUTPUT = OUTPUT_DIR / "batch_output.jsonl"
TRANSLATIONS_DB = OUTPUT_DIR / "translations.jsonl"

SYSTEM_PROMPT = """You are a classicist translating Ancient Greek to English.
Translate the following Ancient Greek sentence to English literally.
Give ONLY the English translation, no explanation or notes.
Preserve the sentence structure as closely as possible.
If a word is unclear, give your best guess."""


def load_greek_passages(limit: int = None) -> list[dict]:
    """Load Greek passages that have human English translations."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    query = """
        SELECT p.passage_id, p.greek_text, a.aligned_text, d.title, d.period
        FROM alignments a
        JOIN passages p ON a.passage_id = p.passage_id
        JOIN documents d ON p.document_id = d.document_id
        WHERE a.alignment_method = 'reference_match'
          AND a.aligned_text NOT LIKE '[awaiting%' AND a.aligned_text != ''
          AND LENGTH(a.aligned_text) BETWEEN 20 AND 300
          AND LENGTH(p.greek_text) BETWEEN 20 AND 400
          AND a.aligned_text NOT LIKE '%[%'
        ORDER BY p.passage_id
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    return [
        {"passage_id": r[0], "greek": r[1].replace("\n", " ").strip(),
         "english_human": r[2], "source": r[3], "period": r[4] or ""}
        for r in rows
    ]


def prepare_batch(passages: list[dict]):
    """Build Anthropic batch request JSONL."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BATCH_INPUT, "w") as f:
        for p in passages:
            request = {
                "custom_id": str(p["passage_id"]),
                "params": {
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 300,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": p["greek"]}
                    ]
                }
            }
            f.write(json.dumps(request) + "\n")
    print(f"Prepared {len(passages)} requests → {BATCH_INPUT}")


def submit_batch():
    """Submit batch to Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    with open(BATCH_INPUT, "r") as f:
        requests = [json.loads(line) for line in f if line.strip()]

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")

    # Save batch ID
    meta = {"batch_id": batch.id, "n_requests": len(requests),
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(OUTPUT_DIR / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return batch.id


def check_batch(batch_id: str):
    """Check batch status."""
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch {batch_id}:")
    print(f"  Status: {batch.processing_status}")
    print(f"  Counts: {batch.request_counts}")
    return batch.processing_status


def retrieve_batch(batch_id: str):
    """Download batch results."""
    import anthropic
    client = anthropic.Anthropic()

    results = []
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text.strip()
            results.append({"passage_id": int(custom_id), "english_auto": text})
        else:
            results.append({"passage_id": int(custom_id), "english_auto": "",
                            "error": str(result.result)})

    with open(BATCH_OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Retrieved {len(results)} results → {BATCH_OUTPUT}")
    errors = sum(1 for r in results if "error" in r)
    if errors:
        print(f"  {errors} errors")


def pilot_translate(n: int = 200):
    """Direct API translation for a small pilot (not batch)."""
    import anthropic

    passages = load_greek_passages(limit=n)
    client = anthropic.Anthropic()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"Translating {len(passages)} passages with Haiku...")

    for i, p in enumerate(passages):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": p["greek"]}],
            )
            en_auto = msg.content[0].text.strip()
        except Exception as e:
            en_auto = ""
            print(f"  Error at {i}: {e}")

        results.append({
            "passage_id": p["passage_id"],
            "greek": p["greek"],
            "english_human": p["english_human"],
            "english_auto": en_auto,
            "source": p["source"],
            "period": p["period"],
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(passages)}")

    with open(TRANSLATIONS_DB, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} translations → {TRANSLATIONS_DB}")
    return results


def verify_alignments():
    """Compute EN(auto) ↔ EN(human) embedding similarity."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Load translations
    translations = []
    with open(TRANSLATIONS_DB) as f:
        for line in f:
            if line.strip():
                translations.append(json.loads(line))

    en_human = [t["english_human"] for t in translations]
    en_auto = [t["english_auto"] for t in translations]

    # Filter out empty auto-translations
    valid = [(h, a, t) for h, a, t in zip(en_human, en_auto, translations) if a]
    if not valid:
        print("No translations to verify.")
        return

    human_texts = [v[0] for v in valid]
    auto_texts = [v[1] for v in valid]

    print(f"Computing embeddings for {len(valid)} pairs...")
    human_embs = model.encode(human_texts, normalize_embeddings=True)
    auto_embs = model.encode(auto_texts, normalize_embeddings=True)

    sims = np.sum(human_embs * auto_embs, axis=1)

    print(f"\nEN(Haiku auto) ↔ EN(human) similarity:")
    print(f"  Mean: {sims.mean():.3f}")
    print(f"  Median: {np.median(sims):.3f}")
    print(f"  >0.5 (good): {(sims > 0.5).sum()} ({(sims > 0.5).mean():.0%})")
    print(f"  >0.3 (acceptable): {(sims > 0.3).sum()} ({(sims > 0.3).mean():.0%})")
    print(f"  <0.2 (likely misaligned): {(sims < 0.2).sum()} ({(sims < 0.2).mean():.0%})")

    # Mark verified pairs
    verified_path = OUTPUT_DIR / "verified_pairs.jsonl"
    n_good = 0
    with open(verified_path, "w") as f:
        for (h, a, t), sim in zip(valid, sims):
            t["similarity"] = round(float(sim), 3)
            t["verified"] = sim > 0.3
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            if sim > 0.3:
                n_good += 1

    print(f"\nVerified pairs (sim > 0.3): {n_good}/{len(valid)}")
    print(f"Saved → {verified_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--check", type=str)
    parser.add_argument("--retrieve", type=str)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--pilot", type=int, help="Pilot run with N passages")
    args = parser.parse_args()

    if args.prepare:
        passages = load_greek_passages()
        prepare_batch(passages)
    elif args.submit:
        submit_batch()
    elif args.check:
        check_batch(args.check)
    elif args.retrieve:
        retrieve_batch(args.retrieve)
    elif args.verify:
        verify_alignments()
    elif args.pilot:
        pilot_translate(args.pilot)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
