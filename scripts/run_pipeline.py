#!/usr/bin/env python3
"""
Master pipeline for the Diachronic Sense Change project.

Orchestrates the full WP1-WP2 workflow:
  1. Initialize database
  2. Ingest corpus (TEI XML -> DB)
  3. Align translations to Greek passages
  4. Find occurrences of pilot lemmata
  5. Generate evidence extraction prompts
  6. (Future) Run LLM extraction and store results
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
DB_PATH = SCRIPTS_DIR.parent / "db" / "diachronic.db"
RAW_DIR = SCRIPTS_DIR.parent / "corpus" / "raw" / "perseus"


def run_step(name: str, cmd: list[str]) -> bool:
    """Run a pipeline step and report status."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR.parent)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return False
    print(f"  DONE")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the full pipeline")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Start from step N")
    parser.add_argument("--to-step", type=int, default=5,
                        help="Stop after step N")
    parser.add_argument("--occurrence-limit", type=int, default=None,
                        help="Limit passages for occurrence search")
    args = parser.parse_args()

    python = sys.executable
    steps = [
        (1, "Initialize database",
         [python, str(SCRIPTS_DIR / "init_db.py")]),
        (2, "Ingest corpus into database",
         [python, str(SCRIPTS_DIR / "ingest_corpus.py")]),
        (3, "Align translations to Greek passages",
         [python, str(SCRIPTS_DIR / "align_translations.py")]),
        (4, "Find occurrences of pilot lemmata",
         [python, str(SCRIPTS_DIR / "find_occurrences.py")]
         + (["--limit", str(args.occurrence_limit)] if args.occurrence_limit else [])),
        (5, "Generate evidence extraction prompts",
         [python, str(SCRIPTS_DIR / "extract_evidence.py"),
          "--limit", "10",
          "--output", str(SCRIPTS_DIR.parent / "corpus" / "structured" / "sample_prompts.json")]),
    ]

    for step_num, name, cmd in steps:
        if step_num < args.from_step:
            continue
        if step_num > args.to_step:
            break
        if not run_step(name, cmd):
            print(f"\nPipeline stopped at step {step_num}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")

    # Print summary stats
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    stats = {
        "Authors": conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0],
        "Documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        "Greek editions": conn.execute("SELECT COUNT(*) FROM editions").fetchone()[0],
        "Translations": conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0],
        "Passages": conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0],
        "Alignments": conn.execute("SELECT COUNT(*) FROM alignments").fetchone()[0],
        "Occurrences": conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0],
        "Sense inventory": conn.execute("SELECT COUNT(*) FROM sense_inventory").fetchone()[0],
    }
    conn.close()

    print("\nDatabase summary:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
