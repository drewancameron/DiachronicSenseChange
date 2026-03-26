#!/usr/bin/env python3
"""
Master segmentation pipeline.

Usage:
  # Dry run (estimate costs):
  python3 scripts/run_segmentation.py --dry-run

  # Run everything:
  export OPENAI_API_KEY="sk-..."
  python3 scripts/run_segmentation.py

  # Run just the free step:
  python3 scripts/run_segmentation.py --step 1

  # Run just note reclassification:
  python3 scripts/run_segmentation.py --step 2

  # Run just PDF segmentation:
  python3 scripts/run_segmentation.py --step 3

  # Test with small batch:
  python3 scripts/run_segmentation.py --step 3 --limit-files 2
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent
python = sys.executable


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run segmentation pipeline")
    parser.add_argument("--step", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--limit-notes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        # Step 1 is free, just show stats
        print("=" * 60)
        print("STEP 1: Rule-based TEI note extraction (FREE)")
        print("=" * 60)
        print("  Already completed: 43,455 notes extracted")
        print()

        # Steps 2+3 cost estimate
        subprocess.run([python, str(SCRIPTS / "segment_with_openai.py"),
                        "--dry-run"])
        return

    if args.step in ("1", "all"):
        print("=" * 60)
        print("STEP 1: Rule-based TEI note extraction (FREE)")
        print("=" * 60)
        subprocess.run([python, str(SCRIPTS / "extract_notes_tei.py")])

    if args.step in ("2", "all"):
        cmd = [python, str(SCRIPTS / "segment_with_openai.py"),
               "--phase", "notes"]
        if args.limit_notes:
            cmd += ["--limit-notes", str(args.limit_notes)]
        subprocess.run(cmd)

    if args.step in ("3", "all"):
        cmd = [python, str(SCRIPTS / "segment_with_openai.py"),
               "--phase", "pdfs"]
        if args.limit_files:
            cmd += ["--limit-files", str(args.limit_files)]
        subprocess.run(cmd)

    # Final stats
    import sqlite3
    db = SCRIPTS.parent / "db" / "diachronic.db"
    conn = sqlite3.connect(db)
    print(f"\n{'='*60}")
    print("Database status:")
    print(f"  Notes: {conn.execute('SELECT COUNT(*) FROM extracted_notes').fetchone()[0]:,}")
    try:
        segs = conn.execute("SELECT segment_type, COUNT(*) FROM segmented_content GROUP BY segment_type ORDER BY COUNT(*) DESC").fetchall()
        total = sum(c for _, c in segs)
        print(f"  Segmented content: {total:,}")
        for t, c in segs:
            print(f"    {t}: {c:,}")
    except Exception:
        print("  Segmented content: not yet created")
    conn.close()


if __name__ == "__main__":
    main()
