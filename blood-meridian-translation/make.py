#!/usr/bin/env python3
"""
Single-command build pipeline for the Blood Meridian translation.

Usage:
  python3 make.py              # full build (check + render, no auto-revise)
  python3 make.py full         # translate → auto-revise → render (end-to-end)
  python3 make.py render       # render only (skip checks)
  python3 make.py check        # checks only (no render)
  python3 make.py grew         # Grew grammar check only
  python3 make.py translate     # guided first-attempt translation (new passages)
  python3 make.py translate-dry # show translation prompt without calling LLM
  python3 make.py revise        # auto-revise: detect → prompt LLM → fix → verify
  python3 make.py revise-dry    # show what revise would do (no LLM calls)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

# Ensure opam env is available for Grew
OPAM_SWITCH = Path.home() / ".opam" / "4.14.2"
if OPAM_SWITCH.is_dir():
    opam_bin = str(OPAM_SWITCH / "bin")
    if opam_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = opam_bin + ":" + os.environ.get("PATH", "")
    os.environ["CAML_LD_LIBRARY_PATH"] = str(OPAM_SWITCH / "lib" / "stublibs")
    os.environ["OCAML_TOPLEVEL_PATH"] = str(OPAM_SWITCH / "lib" / "toplevel")


def run(label: str, script: str, args: list[str] | None = None,
        fatal: bool = False) -> bool:
    """Run a script and report result."""
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    cmd = [sys.executable, str(SCRIPTS / script)]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  ✗ {label} failed (exit {result.returncode})")
        if fatal:
            sys.exit(1)
        return False
    return True


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    start = time.time()

    print(f"╔{'═'*50}╗")
    print(f"║  Blood Meridian — build pipeline                ║")
    print(f"╚{'═'*50}╝")

    if mode in ("all", "fix"):
        run("Sync gloss text ↔ drafts", "sync_gloss_text.py")
        run("Re-index glosses to correct sentences", "reindex_glosses.py")
        run("Fix LLM ellipsis truncations", "fix_gloss_ellipses.py")

    if mode in ("all", "check"):
        run("Check paragraph boundaries", "check_boundaries.py")
        run("Review: grammar + glossary + register", "review_pipeline.py")

    if mode in ("all", "morpheus"):
        run("Morpheus: morphological grammar check", "morpheus_check.py")

    if mode in ("all", "constructions"):
        run("Construction mismatch check", "check_constructions.py")

    if mode in ("all", "grew"):
        run("Grew: treebank-backed grammar check", "grew_check.py",
            args=["--warnings-only"])

    if mode in ("translate", "translate-dry", "full"):
        translate_args = ["--all"]
        if mode == "translate-dry":
            translate_args.append("--dry-run")
        run("Guided translation (structural signposts)", "translate.py",
            args=translate_args, fatal=(mode == "full"))

    if mode in ("revise", "revise-dry", "signposts", "full"):
        run("Generate construction signposts", "generate_signposts.py")

    if mode in ("revise", "revise-dry", "full"):
        revise_args = []
        if mode == "revise-dry":
            revise_args.append("--dry-run")
        run("Auto-revise: detect → prompt LLM → fix → verify",
            "auto_revise.py", args=revise_args)

    if mode in ("all", "revise", "full", "render"):
        run("Mark loanwords and neologisms", "mark_loans.py")
        run("Render HTML", "render_passage.py")

        # Copy to final
        src = ROOT / "output" / "blood_meridian.html"
        dst = ROOT / "output" / "blood_meridian_final_reviewed.html"
        if src.exists():
            import shutil
            shutil.copy2(src, dst)

    elapsed = time.time() - start
    print(f"\n{'═'*50}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
