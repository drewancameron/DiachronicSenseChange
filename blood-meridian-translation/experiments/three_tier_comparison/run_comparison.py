#!/usr/bin/env python3
"""
Three-tier translation comparison experiment.

Translates the same McCarthy passage at three prompt levels:
  Tier 1: Bare — minimal context (register + text only)
  Tier 2: Rules — adds grammatical rules and construction signposting
  Tier 3: Full — adds structural parallels, vocab guidance, polysemy, glossary

Saves all prompts and outputs for journal article use.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config"
GLOSSARY = ROOT / "glossary"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS))

PASSAGE = """He looked about at the dark forest in which they were bivouacked. He nodded toward the specimens he'd collected. These anonymous creatures, he said, may seem little or nothing in the world. Yet the smallest crumb can devour us. Any smallest thing beneath yon rock out of men's knowing. Only nature can enslave man and only when the existence of each last entity is routed out and made to stand naked before him will he be properly suzerain of the earth.

What's a suzerain?

A keeper. A keeper or overlord.

Why not say keeper then?

Because he is a special kind of keeper. A suzerain rules even where there are other rulers. His authority countermands local judgements.

Toadvine spat.

The judge placed his hands on the ground. He looked at his inquisitor. This is my claim, he said. And yet everywhere upon it are pockets of autonomous life. Autonomous. In order for it to be mine nothing must be permitted to occur upon it save by my dispensation."""


def call_opus(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    collected = []
    with client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)
    return "".join(collected).strip()


# ====================================================================
# TIER 1: Bare minimum
# ====================================================================

def build_tier1_prompt() -> str:
    return f"""Translate the following passage from Cormac McCarthy's Blood Meridian into Ancient Greek (Koine register with Attic vocabulary). Use polytonic orthography.

Output ONLY the Greek text.

{PASSAGE}"""


# ====================================================================
# TIER 2: Rules + construction signposting
# ====================================================================

def build_tier2_prompt() -> str:
    rules = (CONFIG / "translation_prompt_rules.md").read_text("utf-8")
    particles = (CONFIG / "particle_guide.md").read_text("utf-8")

    # Build basic construction labels
    from label_constructions import label_english
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', PASSAGE) if s.strip()]
    construction_notes = []
    for i, sent in enumerate(sents, 1):
        labels = label_english(sent)
        if labels:
            construction_notes.append(f"  Sentence {i}: {', '.join(labels)}")

    constr_text = "\n".join(construction_notes) if construction_notes else "  (none detected)"

    return f"""You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine with Attic vocabulary).

## Translation Rules
{rules}

{particles}

## Construction Labels Detected in Source
{constr_text}

## English Source
{PASSAGE}

## Instructions
1. Translate the full passage into Ancient Greek (Koine/Attic register).
2. Follow the construction labels: preserve relative clauses as ὅς/ἥ/ὅ + finite verb, fragments as fragments, etc.
3. McCarthy's comma splices → asyndeton. No δέ unless genuine contrast.
4. Output ONLY the Greek text."""


# ====================================================================
# TIER 3: Full pipeline (parallels, vocab, polysemy, glossary)
# ====================================================================

def build_tier3_prompt() -> str:
    import numpy as np
    from translate import (
        _load_best_index, build_sentence_guidance,
        build_vocab_guidance, load_rules, load_glossary,
        build_domain_notes,
    )
    from conditional_guide import identify_constructions, format_for_prompt

    features_arr, metadata = _load_best_index()

    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', PASSAGE) if s.strip()]
    guidance_parts = []
    for i, sent in enumerate(sents, 1):
        guidance_parts.append(build_sentence_guidance(sent, i, features_arr, metadata))
    guidance = "\n".join(guidance_parts)

    vocab = build_vocab_guidance(PASSAGE)
    rules = load_rules()
    glossary = load_glossary(PASSAGE)
    domain_notes = build_domain_notes(PASSAGE)

    construction_guide = ""
    try:
        findings = identify_constructions(PASSAGE)
        if findings:
            seen = set()
            unique = []
            for f in findings:
                key = (f["type"], f["text"][:40])
                if key not in seen:
                    seen.add(key)
                    unique.append(f)
            construction_guide = format_for_prompt(unique)
    except Exception:
        pass

    return f"""You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine with Attic vocabulary).

## Translation Rules
{rules}

{domain_notes}

## {glossary}

## English Source
{PASSAGE}

## Structural Guidance

For each sentence we describe its grammar and show how translators of classical Greek prose (Thucydides, Herodotus, Xenophon, Plato, Plutarch, Septuagint) handled structurally similar English sentences. Use these as models for construction choices.

{guidance}

{construction_guide}

{vocab}

## Instructions
1. Translate the full passage into Ancient Greek (Koine/Attic register).
2. Follow the structural guidance: match McCarthy's constructions where Greek allows it.
3. Use the vocabulary guidance for word choices — prefer attested forms from the parallel corpus.
4. McCarthy's comma splices → asyndeton. His "and...and...and" → καί chains. No δέ unless genuine contrast.
5. Preserve relative clauses as ὅς/ἥ/ὅ + finite verb. Do NOT convert to articular participles.
6. Preserve fragments as fragments. Do NOT expand into full sentences.
7. Every word must be attestable in Morpheus/LSJ.
7. Output ONLY the Greek text, one continuous paragraph matching McCarthy's formatting."""


def main():
    results = {}

    for tier, name, builder in [
        (1, "bare", build_tier1_prompt),
        (2, "rules_and_constructions", build_tier2_prompt),
        (3, "full_pipeline", build_tier3_prompt),
    ]:
        print(f"\n{'='*60}")
        print(f"  Tier {tier}: {name}")
        print(f"{'='*60}")

        prompt = builder()

        # Save prompt
        prompt_path = OUT / f"tier{tier}_{name}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"  Prompt: {len(prompt)} chars → {prompt_path.name}")

        # Call LLM
        print(f"  Translating with Opus...")
        start = time.time()
        translation = call_opus(prompt)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(translation)} chars)")

        # Save translation
        trans_path = OUT / f"tier{tier}_{name}_output.txt"
        trans_path.write_text(translation, encoding="utf-8")

        results[f"tier{tier}"] = {
            "name": name,
            "prompt_chars": len(prompt),
            "output_chars": len(translation),
            "time_s": round(elapsed, 1),
            "prompt_file": prompt_path.name,
            "output_file": trans_path.name,
        }

    # Save English source
    (OUT / "english_source.txt").write_text(PASSAGE, encoding="utf-8")

    # Save summary
    summary = {
        "experiment": "Three-tier translation comparison",
        "passage": "Blood Meridian, Ch. X — the judge's suzerain speech",
        "model": "claude-opus-4-20250514",
        "date": time.strftime("%Y-%m-%d"),
        "tiers": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Summary saved to summary.json")
    print(f"  All prompts and outputs saved to {OUT}")


if __name__ == "__main__":
    main()
