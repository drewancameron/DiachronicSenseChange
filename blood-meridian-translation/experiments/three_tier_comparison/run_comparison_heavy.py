#!/usr/bin/env python3
"""
Three-tier comparison on a syntactically HEAVY passage.

The mountain torrent passage: long polysyndetic sentences, relative clauses,
participial phrases, figurative language ("bones of trees assassinated").
This is where the full pipeline should outperform the bare prompt.
"""

import json
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

PASSAGE = """In the night they followed a mountain torrent in a wild gorge choked with mossy rocks and they rode under dark grottoes where the water dripped and spattered and tasted of iron and they saw the silver filaments of cascades divided upon the faces of distant buttes that appeared as signs and wonders in the heavens themselves so dark was the ground of their origins. They crossed the blackened wood of a burn and they rode through a region of cloven rock where great boulders lay halved with smooth uncentered faces and on the slopes of those ferric grounds old paths of fire and the blackened bones of trees assassinated in the mountain storms. On the day following they began to encounter holly and oak, hardwood forests much like those they had quit in their youth. In pockets on the north slopes hail lay nested like tectites among the leaves and the nights were cool."""


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


def build_tier1_prompt() -> str:
    return f"""Translate the following passage from Cormac McCarthy's Blood Meridian into Ancient Greek (Koine register with Attic vocabulary). Use polytonic orthography.

Output ONLY the Greek text.

{PASSAGE}"""


def build_tier3_prompt() -> str:
    """Full pipeline with structural parallels and vocab guidance."""
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

    prompt = f"""You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine with Attic vocabulary).

## Translation Rules
{rules}

{domain_notes}

## Vocabulary Notes (preferred, not mandatory)
{glossary}

## English Source
{PASSAGE}

## Structural Guidance

For each sentence we describe its grammar and show how translators of classical Greek prose handled structurally similar sentences. Use these as reference — not templates. Adapt to what sounds natural in Greek.

{guidance}

{construction_guide}

## Vocabulary Reference (from parallel corpus)
These are attested usages, not requirements. Use them if they fit; find better words if they don't.

{vocab}

## Instructions
1. Translate into Ancient Greek (Koine/Attic register).
2. Use the structural guidance as reference for construction choices, not as rigid templates.
3. McCarthy's "and...and...and" chains are CRITICAL to preserve as καί chains. Do NOT subordinate.
4. Preserve relative clauses (where, that) as ὅς/ἥ/ὅ + finite verb where natural.
5. Preserve fragments as fragments.
6. Every word must be attestable in Morpheus/LSJ.
7. Output ONLY the Greek text.
"""
    return prompt


def main():
    results = {}

    for tier, name, builder in [
        (1, "bare", build_tier1_prompt),
        (3, "full_pipeline", build_tier3_prompt),
    ]:
        print(f"\n{'='*60}")
        print(f"  Tier {tier}: {name}")
        print(f"{'='*60}")

        prompt = builder()

        prompt_path = OUT / f"heavy_tier{tier}_{name}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"  Prompt: {len(prompt)} chars → {prompt_path.name}")

        print(f"  Translating with Opus...")
        start = time.time()
        translation = call_opus(prompt)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(translation)} chars)")

        trans_path = OUT / f"heavy_tier{tier}_{name}_output.txt"
        trans_path.write_text(translation, encoding="utf-8")

        results[f"tier{tier}"] = {
            "name": name,
            "prompt_chars": len(prompt),
            "output_chars": len(translation),
            "time_s": round(elapsed, 1),
        }

    (OUT / "heavy_english_source.txt").write_text(PASSAGE, encoding="utf-8")

    summary = {
        "experiment": "Heavy passage comparison (mountain torrent)",
        "passage": "Blood Meridian, Ch. IV — mountain torrent and burn",
        "model": "claude-opus-4-20250514",
        "date": time.strftime("%Y-%m-%d"),
        "complexity": "heavy (avg 31.2 words/sent, 3 subordinations, polysyndetic)",
        "tiers": results,
    }
    (OUT / "heavy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Saved to {OUT}")


if __name__ == "__main__":
    main()
