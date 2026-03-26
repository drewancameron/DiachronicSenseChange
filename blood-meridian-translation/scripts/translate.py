#!/usr/bin/env python3
"""
Guided translation: structural signposts FIRST, not as revision afterthought.

For each McCarthy sentence:
  1. Decompose structure (fingerprint)
  2. Find parallel EN↔GRC pairs from corpus
  3. Build grammatical description + parallels
  4. LLM translates WITH this guidance from the start
  5. Checkers verify
  6. If issues remain: targeted revision pass

Usage:
  python3 scripts/translate.py 001_see_the_child              # translate one passage
  python3 scripts/translate.py 001_see_the_child --dry-run     # show prompt only
  python3 scripts/translate.py --all                           # translate all passages
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PASSAGES = ROOT / "passages"
DRAFTS = ROOT / "drafts"
CONFIG = ROOT / "config"
GLOSSARY = ROOT / "glossary"

sys.path.insert(0, str(SCRIPTS))

import numpy as np

# Prefer stanza index if available (more accurate), fall back to regex
STANZA_INDEX = ROOT / "models" / "fingerprint_index"
FAST_INDEX = ROOT / "models" / "fingerprint_index_fast"

_index_type = None

def _load_best_index():
    global _index_type
    stanza_meta = STANZA_INDEX / "metadata.jsonl"
    if stanza_meta.exists():
        import subprocess
        n_lines = int(subprocess.check_output(["wc", "-l", str(stanza_meta)]).split()[0])
        if n_lines >= 10000:  # use stanza if it has decent coverage
            from build_fingerprint_index import load_index as load_stanza, fingerprint_stanza, fingerprint_label
            _index_type = "stanza"
            return load_stanza()

    from build_fingerprint_index_fast import load_index as load_fast
    _index_type = "fast"
    return load_fast()


def _fingerprint_query(text: str):
    """Fingerprint a query sentence using the best available method."""
    if _index_type == "stanza":
        from build_fingerprint_index import fingerprint_stanza, fingerprint_label, _get_en_nlp
        nlp = _get_en_nlp()
        doc = nlp(text)
        if doc.sentences:
            return fingerprint_stanza(doc.sentences[0]), fingerprint_label(doc.sentences[0])
        return np.zeros(16, dtype=np.float32), {}
    else:
        from build_fingerprint_index_fast import fingerprint, label
        return fingerprint(text), label(text)


# ====================================================================
# 1. Build per-sentence structural guidance
# ====================================================================

def describe_grammar(sent_text: str, lbl: dict) -> str:
    """Describe the grammar of an English sentence in plain language."""
    parts = []

    wc = lbl["word_count"]

    # Main clause type
    if not lbl.get("speech") and wc <= 4 and sent_text.strip().endswith("."):
        # Check for imperative
        first_word = sent_text.split()[0].lower() if sent_text.split() else ""
        if first_word not in ("he", "she", "it", "they", "the", "a", "this", "that", "his", "her"):
            parts.append(f"Imperative, {wc} words")
        else:
            parts.append(f"Short declarative, {wc} words")
    elif lbl["type"] == "fragment":
        parts.append(f"Verbless fragment, {wc} words")
    elif lbl["type"] == "simple":
        parts.append(f"Simple sentence, {wc} words")
    elif lbl["type"] == "compound":
        parts.append(f"Compound sentence (multiple independent clauses), {wc} words")
    elif lbl["type"] == "complex":
        parts.append(f"Complex sentence (main + subordinate clause), {wc} words")
    elif lbl["type"] == "compound_complex":
        parts.append(f"Compound-complex sentence, {wc} words")

    # Subordinate clauses
    if lbl.get("relative"):
        parts.append(f"{lbl['relative']} relative clause(s) — preserve as ὅς/ἥ/ὅ + finite verb")
    if lbl.get("conditional"):
        parts.append(f"conditional clause (εἰ/ἐάν)")
    if lbl.get("temporal"):
        parts.append(f"temporal clause")

    # Coordination
    if lbl.get("coordination", 0) >= 2:
        parts.append(f"{lbl['coordination']} coordinations — preserve καί chain (McCarthy's parataxis)")
    elif lbl.get("coordination", 0) == 1:
        parts.append(f"one coordination (καί)")

    # Voice
    if lbl.get("passive"):
        parts.append("passive voice")

    # Speech
    if lbl.get("speech"):
        parts.append("contains direct speech verb")

    # Comma splice detection
    if ", " in sent_text and lbl["type"] in ("compound", "simple"):
        clause_count = sent_text.count(",") + 1
        if clause_count >= 2:
            parts.append("comma splice → asyndeton in Greek (no δέ)")

    return ". ".join(parts) + "."


def find_parallels(sent_text: str, features_arr, metadata, k: int = 3) -> list[dict]:
    """Find structurally similar parallel pairs."""
    q, _ = _fingerprint_query(sent_text)
    dists = np.sqrt(np.sum((features_arr - q) ** 2, axis=1))
    top_idx = np.argsort(dists)[:k * 5]

    # Diverse source selection
    _, q_lbl = _fingerprint_query(sent_text)
    results = []
    seen_sources = set()

    for idx in top_idx:
        m = metadata[idx]
        src = m["source"]
        if src in seen_sources:
            continue
        # Skip metadata-only entries
        if len(m.get("english", "")) < 15 or len(m.get("greek", "")) < 15:
            continue
        seen_sources.add(src)
        results.append({
            "distance": round(float(dists[idx]), 3),
            "english": m["english"],
            "greek": m["greek"],
            "source": m["source"],
            "label": m.get("label", {}),
            "construction_labels": m.get("construction_labels", []),
        })
        if len(results) >= k:
            break

    return results


_taxonomy = None

def _load_taxonomy() -> dict:
    """Load construction taxonomy for name → pattern/example lookup."""
    global _taxonomy
    if _taxonomy is not None:
        return _taxonomy
    import yaml
    tax_path = CONFIG / "construction_taxonomy.yaml"
    if tax_path.exists():
        raw = yaml.safe_load(open(tax_path))
        _taxonomy = {}
        for category, items in raw.items():
            for item in items:
                _taxonomy[item["name"]] = item
    else:
        _taxonomy = {}
    return _taxonomy


def _construction_guidance(labels: list[str]) -> list[str]:
    """Look up construction labels in taxonomy and return guidance lines."""
    tax = _load_taxonomy()
    lines = []
    for lbl in labels:
        entry = tax.get(lbl)
        if entry:
            lines.append(f"  → **{lbl}**: {entry['greek_pattern']}")
            if entry.get("example_grc"):
                lines.append(f"    e.g. {entry['example_grc']}")
            elif entry.get("example"):
                lines.append(f"    e.g. {entry['example']}")
        else:
            lines.append(f"  → **{lbl}**")
    return lines


def build_sentence_guidance(sent_text: str, sent_idx: int,
                             features_arr, metadata) -> str:
    """Build structural guidance for one sentence."""
    _, lbl = _fingerprint_query(sent_text)
    grammar = describe_grammar(sent_text, lbl)
    parallels = find_parallels(sent_text, features_arr, metadata, k=2)

    # Get construction labels
    try:
        from label_constructions import label_english
        en_labels = label_english(sent_text)
    except Exception:
        en_labels = []

    lines = [f'### {sent_idx}. "{sent_text[:80]}{"..." if len(sent_text) > 80 else ""}"']
    lines.append(f"Grammar: {grammar}")

    # Add named construction guidance (only when we have something helpful)
    if en_labels:
        constr_lines = _construction_guidance(en_labels)
        if constr_lines:
            lines.extend(constr_lines)

    if parallels:
        lines.append("Parallels:")
        for p in parallels:
            pl = p.get("label", {})
            en_short = p["english"][:70]
            gr_short = p["greek"][:70]
            # Check if this parallel has construction labels
            p_labels = p.get("construction_labels", [])
            label_note = ""
            if p_labels:
                grc_labels = [l for l in p_labels if not l.startswith("EN:")]
                if grc_labels:
                    label_note = f" [{', '.join(grc_labels)}]"
            lines.append(f'  "{en_short}"')
            lines.append(f'    → {gr_short}{label_note}')
            lines.append(f'    [{p["source"][:30]}]')

    lines.append("")
    return "\n".join(lines)


# ====================================================================
# 2. Build full translation prompt
# ====================================================================

def load_rules() -> str:
    rules = (CONFIG / "translation_prompt_rules.md").read_text("utf-8")
    particles = (CONFIG / "particle_guide.md").read_text("utf-8")
    return rules + "\n\n" + particles


def load_glossary() -> str:
    glossary_path = GLOSSARY / "idf_glossary.json"
    if not glossary_path.exists():
        return ""
    data = json.load(open(glossary_path))
    lines = ["Locked term translations (MUST use these):"]
    for entry in data.get("terms", []):
        if entry.get("status") == "locked":
            lines.append(f"  {entry['english']} = {entry['greek']}")
    return "\n".join(lines)


def build_vocab_guidance(en_text: str) -> str:
    """Build vocabulary section from parallel corpus lookups."""
    try:
        from vocab_lookup import extract_content_words, lookup_word_in_corpus
    except ImportError:
        return ""

    content_words = extract_content_words(en_text)
    if not content_words:
        return ""

    entries = []
    for w in content_words:
        hits = lookup_word_in_corpus(w["lemma"], w["upos"], w["context"], max_results=2)
        if not hits:
            continue

        role_desc = f" — {w['context']}" if w['context'] else ""
        lines = [f"'{w['text']}' [{w['upos'].lower()}{role_desc}]:"]
        for hit in hits:
            en_short = hit["english"][:60]
            gr_short = hit["greek"][:60]
            lines.append(f"  [{hit['source'][:25]}] \"{en_short}\"")
            lines.append(f"    → {gr_short}")
        entries.append("\n".join(lines))

    if not entries:
        return ""

    return "## Vocabulary Guidance (from parallel corpus)\n\n" + "\n\n".join(entries)


def build_translation_prompt(passage_id: str, features_arr, metadata) -> str:
    """Build a guided first-attempt translation prompt."""
    # Load English source
    p_path = PASSAGES / f"{passage_id}.json"
    if not p_path.exists():
        return ""
    en_text = json.load(open(p_path)).get("text", "")
    if not en_text:
        return ""

    # Split into sentences
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]

    # Build per-sentence guidance
    guidance_parts = []
    for i, sent in enumerate(sents, 1):
        guidance_parts.append(
            build_sentence_guidance(sent, i, features_arr, metadata)
        )

    guidance = "\n".join(guidance_parts)
    vocab = build_vocab_guidance(en_text)
    rules = load_rules()
    glossary = load_glossary()

    # Build construction guide (conditionals, temporals, modals)
    construction_guide = ""
    try:
        from conditional_guide import identify_constructions, format_for_prompt
        findings = identify_constructions(en_text)
        if findings:
            # Deduplicate by (type, text[:40])
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

## {glossary}

## English Source
{en_text}

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
7. Output ONLY the Greek text, one continuous paragraph matching McCarthy's formatting.
"""
    return prompt


def build_revision_prompt(passage_id: str, en_text: str, grc_text: str,
                           issues: str, features_arr, metadata) -> str:
    """Build a revision prompt with structural guidance + specific issues."""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]

    guidance_parts = []
    for i, sent in enumerate(sents, 1):
        guidance_parts.append(
            build_sentence_guidance(sent, i, features_arr, metadata)
        )

    guidance = "\n".join(guidance_parts)
    rules = load_rules()
    glossary = load_glossary()

    prompt = f"""You are revising an Ancient Greek (Koine/Attic) translation of McCarthy's Blood Meridian.

## Translation Rules
{rules}

## {glossary}

## English Source
{en_text}

## Current Greek Translation
{grc_text}

## Structural Guidance
{guidance}

## Issues to Fix
{issues}

## Instructions
1. Fix the flagged issues while preserving everything that is correct.
2. Follow the structural guidance for construction choices.
3. Output ONLY the complete revised Greek text.
4. Every word must be attestable in Morpheus/LSJ.
"""
    return prompt


# ====================================================================
# 3. Call LLM
# ====================================================================

def call_llm(prompt: str, model: str = "claude-opus-4-20250514") -> str:
    import anthropic
    client = anthropic.Anthropic()
    collected = []
    with client.messages.stream(
        model=model,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)
    return "".join(collected).strip()


# ====================================================================
# 4. Main
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="claude-opus-4-20250514")
    args = parser.parse_args()

    # Load fingerprint index (stanza if available, else regex)
    features_arr, metadata = _load_best_index()
    print(f"Loaded fingerprint index ({_index_type}): {len(metadata)} sentences")

    if args.all:
        passage_ids = sorted(p.stem for p in PASSAGES.glob("*.json"))
    elif args.passages:
        passage_ids = args.passages
    else:
        parser.print_help()
        return

    for pid in passage_ids:
        print(f"\n{'='*60}")
        print(f"  {pid}")
        print(f"{'='*60}")

        # Check if draft already exists
        draft_path = DRAFTS / pid / "primary.txt"
        if draft_path.exists():
            existing = draft_path.read_text("utf-8").strip()
            if existing and not args.dry_run:
                print(f"  Draft exists ({len(existing)} chars). Use auto_revise.py to revise.")
                continue

        prompt = build_translation_prompt(pid, features_arr, metadata)
        if not prompt:
            print(f"  No source text found")
            continue

        if args.dry_run:
            print(prompt)
            print(f"\n  [Prompt: {len(prompt)} chars]")
            continue

        print(f"  Translating with {args.model}...")
        result = call_llm(prompt, model=args.model)

        # Save
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(result + "\n", encoding="utf-8")
        print(f"  ✓ Saved ({len(result)} chars)")


if __name__ == "__main__":
    main()
